# adapters/torchscript_serving.py

"""
TorchScript adapter for PyTorch model serving.

This adapter converts trained PyTorch models to TorchScript format for
optimized inference with features like:
- Just-in-time (JIT) compilation
- Model optimization for production
- CPU and GPU inference
- Efficient serialization
- Platform independence
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import torch
    import torch.jit

try:
    import torch
    import torch.jit

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from ..interfaces.serving_interface import (
    IModelServingAdapter,
    ModelInfo,
    PredictionResult,
    ServingStats,
)
from ..schemas.enums import ModelType, TimeFrame
from ..core.config import get_settings
from ..utils.device_manager import create_device_manager_from_settings
from common.logger.logger_factory import LoggerFactory


class TorchScriptServingAdapter(IModelServingAdapter):
    """
    TorchScript adapter for optimized PyTorch model serving.

    This adapter provides production-ready PyTorch model serving using
    TorchScript compilation with features like:
    - JIT compilation for performance
    - Model optimization
    - Cross-platform deployment
    - Efficient batch processing
    - Memory management
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TorchScript serving adapter

        Args:
            config: Configuration dictionary with TorchScript settings
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for TorchScript serving adapter. "
                "Install with: pip install torch"
            )

        super().__init__(config)
        self.settings = get_settings()

        # Initialize device manager
        self.device_manager = create_device_manager_from_settings()

        # TorchScript-specific configuration
        # Use centralized device configuration unless explicitly overridden
        config_device = config.get("device")
        if config_device is None:
            self.device = self.device_manager.device
        else:
            self.device = config_device
            if config_device != self.device_manager.device:
                self.logger.warning(
                    f"Config device '{config_device}' differs from settings device '{self.device_manager.device}'"
                )

        self.optimize_for_inference = config.get("optimize_for_inference", True)
        self.enable_fusion = config.get("enable_fusion", True)
        self.compile_mode = config.get("compile_mode", "trace")  # "trace" or "script"
        self.max_models_in_memory = config.get("max_models_in_memory", 5)
        self.model_timeout_seconds = config.get("model_timeout_seconds", 3600)

        # Model storage
        self._loaded_models: Dict[str, Any] = {}
        self._model_info: Dict[str, ModelInfo] = {}
        self._model_configs: Dict[str, Dict[str, Any]] = {}

        # Validate device availability
        if self.device != "cpu" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        self.logger.info(
            f"TorchScriptServingAdapter initialized - Device: {self.device}, "
            f"Force CPU: {self.device_manager.force_cpu}, "
            f"Max models: {self.max_models_in_memory}"
        )

    async def initialize(self) -> bool:
        """
        Initialize the TorchScript serving adapter.

        Returns:
            bool: True if initialization successful
        """
        try:
            # Verify PyTorch installation
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")

            # Test device
            if self.device != "cpu":
                test_tensor = torch.tensor([1.0]).to(self.device)
                _ = test_tensor + 1

            # Set optimization settings
            if self.optimize_for_inference:
                torch.jit.set_fusion_strategy([("STATIC", 20), ("DYNAMIC", 20)])

            self.logger.info("TorchScript serving adapter initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize TorchScript adapter: {e}")
            return False

    async def load_model(
        self,
        model_path: str,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> ModelInfo:
        """
        Load and compile a model to TorchScript format.

        Args:
            model_path: Path to the trained model
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            model_config: Optional model configuration

        Returns:
            ModelInfo: Information about the loaded model
        """
        try:
            model_id = self.generate_model_id(symbol, timeframe, model_type)
            model_path_obj = Path(model_path)

            # Check if model is already loaded
            if model_id in self._loaded_models:
                self.logger.info(f"Model {model_id} already loaded")
                return self._model_info[model_id]

            # Ensure we don't exceed memory limits
            if len(self._loaded_models) >= self.max_models_in_memory:
                await self._evict_oldest_model()

            # Try to load pre-converted TorchScript model first
            torchscript_path = await self._load_existing_torchscript(model_path_obj)

            if not torchscript_path:
                # Fall back to converting the model
                torchscript_path = await self._convert_to_torchscript(
                    model_path_obj, model_id, model_type, model_config
                )

            # Load the TorchScript model
            scripted_model = torch.jit.load(
                str(torchscript_path), map_location=self.device
            )
            scripted_model.eval()

            # Optimize for inference if enabled
            if self.optimize_for_inference:
                scripted_model = torch.jit.optimize_for_inference(scripted_model)

            # Store model and metadata
            self._loaded_models[model_id] = scripted_model
            self._model_configs[model_id] = model_config or {}

            # Calculate memory usage
            memory_usage = self._estimate_model_memory_usage(scripted_model)

            # Create model info
            model_info = ModelInfo(
                model_id=model_id,
                symbol=symbol,
                timeframe=timeframe.value,
                model_type=model_type.value,
                model_path=str(torchscript_path),
                is_loaded=True,
                loaded_at=datetime.now(),
                memory_usage_mb=memory_usage,
                version="1.0",
                metadata={
                    "device": self.device,
                    "torchscript_path": str(torchscript_path),
                    "original_path": model_path,
                    "compile_mode": self.compile_mode,
                    "optimized": self.optimize_for_inference,
                },
            )

            self._model_info[model_id] = model_info
            self.logger.info(f"Successfully loaded TorchScript model: {model_id}")

            return model_info

        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            raise

    async def predict(
        self,
        model_id: str,
        input_data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
        n_steps: int = 1,
        **kwargs,
    ) -> PredictionResult:
        """
        Make predictions using TorchScript model.

        Args:
            model_id: Model identifier
            input_data: Input data for prediction
            n_steps: Number of prediction steps
            **kwargs: Additional prediction parameters

        Returns:
            PredictionResult: Prediction results
        """
        start_time = datetime.now()

        try:
            # Check if model is loaded
            if model_id not in self._loaded_models:
                raise ValueError(f"Model {model_id} not loaded")

            model = self._loaded_models[model_id]
            model_info = self._model_info[model_id]
            model_config = self._model_configs[model_id]

            # Preprocess input data
            input_tensor = await self._preprocess_input(
                input_data, model_info, model_config
            )

            # Move to device
            input_tensor = input_tensor.to(self.device)

            # Make prediction
            with torch.no_grad():
                if self.compile_mode == "trace":
                    predictions = model(input_tensor)
                else:
                    predictions = model.forward(input_tensor)

            # Post-process predictions
            result = await self._postprocess_predictions(
                predictions, input_data, model_info, model_config, n_steps
            )

            # Calculate inference time
            inference_time = (datetime.now() - start_time).total_seconds() * 1000

            # Update statistics
            self._update_stats(True, inference_time)

            prediction_result = PredictionResult(
                success=True,
                predictions=result["predictions"],
                current_price=result.get("current_price"),
                predicted_change_pct=result.get("predicted_change_pct"),
                confidence_score=result.get("confidence_score"),
                model_info={
                    "model_id": model_id,
                    "device": self.device,
                    "inference_time_ms": inference_time,
                    **model_info.metadata,
                },
                inference_time_ms=inference_time,
            )

            return prediction_result

        except Exception as e:
            # Calculate inference time even for failures
            inference_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(False, inference_time)

            error_msg = f"Prediction failed for model {model_id}: {e}"
            self.logger.error(error_msg)

            return PredictionResult(
                success=False,
                predictions=[],
                error=error_msg,
                inference_time_ms=inference_time,
            )

    async def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from memory.

        Args:
            model_id: Model identifier

        Returns:
            bool: True if unloading successful
        """
        try:
            if model_id in self._loaded_models:
                del self._loaded_models[model_id]
                del self._model_info[model_id]
                del self._model_configs[model_id]

                # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.logger.info(f"Successfully unloaded model: {model_id}")
                return True
            else:
                self.logger.warning(f"Model {model_id} not found for unloading")
                return False

        except Exception as e:
            self.logger.error(f"Failed to unload model {model_id}: {e}")
            return False

    async def list_loaded_models(self) -> List[ModelInfo]:
        """
        List all loaded models.

        Returns:
            List[ModelInfo]: List of loaded model information
        """
        return list(self._model_info.values())

    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Optional[ModelInfo]: Model information or None if not found
        """
        return self._model_info.get(model_id)

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the serving adapter.

        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            # Test basic PyTorch operations
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).to(self.device)
            _ = test_tensor * 2

            # Check memory usage
            if torch.cuda.is_available() and self.device != "cpu":
                memory_allocated = torch.cuda.memory_allocated(self.device)
                memory_reserved = torch.cuda.memory_reserved(self.device)
            else:
                memory_allocated = memory_reserved = 0

            return {
                "status": "healthy",
                "adapter_type": "torchscript",
                "device": self.device,
                "torch_version": torch.__version__,
                "models_loaded": len(self._loaded_models),
                "memory_allocated_mb": memory_allocated / (1024 * 1024),
                "memory_reserved_mb": memory_reserved / (1024 * 1024),
                "cuda_available": torch.cuda.is_available(),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "adapter_type": "torchscript",
                "timestamp": datetime.now().isoformat(),
            }

    async def shutdown(self) -> bool:
        """
        Shutdown the serving adapter and cleanup resources.

        Returns:
            bool: True if shutdown successful
        """
        try:
            # Unload all models
            model_ids = list(self._loaded_models.keys())
            for model_id in model_ids:
                await self.unload_model(model_id)

            # Clear caches
            self._loaded_models.clear()
            self._model_info.clear()
            self._model_configs.clear()

            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info("TorchScript serving adapter shutdown completed")
            return True

        except Exception as e:
            self.logger.error(f"Error during TorchScript adapter shutdown: {e}")
            return False

    # Private helper methods

    async def _convert_to_torchscript(
        self,
        model_path: Path,
        model_id: str,
        model_type: ModelType,
        model_config: Optional[Dict[str, Any]],
    ) -> Path:
        """
        Convert a trained model to TorchScript format.

        Args:
            model_path: Path to original model
            model_id: Model identifier
            model_type: Model type
            model_config: Model configuration

        Returns:
            Path: Path to TorchScript model
        """
        # Create TorchScript directory
        torchscript_dir = self.settings.models_dir / "torchscript" / model_id
        torchscript_dir.mkdir(parents=True, exist_ok=True)

        torchscript_path = torchscript_dir / "model.pt"

        # Check if TorchScript version already exists
        if torchscript_path.exists():
            self.logger.info(f"TorchScript model already exists: {torchscript_path}")
            return torchscript_path

        try:
            # Load the original model
            original_model = await self._load_original_model(
                model_path, model_type, model_config
            )

            # Create example input for tracing
            example_input = self._create_example_input(original_model, model_config)

            # Convert to TorchScript
            if self.compile_mode == "trace":
                scripted_model = torch.jit.trace(
                    original_model,
                    example_input,
                    strict=False,  # Allow dict outputs from models like PatchTSMixer
                )
            else:  # script mode
                scripted_model = torch.jit.script(original_model)

            # Save TorchScript model
            scripted_model.save(str(torchscript_path))

            # Save conversion metadata
            metadata = {
                "original_path": str(model_path),
                "torchscript_path": str(torchscript_path),
                "model_type": model_type.value,
                "compile_mode": self.compile_mode,
                "device": self.device,
                "converted_at": datetime.now().isoformat(),
                "model_config": model_config,
            }

            metadata_path = torchscript_dir / "conversion_metadata.json"
            with open(metadata_path, "w") as f:
                import json

                json.dump(metadata, f, indent=2, default=str)

            self.logger.info(
                f"Successfully converted model to TorchScript: {torchscript_path}"
            )
            return torchscript_path

        except Exception as e:
            self.logger.error(f"Failed to convert model to TorchScript: {e}")
            raise

    async def _load_existing_torchscript(self, model_path: Path) -> Optional[Path]:
        """
        Check if a pre-converted TorchScript model exists and return its path.

        Args:
            model_path: Path to the model directory

        Returns:
            Path to TorchScript model if found, None otherwise
        """
        try:
            # List of possible TorchScript file names (in order of preference)
            torchscript_names = [
                "model_torchscript.pt",
                "scripted_model.pt",
                "model.pt",
            ]

            for torchscript_name in torchscript_names:
                torchscript_path = model_path / torchscript_name
                if torchscript_path.exists():
                    self.logger.info(
                        f"Found existing TorchScript model: {torchscript_path}"
                    )
                    return torchscript_path

            return None

        except Exception as e:
            self.logger.warning(f"Error checking for existing TorchScript model: {e}")
            return None

    async def _load_original_model(
        self,
        model_path: Path,
        model_type: ModelType,
        model_config: Optional[Dict[str, Any]],
    ) -> Any:
        """Load the original PyTorch model for conversion."""
        try:
            # Use ModelFactory approach similar to SimpleServingAdapter
            from ..models.model_factory import ModelFactory
            import json

            self.logger.info(
                f"Loading original model from: {model_path} with type: {model_type}"
            )

            # Read model config if available
            config_path = model_path / "model_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    saved_config = json.load(f)
                # Merge with provided config
                final_config = {**saved_config, **(model_config or {})}
            else:
                final_config = model_config or {}

            # Create model instance using our factory
            model_instance = ModelFactory.create_model(model_type, final_config)
            if model_instance is None:
                raise ValueError(f"Could not create model of type {model_type}")

            # Load the model using our standard interface
            model_instance.load_model(model_path)

            # Return the underlying PyTorch model for TorchScript conversion
            if hasattr(model_instance, "model") and model_instance.model is not None:
                # Ensure the model is on CPU for TorchScript conversion
                pytorch_model = model_instance.model
                if hasattr(pytorch_model, "cpu"):
                    pytorch_model = pytorch_model.cpu()
                # Also move any sub-modules to CPU
                if hasattr(pytorch_model, "to"):
                    pytorch_model = pytorch_model.to("cpu")
                return pytorch_model
            else:
                # For models without a separate .model attribute
                if hasattr(model_instance, "cpu"):
                    model_instance = model_instance.cpu()
                if hasattr(model_instance, "to"):
                    model_instance = model_instance.to("cpu")
                return model_instance

        except Exception as e:
            self.logger.error(f"Failed to load original model: {e}")
            raise

    def _create_example_input(
        self, model: Any, model_config: Optional[Dict[str, Any]]
    ) -> Any:
        """Create example input tensor for model tracing."""
        config = model_config or {}

        # Default configuration
        batch_size = 1
        context_length = config.get("context_length", 64)
        input_dim = config.get("input_dim", 29)

        # Create random input tensor
        example_input = torch.randn(batch_size, context_length, input_dim)
        return example_input

    async def _preprocess_input(
        self,
        input_data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
        model_info: ModelInfo,
        model_config: Dict[str, Any],
    ) -> Any:
        """Preprocess input data for TorchScript model with feature engineering."""
        try:
            # First, convert input to DataFrame format for feature engineering
            if isinstance(input_data, pd.DataFrame):
                df = input_data.copy()
            elif isinstance(input_data, dict):
                if "close" in input_data:
                    df = pd.DataFrame([input_data])
                else:
                    raise ValueError(
                        "Invalid input data format - dict must contain 'close' key"
                    )
            elif isinstance(input_data, np.ndarray):
                # Convert numpy array to DataFrame with appropriate column names
                if input_data.ndim == 1:
                    df = pd.DataFrame({"close": input_data})
                elif input_data.ndim == 2:
                    if input_data.shape[1] >= 5:
                        columns = ["open", "high", "low", "close", "volume"]
                        if input_data.shape[1] > 5:
                            columns.extend(
                                [f"feature_{i}" for i in range(5, input_data.shape[1])]
                            )
                        df = pd.DataFrame(
                            input_data, columns=columns[: input_data.shape[1]]
                        )
                    else:
                        df = pd.DataFrame(
                            input_data,
                            columns=[
                                f"feature_{i}" for i in range(input_data.shape[1])
                            ],
                        )
                else:
                    raise ValueError(
                        f"Unsupported numpy array dimensions: {input_data.ndim}"
                    )
            else:
                raise ValueError(f"Unsupported input data type: {type(input_data)}")

            # Apply feature engineering to match the TorchScript model's expected input
            feature_engineering = self._get_model_feature_engineering(
                model_info.model_id
            )
            if feature_engineering is not None:
                try:
                    # Apply feature engineering transform
                    processed_df = feature_engineering.transform(df)
                    self.logger.info(
                        f"TorchScript: Applied feature engineering, shape: {df.shape} -> {processed_df.shape}"
                    )
                    df = processed_df
                except Exception as fe_error:
                    self.logger.warning(
                        f"Feature engineering failed: {fe_error}. Using raw data."
                    )
                    # Continue with original DataFrame
            else:
                self.logger.warning(
                    "No feature engineering found for TorchScript model"
                )

            # Get the expected input format from the model configuration
            model_config = self._model_configs.get(model_info.model_id, {})
            context_length = model_config.get("context_length", 64)

            # Ensure we have enough data for the context window
            if len(df) < context_length:
                # Repeat the data if we don't have enough
                repeat_count = (context_length // len(df)) + 1
                df = pd.concat([df] * repeat_count, ignore_index=True)
                df = df.tail(context_length)
                self.logger.info(
                    f"TorchScript: Repeated data to meet context length: {len(df)}"
                )
            else:
                # Use the last context_length rows
                df = df.tail(context_length)

            self.logger.info(f"TorchScript: Using {len(df)} rows for context window")

            # Clean the DataFrame (drop non-numeric columns)
            columns_to_drop = []
            for col in df.columns:
                col_dtype = str(df[col].dtype)
                col_name_lower = col.lower()

                # Drop obvious datetime columns
                if any(
                    dt_keyword in col_name_lower
                    for dt_keyword in ["timestamp", "date", "datetime", "time"]
                ):
                    columns_to_drop.append(col)
                    continue

                # Handle datetime64 types
                if "datetime64" in col_dtype:
                    columns_to_drop.append(col)
                    continue

                # Handle object columns
                if col_dtype == "object":
                    try:
                        # Check if it's timestamps
                        sample_values = df[col].dropna().head(5)
                        if len(sample_values) > 0:
                            first_val = sample_values.iloc[0]
                            if (
                                hasattr(first_val, "strftime")
                                or str(type(first_val)).lower().find("timestamp") != -1
                            ):
                                columns_to_drop.append(col)
                                continue

                        # Try to convert to numeric
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                        if df[col].isna().all():
                            columns_to_drop.append(col)

                    except Exception as e:
                        self.logger.warning(f"Failed to process column {col}: {e}")
                        columns_to_drop.append(col)

            # Drop problematic columns
            if columns_to_drop:
                self.logger.info(f"TorchScript: Dropping columns: {columns_to_drop}")
                df = df.drop(columns=columns_to_drop)

            # Ensure we have numeric data only
            df = df.select_dtypes(include=[np.number])

            if df.empty:
                raise ValueError("No numeric columns remaining after preprocessing")

            # Convert to numpy array
            data_array = df.values.astype(np.float32)

            self.logger.info(
                f"TorchScript: Final data shape for model: {data_array.shape}"
            )

            # Ensure correct shape for TorchScript model (batch_size, sequence_length, features)
            if data_array.ndim == 1:
                data_array = data_array.reshape(1, 1, -1)  # (1, 1, features)
            elif data_array.ndim == 2:
                # Assume shape is (sequence_length, features)
                data_array = data_array.reshape(
                    1, data_array.shape[0], data_array.shape[1]
                )  # (1, seq_len, features)

            # Convert to tensor
            input_tensor = torch.from_numpy(data_array).float()

            return input_tensor

        except Exception as e:
            self.logger.error(f"Failed to preprocess input: {e}")
            raise

    async def _postprocess_predictions(
        self,
        predictions: Any,
        input_data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
        model_info: ModelInfo,
        model_config: Dict[str, Any],
        n_steps: int,
    ) -> Dict[str, Any]:
        """Post-process model predictions."""
        try:
            # Handle different prediction formats (tensor or dict)
            if isinstance(predictions, dict):
                # TorchScript model returns dict (from strict=False tracing)
                self.logger.info("TorchScript: Processing dict output from model")

                # Extract prediction tensor from dict
                if "prediction" in predictions:
                    pred_tensor = predictions["prediction"]
                elif "last_hidden_state" in predictions:
                    pred_tensor = predictions["last_hidden_state"]
                elif "logits" in predictions:
                    pred_tensor = predictions["logits"]
                else:
                    # Take the first tensor value from the dict
                    pred_tensor = next(iter(predictions.values()))
                    self.logger.warning(
                        f"Using first dict value as prediction: {list(predictions.keys())}"
                    )

                # Convert tensor to numpy
                pred_array = pred_tensor.detach().cpu().numpy()

            else:
                # Direct tensor output
                pred_array = predictions.detach().cpu().numpy()

            # Extract predictions (assume last n_steps are the predictions)
            self.logger.info(
                f"TorchScript: Prediction tensor shape: {pred_array.shape}"
            )

            if pred_array.ndim > 2:
                # Handle 3D tensors: (batch, sequence, features) -> take last n_steps of sequence
                predictions_array = pred_array[
                    0, -n_steps:, 0
                ]  # First batch, last n_steps, first feature
            elif pred_array.ndim == 2:
                # Handle 2D tensors: (batch, sequence) or (sequence, features)
                if pred_array.shape[0] == 1:
                    # (1, sequence) - take last n_steps
                    predictions_array = pred_array[0, -n_steps:]
                else:
                    # (sequence, features) - take last n_steps, first feature
                    predictions_array = pred_array[-n_steps:, 0]
            else:
                # Handle 1D tensors: (sequence,) - take last n_steps
                predictions_array = pred_array[-n_steps:]

            # Ensure we have scalar values for predictions
            predictions_list = predictions_array.flatten().tolist()

            # Make sure we have exactly n_steps predictions
            if len(predictions_list) > n_steps:
                predictions_list = predictions_list[:n_steps]
            elif len(predictions_list) < n_steps:
                # Pad with the last prediction if we don't have enough
                last_pred = predictions_list[-1] if predictions_list else 0.0
                predictions_list.extend([last_pred] * (n_steps - len(predictions_list)))

            self.logger.info(
                f"TorchScript: Final predictions ({len(predictions_list)}): {predictions_list}"
            )

            # Calculate current price and change percentage
            current_price = None
            predicted_change_pct = None

            if isinstance(input_data, (pd.DataFrame, np.ndarray)):
                if (
                    isinstance(input_data, pd.DataFrame)
                    and "close" in input_data.columns
                ):
                    current_price = float(input_data["close"].iloc[-1])
                elif isinstance(input_data, np.ndarray):
                    current_price = float(
                        input_data[-1, -1]
                    )  # Last value of last feature

                if current_price and predictions_list:
                    try:
                        first_prediction = float(predictions_list[0])
                        predicted_change_pct = (
                            (first_prediction - current_price) / current_price
                        ) * 100
                    except (ValueError, TypeError, IndexError) as e:
                        self.logger.warning(
                            f"Could not calculate change percentage: {e}"
                        )
                        predicted_change_pct = None

            # Simple confidence score (can be improved)
            confidence_score = 0.8  # Placeholder

            return {
                "predictions": predictions_list,
                "current_price": current_price,
                "predicted_change_pct": predicted_change_pct,
                "confidence_score": confidence_score,
            }

        except Exception as e:
            self.logger.error(f"Failed to postprocess predictions: {e}")
            raise

    def _get_model_feature_engineering(self, model_id: str) -> Any:
        """Get the feature engineering instance for a loaded model."""
        try:
            # Try to get from model info metadata first
            if model_id in self._model_info:
                model_info = self._model_info[model_id]
                model_path = Path(model_info.model_path)

                # Try to load feature engineering from pickle file
                fe_path = model_path / "feature_engineering.pkl"
                if fe_path.exists():
                    try:
                        import pickle

                        with open(fe_path, "rb") as f:
                            feature_engineering = pickle.load(f)
                        return feature_engineering
                    except Exception as e:
                        self.logger.warning(
                            f"Could not load pickled feature engineering: {e}"
                        )

                # Try to recreate from config
                fe_config_path = model_path / "feature_engineering_config.json"
                if fe_config_path.exists():
                    try:
                        import json

                        with open(fe_config_path, "r") as f:
                            fe_config = json.load(f)

                        # Recreate feature engineering from config
                        from ..data.feature_engineering import BasicFeatureEngineering

                        feature_engineering = BasicFeatureEngineering(
                            feature_columns=fe_config.get("feature_columns", []),
                            target_column=fe_config.get("target_column", "close"),
                            add_technical_indicators=fe_config.get(
                                "add_technical_indicators", True
                            ),
                            add_datetime_features=fe_config.get(
                                "add_datetime_features", False
                            ),
                            normalize_features=fe_config.get(
                                "normalize_features", True
                            ),
                        )

                        # Set fitted state
                        feature_engineering.fitted_feature_names = fe_config.get(
                            "fitted_feature_names", []
                        )
                        feature_engineering.is_fitted = True

                        self.logger.info(
                            "Recreated feature engineering from config for TorchScript"
                        )
                        return feature_engineering

                    except Exception as fe_error:
                        self.logger.warning(
                            f"Could not recreate feature engineering: {fe_error}"
                        )

            return None

        except Exception as e:
            self.logger.warning(
                f"Failed to get feature engineering for model {model_id}: {e}"
            )
            return None

    def _estimate_model_memory_usage(self, model: Any) -> float:
        """
        Estimate memory usage of a TorchScript model in MB.

        Args:
            model: TorchScript model

        Returns:
            float: Estimated memory usage in MB
        """
        try:
            param_size = 0
            buffer_size = 0

            for param in model.parameters():
                param_size += param.nelement() * param.element_size()

            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

            model_size_mb = (param_size + buffer_size) / (1024 * 1024)
            return model_size_mb

        except Exception:
            return 0.0

    async def _evict_oldest_model(self) -> None:
        """Evict the oldest loaded model to free memory."""
        if not self._model_info:
            return

        # Find oldest model
        oldest_model_id = min(
            self._model_info.keys(),
            key=lambda k: self._model_info[k].loaded_at or datetime.min,
        )

        await self.unload_model(oldest_model_id)
        self.logger.info(f"Evicted oldest model: {oldest_model_id}")
