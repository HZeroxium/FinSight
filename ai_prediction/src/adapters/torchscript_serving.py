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

        # TorchScript-specific configuration
        self.device = config.get("device", "cpu")
        self.optimize_for_inference = config.get("optimize_for_inference", True)
        self.enable_fusion = config.get("enable_fusion", True)
        self.compile_mode = config.get("compile_mode", "trace")  # "trace" or "script"
        self.max_models_in_memory = config.get("max_models_in_memory", 5)
        self.model_timeout_seconds = config.get("model_timeout_seconds", 3600)

        # Model storage
        self._loaded_models: Dict[str, Any] = {}
        self._model_info: Dict[str, ModelInfo] = {}
        self._model_configs: Dict[str, Dict[str, Any]] = {}

        # Ensure device is available
        if self.device != "cpu" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        self.logger.info(
            f"TorchScriptServingAdapter initialized - Device: {self.device}, "
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

            # Convert and load model to TorchScript
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
                scripted_model = torch.jit.trace(original_model, example_input)
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

    async def _load_original_model(
        self,
        model_path: Path,
        model_type: ModelType,
        model_config: Optional[Dict[str, Any]],
    ) -> Any:
        """Load the original PyTorch model for conversion."""
        # This would need to be implemented based on your model types
        # For now, assume the model can be loaded directly
        try:
            # Try loading PyTorch state dict
            checkpoint_path = model_path / "model.pt"
            if not checkpoint_path.exists():
                checkpoint_path = model_path / "pytorch_model.bin"

            if checkpoint_path.exists():
                state_dict = torch.load(checkpoint_path, map_location="cpu")

                # Create model instance (this needs to be adapted per model type)
                from ..models.model_factory import ModelFactory

                model = ModelFactory.create_model(model_type, model_config or {})

                if hasattr(model, "model") and hasattr(model.model, "load_state_dict"):
                    model.model.load_state_dict(state_dict)
                    return model.model
                elif hasattr(model, "load_state_dict"):
                    model.load_state_dict(state_dict)
                    return model
                else:
                    raise ValueError(
                        f"Cannot load state dict for model type: {model_type}"
                    )
            else:
                raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

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
        """Preprocess input data for TorchScript model."""
        try:
            # Convert to numpy array
            if isinstance(input_data, pd.DataFrame):
                data_array = input_data.values.astype(np.float32)
            elif isinstance(input_data, dict):
                # Convert dict to array (assumes it's time series data)
                if "close" in input_data:
                    data_array = np.array(
                        input_data["close"], dtype=np.float32
                    ).reshape(-1, 1)
                else:
                    raise ValueError("Invalid input data format")
            else:
                data_array = np.array(input_data, dtype=np.float32)

            # Ensure correct shape (batch_size, sequence_length, features)
            if data_array.ndim == 1:
                data_array = data_array.reshape(1, -1, 1)
            elif data_array.ndim == 2:
                if data_array.shape[0] == 1:
                    # Already batched
                    data_array = data_array.reshape(
                        1, data_array.shape[0], data_array.shape[1]
                    )
                else:
                    # Add batch dimension
                    data_array = data_array.reshape(
                        1, data_array.shape[0], data_array.shape[1]
                    )

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
            # Convert to numpy
            pred_array = predictions.detach().cpu().numpy()

            # Extract predictions (assume last n_steps are the predictions)
            if pred_array.ndim > 1:
                predictions_list = pred_array[0, -n_steps:].tolist()
            else:
                predictions_list = pred_array[-n_steps:].tolist()

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
                    predicted_change_pct = (
                        (predictions_list[0] - current_price) / current_price
                    ) * 100

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
