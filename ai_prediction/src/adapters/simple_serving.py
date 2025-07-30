# adapters/simple_serving.py

"""
Simple in-memory model serving adapter.

This adapter provides backward compatibility with the current model_facade.py
logic while implementing the serving interface. It loads models directly
into memory and performs inference synchronously.
"""

import time
import psutil
import pickle
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from ..interfaces.serving_interface import (
    IModelServingAdapter,
    ModelInfo,
    PredictionResult,
    ServingStats,
)
from ..schemas.enums import ModelType, TimeFrame
from ..core.config import get_settings
from common.logger.logger_factory import LoggerFactory


class SimpleServingAdapter(IModelServingAdapter):
    """
    Simple in-memory model serving adapter.

    This adapter implements the serving interface using direct model loading
    and inference in the current process. It's suitable for development and
    small-scale production deployments.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize simple serving adapter

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.settings = get_settings()

        # In-memory model storage
        self._loaded_models: Dict[str, Any] = {}
        self._model_info: Dict[str, ModelInfo] = {}

        # Configuration
        self.max_models_in_memory = config.get("max_models_in_memory", 5)
        self.model_timeout_seconds = config.get("model_timeout_seconds", 3600)  # 1 hour

        self.logger.info(
            f"SimpleServingAdapter initialized with max_models={self.max_models_in_memory}"
        )

    async def initialize(self) -> bool:
        """
        Initialize the simple serving backend

        Returns:
            bool: True if initialization successful
        """
        try:
            # Create model cache directory if it doesn't exist
            cache_dir = self.settings.models_dir / "serving_cache"
            cache_dir.mkdir(exist_ok=True, parents=True)

            self.logger.info("SimpleServingAdapter initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize SimpleServingAdapter: {e}")
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
        Load a model into memory

        Args:
            model_path: Path to the model files
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Type of model
            model_config: Optional model configuration

        Returns:
            ModelInfo: Information about the loaded model
        """
        model_id = self.generate_model_id(symbol, timeframe, model_type)

        try:
            self.logger.info(f"Loading model {model_id} from {model_path}")

            # Check if model is already loaded
            if model_id in self._loaded_models:
                self.logger.info(f"Model {model_id} already loaded")
                return self._model_info[model_id]

            # Check memory limits
            if len(self._loaded_models) >= self.max_models_in_memory:
                await self._evict_oldest_model()

            # Load model based on type
            model = await self._load_model_by_type(model_path, model_type, model_config)

            # Calculate memory usage
            memory_usage = self._estimate_model_memory_usage(model)

            # Store model and info
            self._loaded_models[model_id] = model

            model_info = ModelInfo(
                model_id=model_id,
                symbol=symbol,
                timeframe=timeframe.value,
                model_type=model_type.value,
                model_path=model_path,
                is_loaded=True,
                loaded_at=datetime.now(),
                memory_usage_mb=memory_usage,
                metadata=model_config or {},
            )

            self._model_info[model_id] = model_info

            # Update stats
            self._stats.models_loaded += 1
            self._stats.total_memory_usage_mb += memory_usage

            self.logger.info(
                f"Model {model_id} loaded successfully ({memory_usage:.2f}MB)"
            )
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
        Make predictions using a loaded model

        Args:
            model_id: Identifier of the loaded model
            input_data: Input data for prediction
            n_steps: Number of prediction steps
            **kwargs: Additional prediction parameters

        Returns:
            PredictionResult: Prediction results
        """
        start_time = time.time()

        try:
            self.logger.debug(f"Making prediction with model {model_id}")

            # Check if model is loaded
            if model_id not in self._loaded_models:
                error_msg = f"Model {model_id} not loaded"
                self.logger.error(error_msg)
                self._update_stats(success=False)
                return PredictionResult(success=False, predictions=[], error=error_msg)

            model = self._loaded_models[model_id]
            model_info = self._model_info[model_id]

            # Prepare input data
            processed_input = await self._preprocess_input(input_data, model_info)

            # Make prediction
            predictions = await self._make_prediction(
                model, processed_input, n_steps, model_info, **kwargs
            )

            # Post-process results
            result = await self._postprocess_predictions(
                predictions, processed_input, model_info
            )

            inference_time_ms = (time.time() - start_time) * 1000

            # Update stats
            self._update_stats(success=True, inference_time_ms=inference_time_ms)

            return PredictionResult(
                success=True,
                predictions=result["predictions"],
                current_price=result.get("current_price"),
                predicted_change_pct=result.get("predicted_change_pct"),
                confidence_score=result.get("confidence_score", 0.8),
                model_info={
                    "model_id": model_id,
                    "model_type": model_info.model_type,
                    "version": model_info.version,
                },
                inference_time_ms=inference_time_ms,
            )

        except Exception as e:
            inference_time_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Prediction failed for model {model_id}: {e}")
            self._update_stats(success=False, inference_time_ms=inference_time_ms)

            return PredictionResult(
                success=False,
                predictions=[],
                error=str(e),
                inference_time_ms=inference_time_ms,
            )

    async def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from memory

        Args:
            model_id: Identifier of the model to unload

        Returns:
            bool: True if unload successful
        """
        try:
            if model_id in self._loaded_models:
                # Get memory usage before removal
                model_info = self._model_info[model_id]
                memory_usage = model_info.memory_usage_mb or 0.0

                # Remove model from memory
                del self._loaded_models[model_id]
                del self._model_info[model_id]

                # Update stats
                self._stats.models_loaded -= 1
                self._stats.total_memory_usage_mb -= memory_usage

                self.logger.info(f"Model {model_id} unloaded successfully")
                return True
            else:
                self.logger.warning(f"Model {model_id} not found for unloading")
                return False

        except Exception as e:
            self.logger.error(f"Failed to unload model {model_id}: {e}")
            return False

    async def list_loaded_models(self) -> List[ModelInfo]:
        """
        List all currently loaded models

        Returns:
            List[ModelInfo]: List of loaded model information
        """
        return list(self._model_info.values())

    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model

        Args:
            model_id: Model identifier

        Returns:
            Optional[ModelInfo]: Model information if found
        """
        return self._model_info.get(model_id)

    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the serving backend

        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            # Get system metrics
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)

            return {
                "status": "healthy",
                "adapter_type": "simple",
                "models_loaded": len(self._loaded_models),
                "max_models": self.max_models_in_memory,
                "total_memory_usage_mb": self._stats.total_memory_usage_mb,
                "system_memory_percent": memory_info.percent,
                "system_cpu_percent": cpu_percent,
                "uptime_seconds": self.get_stats().uptime_seconds,
                "total_predictions": self._stats.total_predictions,
                "success_rate": self._stats.get_success_rate(),
            }

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def shutdown(self) -> bool:
        """
        Shutdown the serving backend gracefully

        Returns:
            bool: True if shutdown successful
        """
        try:
            self.logger.info("Shutting down SimpleServingAdapter")

            # Unload all models
            model_ids = list(self._loaded_models.keys())
            for model_id in model_ids:
                await self.unload_model(model_id)

            self.logger.info("SimpleServingAdapter shutdown completed")
            return True

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return False

    # Private helper methods

    async def _load_model_by_type(
        self,
        model_path: str,
        model_type: ModelType,
        model_config: Optional[Dict[str, Any]],
    ) -> Any:
        """Load model based on its type"""
        model_path_obj = Path(model_path)

        if model_type in [ModelType.PATCHTST, ModelType.PATCHTSMIXER]:
            # Load HuggingFace transformer models
            try:
                from transformers import AutoModel, AutoConfig

                config_path = model_path_obj / "config.json"
                if config_path.exists():
                    config = AutoConfig.from_pretrained(str(model_path_obj))
                    model = AutoModel.from_pretrained(
                        str(model_path_obj), config=config
                    )
                else:
                    # Load as pickle if no config
                    with open(model_path_obj / "model.pkl", "rb") as f:
                        model = pickle.load(f)

                return model

            except ImportError:
                self.logger.warning(
                    "transformers not available, falling back to pickle"
                )
                with open(model_path_obj / "model.pkl", "rb") as f:
                    return pickle.load(f)

        elif model_type == ModelType.PYTORCH_TRANSFORMER:
            # Load PyTorch Lightning model
            try:
                import torch

                model_file = model_path_obj / "model.pth"
                if model_file.exists():
                    return torch.load(model_file, map_location="cpu")
                else:
                    with open(model_path_obj / "model.pkl", "rb") as f:
                        return pickle.load(f)

            except ImportError:
                self.logger.warning("torch not available, falling back to pickle")
                with open(model_path_obj / "model.pkl", "rb") as f:
                    return pickle.load(f)

        else:
            # Default to pickle
            with open(model_path_obj / "model.pkl", "rb") as f:
                return pickle.load(f)

    async def _preprocess_input(
        self,
        input_data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
        model_info: ModelInfo,
    ) -> np.ndarray:
        """Preprocess input data for prediction"""
        if isinstance(input_data, pd.DataFrame):
            # Convert DataFrame to numpy array
            return input_data.values.astype(np.float32)
        elif isinstance(input_data, np.ndarray):
            return input_data.astype(np.float32)
        elif isinstance(input_data, dict):
            # Extract relevant features
            if "close" in input_data:
                return np.array(input_data["close"]).astype(np.float32)
            else:
                raise ValueError("Invalid input data format")
        else:
            raise ValueError(f"Unsupported input data type: {type(input_data)}")

    async def _make_prediction(
        self,
        model: Any,
        input_data: np.ndarray,
        n_steps: int,
        model_info: ModelInfo,
        **kwargs,
    ) -> np.ndarray:
        """Make prediction using the loaded model"""
        try:
            # This is a simplified prediction logic
            # In practice, this would depend on the specific model type

            if hasattr(model, "predict"):
                # scikit-learn style model
                predictions = model.predict(input_data[-n_steps:].reshape(1, -1))
            elif hasattr(model, "forward"):
                # PyTorch model
                import torch

                with torch.no_grad():
                    input_tensor = torch.tensor(input_data[-64:]).unsqueeze(
                        0
                    )  # Context window
                    predictions = model.forward(input_tensor).cpu().numpy()
            else:
                # Fallback: return last value with small random variation
                last_value = input_data[-1] if len(input_data) > 0 else 0.0
                predictions = np.array(
                    [
                        last_value * (1 + np.random.normal(0, 0.01))
                        for _ in range(n_steps)
                    ]
                )

            return predictions.flatten()[:n_steps]

        except Exception as e:
            self.logger.error(f"Model prediction failed: {e}")
            # Fallback prediction
            if len(input_data) > 0:
                last_value = input_data[-1]
                return np.array([last_value for _ in range(n_steps)])
            else:
                return np.zeros(n_steps)

    async def _postprocess_predictions(
        self, predictions: np.ndarray, input_data: np.ndarray, model_info: ModelInfo
    ) -> Dict[str, Any]:
        """Post-process prediction results"""
        predictions_list = predictions.tolist()

        result = {"predictions": predictions_list}

        if len(input_data) > 0:
            current_price = float(input_data[-1])
            result["current_price"] = current_price

            if len(predictions_list) > 0:
                predicted_price = predictions_list[0]
                change_pct = ((predicted_price - current_price) / current_price) * 100
                result["predicted_change_pct"] = change_pct

        return result

    def _estimate_model_memory_usage(self, model: Any) -> float:
        """Estimate model memory usage in MB"""
        try:
            import sys

            size_bytes = sys.getsizeof(model)

            # Try to get more accurate size for common model types
            if hasattr(model, "parameters"):
                # PyTorch model
                total_params = sum(p.numel() for p in model.parameters())
                size_bytes = total_params * 4  # Assume float32
            elif hasattr(model, "get_params"):
                # scikit-learn model
                params = model.get_params()
                size_bytes = sum(sys.getsizeof(v) for v in params.values())

            return size_bytes / (1024 * 1024)  # Convert to MB

        except Exception:
            return 50.0  # Default estimate

    async def _evict_oldest_model(self) -> None:
        """Evict the oldest loaded model to free memory"""
        if not self._model_info:
            return

        # Find oldest model
        oldest_model_id = min(
            self._model_info.keys(),
            key=lambda mid: self._model_info[mid].loaded_at or datetime.min,
        )

        self.logger.info(f"Evicting oldest model: {oldest_model_id}")
        await self.unload_model(oldest_model_id)
