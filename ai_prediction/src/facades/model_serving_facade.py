# facades/model_serving_facade.py

"""
Model Serving Facade - Specialized facade for model serving operations

This facade handles all serving-related operations including:
- Model serving with adapter integration
- Prediction and inference operations
- Model loading/unloading for serving
- Serving health monitoring and statistics
"""

from typing import Dict, Any, Optional, List, Union
import pandas as pd
from datetime import datetime
import asyncio

from ..interfaces.model_interface import ITimeSeriesModel
from ..interfaces.serving_interface import (
    IModelServingAdapter,
    ModelInfo as ServingModelInfo,
    PredictionResult,
)
from ..models.model_factory import ModelFactory
from ..adapters.adapter_factory import ServingAdapterFactory
from ..schemas.enums import ModelType, TimeFrame
from ..schemas.model_schemas import ModelConfig, ModelInfo
from ..utils.model_utils import ModelUtils
from ..core.constants import FacadeConstants, LoggingConstants
from ..core.config import get_settings
from ..core.serving_config import get_serving_config, get_adapter_config
from common.logger.logger_factory import LoggerFactory, LogLevel


class ModelServingFacade:
    """
    Specialized facade for model serving operations.

    This facade provides:
    - Scalable model serving with adapter integration
    - Async prediction and inference operations
    - Model lifecycle management for serving
    - Serving health monitoring and performance tracking
    - Legacy compatibility for synchronous operations
    """

    def __init__(self, serving_adapter: Optional[IModelServingAdapter] = None):
        """
        Initialize the model serving facade.

        Args:
            serving_adapter: Optional serving adapter instance. If None, creates from config.
        """
        self.logger = LoggerFactory.get_logger(
            "ModelServingFacade",
            level=LogLevel.INFO,
            console_level=LogLevel.INFO,
            log_file="logs/model_serving_facade.log",
        )

        self.settings = get_settings()
        self.model_utils = ModelUtils()

        # Legacy model cache for fallback operations
        self._cached_models: Dict[str, ITimeSeriesModel] = {}

        # Initialize serving adapter
        self.serving_adapter = serving_adapter
        self._adapter_initialized = False
        self._use_serving_adapter = True

        # Get serving configuration
        self.serving_config = get_serving_config()

        if self.serving_adapter is None:
            try:
                # Create serving adapter from configuration
                adapter_config = get_adapter_config(self.serving_config.adapter_type)
                self.serving_adapter = ServingAdapterFactory.create_adapter(
                    self.serving_config.adapter_type, adapter_config
                )
                self.logger.info(
                    f"Created {self.serving_config.adapter_type} serving adapter"
                )
            except Exception as e:
                self.logger.warning(f"Failed to create serving adapter: {e}")
                self.logger.info("Falling back to legacy model loading")
                self._use_serving_adapter = False

        self.logger.info("Model serving facade initialized")

    async def initialize(self) -> bool:
        """
        Initialize the serving adapter (async operation).

        Returns:
            bool: True if initialization successful
        """
        if not self._use_serving_adapter or self.serving_adapter is None:
            self.logger.info("Serving adapter disabled, using legacy mode")
            return True

        if not self._adapter_initialized:
            try:
                success = await self.serving_adapter.initialize()
                if success:
                    self._adapter_initialized = True
                    self.logger.info("Serving adapter initialized successfully")
                else:
                    self.logger.error("Failed to initialize serving adapter")
                    self._use_serving_adapter = False
                return success
            except Exception as e:
                self.logger.error(f"Failed to initialize serving adapter: {e}")
                self._use_serving_adapter = False
                return False
        return True

    async def predict_async(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        recent_data: Union[pd.DataFrame, dict],
        n_steps: int = 1,
        use_serving_adapter: bool = True,
    ) -> Dict[str, Any]:
        """
        Make asynchronous predictions using the serving adapter or legacy model.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            recent_data: Recent data for prediction (DataFrame or dict)
            n_steps: Number of prediction steps
            use_serving_adapter: Whether to use serving adapter (if available)

        Returns:
            Dict containing prediction results
        """
        try:
            self.logger.info(
                f"Making async prediction: {symbol} {timeframe.value} {model_type.value}"
            )

            # Try serving adapter first if enabled and available
            if (
                use_serving_adapter
                and self._use_serving_adapter
                and self._adapter_initialized
                and self.serving_adapter is not None
            ):

                try:
                    return await self._predict_with_serving_adapter(
                        symbol, timeframe, model_type, recent_data, n_steps
                    )
                except Exception as e:
                    self.logger.warning(f"Serving adapter prediction failed: {e}")
                    self.logger.info("Falling back to legacy prediction")

            # Fallback to legacy prediction
            return await self._predict_legacy(
                symbol, timeframe, model_type, recent_data, n_steps
            )

        except Exception as e:
            self.logger.error(f"Async prediction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "predictions": [],
                "model_info": {
                    "symbol": symbol,
                    "timeframe": timeframe.value,
                    "model_type": model_type.value,
                },
            }

    def predict(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        recent_data: pd.DataFrame,
        n_steps: int = 1,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for predictions (for backward compatibility).

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            recent_data: Recent data for prediction
            n_steps: Number of prediction steps

        Returns:
            Dict containing prediction results
        """
        try:
            # Run async prediction in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in an event loop, create a new thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.predict_async(
                            symbol, timeframe, model_type, recent_data, n_steps
                        ),
                    )
                    return future.result(
                        timeout=FacadeConstants.DEFAULT_SERVING_TIMEOUT
                    )
            else:
                return loop.run_until_complete(
                    self.predict_async(
                        symbol, timeframe, model_type, recent_data, n_steps
                    )
                )
        except Exception as e:
            self.logger.error(f"Synchronous prediction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "predictions": [],
            }

    def forecast(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        recent_data: pd.DataFrame,
        n_steps: int = 1,
    ) -> Dict[str, Any]:
        """
        Alias for predict method (for backward compatibility).

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            recent_data: Recent data for forecasting
            n_steps: Number of forecast steps

        Returns:
            Dict containing forecast results
        """
        return self.predict(symbol, timeframe, model_type, recent_data, n_steps)

    async def load_model_to_serving(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        force_reload: bool = False,
    ) -> Dict[str, Any]:
        """
        Load a model to the serving adapter.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            force_reload: Whether to force reload if already loaded

        Returns:
            Dict containing load result
        """
        if not self._use_serving_adapter or not self._adapter_initialized:
            return {"success": False, "error": "Serving adapter not available"}

        try:
            model_id = f"{symbol}_{timeframe.value}_{model_type.value}"
            model_path = self.model_utils.get_model_path(
                symbol,
                timeframe,
                model_type,
                adapter_type=self.serving_config.adapter_type,
            )

            if not model_path.exists():
                return {"success": False, "error": "Model not found"}

            # Check if already loaded
            loaded_models = await self.serving_adapter.list_models()
            if not force_reload and any(m.model_id == model_id for m in loaded_models):
                return {
                    "success": True,
                    "message": "Model already loaded",
                    "model_id": model_id,
                }

            # Load model to serving adapter
            serving_model_info = ServingModelInfo(
                model_id=model_id,
                model_type=model_type.value,
                model_path=str(model_path),
                metadata={
                    "symbol": symbol,
                    "timeframe": timeframe.value,
                    "model_type": model_type.value,
                },
            )

            success = await self.serving_adapter.load_model(serving_model_info)

            if success:
                self.logger.info(f"Model loaded to serving adapter: {model_id}")
                return {"success": True, "model_id": model_id}
            else:
                return {
                    "success": False,
                    "error": "Failed to load model to serving adapter",
                }

        except Exception as e:
            self.logger.error(f"Failed to load model to serving: {e}")
            return {"success": False, "error": str(e)}

    async def unload_model_from_serving(
        self, symbol: str, timeframe: TimeFrame, model_type: ModelType
    ) -> Dict[str, Any]:
        """
        Unload a model from the serving adapter.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type

        Returns:
            Dict containing unload result
        """
        if not self._use_serving_adapter or not self._adapter_initialized:
            return {"success": False, "error": "Serving adapter not available"}

        try:
            model_id = f"{symbol}_{timeframe.value}_{model_type.value}"
            success = await self.serving_adapter.unload_model(model_id)

            if success:
                self.logger.info(f"Model unloaded from serving adapter: {model_id}")
                return {"success": True, "model_id": model_id}
            else:
                return {
                    "success": False,
                    "error": "Failed to unload model from serving adapter",
                }

        except Exception as e:
            self.logger.error(f"Failed to unload model from serving: {e}")
            return {"success": False, "error": str(e)}

    async def list_serving_models(self) -> List[Dict[str, Any]]:
        """
        List all models loaded in the serving adapter.

        Returns:
            List of serving model information
        """
        if not self._use_serving_adapter or not self._adapter_initialized:
            return []

        try:
            models = await self.serving_adapter.list_models()
            return [
                {
                    "model_id": model.model_id,
                    "model_type": model.model_type,
                    "model_path": model.model_path,
                    "metadata": model.metadata,
                    "loaded_at": (
                        model.loaded_at.isoformat() if model.loaded_at else None
                    ),
                }
                for model in models
            ]
        except Exception as e:
            self.logger.error(f"Failed to list serving models: {e}")
            return []

    async def get_serving_health(self) -> Dict[str, Any]:
        """
        Get serving adapter health status.

        Returns:
            Dict containing health information
        """
        if not self._use_serving_adapter or not self._adapter_initialized:
            return {
                "healthy": False,
                "message": "Serving adapter not available",
                "adapter_type": "none",
            }

        try:
            health = await self.serving_adapter.health_check()
            return {
                "healthy": health.get("healthy", False),
                "adapter_type": self.serving_config.adapter_type,
                "details": health,
                "last_check": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "adapter_type": self.serving_config.adapter_type,
                "error": str(e),
                "last_check": datetime.now().isoformat(),
            }

    async def get_serving_stats(self) -> Dict[str, Any]:
        """
        Get serving adapter statistics.

        Returns:
            Dict containing serving statistics
        """
        if not self._use_serving_adapter or not self._adapter_initialized:
            return {"available": False, "message": "Serving adapter not available"}

        try:
            stats = await self.serving_adapter.get_stats()
            return {
                "available": True,
                "adapter_type": self.serving_config.adapter_type,
                "stats": stats,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Failed to get serving stats: {e}")
            return {"available": False, "error": str(e)}

    def list_available_models(self) -> List[ModelInfo]:
        """
        List all available trained models (legacy compatibility).

        Returns:
            List of available model information
        """
        try:
            models = []
            models_dir = self.settings.models_dir

            if not models_dir.exists():
                return models

            # Scan for trained models
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    try:
                        # Parse directory name: symbol_timeframe_modeltype
                        parts = model_dir.name.split("_")
                        if len(parts) >= 3:
                            symbol = parts[0]
                            timeframe_str = parts[1]
                            model_type_str = "_".join(parts[2:])

                            # Validate timeframe and model type
                            try:
                                timeframe = TimeFrame(timeframe_str)
                                model_type = ModelType(model_type_str)

                                # Check if model files exist
                                if self._check_model_files_exist(model_dir):
                                    model_info = ModelInfo(
                                        symbol=symbol,
                                        timeframe=timeframe,
                                        model_type=model_type,
                                        model_path=str(model_dir),
                                        created_at=datetime.fromtimestamp(
                                            model_dir.stat().st_ctime
                                        ),
                                    )
                                    models.append(model_info)
                            except ValueError:
                                # Skip invalid directory names
                                continue
                    except Exception as e:
                        self.logger.warning(
                            f"Error processing model directory {model_dir}: {e}"
                        )
                        continue

            return models

        except Exception as e:
            self.logger.error(f"Failed to list available models: {e}")
            return []

    def model_exists(
        self, symbol: str, timeframe: TimeFrame, model_type: ModelType
    ) -> bool:
        """
        Check if a trained model exists.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type

        Returns:
            bool: True if model exists
        """
        try:
            model_path = self.model_utils.get_model_path(
                symbol,
                timeframe,
                model_type,
                adapter_type=self.serving_config.adapter_type,
            )
            return model_path.exists() and self._check_model_files_exist(model_path)
        except Exception as e:
            self.logger.error(f"Error checking model existence: {e}")
            return False

    def clear_cache(self) -> None:
        """Clear the legacy model cache."""
        self._cached_models.clear()
        self.logger.info("Legacy model cache cleared")

    async def shutdown(self) -> None:
        """Shutdown the serving facade and cleanup resources."""
        try:
            if self.serving_adapter and self._adapter_initialized:
                # Unload all models from serving adapter
                models = await self.list_serving_models()
                for model in models:
                    try:
                        await self.serving_adapter.unload_model(model["model_id"])
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to unload model {model['model_id']}: {e}"
                        )

                # Shutdown serving adapter if it has shutdown method
                if hasattr(self.serving_adapter, "shutdown"):
                    await self.serving_adapter.shutdown()

            # Clear cache
            self.clear_cache()

            self.logger.info("Model serving facade shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during serving facade shutdown: {e}")

    # Private helper methods

    async def _predict_with_serving_adapter(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        recent_data: Union[pd.DataFrame, dict],
        n_steps: int,
    ) -> Dict[str, Any]:
        """Make predictions using the serving adapter."""
        model_id = f"{symbol}_{timeframe.value}_{model_type.value}"

        # Ensure model is loaded in serving adapter
        load_result = await self.load_model_to_serving(symbol, timeframe, model_type)
        if not load_result["success"]:
            raise RuntimeError(
                f"Failed to load model to serving adapter: {load_result.get('error')}"
            )

        # Convert DataFrame to dict if needed
        if isinstance(recent_data, pd.DataFrame):
            input_data = recent_data.to_dict("records")
        else:
            input_data = recent_data

        # Make prediction
        prediction_result = await self.serving_adapter.predict(
            model_id=model_id,
            input_data=input_data,
            n_steps=n_steps,
        )

        # Convert serving result to legacy format
        return self._convert_serving_result_to_legacy(
            prediction_result, symbol, timeframe, model_type
        )

    async def _predict_legacy(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        recent_data: Union[pd.DataFrame, dict],
        n_steps: int,
    ) -> Dict[str, Any]:
        """Make predictions using legacy model loading."""
        try:
            # Load model
            model = self._load_model(symbol, timeframe, model_type)
            if model is None:
                return {
                    "success": False,
                    "error": "Model not found or failed to load",
                    "predictions": [],
                }

            # Convert dict to DataFrame if needed
            if isinstance(recent_data, dict):
                recent_data = pd.DataFrame(recent_data)

            # Make prediction
            predictions = model.forecast(recent_data, n_steps=n_steps)

            return {
                "success": True,
                "predictions": predictions,
                "model_info": {
                    "symbol": symbol,
                    "timeframe": timeframe.value,
                    "model_type": model_type.value,
                },
                "method": "legacy",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Legacy prediction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "predictions": [],
            }

    def _convert_serving_result_to_legacy(
        self,
        serving_result: PredictionResult,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Dict[str, Any]:
        """Convert serving adapter result to legacy format."""
        return {
            "success": serving_result.success,
            "predictions": serving_result.predictions,
            "confidence_scores": getattr(serving_result, "confidence_scores", []),
            "model_info": {
                "symbol": symbol,
                "timeframe": timeframe.value,
                "model_type": model_type.value,
            },
            "method": "serving_adapter",
            "timestamp": datetime.now().isoformat(),
            "error": getattr(serving_result, "error", None),
        }

    def _load_model(
        self, symbol: str, timeframe: TimeFrame, model_type: ModelType
    ) -> Optional[ITimeSeriesModel]:
        """Load a model for legacy prediction."""
        cache_key = f"{symbol}_{timeframe.value}_{model_type.value}"

        # Check cache first
        if cache_key in self._cached_models:
            return self._cached_models[cache_key]

        try:
            # Load from disk
            model_path = self.model_utils.get_model_path(symbol, timeframe, model_type)
            if not model_path.exists():
                return None

            # Create model instance and load
            model = ModelFactory.create_model(model_type, {})
            if model is None:
                return None

            model.load_model(str(model_path))

            # Cache the model (with size limit)
            if len(self._cached_models) >= FacadeConstants.DEFAULT_MODEL_CACHE_SIZE:
                # Remove oldest cached model
                oldest_key = next(iter(self._cached_models))
                del self._cached_models[oldest_key]

            self._cached_models[cache_key] = model
            return model

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return None

    def _check_model_files_exist(self, model_path) -> bool:
        """Check if required model files exist in the model directory."""
        # This is a basic check - can be enhanced based on model type requirements
        if not model_path.exists():
            return False

        # Check for common model files
        required_files = ["pytorch_model.bin", "config.json"]
        existing_files = [f.name for f in model_path.iterdir() if f.is_file()]

        # At least one model file should exist
        return (
            any(req_file in existing_files for req_file in required_files)
            or len(existing_files) > 0
        )
