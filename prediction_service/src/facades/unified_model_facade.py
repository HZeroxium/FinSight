# facades/unified_model_facade.py

"""
Unified Model Facade - Combined interface for both training and serving operations

This facade provides a unified interface that combines both training and serving
functionality, maintaining backward compatibility while providing access to
specialized operations through dedicated sub-facades.
"""

from typing import Dict, Any, Optional, List, Union
import pandas as pd
from datetime import datetime

from .model_training_facade import ModelTrainingFacade
from .model_serving_facade import ModelServingFacade
from ..interfaces.serving_interface import IModelServingAdapter
from ..schemas.enums import ModelType, TimeFrame
from ..schemas.model_schemas import ModelConfig, ModelInfo
from ..core.constants import FacadeConstants
from ..core.config import get_settings
from common.logger.logger_factory import LoggerFactory, LogLevel


class UnifiedModelFacade:
    """
    Unified facade combining training and serving operations.

    This facade provides:
    - Complete backward compatibility with existing model facades
    - Access to specialized training and serving operations
    - Unified interface for common operations
    - Proper lifecycle management for both training and serving
    """

    def __init__(self, serving_adapter: Optional[IModelServingAdapter] = None):
        """
        Initialize the unified model facade.

        Args:
            serving_adapter: Optional serving adapter instance
        """
        self.logger = LoggerFactory.get_logger(
            "UnifiedModelFacade",
            level=LogLevel.INFO,
            console_level=LogLevel.INFO,
            log_file="logs/unified_model_facade.log",
        )

        self.settings = get_settings()

        # Initialize sub-facades
        self.training = ModelTrainingFacade()
        self.serving = ModelServingFacade(serving_adapter)

        self._initialized = False

        self.logger.info("Unified model facade initialized")

    async def initialize(self) -> bool:
        """
        Initialize both training and serving facades.

        Returns:
            bool: True if initialization successful
        """
        if self._initialized:
            return True

        try:
            # Initialize serving facade (async)
            serving_success = await self.serving.initialize()

            # Training facade doesn't need async initialization
            # but we can add it here if needed in the future

            self._initialized = True
            self.logger.info(
                f"Unified facade initialized - Serving: {'✓' if serving_success else '✗'}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize unified facade: {e}")
            return False

    # Training Operations (delegated to training facade)

    async def train_model(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        feature_engineering: Any,
        config: ModelConfig,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train a model (delegates to training facade).

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type to train
            train_data: Training dataset
            val_data: Validation dataset
            feature_engineering: Feature engineering instance
            config: Model configuration
            run_id: Optional experiment run ID for tracking

        Returns:
            Dict containing training results
        """
        return await self.training.train_model(
            symbol,
            timeframe,
            model_type,
            train_data,
            val_data,
            feature_engineering,
            config,
            run_id,
        )

    def evaluate_model(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        test_data: pd.DataFrame,
        config: Optional[ModelConfig] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model (delegates to training facade).

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            test_data: Test dataset
            config: Optional model configuration

        Returns:
            Dict containing evaluation results
        """
        return self.training.evaluate_model(
            symbol, timeframe, model_type, test_data, config
        )

    def get_training_progress(self, training_id: str) -> Dict[str, Any]:
        """Get training progress (delegates to training facade)."""
        return self.training.get_training_progress(training_id)

    def list_active_training_sessions(self) -> List[Dict[str, Any]]:
        """List active training sessions (delegates to training facade)."""
        return self.training.list_active_training_sessions()

    def cancel_training(self, training_id: str) -> Dict[str, Any]:
        """Cancel training session (delegates to training facade)."""
        return self.training.cancel_training(training_id)

    def get_training_history(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get training history (delegates to training facade)."""
        return self.training.get_training_history(symbol)

    # Serving Operations (delegated to serving facade)

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
        Make asynchronous predictions (delegates to serving facade).

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            recent_data: Recent data for prediction
            n_steps: Number of prediction steps
            use_serving_adapter: Whether to use serving adapter

        Returns:
            Dict containing prediction results
        """
        return await self.serving.predict_async(
            symbol, timeframe, model_type, recent_data, n_steps, use_serving_adapter
        )

    def predict(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        recent_data: pd.DataFrame,
        n_steps: int = 1,
    ) -> Dict[str, Any]:
        """
        Make synchronous predictions (delegates to serving facade).

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            recent_data: Recent data for prediction
            n_steps: Number of prediction steps

        Returns:
            Dict containing prediction results
        """
        return self.serving.predict(symbol, timeframe, model_type, recent_data, n_steps)

    def forecast(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        recent_data: pd.DataFrame,
        n_steps: int = 1,
    ) -> Dict[str, Any]:
        """
        Make forecasts (alias for predict, delegates to serving facade).

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            recent_data: Recent data for forecasting
            n_steps: Number of forecast steps

        Returns:
            Dict containing forecast results
        """
        return self.serving.forecast(
            symbol, timeframe, model_type, recent_data, n_steps
        )

    async def load_model_to_serving(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        force_reload: bool = False,
    ) -> Dict[str, Any]:
        """Load model to serving adapter (delegates to serving facade)."""
        return await self.serving.load_model_to_serving(
            symbol, timeframe, model_type, force_reload
        )

    async def unload_model_from_serving(
        self, symbol: str, timeframe: TimeFrame, model_type: ModelType
    ) -> Dict[str, Any]:
        """Unload model from serving adapter (delegates to serving facade)."""
        return await self.serving.unload_model_from_serving(
            symbol, timeframe, model_type
        )

    async def list_serving_models(self) -> List[Dict[str, Any]]:
        """List serving models (delegates to serving facade)."""
        return await self.serving.list_serving_models()

    async def get_serving_health(self) -> Dict[str, Any]:
        """Get serving health (delegates to serving facade)."""
        return await self.serving.get_serving_health()

    async def get_serving_stats(self) -> Dict[str, Any]:
        """Get serving statistics (delegates to serving facade)."""
        return await self.serving.get_serving_stats()

    # Common Operations (implemented in this facade)

    def list_available_models(self) -> List[ModelInfo]:
        """
        List all available trained models.

        Returns:
            List of available model information
        """
        return self.serving.list_available_models()

    async def model_exists(
        self, symbol: str, timeframe: TimeFrame, model_type: ModelType
    ) -> bool:
        """
        Check if a trained model exists with cloud-first strategy.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type

        Returns:
            bool: True if model exists locally or in cloud
        """
        try:
            from ..utils.model_utils import ModelUtils
            from ..core.config import get_settings

            settings = get_settings()
            model_utils = ModelUtils()

            # Check local first
            local_exists = self.serving.model_exists(symbol, timeframe, model_type)

            # If cloud storage is enabled, also check cloud
            if settings.enable_cloud_storage and model_utils.storage_client:
                try:
                    cloud_exists = await model_utils.model_exists_in_cloud(
                        symbol, timeframe, model_type, "simple"
                    )
                    return local_exists or cloud_exists
                except Exception as e:
                    self.logger.warning(f"Failed to check cloud model existence: {e}")
                    # Fallback to local only
                    return local_exists

            return local_exists

        except Exception as e:
            self.logger.error(f"Error checking model existence: {e}")
            return False

    def get_model_info(
        self, symbol: str, timeframe: TimeFrame, model_type: ModelType
    ) -> Dict[str, Any]:
        """
        Get comprehensive information about a specific model.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type

        Returns:
            Dict containing model information
        """
        try:
            if not self.model_exists(symbol, timeframe, model_type):
                return {"exists": False, "error": "Model not found"}

            # Get basic model info
            model_info = {
                "exists": True,
                "symbol": symbol,
                "timeframe": timeframe.value,
                "model_type": model_type.value,
            }

            # Try to get metadata
            try:
                from ..utils.model_utils import ModelUtils

                model_utils = ModelUtils()
                model_path = model_utils.get_model_path(symbol, timeframe, model_type)
                metadata_path = (
                    model_path
                    / f"{symbol}_{timeframe.value}_{model_type.value}_metadata.json"
                )

                if metadata_path.exists():
                    metadata = model_utils.load_json(metadata_path)
                    model_info.update(metadata)

            except Exception as e:
                self.logger.warning(f"Failed to load model metadata: {e}")

            return model_info

        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return {"exists": False, "error": str(e)}

    def get_facade_status(self) -> Dict[str, Any]:
        """
        Get status of both training and serving facades.

        Returns:
            Dict containing facade status information
        """
        return {
            "unified_facade": {
                "initialized": self._initialized,
                "version": "1.0",
            },
            "training_facade": {
                "available": True,
                "active_sessions": len(self.training.list_active_training_sessions()),
            },
            "serving_facade": {
                "available": self.serving._use_serving_adapter,
                "adapter_initialized": self.serving._adapter_initialized,
                "adapter_type": getattr(
                    self.serving.serving_config, "adapter_type", "none"
                ),
            },
            "timestamp": datetime.now().isoformat(),
        }

    def clear_cache(self) -> None:
        """Clear caches in both facades."""
        self.training.clear_cache()
        self.serving.clear_cache()
        self.logger.info("All facade caches cleared")

    async def shutdown(self) -> None:
        """Shutdown both facades and cleanup resources."""
        try:
            # Shutdown serving facade (async)
            await self.serving.shutdown()

            # Training facade cleanup (if needed in the future)
            # await self.training.shutdown()

            # Clear caches
            self.clear_cache()

            self._initialized = False
            self.logger.info("Unified model facade shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during unified facade shutdown: {e}")

    # Backward Compatibility Methods (for existing code)

    async def initialize_serving(self) -> bool:
        """
        Initialize serving (backward compatibility).

        Returns:
            bool: True if serving initialization successful
        """
        return await self.serving.initialize()

    def get_serving_adapter(self) -> Optional[IModelServingAdapter]:
        """
        Get the serving adapter instance (for advanced usage).

        Returns:
            Optional serving adapter instance
        """
        return self.serving.serving_adapter

    def is_serving_available(self) -> bool:
        """
        Check if serving adapter is available and initialized.

        Returns:
            bool: True if serving is available
        """
        return (
            self.serving._use_serving_adapter
            and self.serving._adapter_initialized
            and self.serving.serving_adapter is not None
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics from both facades.

        Returns:
            Dict containing cache statistics
        """
        return {
            "training_cache": {
                "models_cached": len(self.training._cached_models),
                "active_sessions": len(self.training._training_sessions),
            },
            "serving_cache": {
                "models_cached": len(self.serving._cached_models),
            },
            "timestamp": datetime.now().isoformat(),
        }


# Backward compatibility aliases
ModelFacade = UnifiedModelFacade
EnhancedModelFacade = UnifiedModelFacade
