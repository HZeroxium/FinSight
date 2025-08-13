# facades/facade_factory.py

"""
Facade Factory - Factory for creating different types of model facades

This factory provides a centralized way to create and configure
different facade types based on requirements and configuration.
"""

from typing import Optional, Dict, Any, Union
from enum import Enum

from .model_training_facade import ModelTrainingFacade
from .model_serving_facade import ModelServingFacade
from .unified_model_facade import UnifiedModelFacade
from ..interfaces.serving_interface import IModelServingAdapter
from ..core.constants import FacadeConstants
from ..core.config import get_settings
from common.logger.logger_factory import LoggerFactory, LogLevel


class FacadeType(str, Enum):
    """Enumeration of available facade types."""

    TRAINING = FacadeConstants.TRAINING_FACADE
    SERVING = FacadeConstants.SERVING_FACADE
    UNIFIED = FacadeConstants.UNIFIED_FACADE


class FacadeFactory:
    """
    Factory for creating different types of model facades.

    This factory provides:
    - Centralized facade creation and configuration
    - Automatic dependency injection
    - Configuration-based facade selection
    - Singleton pattern for global facade instances
    """

    _instances: Dict[str, Any] = {}
    _logger = LoggerFactory.get_logger(
        "FacadeFactory",
        level=LogLevel.INFO,
        console_level=LogLevel.INFO,
        log_file="logs/facade_factory.log",
    )

    @classmethod
    def create_facade(
        self,
        facade_type: Union[FacadeType, str],
        serving_adapter: Optional[IModelServingAdapter] = None,
        use_singleton: bool = True,
        **kwargs,
    ) -> Union[ModelTrainingFacade, ModelServingFacade, UnifiedModelFacade]:
        """
        Create a facade instance of the specified type.

        Args:
            facade_type: Type of facade to create
            serving_adapter: Optional serving adapter instance
            use_singleton: Whether to use singleton pattern
            **kwargs: Additional arguments for facade initialization

        Returns:
            Facade instance of the requested type

        Raises:
            ValueError: If facade type is invalid
        """
        # Normalize facade type
        if isinstance(facade_type, str):
            try:
                facade_type = FacadeType(facade_type.lower())
            except ValueError:
                raise ValueError(f"Invalid facade type: {facade_type}")

        # Create cache key for singleton pattern
        cache_key = f"{facade_type.value}_{id(serving_adapter) if serving_adapter else 'default'}"

        # Return existing instance if using singleton
        if use_singleton and cache_key in self._instances:
            self._logger.debug(
                f"Returning existing {facade_type.value} facade instance"
            )
            return self._instances[cache_key]

        # Create new facade instance
        self._logger.info(f"Creating new {facade_type.value} facade instance")

        if facade_type == FacadeType.TRAINING:
            facade = ModelTrainingFacade(**kwargs)
        elif facade_type == FacadeType.SERVING:
            facade = ModelServingFacade(serving_adapter=serving_adapter, **kwargs)
        elif facade_type == FacadeType.UNIFIED:
            facade = UnifiedModelFacade(serving_adapter=serving_adapter, **kwargs)
        else:
            raise ValueError(f"Unsupported facade type: {facade_type}")

        # Cache instance if using singleton
        if use_singleton:
            self._instances[cache_key] = facade

        self._logger.info(f"Successfully created {facade_type.value} facade")
        return facade

    @classmethod
    def create_training_facade(
        self, use_singleton: bool = True, **kwargs
    ) -> ModelTrainingFacade:
        """
        Create a training facade instance.

        Args:
            use_singleton: Whether to use singleton pattern
            **kwargs: Additional arguments for facade initialization

        Returns:
            ModelTrainingFacade instance
        """
        return self.create_facade(
            FacadeType.TRAINING, use_singleton=use_singleton, **kwargs
        )

    @classmethod
    def create_serving_facade(
        self,
        serving_adapter: Optional[IModelServingAdapter] = None,
        use_singleton: bool = True,
        **kwargs,
    ) -> ModelServingFacade:
        """
        Create a serving facade instance.

        Args:
            serving_adapter: Optional serving adapter instance
            use_singleton: Whether to use singleton pattern
            **kwargs: Additional arguments for facade initialization

        Returns:
            ModelServingFacade instance
        """
        return self.create_facade(
            FacadeType.SERVING,
            serving_adapter=serving_adapter,
            use_singleton=use_singleton,
            **kwargs,
        )

    @classmethod
    def create_unified_facade(
        self,
        serving_adapter: Optional[IModelServingAdapter] = None,
        use_singleton: bool = True,
        **kwargs,
    ) -> UnifiedModelFacade:
        """
        Create a unified facade instance.

        Args:
            serving_adapter: Optional serving adapter instance
            use_singleton: Whether to use singleton pattern
            **kwargs: Additional arguments for facade initialization

        Returns:
            UnifiedModelFacade instance
        """
        return self.create_facade(
            FacadeType.UNIFIED,
            serving_adapter=serving_adapter,
            use_singleton=use_singleton,
            **kwargs,
        )

    @classmethod
    def get_default_facade(
        self, serving_adapter: Optional[IModelServingAdapter] = None
    ) -> UnifiedModelFacade:
        """
        Get the default facade (unified facade with singleton pattern).

        Args:
            serving_adapter: Optional serving adapter instance

        Returns:
            UnifiedModelFacade instance
        """
        return self.create_unified_facade(
            serving_adapter=serving_adapter, use_singleton=True
        )

    @classmethod
    def create_from_config(
        self, config: Optional[Dict[str, Any]] = None, use_singleton: bool = True
    ) -> UnifiedModelFacade:
        """
        Create a facade based on configuration settings.

        Args:
            config: Optional configuration dict (uses settings if None)
            use_singleton: Whether to use singleton pattern

        Returns:
            Facade instance based on configuration
        """
        if config is None:
            settings = get_settings()
            # Default to unified facade if no specific config
            facade_type = getattr(settings, "default_facade_type", FacadeType.UNIFIED)
        else:
            facade_type = config.get("facade_type", FacadeType.UNIFIED)

        # Always return unified facade for backward compatibility
        # but allow specific facade types in the future if needed
        return self.create_unified_facade(use_singleton=use_singleton)

    @classmethod
    def clear_cache(self) -> None:
        """Clear all cached facade instances."""
        count = len(self._instances)
        self._instances.clear()
        self._logger.info(f"Cleared {count} cached facade instances")

    @classmethod
    def get_cached_instances(self) -> Dict[str, str]:
        """
        Get information about cached facade instances.

        Returns:
            Dict mapping cache keys to facade types
        """
        return {
            key: type(instance).__name__ for key, instance in self._instances.items()
        }

    @classmethod
    async def shutdown_all(self) -> None:
        """Shutdown all cached facade instances."""
        self._logger.info("Shutting down all cached facade instances")

        for key, instance in self._instances.items():
            try:
                if hasattr(instance, "shutdown") and callable(instance.shutdown):
                    await instance.shutdown()
                    self._logger.debug(f"Shutdown facade: {key}")
            except Exception as e:
                self._logger.error(f"Error shutting down facade {key}: {e}")

        self.clear_cache()
        self._logger.info("All facade instances shutdown completed")


# Convenience functions for quick facade creation


def get_training_facade(**kwargs) -> ModelTrainingFacade:
    """Get a training facade instance (singleton)."""
    return FacadeFactory.create_training_facade(**kwargs)


def get_serving_facade(
    serving_adapter: Optional[IModelServingAdapter] = None, **kwargs
) -> ModelServingFacade:
    """Get a serving facade instance (singleton)."""
    return FacadeFactory.create_serving_facade(
        serving_adapter=serving_adapter, **kwargs
    )


def get_unified_facade(
    serving_adapter: Optional[IModelServingAdapter] = None, **kwargs
) -> UnifiedModelFacade:
    """Get a unified facade instance (singleton)."""
    return FacadeFactory.create_unified_facade(
        serving_adapter=serving_adapter, **kwargs
    )


def get_default_facade(
    serving_adapter: Optional[IModelServingAdapter] = None,
) -> UnifiedModelFacade:
    """Get the default facade (unified facade, singleton)."""
    return FacadeFactory.get_default_facade(serving_adapter=serving_adapter)


# Backward compatibility aliases
def get_model_facade(
    serving_adapter: Optional[IModelServingAdapter] = None,
) -> UnifiedModelFacade:
    """Get model facade (backward compatibility alias)."""
    return get_default_facade(serving_adapter=serving_adapter)


def get_enhanced_model_facade(
    serving_adapter: Optional[IModelServingAdapter] = None,
) -> UnifiedModelFacade:
    """Get enhanced model facade (backward compatibility alias)."""
    return get_default_facade(serving_adapter=serving_adapter)
