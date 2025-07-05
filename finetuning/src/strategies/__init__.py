# strategies/__init__.py

from .base import ModelStrategy
from .patchtst import PatchTSTStrategy
from .autoformer import AutoformerStrategy
from .informer import InformerStrategy
from .patchtsmixer import PatchTSMixerStrategy
from .ts_transformer import TimeSeriesTransformerStrategy
from .timesfm import TimesFMStrategy

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common"))
from logger import LoggerFactory, LogLevel, LoggerType

logger = LoggerFactory.get_logger(
    __name__,
    level=LogLevel.INFO,
    logger_type=LoggerType.STANDARD,
)


class ModelStrategyFactory:
    """
    Factory class to create appropriate model strategies.
    """

    _strategies = {
        "patchtst": PatchTSTStrategy,
        "autoformer": AutoformerStrategy,
        "informer": InformerStrategy,
        "patchtsmixer": PatchTSMixerStrategy,
        "ts_transformer": TimeSeriesTransformerStrategy,
        "timesfm": TimesFMStrategy,
    }

    @classmethod
    def create_strategy(cls, model_key: str) -> ModelStrategy:
        """
        Create and return the appropriate strategy for the given model key.

        Args:
            model_key: One of the supported model keys

        Returns:
            ModelStrategy instance for the specified model

        Raises:
            ValueError: If model_key is not supported
        """
        if model_key not in cls._strategies:
            supported_models = list(cls._strategies.keys())
            raise ValueError(
                f"Unsupported model key: {model_key}. Supported models: {supported_models}"
            )

        strategy_class = cls._strategies[model_key]
        strategy = strategy_class()
        logger.info(f"Created strategy for model: {model_key}")
        return strategy

    @classmethod
    def get_supported_models(cls) -> list[str]:
        """
        Get list of all supported model keys.

        Returns:
            List of supported model keys
        """
        return list(cls._strategies.keys())


__all__ = [
    "ModelStrategy",
    "ModelStrategyFactory",
    "get_strategy",
    "PatchTSTStrategy",
    "AutoformerStrategy",
    "InformerStrategy",
    "PatchTSMixerStrategy",
    "TimeSeriesTransformerStrategy",
    "TimesFMStrategy",
]


def get_strategy(model_key: str) -> ModelStrategy:
    """
    Convenience function to get a strategy instance for a model.

    Args:
        model_key: One of the supported model keys

    Returns:
        ModelStrategy instance for the specified model
    """
    return ModelStrategyFactory.create_strategy(model_key)
