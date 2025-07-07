# models/model_factory.py

from typing import Dict, Type
from ..interfaces.model_interface import ITimeSeriesModel
from ..schemas.enums import ModelType
from .adapters.patchtst_adapter import PatchTSTAdapter
from .adapters.patchtsmixer_adapter import PatchTSMixerAdapter
from .adapters.transformer_adapter import TransformerAdapter
from .adapters.enhanced_transformer_adapter import (
    EnhancedTransformerAdapter,
)  # New import
from ..logger.logger_factory import LoggerFactory


class ModelFactory:
    """Factory for creating time series models"""

    _model_registry: Dict[ModelType, Type[ITimeSeriesModel]] = {
        ModelType.PATCHTST: PatchTSTAdapter,
        ModelType.PATCHTSMIXER: PatchTSMixerAdapter,
        ModelType.PYTORCH_TRANSFORMER: TransformerAdapter,
        ModelType.ENHANCED_TRANSFORMER: EnhancedTransformerAdapter,  # New enhanced transformer
    }

    @classmethod
    def create_model(cls, model_type: ModelType, config: Dict) -> ITimeSeriesModel:
        """
        Create a model instance

        Args:
            model_type: Type of model to create
            config: Model configuration

        Returns:
            Model instance

        Raises:
            ValueError: If model type is unsupported or model creation fails
        """
        logger = LoggerFactory.get_logger("ModelFactory")

        if model_type not in cls._model_registry:
            available_models = list(cls._model_registry.keys())
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Available models: {available_models}"
            )

        model_class = cls._model_registry[model_type]
        logger.info(f"Creating model of type: {model_type}")
        logger.debug(f"Model config: {config}")

        try:
            model = model_class(config)
            if model is None:
                raise ValueError(f"Model class {model_class.__name__} returned None")
            logger.info(f"Successfully created model: {model_class.__name__}")
            return model
        except Exception as e:
            logger.error(f"Failed to create model {model_type}: {e}")
            logger.error(f"Config that caused error: {config}")
            raise ValueError(f"Failed to create model {model_type}: {e}")

    @classmethod
    def get_supported_models(cls) -> list[ModelType]:
        """Get list of supported model types"""
        return list(cls._model_registry.keys())

    @classmethod
    def register_model(
        cls, model_type: ModelType, model_class: Type[ITimeSeriesModel]
    ) -> None:
        """
        Register a new model type

        Args:
            model_type: Model type enum
            model_class: Model implementation class
        """
        cls._model_registry[model_type] = model_class
