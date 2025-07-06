# models/model_factory.py

from typing import Dict, Type
from ..interfaces.model_interface import ITimeSeriesModel
from ..schemas.enums import ModelType
from .adapters.patchtst_adapter import PatchTSTAdapter
from .adapters.patchtsmixer_adapter import PatchTSMixerAdapter
from .adapters.transformer_adapter import TransformerAdapter
from ..logger.logger_factory import LoggerFactory


class ModelFactory:
    """Factory for creating time series models"""

    _model_registry: Dict[ModelType, Type[ITimeSeriesModel]] = {
        ModelType.PATCHTST: PatchTSTAdapter,
        ModelType.PATCHTSMIXER: PatchTSMixerAdapter,
        ModelType.PYTORCH_TRANSFORMER: TransformerAdapter,
        # Additional models can be added here
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

        return model_class(config)

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
