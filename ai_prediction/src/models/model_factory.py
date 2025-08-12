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
from ..utils.device_manager import create_device_manager_from_settings
from common.logger.logger_factory import LoggerFactory


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

        # Ensure device configuration is consistent
        device_manager = create_device_manager_from_settings()
        model_config = config.copy()

        # Add device configuration if not explicitly set
        if "device" not in model_config:
            model_config["device"] = device_manager.device
            logger.debug(f"Added device configuration: {device_manager.device}")
        elif model_config["device"] != device_manager.device:
            logger.warning(
                f"Model config device '{model_config['device']}' differs from "
                f"settings device '{device_manager.device}'"
            )

        model_class = cls._model_registry[model_type]
        logger.info(
            f"Creating model of type: {model_type} on device: {model_config.get('device', 'default')}"
        )
        logger.debug(f"Model config: {model_config}")

        try:
            model = model_class(model_config)
            if model is None:
                raise ValueError(f"Model class {model_class.__name__} returned None")
            logger.info(f"Successfully created model: {model_class.__name__}")
            return model
        except Exception as e:
            logger.error(f"Failed to create model {model_type}: {e}")
            logger.error(f"Config that caused error: {model_config}")
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
