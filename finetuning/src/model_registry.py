# model_registry.py

from typing import Dict, Tuple, Type
from transformers import (
    AutoformerConfig,
    AutoformerForPrediction,
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
    InformerConfig,
    InformerForPrediction,
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    PatchTSTConfig,
    PatchTSTModel,
    TimesFmConfig,
    TimesFmModelForPrediction,
    PreTrainedModel,
    PretrainedConfig,
)
from pydantic import BaseModel, Field


class RegistryEntry(BaseModel):
    """
    Associates a HuggingFace Config class with its corresponding Model class.
    """

    config_cls: Type[PretrainedConfig] = Field(
        ..., description="HuggingFace PretrainedConfig subclass"
    )
    model_cls: Type[PreTrainedModel] = Field(
        ..., description="HuggingFace PreTrainedModel subclass"
    )


class ModelRegistry:
    """
    Central registry for all supported time-series models.
    """

    _registry: Dict[str, RegistryEntry] = {
        "autoformer": RegistryEntry(
            config_cls=AutoformerConfig, model_cls=AutoformerForPrediction
        ),
        "ts_transformer": RegistryEntry(
            config_cls=TimeSeriesTransformerConfig,
            model_cls=TimeSeriesTransformerForPrediction,
        ),
        "informer": RegistryEntry(
            config_cls=InformerConfig, model_cls=InformerForPrediction
        ),
        "patchtsmixer": RegistryEntry(
            config_cls=PatchTSMixerConfig, model_cls=PatchTSMixerForPrediction
        ),
        "patchtst": RegistryEntry(config_cls=PatchTSTConfig, model_cls=PatchTSTModel),
        "timesfm": RegistryEntry(
            config_cls=TimesFmConfig, model_cls=TimesFmModelForPrediction
        ),
    }

    @classmethod
    def get(cls, key: str) -> RegistryEntry:
        """
        Retrieve the RegistryEntry for the given model key.
        Raises KeyError if not found.
        """
        try:
            return cls._registry[key]
        except KeyError as e:
            raise KeyError(f"Model '{key}' is not registered.") from e
