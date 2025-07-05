# model_wrappers.py

from typing import Dict, Any
from transformers import PretrainedConfig
from model_registry import ModelRegistry
import torch


class ModelFactory:
    """
    Factory to instantiate any supported time-series model by key.
    """

    @staticmethod
    def create(model_key: str, **config_kwargs: Any) -> torch.nn.Module:
        """
        Instantiate a model given its registry key and config overrides.

        Args:
            model_key: One of ["autoformer","ts_transformer","informer","patchtsmixer","patchtst","timesfm"]
            config_kwargs: All kwargs to pass to the Config constructor.

        Returns:
            A HuggingFace PreTrainedModel ready for training or inference.
        """
        entry = ModelRegistry.get(model_key)
        # Initialize config
        config: PretrainedConfig = entry.config_cls(**config_kwargs)
        # Load model (supports from_pretrained when model_key == "timesfm")
        if model_key == "timesfm":
            return entry.model_cls.from_pretrained(model_key, **config_kwargs)
        return entry.model_cls(config)

    @staticmethod
    def forward_step(model: torch.nn.Module, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified forward pass for all registered models.

        Args:
            model: Instantiated HuggingFace model.
            batch: Dict of input tensors, e.g., past_values, past_time_features, etc.

        Returns:
            Model outputs (distribution or point forecasts).
        """
        return model(**batch)
