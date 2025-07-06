# utils/config_utils.py

"""
Configuration utilities for converting between different config formats
"""

from typing import Dict, Any, Optional, List
from ..schemas.training_schemas import TrainingRequest
from ..schemas.model_schemas import ModelConfig


class ConfigUtils:
    """Utilities for configuration management and conversion"""

    @staticmethod
    def training_request_to_model_config(request: TrainingRequest) -> ModelConfig:
        """
        Convert TrainingRequest to ModelConfig

        Args:
            request: Training request from API

        Returns:
            ModelConfig instance
        """
        return ModelConfig(
            context_length=request.context_length,
            prediction_length=request.prediction_length,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            num_epochs=request.num_epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            use_technical_indicators=request.use_technical_indicators,
            add_datetime_features=request.add_datetime_features,
            normalize_features=request.normalize_features,
            train_ratio=request.train_ratio,
            val_ratio=request.val_ratio,
        )

    @staticmethod
    def get_model_defaults(model_type: str) -> Dict[str, Any]:
        """
        Get default configuration for specific model types

        Args:
            model_type: Model type identifier

        Returns:
            Default configuration dictionary
        """
        defaults = {
            "ibm/patchtst-forecasting": {
                "patch_length": 8,
                "d_model": 128,
                "num_layers": 4,
                "num_attention_heads": 4,
                "dropout": 0.1,
                "head_dropout": 0.1,
                "pooling_type": "mean",
                "norm_type": "batchnorm",
                "activation": "gelu",
                "pre_norm": True,
            },
            "ibm/patchtsmixer-forecasting": {
                "patch_length": 8,
                "d_model": 128,
                "num_layers": 4,
                "expansion_factor": 2,
                "dropout": 0.1,
                "head_dropout": 0.1,
                "pooling_type": "mean",
                "norm_type": "BatchNorm",
                "activation": "gelu",
                "pre_norm": True,
            },
            "pytorch_lightning_transformer": {
                "d_model": 128,
                "n_heads": 8,
                "n_layers": 4,
                "dropout": 0.1,
            },
        }

        return defaults.get(model_type, {})
