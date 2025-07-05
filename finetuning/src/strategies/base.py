# strategies/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from transformers import PreTrainedModel, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from peft_config import PEFTConfig
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common"))
from logger import LoggerFactory, LogLevel, LoggerType

logger = LoggerFactory.get_logger(
    __name__,
    level=LogLevel.INFO,
    logger_type=LoggerType.STANDARD,
)


class ModelStrategy(ABC):
    """
    Base strategy interface for time series models.
    Each model implementation should inherit from this class.
    """

    def __init__(self, model_key: str):
        self.model_key = model_key
        self.logger = LoggerFactory.get_logger(
            f"{__name__}.{self.__class__.__name__}",
            level=LogLevel.INFO,
            logger_type=LoggerType.STANDARD,
        )

    @abstractmethod
    def get_default_config(
        self, context_length: int, prediction_length: int, **kwargs
    ) -> Dict[str, Any]:
        """
        Get default model configuration parameters.

        Args:
            context_length: Number of past timesteps
            prediction_length: Number of future timesteps to predict
            **kwargs: Additional configuration parameters

        Returns:
            Dictionary of model configuration parameters
        """
        pass

    @abstractmethod
    def get_default_peft_config(self) -> PEFTConfig:
        """
        Get default PEFT configuration for this model.

        Returns:
            PEFTConfig with appropriate target modules and parameters
        """
        pass

    @abstractmethod
    def create_model(self, **config_kwargs) -> PreTrainedModel:
        """
        Create and return the model instance.

        Args:
            **config_kwargs: Model configuration parameters

        Returns:
            Instantiated model
        """
        pass

    @abstractmethod
    def prepare_data_collator(self) -> callable:
        """
        Return data collator function specific to this model.

        Returns:
            Data collator function
        """
        pass

    @abstractmethod
    def create_trainer_class(self) -> type:
        """
        Return the appropriate Trainer class for this model.

        Returns:
            Trainer class (can be custom or standard HuggingFace Trainer)
        """
        pass

    def get_default_training_args(self, output_dir: str, **kwargs) -> TrainingArguments:
        """
        Get default training arguments. Can be overridden by specific models.

        Args:
            output_dir: Output directory for model saving
            **kwargs: Additional training arguments

        Returns:
            TrainingArguments instance
        """
        default_args = {
            "output_dir": output_dir,
            "num_train_epochs": 5,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "learning_rate": 3e-4,
            "logging_steps": 100,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "report_to": None,
            "remove_unused_columns": False,
            "dataloader_pin_memory": False,
        }
        default_args.update(kwargs)
        return TrainingArguments(**default_args)

    def load_model_for_inference(
        self, model_path: str, **config_kwargs
    ) -> PreTrainedModel:
        """
        Load a trained model for inference.

        Args:
            model_path: Path to the trained model
            **config_kwargs: Model configuration parameters

        Returns:
            Loaded model ready for inference
        """
        try:
            # Try to load as PEFT model first
            from peft import PeftModel

            base_model = self.create_model(**config_kwargs)
            model = PeftModel.from_pretrained(base_model, model_path)
            self.logger.info(f"Loaded PEFT model from {model_path}")
            return model
        except Exception as e:
            self.logger.warning(f"Failed to load as PEFT model: {e}")
            try:
                # Fallback to regular model loading
                from transformers import AutoModel

                model = AutoModel.from_pretrained(model_path)
                self.logger.info(f"Loaded base model from {model_path}")
                return model
            except Exception as e2:
                self.logger.warning(f"Failed to load from path {model_path}: {e2}")
                try:
                    # Try creating model and loading state dict
                    model = self.create_model(**config_kwargs)
                    import torch

                    state_dict = torch.load(
                        f"{model_path}/pytorch_model.bin", map_location="cpu"
                    )
                    model.load_state_dict(state_dict, strict=False)
                    self.logger.info(f"Loaded model state dict from {model_path}")
                    return model
                except Exception as e3:
                    self.logger.error(f"All loading methods failed: {e3}")
                    # Return newly created model as fallback
                    self.logger.warning("Using newly created model as fallback")
                    return self.create_model(**config_kwargs)

    def extract_predictions(self, model_output: Any) -> torch.Tensor:
        """
        Extract predictions from model output. Can be overridden by specific models.

        Args:
            model_output: Output from model forward pass

        Returns:
            Tensor containing predictions
        """
        if hasattr(model_output, "prediction_outputs"):
            return model_output.prediction_outputs
        elif hasattr(model_output, "forecast"):
            return model_output.forecast
        elif hasattr(model_output, "last_hidden_state"):
            return model_output.last_hidden_state
        elif hasattr(model_output, "predictions"):
            return model_output.predictions
        elif isinstance(model_output, (tuple, list)) and len(model_output) > 0:
            return model_output[0]
        else:
            raise ValueError(
                f"Could not extract predictions from model output: {type(model_output)}"
            )
