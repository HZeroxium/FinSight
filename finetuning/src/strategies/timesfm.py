# strategies/timesfm.py

from typing import Dict, Any
import torch
import torch.nn as nn
import sys
import os
from transformers import TimesFmConfig, TimesFmModelForPrediction, Trainer
from peft_config import PEFTConfig
from .base import ModelStrategy

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common"))
from logger import LoggerFactory, LogLevel, LoggerType

logger = LoggerFactory.get_logger(
    __name__,
    level=LogLevel.INFO,
    logger_type=LoggerType.STANDARD,
)


class TimesFMTimeSeriesTrainer(Trainer):
    """
    Custom trainer for TimesFM model.
    """

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Custom loss computation for TimesFM model.
        """
        past_values = inputs["past_values"]
        future_values = inputs["future_values"]

        # Move tensors to model device
        device = next(model.parameters()).device
        past_values = past_values.to(device)
        future_values = future_values.to(device)

        # TimesFM expects 2D input: [batch_size, sequence_length]
        # If we have 3D input [batch_size, sequence_length, features], squeeze the last dimension
        if len(past_values.shape) == 3:
            if past_values.shape[2] == 1:
                past_values = past_values.squeeze(-1)
            else:
                # Take the mean across features or use first feature
                past_values = past_values.mean(dim=-1)

        try:
            outputs = model(past_values=past_values)
        except Exception as e:
            logger.warning(f"TimesFM forward failed with error: {e}")
            # Try alternative input format
            if len(past_values.shape) == 1:
                past_values = past_values.unsqueeze(0)
            outputs = model(past_values=past_values)

        # Extract predictions
        if hasattr(outputs, "prediction_outputs"):
            predictions = outputs.prediction_outputs
        elif hasattr(outputs, "forecast"):
            predictions = outputs.forecast
        elif hasattr(outputs, "last_hidden_state"):
            predictions = outputs.last_hidden_state
        else:
            predictions = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        # Handle prediction shapes to match target
        target_shape = future_values.shape

        # Ensure predictions have proper dimensions
        if len(predictions.shape) == 2 and len(target_shape) == 3:
            # Add feature dimension: [batch, seq] -> [batch, seq, 1]
            predictions = predictions.unsqueeze(-1)
        elif len(predictions.shape) == 3 and len(target_shape) == 3:
            # Ensure feature dimensions match
            if predictions.shape[2] != target_shape[2]:
                if target_shape[2] == 1:
                    predictions = predictions.mean(dim=2, keepdim=True)
                else:
                    predictions = predictions[:, :, : target_shape[2]]

        # Ensure sequence length matches
        if predictions.shape[1] != target_shape[1]:
            min_len = min(predictions.shape[1], target_shape[1])
            predictions = predictions[:, :min_len]
            future_values = future_values[:, :min_len]
            if len(predictions.shape) == 2:
                predictions = predictions.unsqueeze(-1)
            if len(future_values.shape) == 2:
                future_values = future_values.unsqueeze(-1)

        # Compute MSE loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, future_values)

        return (loss, outputs) if return_outputs else loss


class TimesFMStrategy(ModelStrategy):
    """
    Strategy implementation for TimesFM model.
    """

    def __init__(self):
        super().__init__("timesfm")

    def get_default_config(
        self, context_length: int, prediction_length: int, **kwargs
    ) -> Dict[str, Any]:
        """Get default configuration for TimesFM."""
        config = {
            "context_length": context_length,
            "prediction_length": prediction_length,
            "num_input_channels": 1,
            "num_layers": 12,
            "d_model": 512,
            "num_attention_heads": 8,
            "ffn_dim": 2048,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_function": "gelu",
            "max_position_embeddings": 1024,
            "init_std": 0.02,
            "use_cache": True,
            "vocab_size": 32000,  # For tokenization if needed
        }
        config.update(kwargs)
        return config

    def get_default_peft_config(self) -> PEFTConfig:
        """Get default PEFT configuration for TimesFM."""
        return PEFTConfig(
            peft_method="lora",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )

    def create_model(self, **config_kwargs) -> TimesFmModelForPrediction:
        """Create TimesFM model."""
        # For TimesFM, we might need to load from pretrained
        try:
            model = TimesFmModelForPrediction.from_pretrained("google/timesfm-1.0-200m")
            self.logger.info("Loaded pretrained TimesFM model")
            return model
        except Exception as e:
            self.logger.warning(f"Failed to load pretrained TimesFM, creating new: {e}")
            config = TimesFmConfig(**config_kwargs)
            model = TimesFmModelForPrediction(config)
            self.logger.info(f"Created TimesFM model with config: {config_kwargs}")
            return model

    def prepare_data_collator(self) -> callable:
        """Return data collator for TimesFM."""

        def collate_fn(batch):
            # Extract past_values and future_values from the batch
            past_values = torch.tensor(
                [example["past_values"] for example in batch], dtype=torch.float32
            )
            future_values = torch.tensor(
                [example["future_values"] for example in batch], dtype=torch.float32
            )

            # Add feature dimension if needed
            if len(past_values.shape) == 2:
                past_values = past_values.unsqueeze(-1)
            if len(future_values.shape) == 2:
                future_values = future_values.unsqueeze(-1)

            return {
                "past_values": past_values,
                "future_values": future_values,
            }

        return collate_fn

    def create_trainer_class(self) -> type:
        """Return TimesFM-specific trainer class."""
        return TimesFMTimeSeriesTrainer

    def load_model_for_inference(self, model_path: str, **config_kwargs):
        """Override for TimesFM since it uses pretrained models."""
        try:
            # Try to load as PEFT model first
            from peft import PeftModel

            # Load base pretrained model
            base_model = TimesFmModelForPrediction.from_pretrained(
                "google/timesfm-1.0-200m"
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            self.logger.info(f"Loaded PEFT TimesFM model from {model_path}")
            return model
        except Exception as e:
            self.logger.warning(f"Failed to load PEFT model, using base TimesFM: {e}")
            try:
                return TimesFmModelForPrediction.from_pretrained(
                    "google/timesfm-1.0-200m"
                )
            except Exception as e2:
                self.logger.error(f"Failed to load base TimesFM: {e2}")
                # Fallback to new model
                return self.create_model(**config_kwargs)
