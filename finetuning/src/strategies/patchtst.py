# strategies/patchtst.py

from typing import Dict, Any
import torch
import torch.nn as nn
import sys
import os
from transformers import PatchTSTConfig, PatchTSTModel, Trainer
from peft_config import PEFTConfig
from .base import ModelStrategy

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common"))
from logger import LoggerFactory, LogLevel, LoggerType

logger = LoggerFactory.get_logger(
    __name__,
    level=LogLevel.INFO,
    logger_type=LoggerType.STANDARD,
)


class PatchTSTTimeSeriesTrainer(Trainer):
    """
    Custom trainer for PatchTST model that handles loss computation.
    """

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Custom loss computation for PatchTST model.
        """
        past_values = inputs["past_values"]
        future_values = inputs["future_values"]

        # Move tensors to model device
        device = next(model.parameters()).device
        past_values = past_values.to(device)
        future_values = future_values.to(device)

        # Run forward pass
        outputs = model(past_values=past_values)

        # Extract predictions from PatchTST output
        if hasattr(outputs, "last_hidden_state"):
            predictions = outputs.last_hidden_state
        elif hasattr(outputs, "prediction_outputs"):
            predictions = outputs.prediction_outputs
        else:
            predictions = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        # Handle shape mismatches for PatchTST
        target_shape = future_values.shape

        if len(predictions.shape) == 4:  # (batch, features, num_patches, patch_length)
            batch_size, num_features, num_patches, patch_length = predictions.shape
            # Reshape to (batch, features, sequence_length)
            predictions = predictions.view(batch_size, num_features, -1)
            # Take only the prediction_length timesteps
            if predictions.shape[2] >= target_shape[1]:
                predictions = predictions[:, :, : target_shape[1]]
            # Transpose to (batch, sequence_length, features)
            predictions = predictions.transpose(1, 2)
        elif len(predictions.shape) == 3:
            # Adjust dimensions if needed
            if predictions.shape[1] != target_shape[1]:
                # Take first prediction_length steps
                predictions = predictions[:, : target_shape[1], :]
            if predictions.shape[2] != target_shape[2]:
                predictions = predictions.mean(dim=2, keepdim=True)

        # Ensure shapes match exactly
        if predictions.shape != future_values.shape:
            min_seq_len = min(predictions.shape[1], future_values.shape[1])
            min_features = min(predictions.shape[2], future_values.shape[2])
            predictions = predictions[:, :min_seq_len, :min_features]
            future_values = future_values[:, :min_seq_len, :min_features]

        # Compute MSE loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, future_values)

        return (loss, outputs) if return_outputs else loss


class PatchTSTStrategy(ModelStrategy):
    """
    Strategy implementation for PatchTST model.
    """

    def __init__(self):
        super().__init__("patchtst")

    def get_default_config(
        self, context_length: int, prediction_length: int, **kwargs
    ) -> Dict[str, Any]:
        """Get default configuration for PatchTST."""
        config = {
            "context_length": context_length,
            "prediction_length": prediction_length,
            "num_input_channels": 1,
            "patch_length": 16,
            "stride": 8,
            "hidden_size": 128,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "intermediate_size": 256,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "random_mask_ratio": 0.0,  # No masking for forecasting
        }
        config.update(kwargs)
        return config

    def get_default_peft_config(self) -> PEFTConfig:
        """Get default PEFT configuration for PatchTST."""
        return PEFTConfig(
            peft_method="lora",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )

    def create_model(self, **config_kwargs) -> PatchTSTModel:
        """Create PatchTST model."""
        config = PatchTSTConfig(**config_kwargs)
        model = PatchTSTModel(config)
        self.logger.info(f"Created PatchTST model with config: {config_kwargs}")
        return model

    def prepare_data_collator(self) -> callable:
        """Return data collator for PatchTST."""

        def collate_fn(batch):
            # Extract past_values and future_values from the batch
            past_values = torch.tensor(
                [example["past_values"] for example in batch], dtype=torch.float32
            )
            future_values = torch.tensor(
                [example["future_values"] for example in batch], dtype=torch.float32
            )

            # Add feature dimension: (batch_size, sequence_length) -> (batch_size, sequence_length, num_features)
            past_values = past_values.unsqueeze(-1)
            future_values = future_values.unsqueeze(-1)

            return {
                "past_values": past_values,
                "future_values": future_values,
            }

        return collate_fn

    def create_trainer_class(self) -> type:
        """Return PatchTST-specific trainer class."""
        return PatchTSTTimeSeriesTrainer
