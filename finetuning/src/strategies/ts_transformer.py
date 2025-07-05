# strategies/ts_transformer.py

from typing import Dict, Any
import torch
import torch.nn as nn
import sys
import os
from transformers import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
    Trainer,
)
from peft_config import PEFTConfig
from .base import ModelStrategy

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common"))
from logger import LoggerFactory, LogLevel, LoggerType

logger = LoggerFactory.get_logger(
    __name__,
    level=LogLevel.INFO,
    logger_type=LoggerType.STANDARD,
)


class TimeSeriesTransformerTimeSeriesTrainer(Trainer):
    """
    Custom trainer for TimeSeriesTransformer model.
    """

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Custom loss computation for TimeSeriesTransformer model.
        """
        past_values = inputs["past_values"]
        future_values = inputs["future_values"]

        # Move tensors to model device and ensure gradients are enabled
        device = next(model.parameters()).device
        past_values = past_values.to(device).requires_grad_(True)
        future_values = future_values.to(device)

        # Create required inputs for TimeSeriesTransformer
        batch_size, context_length, num_features = past_values.shape
        prediction_length = future_values.shape[1]

        # Create observed masks with correct dimensions (no feature dimension)
        past_observed_mask = torch.ones(
            batch_size, context_length, device=device, dtype=torch.bool
        )
        future_observed_mask = torch.ones(
            batch_size, prediction_length, device=device, dtype=torch.bool
        )

        # Create time features - single feature of zeros
        past_time_features = torch.zeros(batch_size, context_length, 1, device=device)
        future_time_features = torch.zeros(
            batch_size, prediction_length, 1, device=device
        )

        # For training, create decoder input by shifting future values
        decoder_input = torch.cat(
            [
                past_values[:, -1:, :],  # Use last past value as start
                future_values[:, :-1, :],  # Shift future values
            ],
            dim=1,
        )

        try:
            # TimeSeriesTransformer expects specific inputs during training
            outputs = model(
                past_values=past_values,
                past_time_features=past_time_features,
                past_observed_mask=past_observed_mask,
                future_time_features=future_time_features,
                future_observed_mask=future_observed_mask,
                future_values=future_values,  # For training
            )
        except Exception as e:
            logger.error(f"TimeSeriesTransformer forward failed: {e}")
            # Don't try fallback - raise the error immediately
            raise RuntimeError(
                f"TimeSeriesTransformer model forward pass failed: {e}"
            ) from e

        # Extract predictions
        if hasattr(outputs, "prediction_outputs"):
            predictions = outputs.prediction_outputs
        elif hasattr(outputs, "last_hidden_state"):
            predictions = outputs.last_hidden_state
        else:
            predictions = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        # Ensure predictions match target shape
        if predictions.shape != future_values.shape:
            # Handle shape mismatches
            if len(predictions.shape) == 4:  # (batch, features, seq_len, other_dim)
                predictions = (
                    predictions.squeeze(-1)
                    if predictions.shape[-1] == 1
                    else predictions.mean(dim=-1)
                )
            if predictions.shape[1] != future_values.shape[1]:
                min_seq_len = min(predictions.shape[1], future_values.shape[1])
                predictions = predictions[:, :min_seq_len]
                future_values = future_values[:, :min_seq_len]
            if len(predictions.shape) == 2:
                predictions = predictions.unsqueeze(-1)

        # Compute MSE loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, future_values)

        return (loss, outputs) if return_outputs else loss


class TimeSeriesTransformerStrategy(ModelStrategy):
    """
    Strategy implementation for TimeSeriesTransformer model.
    """

    def __init__(self):
        super().__init__("ts_transformer")

    def get_default_config(
        self, context_length: int, prediction_length: int, **kwargs
    ) -> Dict[str, Any]:
        """Get default configuration for TimeSeriesTransformer."""
        config = {
            "prediction_length": prediction_length,
            "context_length": context_length,
            "lags_sequence": [1, 2, 7, 24],  # Reasonable lags for financial data
            "num_time_features": 1,
            "num_dynamic_real_features": 0,
            "num_static_categorical_features": 0,
            "num_static_real_features": 0,
            "cardinality": [],
            "embedding_dimension": [],
            "d_model": 64,
            "encoder_layers": 2,
            "decoder_layers": 1,
            "encoder_attention_heads": 2,
            "decoder_attention_heads": 2,
            "encoder_ffn_dim": 32,
            "decoder_ffn_dim": 32,
            "activation_function": "gelu",
            "dropout": 0.1,
            "encoder_layerdrop": 0.1,
            "decoder_layerdrop": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.1,
            "num_parallel_samples": 100,
            "init_std": 0.02,
            "use_cache": True,
        }
        config.update(kwargs)
        return config

    def get_default_peft_config(self) -> PEFTConfig:
        """Get default PEFT configuration for TimeSeriesTransformer."""
        return PEFTConfig(
            peft_method="lora",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )

    def create_model(self, **config_kwargs) -> TimeSeriesTransformerForPrediction:
        """Create TimeSeriesTransformer model."""
        config = TimeSeriesTransformerConfig(**config_kwargs)
        model = TimeSeriesTransformerForPrediction(config)
        self.logger.info(
            f"Created TimeSeriesTransformer model with config: {config_kwargs}"
        )
        return model

    def prepare_data_collator(self) -> callable:
        """Return data collator for TimeSeriesTransformer."""

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

            batch_size, context_length, num_features = past_values.shape
            prediction_length = future_values.shape[1]

            # Create required inputs for TimeSeriesTransformer
            past_observed_mask = torch.ones(
                batch_size, context_length, dtype=torch.bool
            )
            past_time_features = torch.zeros(
                batch_size, context_length, 1, dtype=torch.float32
            )

            return {
                "past_values": past_values,
                "future_values": future_values,
                "past_time_features": past_time_features,
                "past_observed_mask": past_observed_mask,
            }

        return collate_fn

    def create_trainer_class(self) -> type:
        """Return TimeSeriesTransformer-specific trainer class."""
        return TimeSeriesTransformerTimeSeriesTrainer
