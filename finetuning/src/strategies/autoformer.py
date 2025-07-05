# strategies/autoformer.py

from typing import Dict, Any
import torch
import torch.nn as nn
import sys
import os
from transformers import AutoformerConfig, AutoformerForPrediction, Trainer
from peft_config import PEFTConfig
from .base import ModelStrategy

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common"))
from logger import LoggerFactory, LogLevel, LoggerType

logger = LoggerFactory.get_logger(
    __name__,
    level=LogLevel.INFO,
    logger_type=LoggerType.STANDARD,
)


class AutoformerTimeSeriesTrainer(Trainer):
    """
    Custom trainer for Autoformer model.
    """

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Custom loss computation for Autoformer model.
        """
        past_values = inputs["past_values"]
        future_values = inputs["future_values"]
        past_time_features = inputs.get("past_time_features")
        past_observed_mask = inputs.get("past_observed_mask")

        # Move tensors to model device and ensure gradients are enabled
        device = next(model.parameters()).device
        past_values = past_values.to(device).requires_grad_(True)
        future_values = future_values.to(device)

        if past_time_features is not None:
            past_time_features = past_time_features.to(device)
        if past_observed_mask is not None:
            past_observed_mask = past_observed_mask.to(device)

        # Create required inputs for Autoformer
        batch_size, context_length, num_features = past_values.shape
        prediction_length = future_values.shape[1]

        # Create required inputs if not provided
        if past_observed_mask is None:
            past_observed_mask = torch.ones(
                batch_size, context_length, device=device, dtype=torch.bool
            )

        if past_time_features is None:
            # Autoformer needs time features to match context length
            past_time_features = torch.zeros(
                batch_size, context_length, 1, device=device, dtype=torch.float32
            )

        # Create future inputs for training
        future_observed_mask = torch.ones(
            batch_size, prediction_length, device=device, dtype=torch.bool
        )
        future_time_features = torch.zeros(
            batch_size, prediction_length, 1, device=device, dtype=torch.float32
        )

        try:
            # Autoformer expects all these inputs during training
            outputs = model(
                past_values=past_values,
                past_time_features=past_time_features,
                past_observed_mask=past_observed_mask,
                future_time_features=future_time_features,
                future_observed_mask=future_observed_mask,
                future_values=future_values,  # For training
            )
        except Exception as e:
            logger.error(f"Autoformer forward failed: {e}")
            # Don't try fallback - raise the error immediately
            raise RuntimeError(f"Autoformer model forward pass failed: {e}") from e

        # Extract predictions
        if hasattr(outputs, "prediction_outputs"):
            predictions = outputs.prediction_outputs
        elif hasattr(outputs, "last_hidden_state"):
            predictions = outputs.last_hidden_state
        elif hasattr(outputs, "logits"):
            predictions = outputs.logits
        else:
            # Try to get the first output
            predictions = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        # Ensure predictions match target shape
        if predictions.shape != future_values.shape:
            # Handle shape mismatches carefully
            if len(predictions.shape) == 4:  # (batch, features, seq_len, other_dim)
                predictions = (
                    predictions.squeeze(-1)
                    if predictions.shape[-1] == 1
                    else predictions.mean(dim=-1)
                )
                predictions = predictions.transpose(1, 2)  # (batch, seq_len, features)
            elif (
                len(predictions.shape) == 3
                and predictions.shape[1] != future_values.shape[1]
            ):
                min_seq_len = min(predictions.shape[1], future_values.shape[1])
                predictions = predictions[:, :min_seq_len]
                future_values = future_values[:, :min_seq_len]
            elif len(predictions.shape) == 2:
                predictions = predictions.unsqueeze(-1)

        # Final shape check
        if predictions.shape != future_values.shape:
            logger.error(
                f"Shape mismatch: predictions {predictions.shape} vs targets {future_values.shape}"
            )
            raise RuntimeError(
                f"Cannot resolve shape mismatch: predictions {predictions.shape} vs targets {future_values.shape}"
            )

        # Compute MSE loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, future_values)

        return (loss, outputs) if return_outputs else loss


class AutoformerStrategy(ModelStrategy):
    """
    Strategy implementation for Autoformer model.
    """

    def __init__(self):
        super().__init__("autoformer")

    def get_default_config(
        self, context_length: int, prediction_length: int, **kwargs
    ) -> Dict[str, Any]:
        """Get default configuration for Autoformer."""
        # Try without lags_sequence to see if that resolves the tensor mismatch
        config = {
            "prediction_length": prediction_length,
            "context_length": context_length,
            # Remove lags_sequence entirely to avoid complex tensor operations
            "num_time_features": 1,
            "num_dynamic_real_features": 0,
            "num_static_categorical_features": 0,
            "num_static_real_features": 0,
            "cardinality": [],
            "embedding_dimension": [],
            "d_model": 32,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "encoder_attention_heads": 2,
            "decoder_attention_heads": 2,
            "encoder_ffn_dim": 16,
            "decoder_ffn_dim": 16,
            "activation_function": "gelu",
            "dropout": 0.1,
            "encoder_layerdrop": 0.0,
            "decoder_layerdrop": 0.0,
            "attention_dropout": 0.1,
            "activation_dropout": 0.1,
            "num_parallel_samples": 100,
            "init_std": 0.02,
            "use_cache": True,
        }
        config.update(kwargs)
        return config

    def get_default_peft_config(self) -> PEFTConfig:
        """Get default PEFT configuration for Autoformer."""
        return PEFTConfig(
            peft_method="lora",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )

    def create_model(self, **config_kwargs) -> AutoformerForPrediction:
        """Create Autoformer model."""
        config = AutoformerConfig(**config_kwargs)
        model = AutoformerForPrediction(config)
        self.logger.info(f"Created Autoformer model with config: {config_kwargs}")
        return model

    def prepare_data_collator(self) -> callable:
        """Return data collator for Autoformer."""

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

            # For Autoformer with lags_sequence=[1], the observed_mask should
            # match the lagged dimension, not the full context_length
            # Based on the error, it expects dimension 8 instead of 128
            # This suggests lags processing creates a different dimension

            # The Autoformer model internally processes lags and creates a different
            # tensor structure. Let's provide the minimum required inputs
            past_observed_mask = torch.ones(
                batch_size, context_length, dtype=torch.bool
            )

            # Create time features (1 dimension to match config)
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
        """Return Autoformer-specific trainer class."""
        return AutoformerTimeSeriesTrainer
