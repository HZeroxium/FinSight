# strategies/patchtsmixer.py

from typing import Dict, Any
import torch
import torch.nn as nn
import sys
import os
from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction, Trainer
from peft_config import PEFTConfig
from .base import ModelStrategy

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common"))
from logger import LoggerFactory, LogLevel, LoggerType

logger = LoggerFactory.get_logger(
    __name__,
    level=LogLevel.INFO,
    logger_type=LoggerType.STANDARD,
)


class PatchTSMixerTimeSeriesTrainer(Trainer):
    """
    Custom trainer for PatchTSMixer model.
    """

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Custom loss computation for PatchTSMixer model.
        """
        past_values = inputs["past_values"]
        future_values = inputs["future_values"]

        # Move tensors to model device and ensure gradients are enabled
        device = next(model.parameters()).device
        past_values = past_values.to(device).requires_grad_(True)
        future_values = future_values.to(device)

        # PatchTSMixer expects input format: [batch_size, sequence_length, num_channels]
        # Do NOT transpose - the input should remain [batch, seq, channels]

        try:
            outputs = model(past_values=past_values)
        except Exception as e:
            logger.error(f"PatchTSMixer forward failed: {e}")
            # Don't try fallback - raise the error immediately
            raise RuntimeError(f"PatchTSMixer model forward pass failed: {e}") from e

        # Extract predictions
        if hasattr(outputs, "prediction_outputs"):
            predictions = outputs.prediction_outputs
        elif hasattr(outputs, "forecast"):
            predictions = outputs.forecast
        elif hasattr(outputs, "last_hidden_state"):
            predictions = outputs.last_hidden_state
        else:
            predictions = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        # Handle shape conversion for loss computation
        target_shape = future_values.shape  # [batch, seq, features]

        # Ensure predictions have the right shape
        if len(predictions.shape) == 3:
            if predictions.shape[1] == 1:  # [batch, 1, seq] -> [batch, seq, 1]
                predictions = predictions.transpose(1, 2)
            elif predictions.shape[2] == 1:  # [batch, seq, 1] - already correct
                pass
            else:  # [batch, channels, seq] -> [batch, seq, 1] (take first channel)
                predictions = predictions[:, :1, :].transpose(1, 2)

        # Ensure lengths match
        if predictions.shape[1] != target_shape[1]:
            min_len = min(predictions.shape[1], target_shape[1])
            predictions = predictions[:, :min_len, :]
            future_values = future_values[:, :min_len, :]

        # Ensure feature dimensions match
        if predictions.shape[2] != target_shape[2]:
            if target_shape[2] == 1:
                predictions = predictions.mean(dim=2, keepdim=True)
            else:
                predictions = predictions[:, :, : target_shape[2]]

        # Compute MSE loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, future_values)

        return (loss, outputs) if return_outputs else loss


class PatchTSMixerStrategy(ModelStrategy):
    """
    Strategy implementation for PatchTSMixer model.
    """

    def __init__(self):
        super().__init__("patchtsmixer")

    def get_default_config(
        self, context_length: int, prediction_length: int, **kwargs
    ) -> Dict[str, Any]:
        """Get default configuration for PatchTSMixer."""
        # PatchTSMixer sequence length should match context_length
        config = {
            "context_length": context_length,
            "prediction_length": prediction_length,
            "num_input_channels": 1,
            "patch_length": 16,
            "num_layers": 8,
            "d_model": 128,
            "expansion_factor": 2,
            "num_patches": context_length
            // 16,  # Will be recalculated based on context_length
            "dropout": 0.2,
            "mode": "forecasting",
            "gated_attn": True,
            "norm_mlp": "LayerNorm",
            "self_attn": False,
            "self_attn_heads": 1,
            "use_positional_encoding": False,
            "positional_encoding_type": "sincos",
            "scaling": True,
            "loss": "mse",
            "init_std": 0.02,
        }
        config.update(kwargs)
        return config

    def get_default_peft_config(self) -> PEFTConfig:
        """Get default PEFT configuration for PatchTSMixer."""
        return PEFTConfig(
            peft_method="lora",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["fc1", "fc2", "attn_layer"],
        )

    def create_model(self, **config_kwargs) -> PatchTSMixerForPrediction:
        """Create PatchTSMixer model."""
        config = PatchTSMixerConfig(**config_kwargs)
        model = PatchTSMixerForPrediction(config)
        self.logger.info(f"Created PatchTSMixer model with config: {config_kwargs}")
        return model

    def prepare_data_collator(self) -> callable:
        """Return data collator for PatchTSMixer."""

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
        """Return PatchTSMixer-specific trainer class."""
        return PatchTSMixerTimeSeriesTrainer

    def load_model_for_inference(
        self, model_path: str, **config_kwargs
    ) -> PatchTSMixerForPrediction:
        """
        Load a trained PatchTSMixer model for inference with consistent configuration.

        Args:
            model_path: Path to the trained model
            **config_kwargs: Model configuration parameters

        Returns:
            Loaded model ready for inference
        """
        try:
            # Try to load as PEFT model first with the exact same config
            from peft import PeftModel

            base_model = self.create_model(**config_kwargs)
            model = PeftModel.from_pretrained(base_model, model_path)
            self.logger.info(f"Loaded PEFT model from {model_path}")
            return model
        except Exception as e:
            self.logger.warning(f"Failed to load as PEFT model: {e}")
            try:
                # Try loading base model configuration from saved files
                import os

                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    import json

                    with open(config_path, "r") as f:
                        saved_config = json.load(f)
                    model = PatchTSMixerForPrediction.from_pretrained(model_path)
                    self.logger.info(f"Loaded saved model from {model_path}")
                    return model
            except Exception as e2:
                self.logger.warning(f"Failed to load from path {model_path}: {e2}")

            # Fallback to newly created model
            self.logger.error(f"All loading methods failed: {e}")
            self.logger.warning("Using newly created model as fallback")
            return self.create_model(**config_kwargs)
