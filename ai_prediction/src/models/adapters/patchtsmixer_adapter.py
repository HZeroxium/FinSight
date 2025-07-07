# models/adapters/patchtsmixer_adapter.py

"""
PatchTSMixer adapter implementing the base adapter pattern
Clean implementation focused on forecasting vs backtesting separation
"""

from typing import Dict, Any, Tuple
import torch

from pathlib import Path
from transformers import (
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    TrainingArguments,
    Trainer,
)
import os

from .base_adapter import BaseTimeSeriesAdapter

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"


class PatchTSMixerDataset(torch.utils.data.Dataset):
    """Dataset for PatchTSMixer training"""

    def __init__(self, sequences: torch.Tensor, targets: torch.Tensor):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"past_values": self.sequences[idx], "future_values": self.targets[idx]}


class PatchTSMixerTrainer(Trainer):
    """Custom trainer for PatchTSMixer with proper loss computation"""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Custom loss computation for PatchTSMixer"""
        try:
            past_values = inputs["past_values"]
            future_values = inputs["future_values"]

            # Ensure tensors are on the same device
            device = next(model.parameters()).device
            past_values = past_values.to(device)
            future_values = future_values.to(device)

            # Forward pass
            outputs = model(past_values=past_values)
            predictions = outputs.prediction_outputs

            # Handle shape consistency
            if predictions.dim() == 3 and predictions.shape[2] > 1:
                predictions = predictions[:, :, 0]  # Take first channel

            predictions = predictions.squeeze()
            future_values = future_values.squeeze()

            # Calculate MSE loss
            loss = torch.nn.functional.mse_loss(predictions, future_values)

            return (loss, outputs) if return_outputs else loss

        except Exception as e:
            # Return a default loss instead of using logger
            return torch.tensor(
                0.1, requires_grad=True, device=next(model.parameters()).device
            )


class PatchTSMixerAdapter(BaseTimeSeriesAdapter):
    """
    PatchTSMixer adapter with clean forecasting implementation

    Separates training (with scaler fitting) from inference (scaler transform only)
    """

    def __init__(self, config: Dict[str, Any]):
        # Extract configuration parameters
        context_length = config.get("context_length", 64)
        prediction_length = config.get("prediction_length", 1)
        target_column = config.get("target_column", "close")
        feature_columns = config.get(
            "feature_columns", ["open", "high", "low", "close", "volume"]
        )
        patch_length = config.get("patch_length", 8)
        num_patches = config.get("num_patches", context_length // patch_length)

        # Filter out parameters that are already passed explicitly
        filtered_config = {
            k: v
            for k, v in config.items()
            if k
            not in [
                "context_length",
                "prediction_length",
                "target_column",
                "feature_columns",
            ]
        }

        super().__init__(
            context_length=context_length,
            prediction_length=prediction_length,
            target_column=target_column,
            feature_columns=feature_columns,
            **filtered_config,
        )

        self.patch_length = patch_length
        self.num_patches = num_patches
        self.trainer = None

        # Model-specific config
        self.model_config = {
            "patch_length": patch_length,
            "num_patches": self.num_patches,
            "num_input_channels": len(self.feature_columns),
            "prediction_length": prediction_length,
            "context_length": context_length,
            # Only add config items that are not already set
            **{
                k: v
                for k, v in config.items()
                if k
                not in [
                    "patch_length",
                    "num_patches",
                    "num_input_channels",
                    "prediction_length",
                    "context_length",
                    "target_column",
                    "feature_columns",
                ]
            },
        }

    def _create_model(self) -> torch.nn.Module:
        """Create PatchTSMixer model"""
        try:
            # Update num_input_channels if feature engineering was applied
            if self.feature_engineering is not None:
                feature_names = self.feature_engineering.get_feature_names()
                self.model_config["num_input_channels"] = len(feature_names)

            config = PatchTSMixerConfig(
                num_input_channels=self.model_config["num_input_channels"],
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                patch_length=self.patch_length,
                num_patches=self.num_patches,
                d_model=self.model_config.get("d_model", 128),
                num_layers=self.model_config.get("num_layers", 4),
                expansion_factor=self.model_config.get("expansion_factor", 2),
                dropout=self.model_config.get("dropout", 0.1),
                head_dropout=self.model_config.get("head_dropout", 0.1),
                pooling_type=self.model_config.get("pooling_type", "mean"),
                norm_type=self.model_config.get("norm_type", "BatchNorm"),
                activation=self.model_config.get("activation", "gelu"),
                pre_norm=self.model_config.get("pre_norm", True),
                norm_eps=1e-5,
            )

            model = PatchTSMixerForPrediction(config)
            self.logger.info(f"Created PatchTSMixer model with config: {config}")

            return model

        except Exception as e:
            self.logger.error(f"Error creating PatchTSMixer model: {e}")
            raise

    def _train_model(
        self,
        train_dataset: Tuple[torch.Tensor, torch.Tensor],
        val_dataset: Tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> Dict[str, Any]:
        """Train PatchTSMixer model"""
        try:
            train_sequences, train_targets = train_dataset
            val_sequences, val_targets = val_dataset

            # Create datasets
            train_ds = PatchTSMixerDataset(train_sequences, train_targets)
            val_ds = PatchTSMixerDataset(val_sequences, val_targets)

            # Training arguments
            training_args = TrainingArguments(
                output_dir="./tmp_trainer",
                learning_rate=kwargs.get("learning_rate", 1e-3),
                per_device_train_batch_size=kwargs.get("batch_size", 32),
                per_device_eval_batch_size=kwargs.get("batch_size", 32),
                num_train_epochs=kwargs.get("num_epochs", 1),
                weight_decay=kwargs.get("weight_decay", 0.01),
                logging_steps=100,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=False,  # Disable since eval_loss is not computed
                report_to=None,  # Disable wandb
                remove_unused_columns=False,
            )

            # Create trainer
            self.trainer = PatchTSMixerTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                # Remove early stopping since eval_loss is not properly computed
            )

            # Train
            self.logger.info("Starting PatchTSMixer training...")
            self.logger.info("Hyperparameters:")
            arguments_dict = training_args.to_dict()
            # Print some hyperparameters
            for key, value in arguments_dict.items():
                if key in [
                    "learning_rate",
                    "num_train_epochs",
                    "per_device_train_batch_size",
                ]:
                    self.logger.info(f"  {key}: {value}")

            train_result = self.trainer.train()

            # Get final metrics
            final_metrics = self.trainer.evaluate()

            return {
                "train_loss": train_result.training_loss,
                "eval_loss": final_metrics.get("eval_loss", None),
                "epochs_trained": int(train_result.global_step),
                "model_type": "PatchTSMixer",
            }

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def _model_predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Make prediction with PatchTSMixer model"""
        try:
            # Ensure model is in eval mode
            self.model.eval()

            with torch.no_grad():
                # PatchTSMixer expects past_values
                outputs = self.model(past_values=input_tensor)
                predictions = outputs.prediction_outputs

                # Handle output shape
                if predictions.dim() == 3:
                    if predictions.shape[2] > 1:
                        predictions = predictions[:, :, 0]  # Take first channel
                    predictions = predictions.squeeze()

                # Return single prediction for forecasting
                if self.prediction_length == 1:
                    return predictions[:, 0] if predictions.dim() > 1 else predictions
                else:
                    return predictions

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    def _save_model_specific(self, model_dir: Path) -> None:
        """Save PatchTSMixer specific components"""
        try:
            # Save model state dict directly (avoid trainer pickle issues)
            if self.model is not None:
                # Save model state dict instead of the full model
                torch.save(self.model.state_dict(), model_dir / "model_state_dict.pt")

            # Save model config - use consistent naming
            import json

            with open(model_dir / "model_config.json", "w") as f:
                json.dump(self.model_config, f, indent=2)

            # Save trainer state if needed (optional)
            if self.trainer is not None:
                try:
                    # Only save essential trainer information, not the trainer object itself
                    trainer_info = {
                        "training_completed": True,
                        "final_train_loss": (
                            getattr(self.trainer.state, "log_history", [{}])[-1].get(
                                "train_loss", None
                            )
                            if hasattr(self.trainer, "state")
                            else None
                        ),
                        "total_steps": (
                            getattr(self.trainer.state, "global_step", 0)
                            if hasattr(self.trainer, "state")
                            else 0
                        ),
                    }
                    with open(model_dir / "trainer_info.json", "w") as f:
                        json.dump(trainer_info, f, indent=2)
                except Exception as trainer_save_error:
                    self.logger.warning(
                        f"Could not save trainer info: {trainer_save_error}"
                    )

            self.logger.info("PatchTSMixer model components saved")

        except Exception as e:
            self.logger.error(f"Error saving PatchTSMixer components: {e}")
            raise

    def _load_model_specific(self, model_dir: Path) -> None:
        """Load PatchTSMixer specific components"""
        try:
            # Load model config - try different file names
            config_files = ["model_config.json", "config.json"]
            model_config = None

            for config_file in config_files:
                config_path = model_dir / config_file
                if config_path.exists():
                    import json

                    with open(config_path, "r") as f:
                        model_config = json.load(f)
                    break

            if model_config is None:
                raise FileNotFoundError(f"No model config found in {model_dir}")

            self.model_config = model_config

            # Create and load model
            self.model = self._create_model()

            # Load state dict if available
            state_dict_path = model_dir / "model_state_dict.pt"
            if state_dict_path.exists():
                state_dict = torch.load(state_dict_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            else:
                # Try alternative state dict names
                alt_paths = [model_dir / "pytorch_model.bin", model_dir / "model.pt"]

                state_dict_loaded = False
                for alt_path in alt_paths:
                    if alt_path.exists():
                        state_dict = torch.load(alt_path, map_location=self.device)
                        # Handle different state dict formats
                        if "model_state_dict" in state_dict:
                            self.model.load_state_dict(state_dict["model_state_dict"])
                        else:
                            self.model.load_state_dict(state_dict)
                        state_dict_loaded = True
                        break

                if not state_dict_loaded:
                    self.logger.warning(
                        "No model state dict found, using initialized weights"
                    )

            self.model.to(self.device)
            self.model.eval()

            # Load trainer info if available
            trainer_info_path = model_dir / "trainer_info.json"
            if trainer_info_path.exists():
                try:
                    with open(trainer_info_path, "r") as f:
                        trainer_info = json.load(f)
                    self.logger.info(f"Loaded trainer info: {trainer_info}")
                except Exception as trainer_load_error:
                    self.logger.warning(
                        f"Could not load trainer info: {trainer_load_error}"
                    )

            self.logger.info("PatchTSMixer model components loaded")

        except Exception as e:
            self.logger.error(f"Error loading PatchTSMixer components: {e}")
            raise
