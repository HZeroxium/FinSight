# models/adapters/transformer_adapter.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .base_adapter import BaseTimeSeriesAdapter
from ...logger.logger_factory import LoggerFactory

# Try to import settings, fall back to defaults if not available
try:
    from ...core.config import get_settings

    settings = get_settings()
except ImportError:
    settings = None


class TimeSeriesTransformer(pl.LightningModule):
    """PyTorch Lightning Time Series Transformer Model"""

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        seq_length: int = 96,
        pred_length: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.dropout = dropout
        self.learning_rate = learning_rate

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding()

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim * pred_length),
        )

        # Loss function
        self.criterion = nn.MSELoss()

    def _create_positional_encoding(self):
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(self.seq_length, self.d_model)
        position = torch.arange(0, self.seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        """Forward pass"""
        batch_size = x.size(0)

        # Input projection
        x = self.input_projection(x)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = x + self.pos_encoding[:, : x.size(1), :]

        # Apply transformer
        x = self.transformer(x)  # [batch, seq_len, d_model]

        # Take the last time step for prediction
        x = x[:, -1, :]  # [batch, d_model]

        # Output projection
        output = self.output_projection(x)  # [batch, output_dim * pred_length]

        # Reshape to [batch, pred_length, output_dim]
        if self.pred_length > 1:
            output = output.view(batch_size, self.pred_length, self.output_dim)
        else:
            output = output.view(batch_size, self.output_dim)

        return output

    def training_step(self, batch, batch_idx):
        """Training step"""
        x, y = batch
        y_hat = self(x)

        # Ensure y and y_hat have the same shape
        if y_hat.dim() != y.dim():
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            elif y_hat.dim() == 1:
                y_hat = y_hat.unsqueeze(-1)

        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, y = batch
        y_hat = self(x)

        # Ensure y and y_hat have the same shape
        if y_hat.dim() != y.dim():
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            elif y_hat.dim() == 1:
                y_hat = y_hat.unsqueeze(-1)

        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class TransformerAdapter(BaseTimeSeriesAdapter):
    """Adapter for PyTorch Lightning Time Series Transformer"""

    def __init__(self, config: Dict[str, Any]):
        # Extract configuration parameters
        context_length = config.get("context_length", 96)
        prediction_length = config.get("prediction_length", 1)
        target_column = config.get("target_column", "close")
        feature_columns = config.get(
            "feature_columns", ["open", "high", "low", "close", "volume"]
        )

        # Ensure feature_columns is not None
        if feature_columns is None:
            feature_columns = ["open", "high", "low", "close", "volume"]

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

        # Model hyperparameters - handle input_dim calculation safely
        default_input_dim = len(feature_columns) if feature_columns else 5
        self.input_dim = config.get("input_dim", default_input_dim)
        self.output_dim = config.get("output_dim", 1)
        self.d_model = config.get("d_model", 128)
        self.n_heads = config.get("n_heads", 8)
        self.n_layers = config.get("n_layers", 4)
        self.dropout = config.get("dropout", 0.1)
        self.learning_rate = config.get("learning_rate", 1e-4)

        # Training parameters
        self.batch_size = config.get("batch_size", 32)
        self.num_epochs = config.get("num_epochs", 1)

    def _create_model(self) -> torch.nn.Module:
        """Create PyTorch Lightning Time Series Transformer model"""
        try:
            # Update input dimension if feature engineering was applied
            if self.feature_engineering is not None:
                feature_names = self.feature_engineering.get_feature_names()
                self.input_dim = len(feature_names)
                self.logger.info(
                    f"Updated input_dim to {self.input_dim} based on feature engineering"
                )

            model = TimeSeriesTransformer(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                seq_length=self.context_length,
                pred_length=self.prediction_length,
                dropout=self.dropout,
                learning_rate=self.learning_rate,
            )

            self.logger.info(
                f"Created TimeSeriesTransformer model with input_dim={self.input_dim}"
            )
            return model

        except Exception as e:
            self.logger.error(f"Error creating TimeSeriesTransformer model: {e}")
            # Don't raise here, return None to be handled by caller
            return None

    def _train_model(
        self,
        train_dataset: Tuple[torch.Tensor, torch.Tensor],
        val_dataset: Tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> Dict[str, Any]:
        """Train PyTorch Lightning Time Series Transformer model"""
        try:
            train_sequences, train_targets = train_dataset
            val_sequences, val_targets = val_dataset

            # Validate inputs
            if train_sequences is None or train_targets is None:
                raise ValueError("Training sequences and targets cannot be None")

            if len(train_sequences) == 0 or len(train_targets) == 0:
                raise ValueError("Training sequences and targets cannot be empty")

            # Update hyperparameters with kwargs if provided
            batch_size = kwargs.get("batch_size", self.batch_size)
            num_epochs = kwargs.get("num_epochs", self.num_epochs)
            learning_rate = kwargs.get("learning_rate", self.learning_rate)

            # Ensure model exists before training
            if self.model is None:
                self.logger.info("Model is None, creating new model...")
                self.model = self._create_model()

            if self.model is None:
                error_msg = (
                    "Failed to create model - model is still None after creation"
                )
                self.logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "model_type": "PyTorchLightningTransformer",
                }

            # Validate model has required methods
            if not hasattr(self.model, "forward"):
                error_msg = "Model does not have forward method"
                self.logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "model_type": "PyTorchLightningTransformer",
                }

            # Update model's learning rate if provided
            if "learning_rate" in kwargs:
                self.model.learning_rate = learning_rate

            # Create data loaders
            train_ds = torch.utils.data.TensorDataset(train_sequences, train_targets)
            val_ds = torch.utils.data.TensorDataset(val_sequences, val_targets)

            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=batch_size, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=batch_size, shuffle=False
            )

            # Create trainer
            trainer = pl.Trainer(
                max_epochs=num_epochs,
                accelerator="auto",
                devices=1 if torch.cuda.is_available() else None,
                enable_progress_bar=True,
                logger=True,
                enable_checkpointing=False,
                log_every_n_steps=100,
            )

            # Train
            self.logger.info("Starting PyTorch Lightning training...")
            self.logger.info("Hyperparameters:")
            self.logger.info(f"  learning_rate: {learning_rate}")
            self.logger.info(f"  num_train_epochs: {num_epochs}")
            self.logger.info(f"  per_device_train_batch_size: {batch_size}")

            trainer.fit(self.model, train_loader, val_loader)

            # After training, ensure model is properly configured for inference
            self.model.eval()
            self.model.to(self.device)

            # Get validation metrics
            val_results = trainer.validate(self.model, val_loader)
            val_loss = val_results[0].get("val_loss", 0.0) if val_results else 0.0

            return {
                "train_loss": 0.0,  # PL doesn't easily expose final train loss
                "eval_loss": val_loss,
                "epochs_trained": num_epochs,
                "model_type": "PyTorchLightningTransformer",
                "success": True,
            }

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_type": "PyTorchLightningTransformer",
            }

    def _model_predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Make prediction with PyTorch Lightning Time Series Transformer model"""
        try:
            # Ensure model is in eval mode
            self.model.eval()

            # Ensure both model and input are on the same device
            # PyTorch Lightning models should already be on the right device after training
            model_device = next(self.model.parameters()).device
            input_tensor = input_tensor.to(model_device)

            with torch.no_grad():
                # Model expects [batch, seq_len, features]
                predictions = self.model(input_tensor)

                # Handle output shape - take only the first prediction step
                if predictions.dim() > 1:
                    predictions = predictions[:, 0]  # Take first time step
                if predictions.dim() > 1:
                    predictions = predictions[:, 0]  # Take first output dimension

                # Return on the same device as model for consistency
                return predictions

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    def _save_model_specific(self, model_dir: Path) -> None:
        """Save PyTorch Lightning Time Series Transformer specific components"""
        try:
            # Save model state dict
            if self.model is not None:
                torch.save(self.model.state_dict(), model_dir / "model_state_dict.pt")

            # Save model config
            model_config = {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
            }

            with open(model_dir / "model_config.json", "w") as f:
                json.dump(model_config, f, indent=2)

            self.logger.info("PyTorch Lightning Transformer model components saved")

        except Exception as e:
            self.logger.error(f"Error saving Transformer components: {e}")
            raise

    def _load_model_specific(self, model_dir: Path) -> None:
        """Load PyTorch Lightning Time Series Transformer specific components"""
        try:
            # Load model config
            with open(model_dir / "model_config.json", "r") as f:
                model_config = json.load(f)

            # Update parameters
            self.input_dim = model_config["input_dim"]
            self.output_dim = model_config["output_dim"]
            self.d_model = model_config["d_model"]
            self.n_heads = model_config["n_heads"]
            self.n_layers = model_config["n_layers"]
            self.dropout = model_config["dropout"]
            self.learning_rate = model_config["learning_rate"]
            self.batch_size = model_config["batch_size"]
            self.num_epochs = model_config["num_epochs"]

            # Create and load model
            self.model = self._create_model()

            # Load state dict if available
            state_dict_path = model_dir / "model_state_dict.pt"
            if state_dict_path.exists():
                # Load state dict to the same device as the model will be created on
                state_dict = torch.load(state_dict_path, map_location=str(self.device))
                self.model.load_state_dict(state_dict)

            # Ensure model is on correct device and in eval mode
            self.model.to(self.device)
            self.model.eval()
            self.logger.info("PyTorch Lightning Transformer model components loaded")

        except Exception as e:
            self.logger.error(f"Error loading Transformer components: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and configuration"""
        try:
            model_info = {
                "model_type": "PyTorchLightningTransformer",
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "target_column": self.target_column,
                "feature_columns": self.feature_columns,
                "is_trained": self.is_trained,
                "device": str(self.device),
            }

            if self.model is not None:
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                )

                model_info.update(
                    {
                        "total_parameters": total_params,
                        "trainable_parameters": trainable_params,
                        "model_size_mb": total_params
                        * 4
                        / (1024 * 1024),  # Assuming float32
                    }
                )

            return model_info

        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return {
                "model_type": "PyTorchLightningTransformer",
                "error": str(e),
                "is_trained": False,
            }
