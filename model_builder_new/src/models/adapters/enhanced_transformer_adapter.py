"""
Enhanced Transformer adapter with hybrid architecture inspired by HybridTransformer
Combines LSTM and Transformer for better time series forecasting
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import json
import pandas as pd

from .base_adapter import BaseTimeSeriesAdapter


class EnhancedTimeSeriesTransformer(pl.LightningModule):
    """
    Enhanced Time Series Transformer with LSTM-Transformer hybrid architecture
    Inspired by HybridTransformer but simplified for financial time series
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        d_model: int = 128,
        n_heads: int = 8,
        n_transformer_layers: int = 4,
        n_lstm_layers: int = 2,
        seq_length: int = 64,
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
        self.n_transformer_layers = n_transformer_layers
        self.n_lstm_layers = n_lstm_layers
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.dropout = dropout
        self.learning_rate = learning_rate

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # LSTM component for temporal patterns
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_lstm_layers,
            dropout=dropout if n_lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=False,
        )

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding()

        # Transformer encoder for attention patterns
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )

        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Combine LSTM and Transformer features
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output projection with residual connection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_dim),
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with Xavier uniform for linear layers"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

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
        """
        Forward pass with hybrid LSTM-Transformer architecture

        Args:
            x: Input tensor [batch, seq_len, features]

        Returns:
            Predictions [batch, output_dim]
        """
        batch_size, seq_len = x.size(0), x.size(1)

        # Input projection
        x = self.input_projection(x)  # [batch, seq_len, d_model]

        # LSTM branch for temporal patterns
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, d_model]

        # Transformer branch for attention patterns
        # Add positional encoding
        x_pos = x + self.pos_encoding[:, :seq_len, :]
        transformer_out = self.transformer(x_pos)  # [batch, seq_len, d_model]

        # Feature fusion - combine LSTM and Transformer outputs
        combined = torch.cat(
            [lstm_out, transformer_out], dim=-1
        )  # [batch, seq_len, d_model*2]
        fused = self.feature_fusion(combined)  # [batch, seq_len, d_model]

        # Global average pooling for sequence aggregation
        pooled = fused.mean(dim=1)  # [batch, d_model]

        # Output projection
        output = self.output_projection(pooled)  # [batch, output_dim]

        return output

    def training_step(self, batch, batch_idx):
        """Training step"""
        x, y = batch
        y_hat = self(x)

        # Ensure y has correct shape
        if y.dim() > 1:
            y = y.view(-1, self.output_dim)
        else:
            y = y.unsqueeze(-1) if self.output_dim > 1 else y

        # Ensure y_hat has correct shape
        if y_hat.dim() > 1 and y_hat.size(1) != self.output_dim:
            y_hat = y_hat.view(-1, self.output_dim)

        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, y = batch
        y_hat = self(x)

        # Ensure y has correct shape
        if y.dim() > 1:
            y = y.view(-1, self.output_dim)
        else:
            y = y.unsqueeze(-1) if self.output_dim > 1 else y

        # Ensure y_hat has correct shape
        if y_hat.dim() > 1 and y_hat.size(1) != self.output_dim:
            y_hat = y_hat.view(-1, self.output_dim)

        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler with warm-up"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy="cos",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


class EnhancedTransformerAdapter(BaseTimeSeriesAdapter):
    """Enhanced Transformer adapter with hybrid LSTM-Transformer architecture"""

    def __init__(self, config: Dict[str, Any]):
        # Extract configuration parameters
        context_length = config.get("context_length", 64)
        prediction_length = config.get("prediction_length", 1)
        target_column = config.get("target_column", "close")
        feature_columns = config.get(
            "feature_columns", ["open", "high", "low", "close", "volume"]
        )

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

        # Model hyperparameters
        default_input_dim = len(feature_columns) if feature_columns else 5
        self.input_dim = config.get("input_dim", default_input_dim)
        self.output_dim = config.get("output_dim", 1)
        self.d_model = config.get("d_model", 128)
        self.n_heads = config.get("n_heads", 8)
        self.n_transformer_layers = config.get("n_transformer_layers", 4)
        self.n_lstm_layers = config.get("n_lstm_layers", 2)
        self.dropout = config.get("dropout", 0.1)
        self.learning_rate = config.get("learning_rate", 1e-4)

        # Training parameters
        self.batch_size = config.get("batch_size", 32)
        self.num_epochs = config.get("num_epochs", 10)

    def _create_model(self) -> torch.nn.Module:
        """Create Enhanced Time Series Transformer model"""
        try:
            # Update input dimension if feature engineering was applied
            if self.feature_engineering is not None:
                feature_names = self.feature_engineering.get_feature_names()
                self.input_dim = len(feature_names)
                self.logger.info(
                    f"Updated input_dim to {self.input_dim} based on feature engineering"
                )

            model = EnhancedTimeSeriesTransformer(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_transformer_layers=self.n_transformer_layers,
                n_lstm_layers=self.n_lstm_layers,
                seq_length=self.context_length,
                pred_length=self.prediction_length,
                dropout=self.dropout,
                learning_rate=self.learning_rate,
            )

            self.logger.info(
                f"Created EnhancedTimeSeriesTransformer model with input_dim={self.input_dim}"
            )
            return model

        except Exception as e:
            self.logger.error(
                f"Error creating EnhancedTimeSeriesTransformer model: {e}"
            )
            raise

    def _train_model(
        self,
        train_dataset: Tuple[torch.Tensor, torch.Tensor],
        val_dataset: Tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> Dict[str, Any]:
        """Train Enhanced Time Series Transformer model"""
        try:
            train_sequences, train_targets = train_dataset
            val_sequences, val_targets = val_dataset

            # Validate inputs
            if train_sequences is None or train_targets is None:
                raise ValueError("Training sequences and targets cannot be None")

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
                    "model_type": "EnhancedTimeSeriesTransformer",
                }

            # Update model's learning rate if provided
            if "learning_rate" in kwargs:
                self.model.learning_rate = learning_rate

            # Reshape targets for training - ensure single prediction per sample
            if train_targets.dim() > 1 and train_targets.size(1) > 1:
                train_targets = train_targets[:, 0]
            if val_targets.dim() > 1 and val_targets.size(1) > 1:
                val_targets = val_targets[:, 0]

            # Create data loaders
            train_ds = torch.utils.data.TensorDataset(train_sequences, train_targets)
            val_ds = torch.utils.data.TensorDataset(val_sequences, val_targets)

            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=0
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=0
            )

            # Create trainer with better configuration
            trainer = pl.Trainer(
                max_epochs=num_epochs,
                accelerator="auto",
                devices=1 if torch.cuda.is_available() else None,
                enable_progress_bar=True,
                logger=False,  # Disable built-in logger to avoid conflicts
                enable_checkpointing=False,
                log_every_n_steps=50,
                gradient_clip_val=1.0,  # Gradient clipping for stability
                accumulate_grad_batches=1,
            )

            # Train
            self.logger.info("Starting Enhanced Transformer training...")
            self.logger.info("Hyperparameters:")
            self.logger.info(f"  learning_rate: {learning_rate}")
            self.logger.info(f"  num_train_epochs: {num_epochs}")
            self.logger.info(f"  per_device_train_batch_size: {batch_size}")
            self.logger.info(f"  d_model: {self.d_model}")
            self.logger.info(f"  n_heads: {self.n_heads}")
            self.logger.info(f"  n_transformer_layers: {self.n_transformer_layers}")
            self.logger.info(f"  n_lstm_layers: {self.n_lstm_layers}")

            trainer.fit(self.model, train_loader, val_loader)

            # After training, ensure model is properly configured for inference
            self.model.eval()
            self.model.to(self.device)

            # Get validation metrics
            val_results = trainer.validate(self.model, val_loader, verbose=False)
            val_loss = val_results[0].get("val_loss", 0.0) if val_results else 0.0

            return {
                "train_loss": 0.0,  # PL doesn't easily expose final train loss
                "eval_loss": val_loss,
                "epochs_trained": num_epochs,
                "model_type": "EnhancedTimeSeriesTransformer",
                "success": True,
            }

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_type": "EnhancedTimeSeriesTransformer",
            }

    def _model_predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Make prediction with Enhanced Time Series Transformer model"""
        try:
            # Ensure model is in eval mode
            self.model.eval()

            # Ensure both model and input are on the same device
            model_device = next(self.model.parameters()).device
            input_tensor = input_tensor.to(model_device)

            with torch.no_grad():
                # Model expects [batch, seq_len, features]
                predictions = self.model(input_tensor)

                # Handle output shape - should be [batch, output_dim] for single prediction
                if predictions.dim() == 2:
                    if predictions.size(1) == 1:
                        predictions = predictions.squeeze(-1)  # [batch]
                    else:
                        predictions = predictions[:, 0]  # Take first output
                elif predictions.dim() == 1:
                    # Already correct shape [batch]
                    pass
                else:
                    # Flatten to batch dimension
                    predictions = predictions.view(predictions.size(0), -1)[:, 0]

                return predictions

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    def _save_model_specific(self, model_dir: Path) -> None:
        """Save Enhanced Time Series Transformer specific components"""
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
                "n_transformer_layers": self.n_transformer_layers,
                "n_lstm_layers": self.n_lstm_layers,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
            }

            with open(model_dir / "model_config.json", "w") as f:
                json.dump(model_config, f, indent=2)

            self.logger.info("Enhanced Transformer model components saved")

        except Exception as e:
            self.logger.error(f"Error saving Enhanced Transformer components: {e}")
            raise

    def _load_model_specific(self, model_dir: Path) -> None:
        """Load Enhanced Time Series Transformer specific components"""
        try:
            # Load model config
            with open(model_dir / "model_config.json", "r") as f:
                model_config = json.load(f)

            # Update parameters
            self.input_dim = model_config["input_dim"]
            self.output_dim = model_config["output_dim"]
            self.d_model = model_config["d_model"]
            self.n_heads = model_config["n_heads"]
            self.n_transformer_layers = model_config["n_transformer_layers"]
            self.n_lstm_layers = model_config["n_lstm_layers"]
            self.dropout = model_config["dropout"]
            self.learning_rate = model_config["learning_rate"]
            self.batch_size = model_config["batch_size"]
            self.num_epochs = model_config["num_epochs"]

            # Create and load model
            self.model = self._create_model()

            # Load state dict if available
            state_dict_path = model_dir / "model_state_dict.pt"
            if state_dict_path.exists():
                state_dict = torch.load(state_dict_path, map_location=str(self.device))
                self.model.load_state_dict(state_dict)

            # Ensure model is on correct device and in eval mode
            self.model.to(self.device)
            self.model.eval()
            self.logger.info("Enhanced Transformer model components loaded")

        except Exception as e:
            self.logger.error(f"Error loading Enhanced Transformer components: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and configuration"""
        try:
            model_info = {
                "model_type": "EnhancedTimeSeriesTransformer",
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
                        "model_size_mb": total_params * 4 / (1024 * 1024),
                        "d_model": self.d_model,
                        "n_heads": self.n_heads,
                        "n_transformer_layers": self.n_transformer_layers,
                        "n_lstm_layers": self.n_lstm_layers,
                    }
                )

            return model_info

        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return {
                "model_type": "EnhancedTimeSeriesTransformer",
                "error": str(e),
                "is_trained": False,
            }
