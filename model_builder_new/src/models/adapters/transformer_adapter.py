# models/adapters/transformer_adapter.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ...interfaces.model_interface import ITimeSeriesModel
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
        x = x + self.pos_encoding

        # Transformer encoding
        x = self.transformer(x)  # [batch, seq_len, d_model]

        # Use the last token for prediction
        x = x[:, -1, :]  # [batch, d_model]

        # Output projection
        output = self.output_projection(x)  # [batch, output_dim * pred_length]

        # Reshape to [batch, pred_length, output_dim]
        output = output.view(batch_size, self.pred_length, self.output_dim)

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class TransformerAdapter(ITimeSeriesModel):
    """Adapter for PyTorch Lightning Time Series Transformer"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_scaler = None
        self.is_trained = False
        self.model_path = None

        # Model hyperparameters
        self.input_dim = config.get("input_dim", config.get("num_input_channels", 1))
        self.output_dim = config.get("output_dim", 1)
        self.d_model = config.get("d_model", 128)
        self.n_heads = config.get("n_heads", 8)
        self.n_layers = config.get("n_layers", 4)
        self.seq_length = config.get("context_length", 96)
        self.pred_length = config.get("prediction_length", 1)
        self.dropout = config.get("dropout", 0.1)
        self.learning_rate = config.get("learning_rate", 1e-4)

        # Training parameters
        self.batch_size = config.get("batch_size", 32)
        self.num_epochs = config.get("num_epochs", 50)

    def train(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame, **kwargs
    ) -> Dict[str, Any]:
        """Train the transformer model"""
        try:
            # Update input dimension based on actual data
            feature_cols = ["open", "high", "low", "close", "volume"]
            actual_input_dim = len(feature_cols)

            if self.input_dim != actual_input_dim:
                print(
                    f"Updating input_dim from {self.input_dim} to {actual_input_dim} based on data"
                )
                self.input_dim = actual_input_dim

            # Prepare data
            train_dataset = self._prepare_data(train_data, is_training=True)
            val_dataset = self._prepare_data(val_data, is_training=False)

            # Ensure datasets have consistent sizes
            train_len = len(train_dataset)
            val_len = len(val_dataset)

            # Use the smaller size to ensure consistency
            min_len = min(train_len, val_len)
            if min_len == 0:
                raise ValueError("Insufficient data for training")

            # Create consistent datasets
            if train_len > min_len:
                train_indices = list(range(min_len))
                train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

            if val_len > min_len:
                val_indices = list(range(min_len))
                val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

            # Initialize model
            self.model = TimeSeriesTransformer(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                seq_length=self.seq_length,
                pred_length=self.pred_length,
                dropout=self.dropout,
                learning_rate=self.learning_rate,
            )

            # Setup trainer
            trainer = pl.Trainer(
                max_epochs=self.num_epochs,
                accelerator="auto",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=True,
            )

            # Train model
            trainer.fit(self.model, train_loader, val_loader)

            # Evaluate on validation set
            val_results = trainer.validate(self.model, val_loader, verbose=False)
            val_loss = val_results[0]["val_loss"] if val_results else 0.0

            # Calculate metrics
            train_predictions = self._predict_dataset_scaled(train_dataset)
            val_predictions = self._predict_dataset_scaled(val_dataset)

            # Extract targets from the actual datasets used (subset if needed)
            train_targets = self._extract_targets_scaled(train_data)
            val_targets = self._extract_targets_scaled(val_data)

            # Ensure same length as predictions
            train_targets = train_targets[: len(train_predictions)]
            val_targets = val_targets[: len(val_predictions)]

            # Calculate metrics using original scale predictions
            train_predictions_orig = self._inverse_transform_targets(train_predictions)
            val_predictions_orig = self._inverse_transform_targets(val_predictions)
            train_targets_orig = self._inverse_transform_targets(train_targets)
            val_targets_orig = self._inverse_transform_targets(val_targets)

            train_mae = mean_absolute_error(train_targets_orig, train_predictions_orig)
            train_rmse = np.sqrt(
                mean_squared_error(train_targets_orig, train_predictions_orig)
            )
            val_mae = mean_absolute_error(val_targets_orig, val_predictions_orig)
            val_rmse = np.sqrt(
                mean_squared_error(val_targets_orig, val_predictions_orig)
            )

            # Calculate directional accuracy for training and validation
            train_accuracy_metrics = self._calculate_prediction_accuracy(
                train_predictions_orig, train_targets_orig
            )
            val_accuracy_metrics = self._calculate_prediction_accuracy(
                val_predictions_orig, val_targets_orig
            )

            self.is_trained = True

            # Safe extraction of training loss
            train_loss_val = trainer.callback_metrics.get("train_loss", 0.0)
            if hasattr(train_loss_val, "item"):
                train_loss_val = train_loss_val.item()
            elif isinstance(train_loss_val, (int, float)):
                train_loss_val = float(train_loss_val)
            else:
                train_loss_val = 0.0

            return {
                "success": True,
                "training_loss": train_loss_val,
                "validation_loss": val_loss,
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "train_directional_accuracy": train_accuracy_metrics.get(
                    "directional_accuracy", 0.0
                ),
                "val_directional_accuracy": val_accuracy_metrics.get(
                    "directional_accuracy", 0.0
                ),
                "epochs_trained": trainer.current_epoch + 1,
            }

        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")

    def predict(self, data: pd.DataFrame, n_steps: int = 1) -> Dict[str, Any]:
        """Make predictions following the standardized pattern"""
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model must be trained before making predictions")

            # Prepare data for prediction
            dataset = self._prepare_data(data, is_training=False)

            # Make predictions (scaled)
            predictions_scaled = self._predict_dataset_scaled(dataset)

            # Get actual targets (scaled)
            actuals_scaled = self._extract_targets_scaled(data)

            # Ensure same length
            min_len = min(len(predictions_scaled), len(actuals_scaled))
            predictions_scaled = predictions_scaled[:min_len]
            actuals_scaled = actuals_scaled[:min_len]

            # Inverse transform to get original scale
            predictions_orig = self._inverse_transform_targets(predictions_scaled)
            actuals_orig = self._inverse_transform_targets(actuals_scaled)

            # Calculate prediction accuracy metrics
            accuracy_metrics = self._calculate_prediction_accuracy(
                predictions_orig, actuals_orig
            )

            # Take only the requested number of steps if needed
            if len(predictions_orig) > n_steps:
                predictions_scaled = predictions_scaled[:n_steps]
                predictions_orig = predictions_orig[:n_steps]
                # Also truncate actuals to match
                actuals_scaled = actuals_scaled[:n_steps]
                actuals_orig = actuals_orig[:n_steps]

            return {
                "predictions_scaled": predictions_scaled.tolist(),
                "predictions_original": predictions_orig.tolist(),
                "actuals_scaled": actuals_scaled.tolist(),
                "actuals_original": actuals_orig.tolist(),
                "accuracy_metrics": accuracy_metrics,
                "predictions": predictions_orig.tolist(),  # Keep for backward compatibility
                "success": True,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance with comprehensive metrics"""
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model must be trained before evaluation")

            # Prepare data for evaluation
            dataset = self._prepare_data(data, is_training=False)

            # Make predictions (scaled)
            predictions_scaled = self._predict_dataset_scaled(dataset)

            # Get actual targets (scaled)
            actuals_scaled = self._extract_targets_scaled(data)

            # Ensure same length
            min_len = min(len(predictions_scaled), len(actuals_scaled))
            predictions_scaled = predictions_scaled[:min_len]
            actuals_scaled = actuals_scaled[:min_len]

            if len(predictions_scaled) == 0:
                return self._get_empty_metrics()

            # Inverse transform to get original scale
            predictions_orig = self._inverse_transform_targets(predictions_scaled)
            actuals_orig = self._inverse_transform_targets(actuals_scaled)

            # Calculate metrics on original scale
            mse = mean_squared_error(actuals_orig, predictions_orig)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actuals_orig, predictions_orig)

            # MAPE (avoid division by zero)
            mask = actuals_orig != 0
            mape = (
                np.mean(
                    np.abs(
                        (actuals_orig[mask] - predictions_orig[mask])
                        / actuals_orig[mask]
                    )
                )
                * 100
                if np.any(mask)
                else 0.0
            )

            # Calculate directional accuracy
            directional_accuracy = 0.0
            if len(predictions_orig) > 1 and len(actuals_orig) > 1:
                pred_direction = np.diff(predictions_orig) > 0
                actual_direction = np.diff(actuals_orig) > 0
                directional_accuracy = (
                    np.mean(pred_direction == actual_direction) * 100
                    if len(pred_direction) > 0
                    else 0.0
                )

            # Calculate price accuracy within tolerance
            tolerance_1pct = (
                np.mean(
                    np.abs(predictions_orig - actuals_orig) / np.abs(actuals_orig)
                    < 0.01
                )
                * 100
                if len(actuals_orig) > 0
                else 0.0
            )
            tolerance_5pct = (
                np.mean(
                    np.abs(predictions_orig - actuals_orig) / np.abs(actuals_orig)
                    < 0.05
                )
                * 100
                if len(actuals_orig) > 0
                else 0.0
            )

            # Statistical correlation
            correlation = (
                np.corrcoef(predictions_orig, actuals_orig)[0, 1]
                if len(predictions_orig) > 1
                else 0.0
            )

            return {
                "eval_loss": float(mse),
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "directional_accuracy": float(directional_accuracy),
                "accuracy": float(
                    directional_accuracy
                ),  # Use directional accuracy as main accuracy
                "within_1pct_tolerance": float(tolerance_1pct),
                "within_5pct_tolerance": float(tolerance_5pct),
                "correlation": float(correlation),
                "mean_prediction": float(np.mean(predictions_orig)),
                "mean_actual": float(np.mean(actuals_orig)),
                "prediction_std": float(np.std(predictions_orig)),
                "actual_std": float(np.std(actuals_orig)),
                "success": True,
            }

        except Exception as e:
            return {**self._get_empty_metrics(), "success": False, "error": str(e)}

    def save_model(self, path: str) -> None:
        """Save the trained model"""
        if not self.is_trained or self.model is None:
            raise ValueError("No trained model to save")

        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)

            # Save model checkpoint
            model_path = save_path / "model.ckpt"
            torch.save(self.model.state_dict(), model_path)

            # Save scalers
            if self.scaler is not None:
                with open(save_path / "scaler.pkl", "wb") as f:
                    pickle.dump(self.scaler, f)

            if self.feature_scaler is not None:
                with open(save_path / "feature_scaler.pkl", "wb") as f:
                    pickle.dump(self.feature_scaler, f)

            # Save config
            config_to_save = {
                **self.config,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "seq_length": self.seq_length,
                "pred_length": self.pred_length,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
            }

            with open(save_path / "config.json", "w") as f:
                json.dump(config_to_save, f, indent=2)

            self.model_path = str(save_path)

        except Exception as e:
            raise RuntimeError(f"Failed to save model: {str(e)}")

    def load_model(self, path: str) -> None:
        """Load a trained model"""
        try:
            load_path = Path(path)

            # Load config
            with open(load_path / "config.json", "r") as f:
                config = json.load(f)

            # Update config
            self.config.update(config)
            self.input_dim = config["input_dim"]
            self.output_dim = config["output_dim"]
            self.d_model = config["d_model"]
            self.n_heads = config["n_heads"]
            self.n_layers = config["n_layers"]
            self.seq_length = config["seq_length"]
            self.pred_length = config["pred_length"]
            self.dropout = config["dropout"]
            self.learning_rate = config["learning_rate"]

            # Initialize model
            self.model = TimeSeriesTransformer(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                seq_length=self.seq_length,
                pred_length=self.pred_length,
                dropout=self.dropout,
                learning_rate=self.learning_rate,
            )

            # Load model state
            model_path = load_path / "model.ckpt"
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.eval()

            # Load scalers
            try:
                with open(load_path / "scaler.pkl", "rb") as f:
                    self.scaler = pickle.load(f)
            except FileNotFoundError:
                pass

            try:
                with open(load_path / "feature_scaler.pkl", "rb") as f:
                    self.feature_scaler = pickle.load(f)
            except FileNotFoundError:
                pass

            self.is_trained = True
            self.model_path = str(load_path)

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            **self.config,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "seq_length": self.seq_length,
            "pred_length": self.pred_length,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "is_trained": self.is_trained,
            "model_path": self.model_path,
        }

    def _prepare_data(
        self, data: pd.DataFrame, is_training: bool = False
    ) -> torch.utils.data.TensorDataset:
        """Prepare data for training/prediction"""
        # Define standard feature columns
        standard_cols = ["open", "high", "low", "close", "volume"]
        target_col = "close"

        # Use only columns that exist in the data
        available_cols = [col for col in standard_cols if col in data.columns]
        if not available_cols:
            # Fallback to numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            available_cols = [col for col in numeric_cols if col != target_col]
            if not available_cols:
                available_cols = [target_col]

        feature_cols = [col for col in available_cols if col != target_col]
        if not feature_cols:
            feature_cols = available_cols[
                : min(4, len(available_cols))
            ]  # Use first 4 as features

        # Update input_dim if it doesn't match the actual number of features
        actual_input_dim = len(feature_cols)
        if is_training and actual_input_dim != self.input_dim:
            print(
                f"Updating input_dim from {self.input_dim} to {actual_input_dim} based on data"
            )
            self.input_dim = actual_input_dim

        # Extract features and targets
        features = data[feature_cols].values.astype(np.float32)
        targets = data[target_col].values.astype(np.float32)

        # Scale features
        if is_training:
            self.feature_scaler = StandardScaler()
            features = self.feature_scaler.fit_transform(features)
        elif self.feature_scaler is not None:
            features = self.feature_scaler.transform(features)

        # Scale targets
        if is_training:
            self.scaler = StandardScaler()
            targets = self.scaler.fit_transform(targets.reshape(-1, 1)).ravel()
        elif self.scaler is not None:
            targets = self.scaler.transform(targets.reshape(-1, 1)).ravel()

        # Create sequences with proper error handling
        X, y = [], []
        min_required_length = self.seq_length + self.pred_length

        if len(features) < min_required_length:
            # Not enough data - create minimum required sequences by repeating data
            num_repeats = (min_required_length // len(features)) + 1
            features = np.tile(features, (num_repeats, 1))[:min_required_length]
            targets = np.tile(targets, num_repeats)[:min_required_length]

        for i in range(len(features) - self.seq_length - self.pred_length + 1):
            # Input sequence
            X.append(features[i : i + self.seq_length])
            # Target sequence
            y.append(
                targets[i + self.seq_length : i + self.seq_length + self.pred_length]
            )

        if not X:
            # Fallback - create at least one sequence
            if len(features) >= self.seq_length:
                X.append(features[: self.seq_length])
                y.append(
                    targets[self.seq_length : self.seq_length + self.pred_length]
                    if len(targets) > self.seq_length
                    else [targets[-1]] * self.pred_length
                )
            else:
                # Pad sequences if necessary
                padded_features = np.pad(
                    features,
                    ((0, self.seq_length - len(features)), (0, 0)),
                    mode="edge",
                )
                X.append(padded_features)
                y.append([targets[-1]] * self.pred_length)

        X = np.array(X)
        y = np.array(y)

        # Ensure y has the right shape
        if y.ndim == 1:
            y = y.reshape(-1, self.pred_length)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # Ensure output dimension is correct
        if y_tensor.dim() == 2:
            y_tensor = y_tensor.unsqueeze(-1)  # Add output_dim dimension

        return torch.utils.data.TensorDataset(X_tensor, y_tensor)

    def _predict_dataset_scaled(
        self, dataset: torch.utils.data.TensorDataset
    ) -> np.ndarray:
        """Make predictions on a dataset returning scaled values"""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for x, _ in torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size
            ):
                pred = self.model(x)
                predictions.append(pred.cpu().numpy())

        if predictions:
            predictions = np.concatenate(predictions, axis=0)
            # Take the first prediction step and output dimension
            predictions = predictions[:, 0, 0]
            return predictions
        else:
            return np.array([])

    def _extract_targets_scaled(self, data: pd.DataFrame) -> np.ndarray:
        """Extract target values from data and apply scaling"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        target_col = "close" if "close" in numeric_cols else numeric_cols[-1]
        targets = data[target_col].values.astype(np.float32)

        # Apply scaling if scaler is available
        if self.scaler is not None:
            targets = self.scaler.transform(targets.reshape(-1, 1)).ravel()

        return targets

    def _inverse_transform_targets(self, targets_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform scaled targets to original scale"""
        if self.scaler is not None:
            return self.scaler.inverse_transform(targets_scaled.reshape(-1, 1)).ravel()
        else:
            return targets_scaled

    def _calculate_prediction_accuracy(
        self, predictions: np.ndarray, actuals: np.ndarray
    ) -> dict:
        """Calculate detailed prediction accuracy metrics"""
        try:
            # Direction accuracy (up/down prediction)
            if len(predictions) > 1 and len(actuals) > 1:
                pred_direction = np.diff(predictions) > 0
                actual_direction = np.diff(actuals) > 0
                direction_accuracy = (
                    np.mean(pred_direction == actual_direction) * 100
                    if len(pred_direction) > 0
                    else 0.0
                )
            else:
                direction_accuracy = 0.0

            # Price accuracy within tolerance
            tolerance_1pct = (
                np.mean(np.abs(predictions - actuals) / np.abs(actuals) < 0.01) * 100
                if len(actuals) > 0
                else 0.0
            )
            tolerance_5pct = (
                np.mean(np.abs(predictions - actuals) / np.abs(actuals) < 0.05) * 100
                if len(actuals) > 0
                else 0.0
            )

            # Statistical metrics
            correlation = (
                np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0.0
            )

            return {
                "direction_accuracy_pct": float(direction_accuracy),
                "directional_accuracy": float(direction_accuracy),
                "within_1pct_tolerance": float(tolerance_1pct),
                "within_5pct_tolerance": float(tolerance_5pct),
                "correlation": float(correlation),
                "mean_prediction": float(np.mean(predictions)),
                "mean_actual": float(np.mean(actuals)),
                "prediction_std": float(np.std(predictions)),
                "actual_std": float(np.std(actuals)),
            }
        except Exception as e:
            return {
                "direction_accuracy_pct": 0.0,
                "directional_accuracy": 0.0,
                "within_1pct_tolerance": 0.0,
                "within_5pct_tolerance": 0.0,
                "correlation": 0.0,
                "mean_prediction": 0.0,
                "mean_actual": 0.0,
                "prediction_std": 0.0,
                "actual_std": 0.0,
            }

    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dict for error cases"""
        return {
            "eval_loss": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "mape": 0.0,
            "directional_accuracy": 0.0,
            "accuracy": 0.0,
            "within_1pct_tolerance": 0.0,
            "within_5pct_tolerance": 0.0,
            "correlation": 0.0,
            "mean_prediction": 0.0,
            "mean_actual": 0.0,
            "prediction_std": 0.0,
            "actual_std": 0.0,
        }

    def _extract_targets(self, data: pd.DataFrame) -> np.ndarray:
        """Extract target values from data"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        target_col = "close" if "close" in numeric_cols else numeric_cols[-1]
        return data[target_col].values
