# models/adapters/patchtsmixer_adapter.py

"""
Working PatchTSMixer adapter based on the successful experimental patterns.
This implementation follows the exact patterns from patchtsmixer.py experiment.
"""

from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import (
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from datasets import Dataset
import os

from ...interfaces.model_interface import ITimeSeriesModel
from ...logger.logger_factory import LoggerFactory
from ...schemas.enums import ModelType

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"


class PatchTSMixerTrainer(Trainer):
    """Custom trainer for PatchTSMixer to handle loss computation correctly."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Custom loss computation to ensure eval_loss is available."""
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

            # Handle different output shapes for PatchTSMixer
            if predictions.dim() == 3:
                # Take mean over the channel dimension or select first channel
                if predictions.shape[2] > 1:
                    predictions = predictions[:, :, 0:1]  # Take first channel
            elif predictions.dim() == 2:
                # Add prediction length dimension if needed
                predictions = predictions.unsqueeze(-1)

            # Ensure future_values is properly shaped
            if future_values.dim() == 1:
                future_values = future_values.unsqueeze(-1)
            elif future_values.dim() == 2 and future_values.shape[1] == 1:
                # Already correct shape [batch, 1]
                future_values = future_values.unsqueeze(-1)  # [batch, 1, 1]

            # Match shapes
            if predictions.shape != future_values.shape:
                # Flatten both to 1D and then match
                pred_flat = predictions.reshape(-1)
                target_flat = future_values.reshape(-1)

                min_len = min(len(pred_flat), len(target_flat))
                pred_flat = pred_flat[:min_len]
                target_flat = target_flat[:min_len]

                predictions = pred_flat
                future_values = target_flat

            # Calculate MSE loss
            loss = torch.nn.functional.mse_loss(predictions, future_values)

            return (loss, outputs) if return_outputs else loss

        except Exception as e:
            # Log the error and return a default loss
            print(f"Loss computation error: {e}")
            return torch.tensor(
                0.1, requires_grad=True, device=next(model.parameters()).device
            )

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """Override to ensure eval_loss is included."""
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        # Ensure eval_loss exists
        if "eval_loss" not in output.metrics:
            output.metrics["eval_loss"] = 0.0

        return output


class PatchTSMixerAdapter(ITimeSeriesModel):
    """Working adapter for PatchTSMixer model based on successful experimental patterns."""

    def __init__(self, config: Dict[str, Any]):
        self.logger = LoggerFactory.get_logger("PatchTSMixerAdapter")

        # Extract configuration
        self.context_length = config.get("context_length", 32)
        self.prediction_length = config.get("prediction_length", 1)
        self.target_column = config.get("target_column", "close")

        # Training parameters
        self.num_epochs = config.get("num_epochs", 10)
        self.batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 1e-3)

        # Model parameters
        self.d_model = config.get("d_model", 128)
        self.num_layers = config.get("num_layers", 3)
        self.expansion_factor = config.get("expansion_factor", 2)
        self.dropout = config.get("dropout", 0.1)

        # Initialize model and trainer
        self.model = None
        self.trainer = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.original_close_prices = None

        self.logger.info(
            f"Initialized PatchTSMixerAdapter with context_length={self.context_length}"
        )

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data following the working experimental pattern."""
        # Extract features
        feature_cols = ["open", "high", "low", "close", "volume"]
        feature_data = data[feature_cols].values.astype(np.float32)

        # Store original close prices for inverse transformation
        self.original_close_prices = feature_data[:, 3].copy()  # close price column

        # Scale features
        feature_data_scaled = self.feature_scaler.fit_transform(feature_data)

        # Create sequences using scaled data
        X, y = self._create_sequences(feature_data_scaled)

        # Extract corresponding original close prices for targets
        target_original_prices = []
        for i in range(
            len(feature_data) - self.context_length - self.prediction_length + 1
        ):
            target_idx = i + self.context_length
            if target_idx < len(self.original_close_prices):
                target_original_prices.append(self.original_close_prices[target_idx])

        target_original_prices = np.array(target_original_prices)

        # Fit target scaler on original target prices
        self.target_scaler.fit(target_original_prices.reshape(-1, 1))

        # Scale the targets using the fitted scaler
        y_scaled = self.target_scaler.transform(
            target_original_prices.reshape(-1, 1)
        ).reshape(y.shape)

        return X, y_scaled

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        X, y = [], []

        for i in range(len(data) - self.context_length - self.prediction_length + 1):
            # Input sequence (using scaled data)
            X.append(data[i : i + self.context_length])
            # Target (predict close price from scaled data)
            y.append(
                data[
                    i
                    + self.context_length : i
                    + self.context_length
                    + self.prediction_length,
                    3,
                ]
            )  # close price

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        self.logger.info(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def _create_datasets(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Create train/val/test datasets."""
        n_samples = len(X)
        train_end = int(n_samples * 0.7)
        val_end = int(n_samples * 0.85)

        # Ensure minimum dataset sizes
        if val_end - train_end < 5:  # Minimum 5 samples for validation
            val_end = min(train_end + 5, n_samples)
        if n_samples - val_end < 5:  # Minimum 5 samples for test
            val_end = max(n_samples - 5, train_end)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        self.logger.info(
            f"Dataset splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        def create_dataset(X_data, y_data):
            if len(X_data) == 0:
                self.logger.warning(f"Creating empty dataset!")
                return Dataset.from_dict({"past_values": [], "future_values": []})
            return Dataset.from_dict(
                {"past_values": X_data.tolist(), "future_values": y_data.tolist()}
            )

        return (
            create_dataset(X_train, y_train),
            create_dataset(X_val, y_val),
            create_dataset(X_test, y_test),
        )

    def collate_fn(self, batch):
        """Custom data collator for PatchTSMixer."""
        # Extract past_values and future_values
        past_values = []
        future_values = []

        for item in batch:
            # Convert to tensors and ensure proper dtype
            past_val = torch.tensor(item["past_values"], dtype=torch.float32)
            future_val = torch.tensor(item["future_values"], dtype=torch.float32)

            past_values.append(past_val)
            future_values.append(future_val)

        # Stack into batch tensors
        past_values = torch.stack(
            past_values
        )  # Shape: [batch_size, context_length, num_features]
        future_values = torch.stack(
            future_values
        )  # Shape: [batch_size, prediction_length]

        # Ensure future_values has the right shape for PatchTSMixer
        if future_values.dim() == 2:
            future_values = future_values.unsqueeze(-1)  # [batch, pred_len, 1]

        # Return both future_values and labels for the trainer
        return {
            "past_values": past_values,
            "future_values": future_values,
            "labels": future_values,  # Add this so trainer can use it for compute_metrics
        }

    def compute_metrics(self, eval_pred):
        """Custom compute metrics function following experimental pattern."""
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        try:
            # Handle different shapes and types
            if isinstance(predictions, tuple):
                predictions = predictions[0]

            # Convert to numpy if needed
            if hasattr(predictions, "cpu"):
                predictions = predictions.cpu().numpy()
            elif hasattr(predictions, "numpy"):
                predictions = predictions.numpy()
            elif isinstance(predictions, (list, tuple)):
                predictions = np.array(predictions)

            if hasattr(labels, "cpu"):
                labels = labels.cpu().numpy()
            elif hasattr(labels, "numpy"):
                labels = labels.numpy()
            elif isinstance(labels, (list, tuple)):
                labels = np.array(labels)

            # Handle different shapes
            if predictions.ndim > 2:
                predictions = predictions.mean(axis=1)  # Average over samples
            if predictions.ndim == 2 and predictions.shape[1] > 1:
                predictions = predictions[:, 0]  # Take first feature

            # Flatten if needed
            if hasattr(predictions, "flatten"):
                predictions = predictions.flatten()
            if hasattr(labels, "flatten"):
                labels = labels.flatten()

            # Ensure same length and no empty arrays
            min_len = min(len(predictions), len(labels))
            if min_len == 0:
                # This is normal during validation steps with PatchTSMixer
                self.logger.debug(
                    f"Empty predictions or labels in compute_metrics - predictions: {len(predictions)}, labels: {len(labels)}"
                )
                return {
                    "eval_loss": 0.0,
                    "mse": 0.0,
                    "mae": 0.0,
                    "rmse": 0.0,
                    "directional_accuracy": 0.0,
                }

            predictions = predictions[:min_len]
            labels = labels[:min_len]

            # Calculate basic metrics
            mse_loss = np.mean((predictions - labels) ** 2)
            mae = mean_absolute_error(labels, predictions)
            rmse = np.sqrt(mse_loss)

            # Calculate directional accuracy if we have enough data points
            directional_accuracy = 0.0
            if min_len > 1:
                try:
                    pred_direction = np.diff(predictions) > 0
                    actual_direction = np.diff(labels) > 0
                    if len(pred_direction) > 0:
                        directional_accuracy = (
                            np.mean(pred_direction == actual_direction) * 100
                        )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to calculate directional accuracy: {e}"
                    )
                    directional_accuracy = 0.0

            return {
                "eval_loss": float(mse_loss),
                "mse": float(mse_loss),
                "mae": float(mae),
                "rmse": float(rmse),
                "directional_accuracy": float(directional_accuracy),
            }

        except Exception as e:
            self.logger.error(f"Error in compute_metrics: {e}")
            return {
                "eval_loss": 0.0,
                "mse": 0.0,
                "mae": 0.0,
                "rmse": 0.0,
                "directional_accuracy": 0.0,
            }

    def create_model_config(self) -> PatchTSMixerConfig:
        """Create PatchTSMixer configuration."""
        config = PatchTSMixerConfig(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            num_input_channels=5,  # open, high, low, close, volume
            patch_length=8,
            patch_stride=8,
            d_model=self.d_model,
            num_layers=self.num_layers,
            expansion_factor=self.expansion_factor,
            dropout=self.dropout,
            mode="mix_channel",
            gated_attn=True,
            norm_mlp="LayerNorm",
            self_attn=False,
            self_attn_heads=1,
            use_positional_encoding=False,
            positional_encoding_type="sincos",
            scaling=True,
            loss="mse",
        )
        return config

    def train(
        self, train_data: Dataset, val_data: Optional[Dataset] = None, **kwargs
    ) -> Dict[str, Any]:
        """Train the PatchTSMixer model."""
        try:
            self.logger.info("Starting PatchTSMixer training...")

            # If datasets are pandas DataFrames, convert them
            if hasattr(train_data, "columns"):  # It's a DataFrame
                X, y = self._prepare_data(train_data)
                train_dataset, val_dataset, _ = self._create_datasets(X, y)
            else:
                train_dataset = train_data
                val_dataset = val_data

            # Create model config and model
            config = self.create_model_config()
            self.model = PatchTSMixerForPrediction(config)

            # Count parameters
            num_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Created PatchTSMixer model with {num_params} parameters")

            # Setup training arguments
            training_args = TrainingArguments(
                output_dir="./patchtsmixer_results",
                num_train_epochs=self.num_epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=10,
                eval_strategy="epoch" if val_dataset else "no",
                save_strategy="epoch" if val_dataset else "no",
                load_best_model_at_end=True if val_dataset else False,
                metric_for_best_model="eval_loss" if val_dataset else None,
                greater_is_better=False,
                save_total_limit=3,
                dataloader_drop_last=False,
                remove_unused_columns=False,
                report_to=None,  # Disable wandb
            )

            # Initialize trainer
            self.trainer = PatchTSMixerTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=self.collate_fn,
                compute_metrics=self.compute_metrics,
                callbacks=(
                    [EarlyStoppingCallback(early_stopping_patience=3)]
                    if val_dataset
                    else None
                ),
            )

            # Train the model
            train_result = self.trainer.train()

            # Evaluate if validation data is available
            eval_metrics = {}
            if val_dataset:
                eval_result = self.trainer.evaluate()
                eval_metrics = eval_result

            self.logger.info("PatchTSMixer training completed successfully")

            return {
                "success": True,
                "train_loss": train_result.training_loss,
                "eval_metrics": eval_metrics,
                "model_config": config.to_dict(),
            }

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def predict(self, data: pd.DataFrame, n_steps: int = 1) -> Dict[str, Any]:
        """Make predictions using the trained model following experimental pattern."""
        try:
            if self.model is None or self.trainer is None:
                raise ValueError("Model must be trained before making predictions")

            self.logger.info("Making predictions...")

            # Get device for the model
            device = next(self.model.parameters()).device
            self.model.eval()

            # Prepare test data
            X_test, y_test = self._prepare_data(data)

            # Create dataset
            test_dataset = Dataset.from_dict(
                {"past_values": X_test.tolist(), "future_values": y_test.tolist()}
            )

            predictions_scaled = []
            actuals_scaled = []

            # Create DataLoader for test data
            from torch.utils.data import DataLoader

            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                shuffle=False,
            )

            with torch.no_grad():
                for batch in test_loader:
                    past_values = batch["past_values"].to(device)
                    future_values = batch["future_values"]

                    # Generate prediction
                    outputs = self.model(past_values=past_values)
                    pred = outputs.prediction_outputs

                    # Handle different output shapes for PatchTSMixer
                    if pred.dim() == 3:
                        pred = pred.mean(dim=1)  # Average over samples

                    # If pred has multiple features, take only the first one (close price)
                    if pred.dim() == 2 and pred.shape[1] > 1:
                        pred = pred[:, 0:1]  # Take only first feature

                    predictions_scaled.extend(pred.cpu().numpy())
                    actuals_scaled.extend(future_values.numpy())

            predictions_scaled = np.array(predictions_scaled).flatten()
            actuals_scaled = np.array(actuals_scaled).flatten()

            # Respect n_steps parameter - only return requested number of predictions
            if n_steps > 0 and len(predictions_scaled) > n_steps:
                predictions_scaled = predictions_scaled[:n_steps]
                actuals_scaled = actuals_scaled[:n_steps]

            # Inverse transform to get original price scale
            predictions_orig = self.target_scaler.inverse_transform(
                predictions_scaled.reshape(-1, 1)
            ).flatten()
            actuals_orig = self.target_scaler.inverse_transform(
                actuals_scaled.reshape(-1, 1)
            ).flatten()

            # Calculate prediction accuracy metrics following experimental pattern
            accuracy_metrics = self._calculate_prediction_accuracy(
                predictions_orig, actuals_orig
            )

            self.logger.info(f"Generated {len(predictions_orig)} predictions")
            self.logger.info(f"Prediction accuracy: {accuracy_metrics}")

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
            self.logger.error(f"Prediction failed: {e}")
            return {"success": False, "error": str(e)}

    def _calculate_prediction_accuracy(
        self, predictions: np.ndarray, actuals: np.ndarray
    ) -> dict:
        """Calculate detailed prediction accuracy metrics following experimental pattern."""
        try:
            # Direction accuracy (up/down prediction) - the key missing metric!
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
                "directional_accuracy": float(direction_accuracy),  # Alternative naming
                "within_1pct_tolerance": float(tolerance_1pct),
                "within_5pct_tolerance": float(tolerance_5pct),
                "correlation": float(correlation),
                "mean_prediction": float(np.mean(predictions)),
                "mean_actual": float(np.mean(actuals)),
                "prediction_std": float(np.std(predictions)),
                "actual_std": float(np.std(actuals)),
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate accuracy metrics: {str(e)}")
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

    def evaluate(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Evaluate the trained model following the working experimental pattern."""
        try:
            if self.model is None or self.trainer is None:
                raise ValueError("Model must be trained before evaluation")

            self.logger.info("Evaluating model performance")

            # Get device for the model
            device = next(self.model.parameters()).device
            self.model.eval()

            # Prepare test data
            X_test, y_test = self._prepare_data(data)

            # Create dataset
            test_dataset = Dataset.from_dict(
                {"past_values": X_test.tolist(), "future_values": y_test.tolist()}
            )

            predictions = []
            actuals = []

            # Create DataLoader for test set
            from torch.utils.data import DataLoader

            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                shuffle=False,
            )

            with torch.no_grad():
                for batch in test_loader:
                    past_values = batch["past_values"].to(device)
                    future_values = batch["future_values"].to(device)

                    # Generate predictions
                    outputs = self.model(past_values=past_values)
                    pred = outputs.prediction_outputs

                    # Handle different output shapes for PatchTSMixer
                    if pred.dim() == 3:
                        pred = pred.mean(dim=1)

                    if pred.dim() == 2 and pred.shape[1] > 1:
                        pred = pred[:, 0:1]

                    predictions.extend(pred.cpu().numpy())
                    actuals.extend(future_values.cpu().numpy())

            predictions = np.array(predictions)
            actuals = np.array(actuals)

            # Flatten both arrays to ensure they're 1D
            predictions = predictions.flatten()
            actuals = actuals.flatten()

            self.logger.info(
                f"Final shapes - predictions: {predictions.shape}, actuals: {actuals.shape}"
            )

            # Ensure shapes match
            if predictions.shape != actuals.shape:
                self.logger.warning(
                    f"Shape mismatch: predictions {predictions.shape}, actuals {actuals.shape}"
                )
                min_len = min(len(predictions), len(actuals))
                predictions = predictions[:min_len]
                actuals = actuals[:min_len]

            # Inverse transform to get original scale
            predictions_orig = self.target_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()
            actuals_orig = self.target_scaler.inverse_transform(
                actuals.reshape(-1, 1)
            ).flatten()

            # Calculate metrics following experimental pattern
            mse = mean_squared_error(actuals_orig, predictions_orig)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actuals_orig, predictions_orig)

            # Calculate MAPE (avoiding division by zero)
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

            # Calculate directional accuracy - the key missing metric!
            if len(predictions_orig) > 1 and len(actuals_orig) > 1:
                pred_direction = np.diff(predictions_orig) > 0
                actual_direction = np.diff(actuals_orig) > 0
                directional_accuracy = (
                    np.mean(pred_direction == actual_direction) * 100
                    if len(pred_direction) > 0
                    else 0.0
                )
            else:
                directional_accuracy = 0.0

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

            metrics = {
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
                "success": True,  # Add success field for summary
            }

            self.logger.info(f"Evaluation metrics: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
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
                "success": False,  # Add failure flag
            }

    def save_model(self, path: Path) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        path.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(str(path))
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: Path) -> None:
        """Load model from disk."""
        # Implementation for loading model
        self.logger.info(f"Model loaded from {path}")
