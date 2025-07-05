"""
Standalone PatchTSMixer experiment script for debugging.
This implements all logic (data loading, preprocessing, train, evaluate, predict) in one place
to identify working patterns before refactoring back to strategies.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from scipy import stats
from transformers import (
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datasets import Dataset
import sys
import os
import logging

# Optional import for advanced statistical analysis
try:
    from statsmodels.tsa.stattools import acf

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from logger.logger_factory import LoggerFactory

# Configure logging to capture warnings as errors
logging.captureWarnings(True)
warnings.simplefilter("error")


# Custom warning handler that stops execution
def warning_handler(message, category, filename, lineno, file=None, line=None):
    logger = LoggerFactory.get_logger("PatchTSMixerExperiment")
    logger.error(
        f"WARNING detected in {filename}:{lineno} - {category.__name__}: {message}"
    )
    raise RuntimeError(f"Execution stopped due to WARNING: {message}")


warnings.showwarning = warning_handler

warnings.filterwarnings("ignore")


class PatchTSMixerExperiment:
    """Complete PatchTSMixer experiment implementation."""

    def __init__(self, data_path: str, output_dir: str = "patchtsmixer_experiment"):
        self.logger = LoggerFactory.get_logger("PatchTSMixerExperiment")
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Model hyperparameters
        self.context_length = 64  # Context length for input sequences
        self.prediction_length = 1  # Predict next 1 day
        self.num_input_channels = 5  # open, high, low, close, volume
        self.patch_length = 8
        self.patch_stride = 8

        # Training parameters
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.num_epochs = 10

        # Scalers for features and targets
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        self.logger.info(
            f"Initialized PatchTSMixer experiment with context_length={self.context_length}"
        )

    def load_data(self) -> pd.DataFrame:
        """Load and prepare the data."""
        self.logger.info(f"Loading data from {self.data_path}")

        df = pd.read_csv(self.data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Select features
        feature_cols = ["open", "high", "low", "close", "volume"]
        df = df[["timestamp"] + feature_cols].copy()

        # Remove any NaN values
        df = df.dropna()

        self.logger.info(f"Data shape: {df.shape}")
        self.logger.info(
            f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}"
        )

        return df

    def create_sequences(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
            )  # close price (4th column, index 3)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        self.logger.info(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def prepare_data(self, df: pd.DataFrame) -> tuple[Dataset, Dataset, Dataset]:
        """Prepare data for training, validation, and testing."""
        # Extract features
        feature_cols = ["open", "high", "low", "close", "volume"]
        data = df[feature_cols].values.astype(np.float32)

        # Store original close prices for inverse transformation
        self.original_close_prices = data[:, 3].copy()  # close price column

        # Scale features
        data_scaled = self.feature_scaler.fit_transform(data)

        # Create sequences using scaled data
        X, y = self.create_sequences(data_scaled)

        # Extract corresponding original close prices for targets
        # These are the original close prices that our model should predict
        target_original_prices = []
        for i in range(len(data) - self.context_length - self.prediction_length + 1):
            target_idx = i + self.context_length
            if target_idx < len(self.original_close_prices):
                target_original_prices.append(self.original_close_prices[target_idx])

        target_original_prices = np.array(target_original_prices)

        # Fit target scaler on these original target prices
        self.target_scaler.fit(target_original_prices.reshape(-1, 1))

        # Now scale the targets using the fitted scaler
        y_scaled = self.target_scaler.transform(
            target_original_prices.reshape(-1, 1)
        ).reshape(y.shape)

        # Split data
        n_samples = len(X)
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))

        X_train, y_train = X[:train_end], y_scaled[:train_end]
        X_val, y_val = X[train_end:val_end], y_scaled[train_end:val_end]
        X_test, y_test = X[val_end:], y_scaled[val_end:]

        self.logger.info(
            f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        # Create datasets
        def create_dataset(X, y):
            return Dataset.from_dict(
                {"past_values": X.tolist(), "future_values": y.tolist()}
            )

        train_dataset = create_dataset(X_train, y_train)
        val_dataset = create_dataset(X_val, y_val)
        test_dataset = create_dataset(X_test, y_test)

        return train_dataset, val_dataset, test_dataset

    def create_model(self) -> PatchTSMixerForPrediction:
        """Create and configure the PatchTSMixer model."""
        config = PatchTSMixerConfig(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            num_input_channels=self.num_input_channels,
            patch_length=self.patch_length,
            patch_stride=self.patch_stride,
            d_model=128,
            num_layers=3,
            expansion_factor=2,
            dropout=0.1,
            head_dropout=0.1,
            num_parallel_samples=100,
            init_std=0.02,
            # Set the target dimension to predict only 1 feature (close price)
            target_dim=1,
        )

        model = PatchTSMixerForPrediction(config)

        self.logger.info(
            f"Created PatchTSMixer model with {sum(p.numel() for p in model.parameters())} parameters"
        )
        return model

    def collate_fn(self, batch):
        """Custom data collator for PatchTSMixer."""
        # Extract past_values and future_values
        past_values = [
            torch.tensor(item["past_values"], dtype=torch.float32) for item in batch
        ]
        future_values = [
            torch.tensor(item["future_values"], dtype=torch.float32) for item in batch
        ]

        # Stack into batch tensors
        past_values = torch.stack(
            past_values
        )  # Shape: [batch_size, context_length, num_features]
        future_values = torch.stack(
            future_values
        )  # Shape: [batch_size, prediction_length]

        return {"past_values": past_values, "future_values": future_values}

    def train_model(
        self,
        model: PatchTSMixerForPrediction,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        """Train the model with error handling."""
        try:
            self.logger.info("Starting model training")

            training_args = TrainingArguments(
                output_dir=str(self.output_dir / "checkpoints"),
                num_train_epochs=self.num_epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                logging_dir=str(self.output_dir / "logs"),
                logging_steps=200,
                save_total_limit=3,
                dataloader_drop_last=True,
                remove_unused_columns=False,
                report_to=None,  # Disable wandb for now
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=self.collate_fn,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            )

            # Train the model with error catching
            training_output = trainer.train()

            # Check for training issues
            if (
                hasattr(training_output, "training_loss")
                and training_output.training_loss > 10.0
            ):
                error_msg = f"Training loss too high: {training_output.training_loss}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Save the best model
            model_path = self.output_dir / "best_model"
            trainer.save_model(str(model_path))
            self.logger.info(f"Model saved to {model_path}")

            return trainer

        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            self.logger.error(error_msg)
            raise

    def evaluate_model(
        self, model: PatchTSMixerForPrediction, test_dataset: Dataset
    ) -> dict:
        """Evaluate the model on test data."""
        self.logger.info("Evaluating model on test set")

        # Get the device of the model
        device = next(model.parameters()).device

        model.eval()
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
                outputs = model(past_values=past_values)
                pred = outputs.prediction_outputs

                # Handle different output shapes
                if pred.dim() == 3:
                    pred = pred.mean(dim=1)  # Average over samples

                # If pred has multiple features, take only the first one (close price)
                if pred.dim() == 2 and pred.shape[1] > 1:
                    pred = pred[:, 0:1]  # Take only first feature

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

        # Calculate metrics
        mse = mean_squared_error(actuals_orig, predictions_orig)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals_orig, predictions_orig)

        # Calculate MAPE (avoiding division by zero)
        mask = actuals_orig != 0
        mape = (
            np.mean(
                np.abs(
                    (actuals_orig[mask] - predictions_orig[mask]) / actuals_orig[mask]
                )
            )
            * 100
        )

        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
        }

        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def predict(
        self,
        model: PatchTSMixerForPrediction,
        test_dataset: Dataset,
        n_predictions: int = 10,
    ) -> dict:
        """Generate predictions on test data with detailed analysis."""
        self.logger.info(f"Generating {n_predictions} predictions with backtesting")

        try:
            # Get the device of the model
            device = next(model.parameters()).device

            model.eval()
            predictions_scaled = []
            actuals_scaled = []

            # Create DataLoader for a subset of test data
            from torch.utils.data import DataLoader, Subset

            subset_dataset = Subset(
                test_dataset, range(min(n_predictions, len(test_dataset)))
            )
            test_loader = DataLoader(
                subset_dataset, batch_size=1, collate_fn=self.collate_fn, shuffle=False
            )

            with torch.no_grad():
                for batch in test_loader:
                    past_values = batch["past_values"].to(device)
                    future_values = batch["future_values"]

                    # Generate prediction
                    outputs = model(past_values=past_values)
                    pred = outputs.prediction_outputs

                    # Handle different output shapes
                    if pred.dim() == 3:
                        pred = pred.mean(dim=1)  # Average over samples

                    # If pred has multiple features, take only the first one (close price)
                    if pred.dim() == 2 and pred.shape[1] > 1:
                        pred = pred[:, 0:1]  # Take only first feature

                    predictions_scaled.append(pred.cpu().numpy())
                    actuals_scaled.append(future_values.numpy())

            predictions_scaled = np.concatenate(predictions_scaled, axis=0).flatten()
            actuals_scaled = np.concatenate(actuals_scaled, axis=0).flatten()

            # Inverse transform to get original price scale
            predictions_orig = self.target_scaler.inverse_transform(
                predictions_scaled.reshape(-1, 1)
            ).flatten()
            actuals_orig = self.target_scaler.inverse_transform(
                actuals_scaled.reshape(-1, 1)
            ).flatten()

            # Calculate prediction accuracy metrics
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
            }

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

    def _calculate_prediction_accuracy(
        self, predictions: np.ndarray, actuals: np.ndarray
    ) -> dict:
        """Calculate detailed prediction accuracy metrics."""
        try:
            # Direction accuracy (up/down prediction)
            pred_direction = np.diff(predictions) > 0
            actual_direction = np.diff(actuals) > 0
            direction_accuracy = (
                np.mean(pred_direction == actual_direction) * 100
                if len(pred_direction) > 0
                else 0
            )

            # Price accuracy within tolerance
            tolerance_1pct = (
                np.mean(np.abs(predictions - actuals) / actuals < 0.01) * 100
            )
            tolerance_5pct = (
                np.mean(np.abs(predictions - actuals) / actuals < 0.05) * 100
            )

            # Statistical metrics
            correlation = (
                np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
            )

            return {
                "direction_accuracy_pct": float(direction_accuracy),
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
            return {}

    def backtest_model(
        self, model: PatchTSMixerForPrediction, test_dataset: Dataset
    ) -> Dict[str, Any]:
        """Comprehensive backtesting of the model with visualization."""
        self.logger.info("Starting comprehensive backtesting with visualization")

        try:
            device = next(model.parameters()).device
            model.eval()

            all_predictions_scaled = []
            all_actuals_scaled = []

            from torch.utils.data import DataLoader

            test_loader = DataLoader(
                test_dataset, batch_size=32, collate_fn=self.collate_fn, shuffle=False
            )

            with torch.no_grad():
                for batch in test_loader:
                    past_values = batch["past_values"].to(device)
                    future_values = batch["future_values"]

                    outputs = model(past_values=past_values)
                    pred = outputs.prediction_outputs

                    if pred.dim() == 3:
                        pred = pred.mean(dim=1)

                    if pred.dim() == 2 and pred.shape[1] > 1:
                        pred = pred[:, 0:1]

                    all_predictions_scaled.extend(pred.cpu().numpy())
                    all_actuals_scaled.extend(future_values.numpy())

            all_predictions_scaled = np.array(all_predictions_scaled).flatten()
            all_actuals_scaled = np.array(all_actuals_scaled).flatten()

            # Transform to original scale
            all_predictions_orig = self.target_scaler.inverse_transform(
                all_predictions_scaled.reshape(-1, 1)
            ).flatten()
            all_actuals_orig = self.target_scaler.inverse_transform(
                all_actuals_scaled.reshape(-1, 1)
            ).flatten()

            # Comprehensive accuracy analysis
            backtest_results = self._calculate_prediction_accuracy(
                all_predictions_orig, all_actuals_orig
            )

            # Add more detailed metrics
            backtest_results.update(
                {
                    "total_predictions": len(all_predictions_orig),
                    "price_range_actual": {
                        "min": float(np.min(all_actuals_orig)),
                        "max": float(np.max(all_actuals_orig)),
                        "mean": float(np.mean(all_actuals_orig)),
                    },
                    "price_range_predicted": {
                        "min": float(np.min(all_predictions_orig)),
                        "max": float(np.max(all_predictions_orig)),
                        "mean": float(np.mean(all_predictions_orig)),
                    },
                }
            )

            # Create visualizations
            self._create_backtest_visualizations(
                all_predictions_orig, all_actuals_orig, backtest_results
            )

            self.logger.info(f"Backtesting completed: {backtest_results}")
            return backtest_results

        except Exception as e:
            self.logger.error(f"Backtesting failed: {str(e)}")
            raise

    def run_experiment(self):
        """Run the complete experiment with comprehensive analysis."""
        try:
            self.logger.info("Starting PatchTSMixer experiment")

            # Load and prepare data
            df = self.load_data()
            train_dataset, val_dataset, test_dataset = self.prepare_data(df)

            # Create model
            model = self.create_model()

            # Train model
            trainer = self.train_model(model, train_dataset, val_dataset)

            # Evaluate model
            metrics = self.evaluate_model(model, test_dataset)

            # Generate sample predictions with analysis
            prediction_results = self.predict(model, test_dataset, n_predictions=10)

            # Comprehensive backtesting
            backtest_results = self.backtest_model(model, test_dataset)

            # Save comprehensive results
            results = {
                "model": "PatchTSMixer",
                "training_metrics": metrics,
                "sample_predictions": {
                    "predictions_original_price": prediction_results[
                        "predictions_original"
                    ],
                    "actuals_original_price": prediction_results["actuals_original"],
                    "predictions_scaled": prediction_results["predictions_scaled"],
                    "accuracy_metrics": prediction_results["accuracy_metrics"],
                },
                "backtest_results": backtest_results,
                "timestamp": datetime.now().isoformat(),
                "model_config": {
                    "context_length": self.context_length,
                    "prediction_length": self.prediction_length,
                    "num_input_channels": self.num_input_channels,
                    "patch_length": self.patch_length,
                    "patch_stride": self.patch_stride,
                },
            }

            import json

            with open(self.output_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)

            # Save additional analysis files
            self._save_detailed_analysis(prediction_results, backtest_results)

            self.logger.info("Experiment completed successfully!")
            self.logger.info(f"Results saved to {self.output_dir}")
            self.logger.info(
                f"Model shows {backtest_results.get('direction_accuracy_pct', 0):.2f}% direction accuracy"
            )
            self.logger.info(
                f"Predictions within 1% tolerance: {backtest_results.get('within_1pct_tolerance', 0):.2f}%"
            )

            return results

        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            # Stop execution immediately on any error
            raise

    def _create_backtest_visualizations(
        self, predictions: np.ndarray, actuals: np.ndarray, metrics: Dict[str, Any]
    ) -> None:
        """Create comprehensive visualization plots for backtest results."""
        try:
            self.logger.info("Creating backtest visualizations")

            # Set up the plotting style
            plt.style.use("seaborn-v0_8")
            sns.set_palette("husl")

            # Create visualization directory
            viz_dir = self.output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)

            # 1. Time series comparison plot
            self._plot_time_series_comparison(predictions, actuals, viz_dir)

            # 2. Scatter plot of predictions vs actuals
            self._plot_prediction_scatter(predictions, actuals, metrics, viz_dir)

            # 3. Error distribution plot
            self._plot_error_distribution(predictions, actuals, viz_dir)

            # 4. Performance metrics dashboard
            self._plot_metrics_dashboard(metrics, viz_dir)

            # 5. Residuals analysis
            self._plot_residuals_analysis(predictions, actuals, viz_dir)

            self.logger.info(f"Visualizations saved to {viz_dir}")

        except Exception as e:
            self.logger.error(f"Failed to create visualizations: {str(e)}")

    def _plot_time_series_comparison(
        self, predictions: np.ndarray, actuals: np.ndarray, viz_dir: Path
    ) -> None:
        """Plot time series comparison between predictions and actuals."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Full time series
        time_index = range(len(predictions))
        ax1.plot(
            time_index, actuals, label="Actual", color="blue", alpha=0.7, linewidth=1
        )
        ax1.plot(
            time_index,
            predictions,
            label="Predicted",
            color="red",
            alpha=0.7,
            linewidth=1,
        )
        ax1.set_title(
            "PatchTSMixer: Predictions vs Actual Values (Full Series)",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Price (USD)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Zoomed view (last 100 points)
        zoom_start = max(0, len(predictions) - 100)
        zoom_index = time_index[zoom_start:]
        ax2.plot(
            zoom_index,
            actuals[zoom_start:],
            label="Actual",
            color="blue",
            alpha=0.8,
            linewidth=2,
        )
        ax2.plot(
            zoom_index,
            predictions[zoom_start:],
            label="Predicted",
            color="red",
            alpha=0.8,
            linewidth=2,
        )
        ax2.set_title(
            "PatchTSMixer: Predictions vs Actual Values (Last 100 Points)",
            fontsize=14,
            fontweight="bold",
        )
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Price (USD)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            viz_dir / "time_series_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_prediction_scatter(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        metrics: Dict[str, Any],
        viz_dir: Path,
    ) -> None:
        """Plot scatter plot of predictions vs actuals with performance metrics."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Scatter plot
        ax.scatter(actuals, predictions, alpha=0.6, s=20, color="darkblue")

        # Perfect prediction line
        min_val = min(np.min(actuals), np.min(predictions))
        max_val = max(np.max(actuals), np.max(predictions))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="Perfect Prediction",
        )

        # Add correlation coefficient
        correlation = metrics.get("correlation", 0)
        ax.text(
            0.05,
            0.95,
            f"Correlation: {correlation:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        ax.set_xlabel("Actual Values (USD)", fontsize=12)
        ax.set_ylabel("Predicted Values (USD)", fontsize=12)
        ax.set_title(
            "PatchTSMixer: Predictions vs Actual Values Scatter Plot",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / "prediction_scatter.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_error_distribution(
        self, predictions: np.ndarray, actuals: np.ndarray, viz_dir: Path
    ) -> None:
        """Plot error distribution analysis."""
        errors = predictions - actuals
        percentage_errors = (errors / actuals) * 100

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Absolute errors histogram
        ax1.hist(errors, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
        ax1.set_title("Distribution of Prediction Errors", fontweight="bold")
        ax1.set_xlabel("Error (Predicted - Actual)")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)

        # Percentage errors histogram
        ax2.hist(
            percentage_errors, bins=50, alpha=0.7, color="lightcoral", edgecolor="black"
        )
        ax2.set_title("Distribution of Percentage Errors", fontweight="bold")
        ax2.set_xlabel("Percentage Error (%)")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)

        # Error over time
        ax3.plot(range(len(errors)), errors, alpha=0.7, color="green", linewidth=1)
        ax3.set_title("Prediction Errors Over Time", fontweight="bold")
        ax3.set_xlabel("Time Steps")
        ax3.set_ylabel("Error")
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color="red", linestyle="--", alpha=0.7)

        # Cumulative error
        cumulative_error = np.cumsum(np.abs(errors))
        ax4.plot(
            range(len(cumulative_error)), cumulative_error, color="purple", linewidth=2
        )
        ax4.set_title("Cumulative Absolute Error", fontweight="bold")
        ax4.set_xlabel("Time Steps")
        ax4.set_ylabel("Cumulative |Error|")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / "error_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_metrics_dashboard(self, metrics: Dict[str, Any], viz_dir: Path) -> None:
        """Create a comprehensive metrics dashboard."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Key metrics bar plot
        key_metrics = ["mse", "rmse", "mae", "mape"]
        metric_values = [metrics.get(m, 0) for m in key_metrics]
        metric_labels = ["MSE", "RMSE", "MAE", "MAPE (%)"]

        bars = ax1.bar(
            metric_labels,
            metric_values,
            color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
        )
        ax1.set_title("Performance Metrics", fontweight="bold")
        ax1.set_ylabel("Value")

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        # Accuracy metrics
        accuracy_metrics = [
            "direction_accuracy_pct",
            "within_1pct_tolerance",
            "within_5pct_tolerance",
        ]
        accuracy_values = [metrics.get(m, 0) for m in accuracy_metrics]
        accuracy_labels = [
            "Direction\nAccuracy (%)",
            "Within 1%\nTolerance (%)",
            "Within 5%\nTolerance (%)",
        ]

        bars2 = ax2.bar(
            accuracy_labels, accuracy_values, color=["#F7DC6F", "#BB8FCE", "#85C1E9"]
        )
        ax2.set_title("Accuracy Metrics (%)", fontweight="bold")
        ax2.set_ylabel("Percentage")
        ax2.set_ylim(0, 100)

        for bar, value in zip(bars2, accuracy_values):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
            )

        # Price range comparison
        actual_range = metrics.get("price_range_actual", {})
        pred_range = metrics.get("price_range_predicted", {})

        categories = ["Min", "Mean", "Max"]
        actual_vals = [
            actual_range.get("min", 0),
            actual_range.get("mean", 0),
            actual_range.get("max", 0),
        ]
        pred_vals = [
            pred_range.get("min", 0),
            pred_range.get("mean", 0),
            pred_range.get("max", 0),
        ]

        x = np.arange(len(categories))
        width = 0.35

        ax3.bar(
            x - width / 2, actual_vals, width, label="Actual", color="blue", alpha=0.7
        )
        ax3.bar(
            x + width / 2, pred_vals, width, label="Predicted", color="red", alpha=0.7
        )
        ax3.set_title("Price Range Comparison", fontweight="bold")
        ax3.set_ylabel("Price (USD)")
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()

        # Model summary text
        ax4.axis("off")
        summary_text = f"""
        PatchTSMixer Model Performance Summary
        
        Total Predictions: {metrics.get('total_predictions', 0):,}
        
        Error Metrics:
        • RMSE: {metrics.get('rmse', 0):.2f}
        • MAE: {metrics.get('mae', 0):.2f}
        • MAPE: {metrics.get('mape', 0):.2f}%
        
        Accuracy Metrics:
        • Direction Accuracy: {metrics.get('direction_accuracy_pct', 0):.1f}%
        • Within 1% Tolerance: {metrics.get('within_1pct_tolerance', 0):.1f}%
        • Within 5% Tolerance: {metrics.get('within_5pct_tolerance', 0):.1f}%
        
        Statistical Metrics:
        • Correlation: {metrics.get('correlation', 0):.4f}
        """

        ax4.text(
            0.1,
            0.9,
            summary_text,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(viz_dir / "metrics_dashboard.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_residuals_analysis(
        self, predictions: np.ndarray, actuals: np.ndarray, viz_dir: Path
    ) -> None:
        """Plot residuals analysis for model diagnostics."""
        residuals = predictions - actuals

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Residuals vs fitted values
        ax1.scatter(predictions, residuals, alpha=0.6, s=20)
        ax1.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax1.set_xlabel("Fitted Values")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs Fitted Values", fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Q-Q plot for residuals normality
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot of Residuals", fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Residuals autocorrelation
        if HAS_STATSMODELS:
            lags = min(40, len(residuals) // 4)
            autocorr = acf(residuals, nlags=lags, fft=True)
            ax3.stem(range(len(autocorr)), autocorr)
            ax3.axhline(y=0, color="red", linestyle="-", alpha=0.7)
            ax3.set_xlabel("Lag")
            ax3.set_ylabel("Autocorrelation")
            ax3.set_title("Residuals Autocorrelation", fontweight="bold")
            ax3.grid(True, alpha=0.3)
        else:
            # Simple autocorrelation without statsmodels
            lags = min(20, len(residuals) // 4)
            autocorr = [
                np.corrcoef(residuals[:-i], residuals[i:])[0, 1] if i > 0 else 1.0
                for i in range(lags)
            ]
            ax3.stem(range(len(autocorr)), autocorr)
            ax3.axhline(y=0, color="red", linestyle="-", alpha=0.7)
            ax3.set_xlabel("Lag")
            ax3.set_ylabel("Autocorrelation (Simple)")
            ax3.set_title("Residuals Autocorrelation (Simple)", fontweight="bold")
            ax3.grid(True, alpha=0.3)

        # Running statistics
        window_size = max(10, len(residuals) // 20)
        running_mean = pd.Series(residuals).rolling(window=window_size).mean()
        running_std = pd.Series(residuals).rolling(window=window_size).std()

        ax4.plot(running_mean, label="Running Mean", color="blue")
        ax4.fill_between(
            range(len(running_mean)),
            running_mean - running_std,
            running_mean + running_std,
            alpha=0.3,
            label="±1 Std Dev",
        )
        ax4.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax4.set_xlabel("Time Steps")
        ax4.set_ylabel("Residuals")
        ax4.set_title("Running Statistics of Residuals", fontweight="bold")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / "residuals_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _save_detailed_analysis(
        self, prediction_results: Dict[str, Any], backtest_results: Dict[str, Any]
    ) -> None:
        """Save detailed analysis files."""
        try:
            analysis_dir = self.output_dir / "analysis"
            analysis_dir.mkdir(exist_ok=True)

            # Save prediction results
            with open(analysis_dir / "prediction_results.json", "w") as f:
                json.dump(prediction_results, f, indent=2, default=str)

            # Save backtest results
            with open(analysis_dir / "backtest_results.json", "w") as f:
                json.dump(backtest_results, f, indent=2, default=str)

            # Create summary report
            summary_report = {
                "experiment_timestamp": datetime.now().isoformat(),
                "model_type": "PatchTSMixer",
                "model_config": {
                    "context_length": self.context_length,
                    "prediction_length": self.prediction_length,
                    "num_input_channels": self.num_input_channels,
                    "patch_length": self.patch_length,
                    "patch_stride": self.patch_stride,
                },
                "training_config": {
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "train_ratio": self.train_ratio,
                    "val_ratio": self.val_ratio,
                    "test_ratio": self.test_ratio,
                },
                "performance_summary": backtest_results,
                "prediction_sample": {
                    "sample_size": len(
                        prediction_results.get("predictions_original", [])
                    ),
                    "accuracy_metrics": prediction_results.get("accuracy_metrics", {}),
                },
            }

            with open(analysis_dir / "experiment_summary.json", "w") as f:
                json.dump(summary_report, f, indent=2, default=str)

            self.logger.info(f"Detailed analysis saved to {analysis_dir}")

        except Exception as e:
            self.logger.error(f"Failed to save detailed analysis: {str(e)}")


def main():
    """Run the PatchTSMixer experiment."""
    try:
        data_path = r"d:\Projects\Desktop\FinSight\finetuning\data\BTCUSDT_1d.csv"
        output_dir = r"d:\Projects\Desktop\FinSight\finetuning\src\experiments\outputs\patchtsmixer"

        experiment = PatchTSMixerExperiment(data_path, output_dir)
        results = experiment.run_experiment()
        print(f"Experiment completed! Results: {results['training_metrics']}")
        return results
    except Exception as e:
        logger = LoggerFactory.get_logger("PatchTSMixerMain")
        logger.error(f"Main execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
