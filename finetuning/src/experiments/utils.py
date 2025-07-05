"""
Shared utilities for time series experiments.
Contains common functionality used across different model implementations.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import warnings
import logging
from typing import Tuple, Dict, Any, List, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datasets import Dataset
from datetime import datetime
import json
import sys
import os

# Add parent directory to path for logger access
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from logger.logger_factory import LoggerFactory


class ExperimentConfig:
    """Configuration class for time series experiments."""

    def __init__(
        self,
        context_length: int = 64,
        prediction_length: int = 1,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        num_epochs: int = 10,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio


def setup_error_handling(experiment_name: str) -> None:
    """Setup error handling to stop execution on warnings."""
    logger = LoggerFactory.get_logger(experiment_name)

    def warning_handler(message, category, filename, lineno, file=None, line=None):
        logger.error(
            f"WARNING detected in {filename}:{lineno} - {category.__name__}: {message}"
        )
        raise RuntimeError(f"Execution stopped due to WARNING: {message}")

    warnings.showwarning = warning_handler
    warnings.simplefilter("error")
    logging.captureWarnings(True)


def load_btcusdt_data(data_path: Union[str, Path], logger: Any) -> pd.DataFrame:
    """Load and prepare BTCUSDT data."""
    logger.info(f"Loading data from {data_path}")

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Select features
    feature_cols = ["open", "high", "low", "close", "volume"]
    df = df[["timestamp"] + feature_cols].copy()

    # Remove any NaN values
    df = df.dropna()

    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def create_sequences(
    data: np.ndarray, context_length: int, prediction_length: int, logger: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for time series prediction."""
    X, y = [], []

    for i in range(len(data) - context_length - prediction_length + 1):
        # Input sequence
        X.append(data[i : i + context_length])
        # Target (predict close price)
        y.append(
            data[
                i + context_length : i + context_length + prediction_length,
                3,  # close price index
            ]
        )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    logger.info(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def prepare_scalers_and_data(
    df: pd.DataFrame,
    config: ExperimentConfig,
    logger: Any,
    num_input_channels: Optional[int] = None,
) -> Tuple[Dataset, Dataset, Dataset, StandardScaler, StandardScaler]:
    """Prepare data with proper scaling for time series prediction."""

    # Extract features
    feature_cols = ["open", "high", "low", "close", "volume"]
    data = df[feature_cols].values.astype(np.float32)

    # Initialize scalers
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Scale features
    data_scaled = feature_scaler.fit_transform(data)

    # Create sequences using scaled data
    X, y = create_sequences(
        data_scaled, config.context_length, config.prediction_length, logger
    )

    # Extract corresponding original close prices for targets
    target_original_prices = []
    for i in range(len(data) - config.context_length - config.prediction_length + 1):
        target_idx = i + config.context_length
        if target_idx < len(data):
            target_original_prices.append(data[target_idx, 3])  # close price

    target_original_prices = np.array(target_original_prices)

    # Fit target scaler on original target prices
    target_scaler.fit(target_original_prices.reshape(-1, 1))

    # Scale the targets using the fitted scaler
    y_scaled = target_scaler.transform(target_original_prices.reshape(-1, 1)).reshape(
        y.shape
    )

    # Handle different input channel requirements
    if num_input_channels == 1:
        # Use only close price for models that require single channel
        X = X[:, :, 3:4]  # Take only close price (index 3)
    elif num_input_channels is not None and num_input_channels != 5:
        # Handle other specific channel requirements
        X = X[:, :, :num_input_channels]

    # Split data
    n_samples = len(X)
    train_end = int(n_samples * config.train_ratio)
    val_end = int(n_samples * (config.train_ratio + config.val_ratio))

    X_train, y_train = X[:train_end], y_scaled[:train_end]
    X_val, y_val = X[train_end:val_end], y_scaled[train_end:val_end]
    X_test, y_test = X[val_end:], y_scaled[val_end:]

    logger.info(
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

    return train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler


def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive metrics for time series predictions."""

    # Basic regression metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)

    # Calculate MAPE (avoiding division by zero)
    mask = actuals != 0
    mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
    }


def calculate_prediction_accuracy(
    predictions: np.ndarray, actuals: np.ndarray
) -> Dict[str, Any]:
    """Calculate detailed prediction accuracy metrics."""

    # Direction accuracy
    pred_direction = np.diff(predictions) > 0
    actual_direction = np.diff(actuals) > 0
    direction_accuracy = np.mean(pred_direction == actual_direction) * 100

    # Tolerance accuracy
    tolerance_1pct = np.mean(np.abs(predictions - actuals) / actuals <= 0.01) * 100
    tolerance_5pct = np.mean(np.abs(predictions - actuals) / actuals <= 0.05) * 100

    # Correlation
    correlation = (
        np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0.0
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


def save_experiment_results(
    output_dir: Path,
    model_name: str,
    training_metrics: Dict[str, float],
    sample_predictions: Dict[str, Any],
    backtest_results: Dict[str, Any],
    logger: Any,
) -> None:
    """Save comprehensive experiment results."""

    try:
        # Main results file
        results = {
            "model": model_name,
            "training_metrics": training_metrics,
            "sample_predictions": sample_predictions,
            "backtest_summary": {
                "total_predictions": backtest_results.get("total_predictions", 0),
                "direction_accuracy_pct": backtest_results.get(
                    "direction_accuracy_pct", 0
                ),
                "within_1pct_tolerance": backtest_results.get(
                    "within_1pct_tolerance", 0
                ),
                "within_5pct_tolerance": backtest_results.get(
                    "within_5pct_tolerance", 0
                ),
                "correlation": backtest_results.get("correlation", 0),
                "price_range_actual": backtest_results.get("price_range_actual", {}),
                "price_range_predicted": backtest_results.get(
                    "price_range_predicted", {}
                ),
            },
        }

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Detailed predictions CSV
        if (
            "predictions_original_price" in sample_predictions
            and "actuals_original_price" in sample_predictions
        ):
            predictions_df = pd.DataFrame(
                {
                    "prediction": sample_predictions["predictions_original_price"],
                    "actual": sample_predictions["actuals_original_price"],
                    "error": np.array(sample_predictions["predictions_original_price"])
                    - np.array(sample_predictions["actuals_original_price"]),
                    "error_pct": (
                        np.array(sample_predictions["predictions_original_price"])
                        - np.array(sample_predictions["actuals_original_price"])
                    )
                    / np.array(sample_predictions["actuals_original_price"])
                    * 100,
                }
            )
            predictions_df.to_csv(output_dir / "detailed_predictions.csv", index=False)

        # Backtest summary
        with open(output_dir / "backtest_summary.txt", "w") as f:
            f.write(f"{model_name} Model Backtest Summary\n")
            f.write("=" * 35 + "\n\n")
            f.write(
                f"Total Predictions: {backtest_results.get('total_predictions', 0)}\n"
            )
            f.write(
                f"Direction Accuracy: {backtest_results.get('direction_accuracy_pct', 0):.2f}%\n"
            )
            f.write(
                f"Within 1% Tolerance: {backtest_results.get('within_1pct_tolerance', 0):.2f}%\n"
            )
            f.write(
                f"Within 5% Tolerance: {backtest_results.get('within_5pct_tolerance', 0):.2f}%\n"
            )
            f.write(f"Correlation: {backtest_results.get('correlation', 0):.4f}\n\n")

            price_actual = backtest_results.get("price_range_actual", {})
            price_pred = backtest_results.get("price_range_predicted", {})
            f.write(
                f"Actual Price Range: ${price_actual.get('min', 0):.2f} - ${price_actual.get('max', 0):.2f}\n"
            )
            f.write(
                f"Predicted Price Range: ${price_pred.get('min', 0):.2f} - ${price_pred.get('max', 0):.2f}\n"
            )

        logger.info(f"Results saved to {output_dir}")

    except Exception as e:
        logger.warning(f"Failed to save results: {str(e)}")


def create_standard_collate_fn(add_static_features: bool = False):
    """Create a standard collate function for time series models."""

    def collate_fn(batch):
        # Extract past_values and future_values
        past_values = [
            torch.tensor(item["past_values"], dtype=torch.float32) for item in batch
        ]
        future_values = [
            torch.tensor(item["future_values"], dtype=torch.float32) for item in batch
        ]

        # Stack into batch tensors
        past_values = torch.stack(past_values)
        future_values = torch.stack(future_values)

        batch_size, context_length, num_features = past_values.shape

        # Basic inputs
        inputs = {
            "past_values": past_values,
            "future_values": future_values,
        }

        # Add required inputs for certain models
        if add_static_features:
            inputs.update(
                {
                    "past_time_features": torch.zeros((batch_size, context_length, 0)),
                    "past_observed_mask": torch.ones(
                        (batch_size, context_length), dtype=torch.bool
                    ),
                    "static_categorical_features": torch.zeros(
                        (batch_size, 1), dtype=torch.long
                    ),
                }
            )

        return inputs

    return collate_fn


def run_model_evaluation(
    model: Any,
    test_dataset: Dataset,
    collate_fn: callable,
    target_scaler: StandardScaler,
    batch_size: int,
    logger: Any,
    model_name: str = "Model",
) -> Dict[str, float]:
    """Standard model evaluation procedure."""

    logger.info("Evaluating model on test set")

    device = next(model.parameters()).device
    model.eval()
    predictions = []
    actuals = []

    from torch.utils.data import DataLoader

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            past_values = inputs["past_values"]
            future_values = inputs["future_values"]

            # Generate predictions (exclude future_values for inference)
            inference_inputs = {k: v for k, v in inputs.items() if k != "future_values"}

            try:
                outputs = model(**inference_inputs)
                pred = outputs.prediction_outputs

                # Handle different output shapes
                if pred.dim() == 3:
                    pred = pred.mean(dim=1)  # Average over samples

                # If pred has multiple features, take only the first one
                if pred.dim() == 2 and pred.shape[1] > 1:
                    pred = pred[:, 0:1]

                predictions.extend(pred.cpu().numpy())
                actuals.extend(future_values.cpu().numpy())

            except Exception as e:
                logger.error(f"Evaluation failed for batch: {str(e)}")
                raise RuntimeError(f"{model_name} evaluation failed: {str(e)}") from e

    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    logger.info(
        f"Final shapes - predictions: {predictions.shape}, actuals: {actuals.shape}"
    )

    # Ensure shapes match
    if predictions.shape != actuals.shape:
        logger.warning(
            f"Shape mismatch: predictions {predictions.shape}, actuals {actuals.shape}"
        )
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]

    # Inverse transform to get original scale
    predictions_orig = target_scaler.inverse_transform(
        predictions.reshape(-1, 1)
    ).flatten()
    actuals_orig = target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

    # Calculate metrics
    metrics = calculate_metrics(predictions_orig, actuals_orig)

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics
