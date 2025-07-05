"""
Standalone TimesFM experiment script for debugging.
This implements all logic (data loading, preprocessing, train, evaluate, predict) in one place
to identify working patterns before refactoring back to strategies.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import (
    TimesfmConfig,
    TimesfmForPrediction,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datasets import Dataset
from datetime import datetime
import json

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from logger.logger_factory import LoggerFactory

warnings.filterwarnings("ignore")


class TimesfmExperiment:
    """Complete TimesFM experiment implementation."""

    def __init__(self, data_path: str, output_dir: str = "timesfm_experiment"):
        self.logger = LoggerFactory.get_logger("TimesfmExperiment")
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Model hyperparameters
        self.context_length = 64  # Context length for input sequences
        self.prediction_length = 1  # Predict next 1 day
        self.num_input_channels = 1  # Only predict close price

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
            f"Initialized TimesFM experiment with context_length={self.context_length}"
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
            # Input sequence (only close price for TimesFM)
            X.append(data[i : i + self.context_length, 3:4])  # close price only
            # Target (predict close price)
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

    def prepare_data(self, df: pd.DataFrame) -> tuple[Dataset, Dataset, Dataset]:
        """Prepare data for training, validation, and testing."""
        # Extract features
        feature_cols = ["open", "high", "low", "close", "volume"]
        data = df[feature_cols].values.astype(np.float32)

        # Scale features
        data_scaled = self.feature_scaler.fit_transform(data)

        # Create sequences
        X, y = self.create_sequences(data_scaled)

        # Scale targets separately
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).reshape(y.shape)

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

    def create_model(self) -> TimesfmForPrediction:
        """Create and configure the TimesFM model."""
        try:
            config = TimesfmConfig(
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                num_input_channels=self.num_input_channels,
                # Add other TimesFM specific parameters as needed
            )

            model = TimesfmForPrediction(config)

            self.logger.info(
                f"Created TimesFM model with {sum(p.numel() for p in model.parameters())} parameters"
            )
            return model
        except Exception as e:
            self.logger.error(f"Failed to create TimesFM model: {e}")
            raise

    def collate_fn(self, batch):
        """Custom data collator for TimesFM."""
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
        model: TimesfmForPrediction,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        """Train the model."""
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
            load_best_model_at_end=False,  # Disable for now
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
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
        )

        # Train the model
        trainer.train()

        # Save the best model
        model_path = self.output_dir / "best_model"
        trainer.save_model(str(model_path))
        self.logger.info(f"Model saved to {model_path}")

        return trainer

    def evaluate_model(
        self, model: TimesfmForPrediction, test_dataset: Dataset
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
            test_dataset, batch_size=32, collate_fn=self.collate_fn, shuffle=False
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
        mape = (
            np.mean(
                np.abs(
                    (actuals_orig - predictions_orig)
                    / np.where(actuals_orig != 0, actuals_orig, 1e-8)
                )
            )
            * 100
        )

        metrics = {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}

        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def predict(
        self,
        model: TimesfmForPrediction,
        test_dataset: Dataset,
        n_predictions: int = 10,
    ) -> np.ndarray:
        """Generate predictions on test data."""
        self.logger.info(f"Generating {n_predictions} predictions")

        # Get the device of the model
        device = next(model.parameters()).device

        model.eval()
        predictions = []

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

                # Generate prediction
                outputs = model(past_values=past_values)
                pred = outputs.prediction_outputs

                # Handle different output shapes
                if pred.dim() == 3:
                    pred = pred.mean(dim=1)  # Average over samples

                # If pred has multiple features, take only the first one (close price)
                if pred.dim() == 2 and pred.shape[1] > 1:
                    pred = pred[:, 0:1]  # Take only first feature

                predictions.append(pred.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)

        # Flatten to ensure 1D
        predictions = predictions.flatten()

        # Inverse transform to get original scale
        predictions_orig = self.target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()

        self.logger.info(f"Generated {len(predictions_orig)} predictions")
        return predictions_orig

    def run_experiment(self):
        """Run the complete experiment."""
        try:
            self.logger.info("Starting TimesFM experiment")

            # Load and prepare data
            df = self.load_data()
            train_dataset, val_dataset, test_dataset = self.prepare_data(df)

            # Create model
            model = self.create_model()

            # Train model
            trainer = self.train_model(model, train_dataset, val_dataset)

            # Evaluate model
            metrics = self.evaluate_model(model, test_dataset)

            # Generate some predictions
            predictions = self.predict(model, test_dataset)

            # Save results
            results = {
                "model": "TimesFM",
                "metrics": metrics,
                "sample_predictions": [float(x) for x in predictions],
                "timestamp": datetime.now().isoformat(),
            }

            with open(self.output_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)

            self.logger.info("Experiment completed successfully!")
            self.logger.info(f"Results saved to {self.output_dir}")

            return results

        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            raise


def main():
    """Run the TimesFM experiment."""
    data_path = r"d:\Projects\Desktop\FinSight\finetuning\data\BTCUSDT_1d.csv"
    output_dir = (
        r"d:\Projects\Desktop\FinSight\finetuning\src\experiments\outputs\timesfm"
    )

    experiment = TimesfmExperiment(data_path, output_dir)
    results = experiment.run_experiment()

    print(f"Experiment completed! Results: {results['metrics']}")


if __name__ == "__main__":
    main()
