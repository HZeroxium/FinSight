"""
Standalone TimeSeriesTransformer experiment script for debugging.
This implements all logic (data loading, preprocessing, train, evaluate, predict) in one place
to identify working patterns before refactoring back to strategies.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import warnings
from transformers import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datasets import Dataset
from datetime import datetime
import sys
import os

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from logger.logger_factory import LoggerFactory

warnings.filterwarnings("ignore")


class TimeSeriesTransformerTrainer(Trainer):
    """Custom trainer for TimeSeriesTransformer to handle specific loss computation."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Compute loss for TimeSeriesTransformer.
        """
        past_values = inputs["past_values"]
        future_values = inputs["future_values"]
        past_time_features = inputs.get("past_time_features")
        past_observed_mask = inputs.get("past_observed_mask")

        # Create required inputs if not provided
        batch_size, context_length, num_features = past_values.shape
        device = past_values.device

        if past_time_features is None:
            # No time features for now
            past_time_features = torch.zeros(
                (batch_size, context_length, 0), device=device
            )

        if past_observed_mask is None:
            # All values are observed
            past_observed_mask = torch.ones(
                (batch_size, context_length), device=device, dtype=torch.bool
            )

        # Forward pass
        outputs = model(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            future_values=future_values,
        )

        # Extract predictions
        predictions = outputs.prediction_outputs

        # Handle different output shapes
        if predictions.dim() == 3:
            # Take the mean over num_parallel_samples dimension if present
            predictions = predictions.mean(dim=1)

        # Ensure predictions match target shape
        if predictions.shape != future_values.shape:
            # Reshape predictions to match targets
            predictions = predictions.view(future_values.shape)

        # Calculate MSE loss
        loss = torch.nn.functional.mse_loss(predictions, future_values)

        return (loss, outputs) if return_outputs else loss


class TimeSeriesTransformerExperiment:
    """Complete TimeSeriesTransformer experiment implementation."""

    def __init__(self, data_path: str, output_dir: str = "ts_transformer_experiment"):
        self.logger = LoggerFactory.get_logger("TimeSeriesTransformerExperiment")
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
            f"Initialized TimeSeriesTransformer experiment with context_length={self.context_length}"
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
            # Input sequence (only close price for TimeSeriesTransformer)
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

    def create_model(self) -> TimeSeriesTransformerForPrediction:
        """Create and configure the TimeSeriesTransformer model."""
        config = TimeSeriesTransformerConfig(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            num_time_features=0,  # No additional time features
            lags_sequence=[1],  # Simplified lags sequence
            num_dynamic_real_features=0,
            num_static_categorical_features=0,
            num_static_real_features=0,
            cardinality=[],
            embedding_dimension=[],
            encoder_layers=2,
            decoder_layers=1,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            activation_function="gelu",
            dropout=0.1,
            encoder_layerdrop=0.1,
            decoder_layerdrop=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,
            num_parallel_samples=100,
            init_std=0.02,
            use_cache=True,
            # Set target dimension to predict only 1 feature (close price)
            num_input_channels=1,  # Only predict close price
        )

        model = TimeSeriesTransformerForPrediction(config)

        self.logger.info(
            f"Created TimeSeriesTransformer model with {sum(p.numel() for p in model.parameters())} parameters"
        )
        return model

    def collate_fn(self, batch):
        """Custom data collator for TimeSeriesTransformer."""
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

        batch_size, context_length, num_features = past_values.shape

        # Create required inputs for TimeSeriesTransformer based on configuration
        return_dict = {
            "past_values": past_values,
            "future_values": future_values,
        }

        # Add required tensors for TimeSeriesTransformer model
        # past_time_features - empty tensor since num_time_features=0
        if hasattr(self, "model") and hasattr(self.model.config, "num_time_features"):
            num_time_features = self.model.config.num_time_features
            if num_time_features > 0:
                return_dict["past_time_features"] = torch.zeros(
                    (batch_size, context_length, num_time_features)
                )

        # past_observed_mask - all values are observed
        return_dict["past_observed_mask"] = torch.ones(
            (batch_size, context_length), dtype=torch.bool
        )

        # static features - not used in our case
        if hasattr(self, "model") and hasattr(
            self.model.config, "num_static_categorical_features"
        ):
            num_static_cat = self.model.config.num_static_categorical_features
            if num_static_cat > 0:
                return_dict["static_categorical_features"] = torch.zeros(
                    (batch_size, num_static_cat), dtype=torch.long
                )

        if hasattr(self, "model") and hasattr(
            self.model.config, "num_static_real_features"
        ):
            num_static_real = self.model.config.num_static_real_features
            if num_static_real > 0:
                return_dict["static_real_features"] = torch.zeros(
                    (batch_size, num_static_real)
                )

        return return_dict

    def train_model(
        self,
        model: TimeSeriesTransformerForPrediction,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        """Train the model."""
        self.logger.info("Starting model training")

        # Store model reference for collate function
        self.model = model

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

        trainer = TimeSeriesTransformerTrainer(
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
        return trainer

    def evaluate_model(
        self, model: TimeSeriesTransformerForPrediction, test_dataset: Dataset
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
                past_time_features = batch["past_time_features"].to(device)
                past_observed_mask = batch["past_observed_mask"].to(device)

                # Generate predictions (no future_values for prediction)
                outputs = model(
                    past_values=past_values,
                    past_time_features=past_time_features,
                    past_observed_mask=past_observed_mask,
                )
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
        model: TimeSeriesTransformerForPrediction,
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
                past_time_features = batch["past_time_features"].to(device)
                past_observed_mask = batch["past_observed_mask"].to(device)

                # Generate prediction
                outputs = model(
                    past_values=past_values,
                    past_time_features=past_time_features,
                    past_observed_mask=past_observed_mask,
                )
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
            self.logger.info("Starting TimeSeriesTransformer experiment")

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
                "model": "TimeSeriesTransformer",
                "metrics": metrics,
                "sample_predictions": [float(x) for x in predictions],
                "timestamp": datetime.now().isoformat(),
            }

            import json

            with open(self.output_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)

            self.logger.info("Experiment completed successfully!")
            self.logger.info(f"Results saved to {self.output_dir}")

            return results

        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            raise


def main():
    """Run the TimeSeriesTransformer experiment."""
    data_path = r"d:\Projects\Desktop\FinSight\finetuning\data\BTCUSDT_1d.csv"
    output_dir = r"d:\Projects\Desktop\FinSight\finetuning\src\experiments\outputs\ts_transformer"

    experiment = TimeSeriesTransformerExperiment(data_path, output_dir)
    results = experiment.run_experiment()

    print(f"Experiment completed! Results: {results['metrics']}")


if __name__ == "__main__":
    main()
