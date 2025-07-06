# models/adapters/base_adapter.py

"""
Base adapter for time series models with common functionality
Implements the Template Method pattern for shared operations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from sklearn.preprocessing import StandardScaler

from ...interfaces.model_interface import ITimeSeriesModel
from ...logger.logger_factory import LoggerFactory
from ...utils.metrics_utils import MetricUtils


class BaseTimeSeriesAdapter(ITimeSeriesModel):
    """
    Base adapter implementing common time series model functionality

    This class provides:
    - Scaler management (fit during training, transform during inference)
    - Data preparation utilities
    - Common evaluation metrics
    - Model persistence (save/load)
    - Device management

    Subclasses only need to implement model-specific operations
    """

    def __init__(
        self,
        context_length: int = 64,
        prediction_length: int = 1,
        target_column: str = "close",
        feature_columns: Optional[List[str]] = None,
        device: Optional[str] = None,
        **config,
    ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_column = target_column
        self.feature_columns = feature_columns or [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        self.config = config

        # Device management
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model and scalers (to be set by subclasses)
        self.model = None
        self.feature_scaler: Optional[StandardScaler] = None
        self.target_scaler: Optional[StandardScaler] = None
        self.is_trained = False

        # Feature engineering strategy (injected)
        self.feature_engineering = None

        self.logger = LoggerFactory.get_logger(f"{self.__class__.__name__}")

    # ================================
    # Abstract methods for subclasses
    # ================================

    @abstractmethod
    def _create_model(self) -> torch.nn.Module:
        """Create the specific model instance"""
        pass

    @abstractmethod
    def _train_model(
        self, train_dataset: Any, val_dataset: Any, **kwargs
    ) -> Dict[str, Any]:
        """Train the specific model implementation"""
        pass

    @abstractmethod
    def _model_predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Make prediction with the specific model"""
        pass

    @abstractmethod
    def _save_model_specific(self, model_dir: Path) -> None:
        """Save model-specific components"""
        pass

    @abstractmethod
    def _load_model_specific(self, model_dir: Path) -> None:
        """Load model-specific components"""
        pass

    # ================================
    # Data preparation utilities
    # ================================

    def _fit_scalers(self, data: pd.DataFrame) -> None:
        """Fit scalers on training data only"""
        try:
            # Prepare feature columns
            if self.feature_engineering is not None:
                processed_data = self.feature_engineering.fit_transform(data)
                feature_cols = self.feature_engineering.get_feature_names()
            else:
                processed_data = data.copy()
                feature_cols = [
                    col for col in self.feature_columns if col in processed_data.columns
                ]

            # Fit feature scaler
            if len(feature_cols) > 0:
                self.feature_scaler = StandardScaler()
                self.feature_scaler.fit(processed_data[feature_cols])
                self.logger.info(
                    f"Fitted feature scaler on {len(feature_cols)} features"
                )

            # Fit target scaler
            if self.target_column in processed_data.columns:
                self.target_scaler = StandardScaler()
                target_values = processed_data[self.target_column].values.reshape(-1, 1)
                self.target_scaler.fit(target_values)
                self.logger.info(f"Fitted target scaler on '{self.target_column}'")

        except Exception as e:
            self.logger.error(f"Error fitting scalers: {e}")
            raise

    def _transform_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scalers"""
        try:
            # Apply feature engineering
            if self.feature_engineering is not None:
                processed_data = self.feature_engineering.transform(data)
                feature_cols = self.feature_engineering.get_feature_names()
            else:
                processed_data = data.copy()
                feature_cols = [
                    col for col in self.feature_columns if col in processed_data.columns
                ]

            # Transform features
            if self.feature_scaler is not None and len(feature_cols) > 0:
                processed_data[feature_cols] = self.feature_scaler.transform(
                    processed_data[feature_cols]
                )

            # Transform target
            if (
                self.target_scaler is not None
                and self.target_column in processed_data.columns
            ):
                target_values = processed_data[self.target_column].values.reshape(-1, 1)
                processed_data[self.target_column] = self.target_scaler.transform(
                    target_values
                ).flatten()

            return processed_data

        except Exception as e:
            self.logger.error(f"Error transforming features: {e}")
            raise

    def _inverse_transform_target(self, scaled_values: np.ndarray) -> np.ndarray:
        """Inverse transform target values to original scale"""
        if self.target_scaler is not None:
            if scaled_values.ndim == 1:
                scaled_values = scaled_values.reshape(-1, 1)
            return self.target_scaler.inverse_transform(scaled_values).flatten()
        return scaled_values

    def _create_sequences(
        self, data: pd.DataFrame, for_training: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Create sequences for training or inference"""
        try:
            # Get feature columns
            if self.feature_engineering is not None:
                feature_cols = self.feature_engineering.get_feature_names()
            else:
                feature_cols = [
                    col for col in self.feature_columns if col in data.columns
                ]

            feature_data = data[feature_cols].values

            if for_training:
                # Create overlapping sequences for training
                sequences = []
                targets = []

                for i in range(
                    len(data) - self.context_length - self.prediction_length + 1
                ):
                    seq = feature_data[i : i + self.context_length]
                    target = (
                        data[self.target_column]
                        .iloc[
                            i
                            + self.context_length : i
                            + self.context_length
                            + self.prediction_length
                        ]
                        .values
                    )

                    sequences.append(seq)
                    targets.append(target)

                if len(sequences) == 0:
                    raise ValueError(
                        f"No sequences created. Data length: {len(data)}, "
                        f"Context length: {self.context_length}, "
                        f"Prediction length: {self.prediction_length}"
                    )

                sequences_tensor = torch.FloatTensor(np.array(sequences))
                targets_tensor = torch.FloatTensor(np.array(targets))

                return sequences_tensor, targets_tensor

            else:
                # Create single sequence from end of data for inference
                if len(data) < self.context_length:
                    raise ValueError(
                        f"Need at least {self.context_length} data points for inference"
                    )

                sequence = feature_data[-self.context_length :]
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(
                    0
                )  # Add batch dimension

                return sequence_tensor, None

        except Exception as e:
            self.logger.error(f"Error creating sequences: {e}")
            raise

    # ================================
    # Core interface implementations
    # ================================

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        feature_engineering: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train the model with proper scaler fitting"""
        try:
            self.logger.info(f"Starting training for {self.__class__.__name__}")

            # Store feature engineering strategy
            self.feature_engineering = feature_engineering

            # Fit scalers on training data only
            self._fit_scalers(train_data)

            # Transform both training and validation data
            train_transformed = self._transform_features(train_data)
            val_transformed = self._transform_features(val_data)

            # Create sequences
            train_sequences, train_targets = self._create_sequences(
                train_transformed, for_training=True
            )
            val_sequences, val_targets = self._create_sequences(
                val_transformed, for_training=True
            )

            self.logger.info(f"Created {len(train_sequences)} training sequences")
            self.logger.info(f"Created {len(val_sequences)} validation sequences")

            # Create model if not exists
            if self.model is None:
                self.model = self._create_model()

            # Move model to device
            self.model.to(self.device)

            # Train model (implementation specific)
            training_result = self._train_model(
                (train_sequences, train_targets), (val_sequences, val_targets), **kwargs
            )

            self.is_trained = True
            training_result["success"] = True

            return training_result

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {"success": False, "error": str(e)}

    def forecast(self, recent_data: pd.DataFrame, n_steps: int = 1) -> Dict[str, Any]:
        """Make true forecasting predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            self.logger.info(f"Making {n_steps}-step forecast")

            # Transform data using fitted scalers
            transformed_data = self._transform_features(recent_data)

            # Create sequence from last context_length points
            sequence_tensor, _ = self._create_sequences(
                transformed_data, for_training=False
            )
            sequence_tensor = sequence_tensor.to(self.device)

            # Generate predictions
            self.model.eval()
            predictions_scaled = []

            with torch.no_grad():
                if n_steps == 1:
                    # Single step prediction
                    pred = self._model_predict(sequence_tensor)
                    predictions_scaled = pred.cpu().numpy().flatten()
                else:
                    # Multi-step prediction (iterative approach)
                    current_sequence = sequence_tensor.clone()
                    target_device = current_sequence.device

                    for step in range(n_steps):
                        pred = self._model_predict(current_sequence)
                        predictions_scaled.append(pred.cpu().item())

                        # Update sequence for next prediction
                        # Ensure consistent device and shape handling
                        pred_expanded = pred.view(1, 1, 1).to(target_device)

                        # Ensure new_step matches the feature dimension of current_sequence
                        if current_sequence.shape[2] > 1:
                            # For multivariate data, replicate prediction across all features
                            # This is a simplified approach - in practice you might want
                            # different logic for different features
                            new_step = pred_expanded.expand(
                                1, 1, current_sequence.shape[2]
                            )
                        else:
                            new_step = pred_expanded

                        # Ensure both tensors are on the same device
                        current_sequence = current_sequence.to(target_device)
                        new_step = new_step.to(target_device)

                        current_sequence = torch.cat(
                            [
                                current_sequence[:, 1:, :],  # Remove first step
                                new_step,  # Add prediction
                            ],
                            dim=1,
                        )

            # Inverse transform predictions
            predictions = self._inverse_transform_target(np.array(predictions_scaled))

            # Calculate metadata
            current_price = recent_data[self.target_column].iloc[-1]

            result = {
                "predictions": predictions.tolist(),
                "current_price": float(current_price),
                "n_steps": n_steps,
                "model_type": self.__class__.__name__,
                "success": True,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            if n_steps == 1:
                prediction_change = predictions[0] - current_price
                result.update(
                    {
                        "predicted_price": float(predictions[0]),
                        "predicted_change": float(prediction_change),
                        "predicted_change_pct": float(
                            (prediction_change / current_price) * 100
                        ),
                        "direction": "up" if prediction_change > 0 else "down",
                    }
                )

            return result

        except Exception as e:
            self.logger.error(f"Forecasting failed: {e}")
            return {"success": False, "error": str(e)}

    def backtest(self, test_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Perform backtesting evaluation"""
        if not self.is_trained:
            raise ValueError("Model must be trained before backtesting")

        try:
            self.logger.info("Starting backtest evaluation")

            # Transform test data
            test_transformed = self._transform_features(test_data)

            # Create sequences for backtesting
            test_sequences, test_targets = self._create_sequences(
                test_transformed, for_training=True
            )

            # Generate predictions
            self.model.eval()
            predictions_scaled = []

            with torch.no_grad():
                for i in range(len(test_sequences)):
                    sequence = test_sequences[i : i + 1].to(self.device)
                    pred = self._model_predict(sequence)
                    predictions_scaled.append(pred.cpu().numpy().flatten())

            predictions_scaled = np.concatenate(predictions_scaled)
            targets_scaled = test_targets.numpy().flatten()

            # Inverse transform
            predictions = self._inverse_transform_target(predictions_scaled)
            targets = self._inverse_transform_target(targets_scaled)

            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(targets, predictions)

            return {
                "success": True,
                "metrics": metrics,
                "predictions": predictions.tolist(),
                "targets": targets.tolist(),
                "n_samples": len(predictions),
            }

        except Exception as e:
            self.logger.error(f"Backtesting failed: {e}")
            return {"success": False, "error": str(e)}

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model using backtesting"""
        return self.backtest(test_data)

    def _calculate_comprehensive_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        try:
            metrics = {}

            # Basic regression metrics
            metrics["mse"] = MetricUtils.calculate_mse(y_true, y_pred)
            metrics["rmse"] = MetricUtils.calculate_rmse(y_true, y_pred)
            metrics["mae"] = MetricUtils.calculate_mae(y_true, y_pred)
            metrics["mape"] = MetricUtils.calculate_mape(y_true, y_pred)
            metrics["smape"] = MetricUtils.calculate_smape(y_true, y_pred)
            metrics["r2"] = MetricUtils.calculate_r2(y_true, y_pred)

            # Tolerance-based accuracy metrics
            metrics["tolerance_accuracy_1pct"] = (
                MetricUtils.calculate_tolerance_accuracy(y_true, y_pred, 1.0)
            )
            metrics["tolerance_accuracy_5pct"] = (
                MetricUtils.calculate_tolerance_accuracy(y_true, y_pred, 5.0)
            )

            # Financial metrics
            metrics["directional_accuracy"] = (
                MetricUtils.calculate_directional_accuracy(y_true, y_pred)
            )

            # Risk metrics
            drawdown_metrics = MetricUtils.calculate_max_drawdown(y_true, y_pred)
            metrics.update(drawdown_metrics)

            volatility_metrics = MetricUtils.calculate_volatility(y_true, y_pred)
            metrics.update(volatility_metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {"error": str(e)}

    # ================================
    # Model persistence
    # ================================

    def save_model(self, path: Path) -> None:
        """Save complete model state"""
        try:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            # Save model configuration - use standardized name
            config = {
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "target_column": self.target_column,
                "feature_columns": self.feature_columns,
                "model_class": self.__class__.__name__,
                "is_trained": self.is_trained,
                **self.config,
            }

            with open(path / "config.json", "w") as f:
                json.dump(config, f, indent=2)

            # Save scalers
            if self.feature_scaler is not None:
                with open(path / "feature_scaler.pkl", "wb") as f:
                    pickle.dump(self.feature_scaler, f)

            if self.target_scaler is not None:
                with open(path / "target_scaler.pkl", "wb") as f:
                    pickle.dump(self.target_scaler, f)

            # Save feature engineering if available
            if self.feature_engineering is not None:
                try:
                    with open(path / "feature_engineering.pkl", "wb") as f:
                        pickle.dump(self.feature_engineering, f)
                except Exception as e:
                    self.logger.warning(f"Could not pickle feature engineering: {e}")
                    # Save as JSON fallback if possible
                    try:
                        fe_config = {
                            "feature_columns": getattr(
                                self.feature_engineering, "feature_columns", []
                            ),
                            "add_technical_indicators": getattr(
                                self.feature_engineering,
                                "add_technical_indicators",
                                True,
                            ),
                            "add_datetime_features": getattr(
                                self.feature_engineering, "add_datetime_features", False
                            ),
                            "normalize_features": getattr(
                                self.feature_engineering, "normalize_features", True
                            ),
                            "fitted_feature_names": getattr(
                                self.feature_engineering, "fitted_feature_names", []
                            ),
                        }
                        with open(path / "feature_engineering_config.json", "w") as f:
                            json.dump(fe_config, f, indent=2)
                    except Exception as fe_error:
                        self.logger.error(
                            f"Could not save feature engineering config: {fe_error}"
                        )

            # Save model-specific components
            self._save_model_specific(path)

            self.logger.info(f"Model saved to {path}")

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, path: Path) -> None:
        """Load complete model state"""
        try:
            path = Path(path)

            # Load configuration - try different config file names
            config_files = ["config.json", "model_config.json", "adapter_config.json"]
            config = None

            for config_file in config_files:
                config_path = path / config_file
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    break

            if config is None:
                raise FileNotFoundError(f"No config file found in {path}")

            self.context_length = config["context_length"]
            self.prediction_length = config["prediction_length"]
            self.target_column = config["target_column"]
            self.feature_columns = config["feature_columns"]
            self.is_trained = config.get("is_trained", True)
            self.config.update(
                {
                    k: v
                    for k, v in config.items()
                    if k
                    not in [
                        "context_length",
                        "prediction_length",
                        "target_column",
                        "feature_columns",
                        "is_trained",
                        "model_class",
                    ]
                }
            )

            # Load scalers
            feature_scaler_path = path / "feature_scaler.pkl"
            if feature_scaler_path.exists():
                with open(feature_scaler_path, "rb") as f:
                    self.feature_scaler = pickle.load(f)

            target_scaler_path = path / "target_scaler.pkl"
            if target_scaler_path.exists():
                with open(target_scaler_path, "rb") as f:
                    self.target_scaler = pickle.load(f)

            # Load feature engineering
            fe_path = path / "feature_engineering.pkl"
            if fe_path.exists():
                try:
                    with open(fe_path, "rb") as f:
                        self.feature_engineering = pickle.load(f)
                except Exception as e:
                    self.logger.warning(
                        f"Could not load pickled feature engineering: {e}"
                    )
                    # Try loading from JSON config fallback
                    fe_config_path = path / "feature_engineering_config.json"
                    if fe_config_path.exists():
                        try:
                            with open(fe_config_path, "r") as f:
                                fe_config = json.load(f)
                            # Recreate feature engineering from config
                            from ...data.feature_engineering import (
                                BasicFeatureEngineering,
                            )

                            self.feature_engineering = BasicFeatureEngineering(
                                feature_columns=fe_config.get("feature_columns"),
                                add_technical_indicators=fe_config.get(
                                    "add_technical_indicators", True
                                ),
                                add_datetime_features=fe_config.get(
                                    "add_datetime_features", False
                                ),
                                normalize_features=fe_config.get(
                                    "normalize_features", True
                                ),
                            )
                            # Restore fitted state
                            self.feature_engineering.fitted_feature_names = (
                                fe_config.get("fitted_feature_names", [])
                            )
                            self.feature_engineering.is_fitted = True
                            self.logger.info(
                                "Recreated feature engineering from config"
                            )
                        except Exception as fe_error:
                            self.logger.error(
                                f"Could not recreate feature engineering: {fe_error}"
                            )

            # Load model-specific components
            self._load_model_specific(path)

            self.logger.info(f"Model loaded from {path}")

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_class": self.__class__.__name__,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "is_trained": self.is_trained,
            "device": str(self.device),
            "has_feature_scaler": self.feature_scaler is not None,
            "has_target_scaler": self.target_scaler is not None,
            "has_feature_engineering": self.feature_engineering is not None,
            **self.config,
        }
