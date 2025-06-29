# utils/prediction_utils.py

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import DataLoader

from ..common.logger.logger_factory import LoggerFactory
from .metric_utils import MetricUtils


class PredictionUtils:
    """Utility class for model prediction operations and analysis"""

    _logger = LoggerFactory.get_logger(__name__)

    @staticmethod
    def generate_predictions(
        model: torch.nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """
        Generate comprehensive predictions with analysis

        Args:
            model: PyTorch model
            test_loader: Test data loader
            device: PyTorch device
            model_name: Name of the model for logging

        Returns:
            Dict containing predictions and analysis
        """
        PredictionUtils._logger.info(f"🔮 Generating predictions using {model_name}...")

        # Get predictions
        predictions, targets = model.predict_batch(test_loader, device)

        # Analyze predictions
        prediction_analysis = PredictionUtils.analyze_predictions(
            predictions.numpy(), targets.numpy()
        )

        # Generate future predictions (simplified)
        future_predictions = PredictionUtils.generate_future_predictions(
            model, test_loader, device
        )

        result = {
            "test_predictions": {
                "predictions": predictions.numpy(),
                "targets": targets.numpy(),
                "analysis": prediction_analysis,
            },
            "future_predictions": future_predictions,
            "model_used": model_name,
        }

        PredictionUtils._logger.info(f"✓ Predictions generated using {model_name}:")
        PredictionUtils._logger.info(f"  Test samples: {len(predictions)}")
        PredictionUtils._logger.info(
            f"  Prediction accuracy: {prediction_analysis['accuracy']:.2f}%"
        )
        PredictionUtils._logger.info(
            f"  Average error: {prediction_analysis['mean_error']:.4f}"
        )

        return result

    @staticmethod
    def analyze_predictions(
        predictions: np.ndarray, targets: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction quality and return comprehensive metrics"""
        errors = predictions.flatten() - targets.flatten()

        # Avoid division by zero in accuracy calculation
        safe_targets = np.where(targets.flatten() == 0, 1e-8, targets.flatten())

        return {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "accuracy": float(100 - np.mean(np.abs(errors / safe_targets)) * 100),
            "directional_accuracy": PredictionUtils._calculate_directional_accuracy(
                predictions, targets
            ),
            "error_distribution": {
                "min": float(np.min(errors)),
                "max": float(np.max(errors)),
                "q25": float(np.percentile(errors, 25)),
                "q50": float(np.percentile(errors, 50)),
                "q75": float(np.percentile(errors, 75)),
            },
        }

    @staticmethod
    def _calculate_directional_accuracy(
        predictions: np.ndarray, targets: np.ndarray
    ) -> float:
        """Calculate directional accuracy for time series predictions"""
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()

        if len(pred_flat) < 2:
            return 0.0

        pred_direction = np.diff(pred_flat) > 0
        target_direction = np.diff(target_flat) > 0

        return float(np.mean(pred_direction == target_direction) * 100)

    @staticmethod
    def generate_future_predictions(
        model: torch.nn.Module, test_loader: DataLoader, device: torch.device
    ) -> Dict[str, Any]:
        """Generate future predictions using the last sequence from test data"""
        try:
            # Get last batch from test loader
            last_batch = None
            for batch in test_loader:
                last_batch = batch

            if last_batch is None:
                return {"error": "No data available for future prediction"}

            # Take last sequence
            last_sequence = last_batch[0][-1:].to(device)

            # Generate prediction
            model.eval()
            with torch.no_grad():
                future_pred = model(last_sequence)

            return {
                "prediction": future_pred.cpu().numpy().tolist(),
                "confidence": "N/A",  # Would need uncertainty quantification
                "note": "Simplified demonstration - real implementation would use rolling predictions",
            }

        except Exception as e:
            PredictionUtils._logger.error(f"Error generating future predictions: {e}")
            return {"error": str(e)}

    @staticmethod
    def calculate_prediction_metrics(
        predictions: np.ndarray, targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive prediction metrics"""
        return MetricUtils.calculate_all_metrics(targets, predictions)

    @staticmethod
    def compare_prediction_quality(
        predictions_dict: Dict[str, np.ndarray], targets: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare prediction quality across multiple models

        Args:
            predictions_dict: Dict mapping model names to predictions
            targets: True target values

        Returns:
            Dict containing comparison results
        """
        comparison = {}

        for model_name, predictions in predictions_dict.items():
            metrics = PredictionUtils.calculate_prediction_metrics(predictions, targets)
            comparison[model_name] = metrics

        # Find best performing model for each metric
        best_models = {}
        metrics_to_compare = ["rmse", "mae", "mape", "directional_accuracy"]

        for metric in metrics_to_compare:
            model_values = {
                name: results[metric] for name, results in comparison.items()
            }

            if model_values:
                # For directional accuracy, higher is better; for others, lower is better
                reverse = metric == "directional_accuracy"
                best_model = (
                    max(model_values.items(), key=lambda x: x[1])
                    if reverse
                    else min(model_values.items(), key=lambda x: x[1])
                )
                best_models[metric] = {
                    "model": best_model[0],
                    "value": best_model[1],
                }

        return {"individual_metrics": comparison, "best_models": best_models}
