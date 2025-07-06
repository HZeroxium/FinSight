# utils/metrics_utils.py

"""
Comprehensive metrics utilities for time series model evaluation
"""

from typing import Union, Tuple, Dict, Any
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..logger.logger_factory import LoggerFactory


class MetricUtils:
    """Utility class for calculating various time series evaluation metrics"""

    _logger = LoggerFactory.get_logger(__name__)

    @staticmethod
    def _to_numpy(data: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
        """
        Convert input data to numpy array

        Args:
            data: Input data in various formats

        Returns:
            np.ndarray: Converted numpy array
        """
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, list):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    @staticmethod
    def _validate_inputs(
        y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and convert inputs to numpy arrays

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Tuple of validated numpy arrays

        Raises:
            ValueError: If inputs have mismatched shapes or contain invalid values
        """
        y_true_np = MetricUtils._to_numpy(y_true)
        y_pred_np = MetricUtils._to_numpy(y_pred)

        # Flatten arrays to 1D for metric calculations
        y_true_np = y_true_np.flatten()
        y_pred_np = y_pred_np.flatten()

        if y_true_np.shape != y_pred_np.shape:
            raise ValueError(
                f"Shape mismatch after flattening: y_true {y_true_np.shape} vs y_pred {y_pred_np.shape}"
            )

        # Remove any NaN or infinite values
        valid_mask = np.isfinite(y_true_np) & np.isfinite(y_pred_np)

        if not valid_mask.any():
            raise ValueError("No valid (finite) values found in inputs")

        if not valid_mask.all():
            MetricUtils._logger.warning(
                f"Removing {(~valid_mask).sum()} invalid values from inputs"
            )
            y_true_np = y_true_np[valid_mask]
            y_pred_np = y_pred_np[valid_mask]

        return y_true_np, y_pred_np

    @staticmethod
    def calculate_mse(
        y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calculate Mean Squared Error

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float: MSE value
        """
        y_true_np, y_pred_np = MetricUtils._validate_inputs(y_true, y_pred)
        return float(mean_squared_error(y_true_np, y_pred_np))

    @staticmethod
    def calculate_rmse(
        y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calculate Root Mean Squared Error

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float: RMSE value
        """
        mse = MetricUtils.calculate_mse(y_true, y_pred)
        return float(np.sqrt(mse))

    @staticmethod
    def calculate_mae(
        y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calculate Mean Absolute Error

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float: MAE value
        """
        y_true_np, y_pred_np = MetricUtils._validate_inputs(y_true, y_pred)
        return float(mean_absolute_error(y_true_np, y_pred_np))

    @staticmethod
    def calculate_mape(
        y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calculate Mean Absolute Percentage Error

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float: MAPE value (percentage)
        """
        y_true_np, y_pred_np = MetricUtils._validate_inputs(y_true, y_pred)

        mask = y_true_np != 0
        if not mask.any():
            MetricUtils._logger.warning(
                "All true values are zero, MAPE cannot be calculated"
            )
            return float("inf")

        return float(
            np.mean(np.abs((y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask])) * 100
        )

    @staticmethod
    def calculate_smape(
        y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float: SMAPE value (percentage)
        """
        y_true_np, y_pred_np = MetricUtils._validate_inputs(y_true, y_pred)

        denominator = (np.abs(y_true_np) + np.abs(y_pred_np)) / 2
        mask = denominator != 0

        if not mask.any():
            return 0.0

        return float(
            np.mean(np.abs(y_true_np[mask] - y_pred_np[mask]) / denominator[mask]) * 100
        )

    @staticmethod
    def calculate_r2(
        y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calculate R-squared (coefficient of determination)

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float: R2 value
        """
        y_true_np, y_pred_np = MetricUtils._validate_inputs(y_true, y_pred)
        return float(r2_score(y_true_np, y_pred_np))

    @staticmethod
    def calculate_directional_accuracy(
        y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calculate directional accuracy (percentage of correct direction predictions)

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float: Directional accuracy percentage
        """
        y_true_np, y_pred_np = MetricUtils._validate_inputs(y_true, y_pred)

        if len(y_true_np) < 2:
            MetricUtils._logger.warning(
                "Need at least 2 values to calculate directional accuracy"
            )
            return 0.0

        true_direction = np.diff(y_true_np) > 0
        pred_direction = np.diff(y_pred_np) > 0

        return float(np.mean(true_direction == pred_direction) * 100)

    @staticmethod
    def calculate_max_drawdown(
        y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Calculate maximum drawdown for both true and predicted values

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            dict: Maximum drawdown for true and predicted values
        """
        y_true_np, y_pred_np = MetricUtils._validate_inputs(y_true, y_pred)

        def _max_drawdown(values: np.ndarray) -> float:
            if len(values) < 2:
                return 0.0

            # Ensure we have a 1D array
            values = values.flatten()

            # Convert to cumulative series if needed
            if np.any(values <= 0):
                # For series with negative values, use cumulative sum
                cumulative = np.cumsum(values)
            else:
                cumulative = values

            peak = np.maximum.accumulate(cumulative)
            drawdown = cumulative - peak

            # Avoid division by zero
            valid_peak = peak != 0
            if valid_peak.any():
                drawdown_pct = np.where(valid_peak, drawdown / peak, 0)
                return float(np.min(drawdown_pct) * 100)
            else:
                return 0.0

        return {
            "true_max_drawdown": _max_drawdown(y_true_np),
            "pred_max_drawdown": _max_drawdown(y_pred_np),
        }

    @staticmethod
    def calculate_volatility(
        y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Calculate volatility metrics for both true and predicted values

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            dict: Volatility metrics for true and predicted values
        """
        y_true_np, y_pred_np = MetricUtils._validate_inputs(y_true, y_pred)

        def _calculate_vol(values: np.ndarray) -> float:
            if len(values) < 2:
                return 0.0
            returns = np.diff(values) / values[:-1]
            return float(np.std(returns))

        return {
            "true_volatility": _calculate_vol(y_true_np),
            "pred_volatility": _calculate_vol(y_pred_np),
        }

    @staticmethod
    def calculate_all_metrics(
        y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Calculate all available metrics

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            dict: All calculated metrics
        """
        try:
            metrics = {}

            # Basic metrics
            metrics["mse"] = MetricUtils.calculate_mse(y_true, y_pred)
            metrics["rmse"] = MetricUtils.calculate_rmse(y_true, y_pred)
            metrics["mae"] = MetricUtils.calculate_mae(y_true, y_pred)
            metrics["mape"] = MetricUtils.calculate_mape(y_true, y_pred)
            metrics["smape"] = MetricUtils.calculate_smape(y_true, y_pred)
            metrics["r2"] = MetricUtils.calculate_r2(y_true, y_pred)

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
            MetricUtils._logger.error(f"Error calculating metrics: {e}")
            return {"error": str(e)}
