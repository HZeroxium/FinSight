# utils/metrics_utils.py

"""
Comprehensive metrics utilities for time series model evaluation
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from common.logger.logger_factory import LoggerFactory
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    @staticmethod
    def calculate_tolerance_accuracy(
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        tolerance_pct: float = 5.0,
    ) -> float:
        """
        Calculate accuracy within tolerance percentage

        Args:
            y_true: True values
            y_pred: Predicted values
            tolerance_pct: Tolerance percentage (e.g., 5.0 for 5%)

        Returns:
            float: Accuracy percentage within tolerance
        """
        y_true_np, y_pred_np = MetricUtils._validate_inputs(y_true, y_pred)

        # Calculate percentage error
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_error = np.abs((y_pred_np - y_true_np) / y_true_np) * 100
            # Handle division by zero
            pct_error = np.where(
                y_true_np == 0, np.abs(y_pred_np - y_true_np), pct_error
            )

        # Count predictions within tolerance
        within_tolerance = pct_error <= tolerance_pct
        accuracy = np.mean(within_tolerance) * 100

        return float(accuracy)

    @staticmethod
    def calculate_directional_accuracy(
        y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calculate directional accuracy (sign prediction accuracy)

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float: Directional accuracy percentage
        """
        y_true_np, y_pred_np = MetricUtils._validate_inputs(y_true, y_pred)

        if len(y_true_np) < 2:
            return 0.0

        # Calculate direction changes
        true_directions = np.diff(y_true_np) > 0
        pred_directions = np.diff(y_pred_np) > 0

        # Calculate accuracy
        directional_accuracy = np.mean(true_directions == pred_directions) * 100

        return float(directional_accuracy)

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
    def calculate_tolerance_accuracy(
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        tolerance: float = 0.01,
    ) -> float:
        """
        Calculate accuracy based on tolerance threshold

        Args:
            y_true: True values
            y_pred: Predicted values
            tolerance: Tolerance as fraction (e.g., 0.01 for 1%, 0.05 for 5%)

        Returns:
            float: Accuracy percentage (0-100)
        """
        y_true_np, y_pred_np = MetricUtils._validate_inputs(y_true, y_pred)

        # Calculate absolute percentage error
        abs_pct_error = np.abs(
            (y_true_np - y_pred_np) / np.maximum(np.abs(y_true_np), 1e-8)
        )

        # Calculate accuracy based on tolerance
        within_tolerance = abs_pct_error <= tolerance
        accuracy = np.mean(within_tolerance) * 100

        return float(accuracy)

    @staticmethod
    def calculate_multiple_tolerance_accuracies(
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        tolerances: List[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate accuracy for multiple tolerance levels

        Args:
            y_true: True values
            y_pred: Predicted values
            tolerances: List of tolerance levels (default: [0.01, 0.05, 0.10])

        Returns:
            Dict with tolerance accuracies
        """
        if tolerances is None:
            tolerances = [0.01, 0.05, 0.10]  # 1%, 5%, 10%

        accuracies = {}
        for tolerance in tolerances:
            pct = int(tolerance * 100)
            key = f"accuracy_{pct}pct_tolerance"
            accuracies[key] = MetricUtils.calculate_tolerance_accuracy(
                y_true, y_pred, tolerance
            )

        return accuracies

    @staticmethod
    def calculate_hit_rate(
        y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calculate hit rate (percentage of predictions with correct direction)

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float: Hit rate percentage (0-100)
        """
        return MetricUtils.calculate_directional_accuracy(y_true, y_pred)

    @staticmethod
    def calculate_sharpe_ratio(
        returns: Union[np.ndarray, torch.Tensor], risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate Sharpe ratio for returns

        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (default 0.0)

        Returns:
            float: Sharpe ratio
        """
        returns_np = MetricUtils._to_numpy(returns)

        excess_returns = returns_np - risk_free_rate

        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        return float(sharpe)

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

            # Tolerance-based metrics
            tolerance_metrics = MetricUtils.calculate_multiple_tolerance_accuracies(
                y_true, y_pred
            )
            metrics.update(tolerance_metrics)

            # Hit rate
            metrics["hit_rate"] = MetricUtils.calculate_hit_rate(y_true, y_pred)

            return metrics

        except Exception as e:
            MetricUtils._logger.error(f"Error calculating metrics: {e}")
            return {"error": str(e)}
