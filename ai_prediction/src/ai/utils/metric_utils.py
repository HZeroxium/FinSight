import numpy as np
import torch
from typing import Dict, Tuple, Union, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

from common.logger.logger_factory import LoggerFactory


class MetricUtils:
    """Utility class for calculating various metrics for financial predictions"""

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

        if y_true_np.shape != y_pred_np.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true_np.shape} vs y_pred {y_pred_np.shape}"
            )

        if np.isnan(y_true_np).any() or np.isnan(y_pred_np).any():
            MetricUtils._logger.warning("NaN values detected in inputs")

        if np.isinf(y_true_np).any() or np.isinf(y_pred_np).any():
            MetricUtils._logger.warning("Infinite values detected in inputs")

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
            peak = np.maximum.accumulate(values)
            drawdown = (values - peak) / peak
            return float(np.min(drawdown) * 100)

        return {
            "true_max_drawdown": _max_drawdown(y_true_np),
            "pred_max_drawdown": _max_drawdown(y_pred_np),
        }

    @staticmethod
    def calculate_volatility(
        y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Calculate volatility (standard deviation of returns) for both series

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            dict: Volatility for true and predicted values
        """
        y_true_np, y_pred_np = MetricUtils._validate_inputs(y_true, y_pred)

        def _volatility(values: np.ndarray) -> float:
            if len(values) < 2:
                return 0.0
            returns = np.diff(values) / values[:-1]
            return float(np.std(returns) * 100)

        return {
            "true_volatility": _volatility(y_true_np),
            "pred_volatility": _volatility(y_pred_np),
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
            dict: Dictionary containing all metrics
        """
        try:
            metrics = {
                "mse": MetricUtils.calculate_mse(y_true, y_pred),
                "rmse": MetricUtils.calculate_rmse(y_true, y_pred),
                "mae": MetricUtils.calculate_mae(y_true, y_pred),
                "mape": MetricUtils.calculate_mape(y_true, y_pred),
                "smape": MetricUtils.calculate_smape(y_true, y_pred),
                "r2": MetricUtils.calculate_r2(y_true, y_pred),
                "directional_accuracy": MetricUtils.calculate_directional_accuracy(
                    y_true, y_pred
                ),
            }

            # Add financial metrics
            drawdown_metrics = MetricUtils.calculate_max_drawdown(y_true, y_pred)
            volatility_metrics = MetricUtils.calculate_volatility(y_true, y_pred)

            metrics.update(drawdown_metrics)
            metrics.update(volatility_metrics)

            return metrics

        except Exception as e:
            MetricUtils._logger.error(f"Error calculating metrics: {str(e)}")
            raise
