"""
Data Validation Utilities

This module provides utilities for validating and processing prediction data,
ensuring data quality and compatibility with model requirements.
"""

from typing import List, Optional, Tuple, Union
import pandas as pd
from pathlib import Path

from ..schemas.enums import TimeFrame, ModelType
from ..core.config import get_settings
from common.logger.logger_factory import LoggerFactory


class DataValidationUtils:
    """Utilities for validating and processing prediction data"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger("DataValidationUtils")
        self.settings = get_settings()

    def validate_prediction_data(
        self,
        data: Union[List[float], pd.DataFrame],
        timeframe: TimeFrame,
        model_type: ModelType,
        n_steps: int,
        required_context_length: Optional[int] = None,
    ) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """
        Validate prediction data for compatibility with model requirements.

        Args:
            data: Input data (list of floats or DataFrame)
            timeframe: Data timeframe
            model_type: Model type
            n_steps: Number of prediction steps
            required_context_length: Required context length (if None, uses default)

        Returns:
            Tuple of (is_valid, error_message, processed_dataframe)
        """
        try:
            # Convert list to DataFrame if needed
            if isinstance(data, list):
                df = self._convert_list_to_dataframe(data, timeframe)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                return False, f"Unsupported data type: {type(data)}", None

            # Validate DataFrame structure
            if df.empty:
                return False, "DataFrame is empty", None

            # Check minimum data requirements
            min_required = self._get_minimum_data_requirements(
                timeframe, model_type, n_steps, required_context_length
            )

            if len(df) < min_required:
                return (
                    False,
                    f"Insufficient data points. Required: {min_required}, Provided: {len(df)}",
                    None,
                )

            # Validate data quality
            quality_check = self._check_data_quality(df)
            if not quality_check[0]:
                return False, f"Data quality issues: {quality_check[1]}", None

            # Ensure proper column structure
            processed_df = self._ensure_proper_structure(df, timeframe)

            return True, "Data validation successful", processed_df

        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return False, f"Data validation error: {str(e)}", None

    def _convert_list_to_dataframe(
        self, data: List[float], timeframe: TimeFrame
    ) -> pd.DataFrame:
        """Convert list of floats to properly structured DataFrame."""
        import pandas as pd
        from datetime import datetime, timedelta

        # Generate timestamps based on timeframe
        base_time = datetime.now()
        timestamps = []

        # Calculate time delta based on timeframe
        time_deltas = {
            TimeFrame.MINUTE_1: timedelta(minutes=1),
            TimeFrame.MINUTE_5: timedelta(minutes=5),
            TimeFrame.MINUTE_15: timedelta(minutes=15),
            TimeFrame.HOUR_1: timedelta(hours=1),
            TimeFrame.HOUR_4: timedelta(hours=4),
            TimeFrame.HOUR_12: timedelta(hours=12),
            TimeFrame.DAY_1: timedelta(days=1),
            TimeFrame.WEEK_1: timedelta(weeks=1),
        }

        delta = time_deltas.get(timeframe, timedelta(hours=1))

        # Generate timestamps going backwards from current time
        for i in range(len(data)):
            timestamp = base_time - (delta * (len(data) - i - 1))
            timestamps.append(timestamp)

        # Create DataFrame with proper structure
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "close": data,
                "open": data,  # Use close as default for open
                "high": data,  # Use close as default for high
                "low": data,  # Use close as default for low
                "volume": [1000.0] * len(data),  # Default volume
            }
        )

        df.set_index("timestamp", inplace=True)
        return df

    def _get_minimum_data_requirements(
        self,
        timeframe: TimeFrame,
        model_type: ModelType,
        n_steps: int,
        required_context_length: Optional[int] = None,
    ) -> int:
        """Calculate minimum required data points for prediction."""
        # Default context lengths for different model types
        default_context_lengths = {
            ModelType.PATCHTST: 64,
            ModelType.PATCHTSMIXER: 64,
            ModelType.PYTORCH_TRANSFORMER: 32,
            ModelType.ENHANCED_TRANSFORMER: 64,
        }

        context_length = required_context_length or default_context_lengths.get(
            model_type, 64
        )

        # Add buffer for safety and ensure we have enough for prediction
        min_required = max(context_length + n_steps + 10, 50)

        self.logger.debug(
            f"Minimum data requirements: context_length={context_length}, "
            f"n_steps={n_steps}, min_required={min_required}"
        )

        return min_required

    def _check_data_quality(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check data quality and consistency."""
        try:
            # Check for NaN values
            if df.isnull().any().any():
                return False, "Data contains NaN values"

            # Check for infinite values
            if not df.isfinite().all().all():
                return False, "Data contains infinite values"

            # Check for negative prices (if close column exists)
            if "close" in df.columns and (df["close"] < 0).any():
                return False, "Data contains negative prices"

            # Check for zero prices
            if "close" in df.columns and (df["close"] == 0).any():
                return False, "Data contains zero prices"

            # Check for reasonable price ranges
            if "close" in df.columns:
                price_range = df["close"].max() - df["close"].min()
                if price_range == 0:
                    return False, "All prices are identical (no variation)"

            return True, "Data quality check passed"

        except Exception as e:
            return False, f"Data quality check failed: {str(e)}"

    def _ensure_proper_structure(
        self, df: pd.DataFrame, timeframe: TimeFrame
    ) -> pd.DataFrame:
        """Ensure DataFrame has proper structure for prediction."""
        # Ensure we have required columns
        required_columns = ["close", "open", "high", "low", "volume"]

        for col in required_columns:
            if col not in df.columns:
                if col == "close":
                    # If no close column, use the first numeric column
                    numeric_cols = df.select_dtypes(include=["number"]).columns
                    if len(numeric_cols) > 0:
                        df[col] = df[numeric_cols[0]]
                    else:
                        df[col] = 100.0  # Default value
                else:
                    # For other columns, use close values as defaults
                    df[col] = df.get("close", 100.0)

        # Ensure proper data types
        for col in required_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort by timestamp to ensure chronological order
        if df.index.name == "timestamp":
            df.sort_index(inplace=True)

        return df

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics for the data."""
        try:
            summary = {
                "total_points": len(df),
                "date_range": {
                    "start": (
                        df.index.min().isoformat()
                        if hasattr(df.index.min(), "isoformat")
                        else str(df.index.min())
                    ),
                    "end": (
                        df.index.max().isoformat()
                        if hasattr(df.index.max(), "isoformat")
                        else str(df.index.max())
                    ),
                },
                "columns": list(df.columns),
                "data_types": df.dtypes.to_dict(),
            }

            # Add numeric column statistics
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                summary["numeric_stats"] = df[numeric_cols].describe().to_dict()

            return summary

        except Exception as e:
            self.logger.error(f"Failed to generate data summary: {e}")
            return {"error": str(e)}
