# features.py

from typing import Dict, List, Union
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from datasets import Dataset
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from logger import LoggerFactory, LogLevel, LoggerType

logger = LoggerFactory.get_logger(
    __name__,
    level=LogLevel.INFO,
    logger_type=LoggerType.STANDARD,
)


class FeatureConfig(BaseModel):
    """
    Configuration for computing technical & time features.
    """

    use_log_return: bool = True
    use_rsi: bool = True
    rsi_period: int = Field(14, description="Period for RSI calculation")
    use_macd: bool = True
    macd_fast: int = Field(12, description="Fast EMA window for MACD")
    macd_slow: int = Field(26, description="Slow EMA window for MACD")
    macd_signal: int = Field(9, description="Signal EMA window for MACD")


class FeatureEngineer:
    """
    Adds features: log-returns, RSI, MACD, time embeddings (hour, dow).
    """

    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg

    def _compute_batch(
        self, batch: Dict[str, List[Union[str, float, int]]]
    ) -> Dict[str, List[Union[str, float, int]]]:
        """
        Compute technical indicators and time features for a batch of data.

        Args:
            batch: Dictionary containing lists of values for each column

        Returns:
            Dictionary with computed features
        """
        # Convert LazyBatch to proper dict if needed
        if hasattr(batch, "keys") and hasattr(batch, "__getitem__"):
            batch_dict = {key: batch[key] for key in batch.keys()}
        else:
            batch_dict = dict(batch)

        # Create DataFrame from the converted batch
        df = pd.DataFrame(batch_dict)

        # Convert timestamp to datetime if it's not already
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Ensure we have the close price for calculations
        if "close" not in df.columns:
            raise ValueError("'close' column is required for feature engineering")

        # log-return
        if self.cfg.use_log_return:
            df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # RSI
        if self.cfg.use_rsi:
            rsi = RSIIndicator(df["close"], window=self.cfg.rsi_period).rsi()
            df["rsi"] = rsi

        # MACD
        if self.cfg.use_macd:
            macd = MACD(
                df["close"],
                window_slow=self.cfg.macd_slow,
                window_fast=self.cfg.macd_fast,
                window_sign=self.cfg.macd_signal,
            )
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()

        # time embeddings
        if "timestamp" in df.columns:
            df["hour"] = df["timestamp"].dt.hour
            df["dayofweek"] = df["timestamp"].dt.dayofweek

        # drop rows with NaNs from technical indicators only, not from original data columns
        # Don't drop based on original columns that might have NaN (like exchange)
        technical_cols = []
        if self.cfg.use_log_return:
            technical_cols.append("log_return")
        if self.cfg.use_rsi:
            technical_cols.append("rsi")
        if self.cfg.use_macd:
            technical_cols.extend(["macd", "macd_signal"])

        initial_length = len(df)
        if technical_cols:
            # Only drop rows where technical indicators have NaN
            df = df.dropna(subset=technical_cols).reset_index(drop=True)
            if initial_length != len(df):
                logger.info(
                    f"Dropped {initial_length - len(df)} rows with NaN values from technical indicators"
                )

        # Return only the relevant columns for time series modeling
        feature_cols = ["close", "timestamp"]
        if self.cfg.use_log_return:
            feature_cols.append("log_return")
        if self.cfg.use_rsi:
            feature_cols.append("rsi")
        if self.cfg.use_macd:
            feature_cols.extend(["macd", "macd_signal"])
        if "timestamp" in df.columns:
            feature_cols.extend(["hour", "dayofweek"])

        # Only return columns that exist in the dataframe
        final_cols = [col for col in feature_cols if col in df.columns]

        # Return the processed data
        return {k: df[k].tolist() for k in final_cols}

    def transform(self, ds: Dataset) -> Dataset:
        """
        Apply feature engineering to HuggingFace Dataset.

        Args:
            ds: HuggingFace Dataset with OHLCV data

        Returns:
            Dataset with computed technical indicators and time features
        """
        logger.info(f"Columns in dataset before feature engineering: {ds.column_names}")

        # Convert entire dataset to pandas for proper time series processing
        df = ds.to_pandas()

        # Sort by timestamp to ensure proper time series order
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Apply feature engineering to the entire time series
        processed_batch = self._compute_batch(
            {col: df[col].tolist() for col in df.columns}
        )

        # Convert back to HuggingFace dataset
        from datasets import Dataset

        return Dataset.from_dict(processed_batch)
