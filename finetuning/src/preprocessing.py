# preprocessing.py
from typing import Dict, List
from pydantic import BaseModel, Field
from datasets import Dataset
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common"))
from logger import LoggerFactory, LogLevel, LoggerType

logger = LoggerFactory.get_logger(
    __name__,
    level=LogLevel.INFO,
    logger_type=LoggerType.STANDARD,
)


class PreprocessorConfig(BaseModel):
    """
    Configuration for sliding-window creation.
    """

    context_length: int = Field(..., description="Number of past timesteps")
    prediction_length: int = Field(
        ..., description="Number of future timesteps to predict"
    )
    stride: int = Field(1, description="Step size between windows")


class Preprocessor:
    """
    Creates sliding windows of past/future series for model inputs.
    """

    def __init__(self, cfg: PreprocessorConfig):
        self.cfg = cfg

    def _make_windows(
        self, batch: Dict[str, List[float]]
    ) -> Dict[str, List[List[float]]]:
        """
        Create sliding windows from time series data.

        Args:
            batch: Dictionary containing time series data

        Returns:
            Dictionary with past_values and future_values windows
        """
        c, p, s = (
            self.cfg.context_length,
            self.cfg.prediction_length,
            self.cfg.stride,
        )

        # Handle both single series and batch processing
        if isinstance(batch["close"][0], list):
            # Multiple series in batch - process each separately
            all_pasts: List[List[float]] = []
            all_futures: List[List[float]] = []

            for series_idx in range(len(batch["close"])):
                closes = batch["close"][series_idx]
                n = len(closes)

                for start in range(0, n - c - p + 1, s):
                    all_pasts.append(closes[start : start + c])
                    all_futures.append(closes[start + c : start + c + p])

            return {"past_values": all_pasts, "future_values": all_futures}
        else:
            # Single series
            closes = batch["close"]
            n = len(closes)
            pasts: List[List[float]] = []
            futures: List[List[float]] = []

            for start in range(0, n - c - p + 1, s):
                pasts.append(closes[start : start + c])
                futures.append(closes[start + c : start + c + p])

            return {"past_values": pasts, "future_values": futures}

    def transform(self, ds: Dataset) -> Dataset:
        """
        Convert full-series dataset into windowed dataset for training.

        Args:
            ds: Dataset with time series features

        Returns:
            Dataset with sliding windows
        """
        logger.info(f"Transforming dataset with {len(ds)} examples to sliding windows")
        logger.info(
            f"Window config: context_length={self.cfg.context_length}, prediction_length={self.cfg.prediction_length}, stride={self.cfg.stride}"
        )

        total = len(ds)
        windowed_ds = ds.map(
            self._make_windows,
            batched=True,
            batch_size=total,
            remove_columns=ds.column_names,
        )

        logger.info(f"Created {len(windowed_ds)} windows from dataset")
        if len(windowed_ds) > 0:
            sample_window = windowed_ds[0]
            logger.info(
                f"Sample window - past_values length: {len(sample_window['past_values'])}, future_values length: {len(sample_window['future_values'])}"
            )

        return windowed_ds
