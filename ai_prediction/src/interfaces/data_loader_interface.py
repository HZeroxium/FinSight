# interfaces/data_loader_interface.py

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import pandas as pd
from pathlib import Path
from ..schemas.enums import TimeFrame


class IDataLoader(ABC):
    """Interface for data loading strategies"""

    @abstractmethod
    async def load_data(
        self, symbol: str, timeframe: TimeFrame, data_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Load data for given symbol and timeframe

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            data_path: Optional path to data file

        Returns:
            Loaded dataset
        """
        pass

    @abstractmethod
    async def check_data_exists(self, symbol: str, timeframe: TimeFrame) -> bool:
        """
        Check if data exists for given symbol and timeframe

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            True if data exists, False otherwise
        """
        pass

    @abstractmethod
    def split_data(
        self, data: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets

        Args:
            data: Input dataset
            train_ratio: Ratio for training data
            val_ratio: Ratio for validation data

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        pass
