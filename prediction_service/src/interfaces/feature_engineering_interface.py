# interfaces/feature_engineering_interface.py

from abc import ABC, abstractmethod
from typing import List, Tuple
import pandas as pd
import numpy as np


class IFeatureEngineering(ABC):
    """Interface for feature engineering strategies"""

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit feature engineering on training data

        Args:
            data: Training dataset
        """
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted feature engineering

        Args:
            data: Dataset to transform

        Returns:
            Transformed dataset
        """
        pass

    @abstractmethod
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform data

        Args:
            data: Dataset to fit and transform

        Returns:
            Transformed dataset
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get names of output features

        Returns:
            List of feature names
        """
        pass

    @abstractmethod
    def create_sequences(
        self,
        data: pd.DataFrame,
        context_length: int,
        prediction_length: int,
        target_column: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences and targets for time series modeling

        Args:
            data: Input dataset
            context_length: Length of input sequences
            prediction_length: Length of prediction sequences
            target_column: Target column name

        Returns:
            Tuple of (input_sequences, targets)
        """
        pass
