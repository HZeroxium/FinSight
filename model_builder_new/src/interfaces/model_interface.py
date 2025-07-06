# interfaces/model_interface.py

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path


class ITimeSeriesModel(ABC):
    """Interface for time series models"""

    @abstractmethod
    def train(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame, **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model

        Args:
            train_data: Training dataset
            val_data: Validation dataset
            **kwargs: Additional training parameters

        Returns:
            Training results and metrics
        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame, n_steps: int = 1) -> Dict[str, Any]:
        """
        Make predictions

        Args:
            data: Input data for prediction
            n_steps: Number of prediction steps

        Returns:
            Prediction results
        """
        pass

    @abstractmethod
    def save_model(self, path: Path) -> None:
        """
        Save model to disk

        Args:
            path: Path to save the model
        """
        pass

    @abstractmethod
    def load_model(self, path: Path) -> None:
        """
        Load model from disk

        Args:
            path: Path to load the model from
        """
        pass

    @abstractmethod
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            test_data: Test dataset

        Returns:
            Evaluation metrics
        """
        pass
