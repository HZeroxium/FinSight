# interfaces/model_interface.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path


class ITimeSeriesModel(ABC):
    """Interface for time series models with clear separation between training and inference"""

    @abstractmethod
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        feature_engineering: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model

        Args:
            train_data: Training dataset
            val_data: Validation dataset
            feature_engineering: Feature engineering strategy
            **kwargs: Additional training parameters

        Returns:
            Training results and metrics
        """
        pass

    @abstractmethod
    def forecast(self, recent_data: pd.DataFrame, n_steps: int = 1) -> Dict[str, Any]:
        """
        Make true forecasting predictions using only historical data

        This method should:
        1. Use pre-fitted scalers from training
        2. Take only the last context_length window
        3. Generate n_steps future predictions
        4. NOT use any future information

        Args:
            recent_data: Recent historical data (at least context_length rows)
            n_steps: Number of future steps to predict

        Returns:
            Forecasting results with predictions and metadata
        """
        pass

    @abstractmethod
    def backtest(self, test_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Perform backtesting evaluation on historical data

        This method should:
        1. Create sliding windows over test_data
        2. Use pre-fitted scalers from training
        3. Generate predictions for each window
        4. Calculate comprehensive metrics

        Args:
            test_data: Historical test dataset
            **kwargs: Backtesting parameters

        Returns:
            Backtesting results with metrics and analysis
        """
        pass

    @abstractmethod
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate model performance using backtesting

        Args:
            test_data: Test dataset

        Returns:
            Evaluation metrics and analysis
        """
        pass

    @abstractmethod
    def save_model(self, path: Path) -> None:
        """
        Save complete model state including weights, scalers, and config

        Args:
            path: Path to save the model
        """
        pass

    @abstractmethod
    def load_model(self, path: Path) -> None:
        """
        Load complete model state including weights, scalers, and config

        Args:
            path: Path to load the model from
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and configuration

        Returns:
            Model metadata and configuration
        """
        pass
