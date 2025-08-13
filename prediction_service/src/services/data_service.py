# services/data_service.py

"""
Data service for encapsulating data loading and feature engineering operations.
"""

from typing import Dict, Any, Tuple
import pandas as pd

from ..data.feature_engineering import BasicFeatureEngineering
from ..schemas.enums import TimeFrame
from ..schemas.model_schemas import ModelConfig
from ..core.config import get_settings
from ..utils.dependencies import get_data_loader
from common.logger.logger_factory import LoggerFactory


class DataService:
    """Service for data loading and feature engineering operations"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger("DataService")
        self.settings = get_settings()
        self.data_loader = (
            get_data_loader()
        )  # Use dependency injection for cloud-first loading
        self._feature_engineering_cache: Dict[str, BasicFeatureEngineering] = {}

    async def load_and_prepare_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        config: ModelConfig,
    ) -> Dict[str, Any]:
        """
        Load and prepare data for model training/prediction

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            config: Model configuration

        Returns:
            Dictionary containing prepared data and metadata
        """
        try:
            self.logger.info(f"Loading and preparing data for {symbol} {timeframe}")

            # Load raw data
            raw_data = await self.data_loader.load_data(symbol, timeframe)

            if raw_data.empty:
                raise ValueError(f"No data found for {symbol} {timeframe}")

            # Create feature engineering instance
            feature_engineering = self._get_feature_engineering(config)

            # Process data
            processed_data = feature_engineering.fit_transform(raw_data)

            # Split data
            train_data, val_data, test_data = self._split_data(
                processed_data, config.train_ratio, config.val_ratio
            )

            # Get feature information
            feature_names = feature_engineering.get_feature_names()

            # Validate target column
            if config.target_column not in processed_data.columns:
                available_columns = list(processed_data.columns)
                raise ValueError(
                    f"Target column '{config.target_column}' not found. "
                    f"Available columns: {available_columns}"
                )

            result = {
                "raw_data": raw_data,
                "processed_data": processed_data,
                "train_data": train_data,
                "val_data": val_data,
                "test_data": test_data,
                "feature_engineering": feature_engineering,
                "feature_names": feature_names,
                "target_column": config.target_column,
                "data_info": {
                    "total_samples": len(processed_data),
                    "train_samples": len(train_data),
                    "val_samples": len(val_data),
                    "test_samples": len(test_data),
                    "feature_count": len(feature_names),
                    "date_range": {
                        "start": (
                            str(processed_data.index.min())
                            if hasattr(processed_data.index, "min")
                            else "N/A"
                        ),
                        "end": (
                            str(processed_data.index.max())
                            if hasattr(processed_data.index, "max")
                            else "N/A"
                        ),
                    },
                },
            }

            self.logger.info(
                f"Data preparation completed: {len(processed_data)} samples, "
                f"{len(feature_names)} features"
            )

            return result

        except Exception as e:
            self.logger.error(f"Failed to load and prepare data: {e}")
            raise

    async def load_data_for_prediction(
        self,
        symbol: str,
        timeframe: TimeFrame,
        fitted_feature_engineering: BasicFeatureEngineering,
    ) -> pd.DataFrame:
        """
        Load and prepare data for prediction using fitted feature engineering

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            fitted_feature_engineering: Pre-fitted feature engineering instance

        Returns:
            Processed data ready for prediction
        """
        try:
            self.logger.info(f"Loading data for prediction: {symbol} {timeframe}")

            # Load raw data
            raw_data = await self.data_loader.load_data(symbol, timeframe)

            if raw_data.empty:
                raise ValueError(f"No data found for {symbol} {timeframe}")

            # Transform using fitted feature engineering
            processed_data = fitted_feature_engineering.transform(raw_data)

            self.logger.info(
                f"Data loaded for prediction: {len(processed_data)} samples"
            )

            return processed_data

        except Exception as e:
            self.logger.error(f"Failed to load data for prediction: {e}")
            raise

    async def check_data_availability(
        self, symbol: str, timeframe: TimeFrame
    ) -> Dict[str, Any]:
        """
        Check data availability for a symbol and timeframe

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            Dictionary with availability information
        """
        try:
            exists = await self.data_loader.check_data_exists(symbol, timeframe)

            if exists:
                # Load basic info
                data = await self.data_loader.load_data(symbol, timeframe)
                info = {
                    "exists": True,
                    "sample_count": len(data),
                    "columns": list(data.columns),
                    "date_range": {
                        "start": (
                            str(data.index.min())
                            if hasattr(data.index, "min")
                            else "N/A"
                        ),
                        "end": (
                            str(data.index.max())
                            if hasattr(data.index, "max")
                            else "N/A"
                        ),
                    },
                }
            else:
                info = {"exists": False}

            return info

        except Exception as e:
            self.logger.error(f"Failed to check data availability: {e}")
            return {"exists": False, "error": str(e)}

    def _get_feature_engineering(self, config: ModelConfig) -> BasicFeatureEngineering:
        """
        Get or create feature engineering instance with caching

        Args:
            config: Model configuration

        Returns:
            Feature engineering instance
        """
        # Create cache key based on feature engineering config
        cache_key = (
            f"{hash(str(config.feature_columns))}_"
            f"{config.use_technical_indicators}_"
            f"{config.add_datetime_features}_"
            f"{config.normalize_features}"
        )

        if cache_key not in self._feature_engineering_cache:
            self._feature_engineering_cache[cache_key] = BasicFeatureEngineering(
                feature_columns=config.feature_columns,
                add_technical_indicators=config.use_technical_indicators,
                add_datetime_features=config.add_datetime_features,
                normalize_features=config.normalize_features,
            )

        return self._feature_engineering_cache[cache_key]

    def _split_data(
        self, data: pd.DataFrame, train_ratio: float, val_ratio: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets

        Args:
            data: Input data
            train_ratio: Training data ratio
            val_ratio: Validation data ratio

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        return self.data_loader.split_data(data, train_ratio, val_ratio)

    def clear_cache(self):
        """Clear feature engineering cache"""
        self._feature_engineering_cache.clear()
        self.logger.info("Feature engineering cache cleared")
