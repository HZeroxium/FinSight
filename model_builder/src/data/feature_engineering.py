# data/feature_engineering.py

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from ..common.logger.logger_factory import LoggerFactory
from ..core.config import Config
from ..utils import ValidationUtils, FileUtils


class FeatureEngineering:
    """Feature engineering for financial time series data with comprehensive technical indicators"""

    def __init__(self, config: Config):
        """
        Initialize feature engineering

        Args:
            config: Configuration object containing feature engineering parameters
        """
        self.config = config
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)
        self.scalers: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.original_columns: List[str] = []

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators to the dataset

        Args:
            data: Input dataframe with OHLCV data

        Returns:
            pd.DataFrame: Dataframe with technical indicators added

        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ["Open", "High", "Low", "Close"]
        ValidationUtils.validate_dataframe(data, required_columns=required_columns)

        if not self.config.data.add_technical_indicators:
            return data

        df = data.copy()
        indicators = self.config.data.technical_indicators

        self.logger.info("Adding technical indicators...")

        try:
            # Simple Moving Averages
            if "sma" in indicators:
                for period in indicators["sma"]:
                    df[f"SMA_{period}"] = (
                        df["Close"].rolling(window=period, min_periods=1).mean()
                    )

            # Exponential Moving Averages
            if "ema" in indicators:
                for period in indicators["ema"]:
                    df[f"EMA_{period}"] = df["Close"].ewm(span=period).mean()

            # RSI (Relative Strength Index)
            if "rsi" in indicators:
                for period in indicators["rsi"]:
                    df[f"RSI_{period}"] = self._calculate_rsi(df["Close"], period)

            # Bollinger Bands
            if "bollinger" in indicators:
                for period in indicators["bollinger"]:
                    sma = df["Close"].rolling(window=period, min_periods=1).mean()
                    std = df["Close"].rolling(window=period, min_periods=1).std()
                    df[f"BB_Upper_{period}"] = sma + (2 * std)
                    df[f"BB_Lower_{period}"] = sma - (2 * std)
                    df[f"BB_Width_{period}"] = (
                        df[f"BB_Upper_{period}"] - df[f"BB_Lower_{period}"]
                    )
                    # Avoid division by zero
                    bb_width_safe = df[f"BB_Width_{period}"].replace(0, np.nan)
                    df[f"BB_Position_{period}"] = (
                        df["Close"] - df[f"BB_Lower_{period}"]
                    ) / bb_width_safe

            # MACD
            if indicators.get("macd", False):
                ema_12 = df["Close"].ewm(span=12).mean()
                ema_26 = df["Close"].ewm(span=26).mean()
                df["MACD"] = ema_12 - ema_26
                df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
                df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

            # Stochastic Oscillator
            if indicators.get("stochastic", False):
                df = self._add_stochastic_oscillator(df)

            # Williams %R
            if indicators.get("williams_r", False):
                df = self._add_williams_r(df)

            # Average True Range (ATR)
            if indicators.get("atr", False):
                df = self._add_atr(df)

            # Price-based features
            df["High_Low_Ratio"] = df["High"] / df["Low"]
            df["Close_Open_Ratio"] = df["Close"] / df["Open"]
            df["High_Close_Ratio"] = df["High"] / df["Close"]
            df["Low_Close_Ratio"] = df["Low"] / df["Close"]

            # Volume-based features
            if "Volume" in df.columns:
                df["Volume_SMA_5"] = (
                    df["Volume"].rolling(window=5, min_periods=1).mean()
                )
                df["Volume_SMA_20"] = (
                    df["Volume"].rolling(window=20, min_periods=1).mean()
                )
                volume_sma_safe = df["Volume_SMA_5"].replace(0, np.nan)
                df["Volume_Ratio"] = df["Volume"] / volume_sma_safe
                df["Volume_Price_Trend"] = df["Volume"] * df["Close"].pct_change()

            # Return features
            df["Returns"] = df["Close"].pct_change()
            df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
            df["Returns_2"] = df["Returns"].shift(1)
            df["Returns_3"] = df["Returns"].shift(2)

            # Volatility features
            df["Volatility_5"] = df["Returns"].rolling(window=5, min_periods=1).std()
            df["Volatility_20"] = df["Returns"].rolling(window=20, min_periods=1).std()
            df["Volatility_60"] = df["Returns"].rolling(window=60, min_periods=1).std()

            # Price momentum
            df["Price_Momentum_5"] = df["Close"] / df["Close"].shift(5) - 1
            df["Price_Momentum_10"] = df["Close"] / df["Close"].shift(10) - 1
            df["Price_Momentum_20"] = df["Close"] / df["Close"].shift(20) - 1

            self.logger.info(f"Added technical indicators. New shape: {df.shape}")

        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            raise

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index

        Args:
            prices: Price series
            period: RSI period

        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()

        # Avoid division by zero
        loss_safe = loss.replace(0, np.nan)
        rs = gain / loss_safe
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI value

    def _add_stochastic_oscillator(
        self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> pd.DataFrame:
        """
        Add Stochastic Oscillator indicators

        Args:
            df: Input dataframe
            k_period: Period for %K calculation
            d_period: Period for %D calculation

        Returns:
            pd.DataFrame: Dataframe with stochastic indicators
        """
        low_min = df["Low"].rolling(window=k_period, min_periods=1).min()
        high_max = df["High"].rolling(window=k_period, min_periods=1).max()

        # Avoid division by zero
        range_safe = (high_max - low_min).replace(0, np.nan)
        df["Stoch_K"] = 100 * (df["Close"] - low_min) / range_safe
        df["Stoch_D"] = df["Stoch_K"].rolling(window=d_period, min_periods=1).mean()

        return df

    def _add_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Williams %R indicator

        Args:
            df: Input dataframe
            period: Lookback period

        Returns:
            pd.DataFrame: Dataframe with Williams %R
        """
        high_max = df["High"].rolling(window=period, min_periods=1).max()
        low_min = df["Low"].rolling(window=period, min_periods=1).min()

        # Avoid division by zero
        range_safe = (high_max - low_min).replace(0, np.nan)
        df["Williams_R"] = -100 * (high_max - df["Close"]) / range_safe

        return df

    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average True Range indicator

        Args:
            df: Input dataframe
            period: ATR period

        Returns:
            pd.DataFrame: Dataframe with ATR
        """
        high_low = df["High"] - df["Low"]
        high_close_prev = np.abs(df["High"] - df["Close"].shift(1))
        low_close_prev = np.abs(df["Low"] - df["Close"].shift(1))

        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        df["ATR"] = true_range.rolling(window=period, min_periods=1).mean()

        return df

    def add_cyclical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical time-based features

        Args:
            data: Input dataframe with date column

        Returns:
            pd.DataFrame: Dataframe with cyclical features
        """
        if self.config.data.date_column not in data.columns:
            self.logger.warning("Date column not found, skipping cyclical features")
            return data

        df = data.copy()
        date_col = self.config.data.date_column

        try:
            # Ensure datetime
            df[date_col] = pd.to_datetime(df[date_col])

            # Day of week (0=Monday, 6=Sunday)
            df["DayOfWeek"] = df[date_col].dt.dayofweek
            df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
            df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

            # Month
            df["Month"] = df[date_col].dt.month
            df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
            df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

            # Day of month
            df["DayOfMonth"] = df[date_col].dt.day
            df["DayOfMonth_sin"] = np.sin(2 * np.pi * df["DayOfMonth"] / 31)
            df["DayOfMonth_cos"] = np.cos(2 * np.pi * df["DayOfMonth"] / 31)

            # Hour (if available)
            if df[date_col].dt.hour.nunique() > 1:
                df["Hour"] = df[date_col].dt.hour
                df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
                df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

            self.logger.info("Added cyclical time features")

        except Exception as e:
            self.logger.error(f"Error adding cyclical features: {str(e)}")
            raise

        return df

    def add_lag_features(
        self, data: pd.DataFrame, columns: List[str], lags: List[int]
    ) -> pd.DataFrame:
        """
        Add lag features for specified columns

        Args:
            data: Input dataframe
            columns: List of columns to create lags for
            lags: List of lag periods

        Returns:
            pd.DataFrame: Dataframe with lag features
        """
        df = data.copy()

        for col in columns:
            if col not in df.columns:
                self.logger.warning(f"Column {col} not found, skipping lag features")
                continue

            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        self.logger.info(
            f"Added lag features for {len(columns)} columns with lags {lags}"
        )
        return df

    def add_rolling_statistics(
        self, data: pd.DataFrame, columns: List[str], windows: List[int]
    ) -> pd.DataFrame:
        """
        Add rolling statistics for specified columns

        Args:
            data: Input dataframe
            columns: List of columns to create rolling stats for
            windows: List of window sizes

        Returns:
            pd.DataFrame: Dataframe with rolling statistics
        """
        df = data.copy()

        for col in columns:
            if col not in df.columns:
                self.logger.warning(f"Column {col} not found, skipping rolling stats")
                continue

            for window in windows:
                df[f"{col}_rolling_mean_{window}"] = (
                    df[col].rolling(window=window, min_periods=1).mean()
                )
                df[f"{col}_rolling_std_{window}"] = (
                    df[col].rolling(window=window, min_periods=1).std()
                )
                df[f"{col}_rolling_min_{window}"] = (
                    df[col].rolling(window=window, min_periods=1).min()
                )
                df[f"{col}_rolling_max_{window}"] = (
                    df[col].rolling(window=window, min_periods=1).max()
                )

        self.logger.info(
            f"Added rolling statistics for {len(columns)} columns with windows {windows}"
        )
        return df

    def add_price_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add price pattern features

        Args:
            data: Input dataframe

        Returns:
            pd.DataFrame: Dataframe with price pattern features
        """
        df = data.copy()

        try:
            # Candlestick patterns
            df["Doji"] = (
                np.abs(df["Close"] - df["Open"]) / (df["High"] - df["Low"]) < 0.1
            ).astype(int)
            df["Hammer"] = (
                (df["Close"] > df["Open"])
                & ((df["Open"] - df["Low"]) > 2 * (df["Close"] - df["Open"]))
            ).astype(int)
            df["Shooting_Star"] = (
                (df["Open"] > df["Close"])
                & ((df["High"] - df["Open"]) > 2 * (df["Open"] - df["Close"]))
            ).astype(int)

            # Gap features
            df["Gap_Up"] = (df["Open"] > df["High"].shift(1)).astype(int)
            df["Gap_Down"] = (df["Open"] < df["Low"].shift(1)).astype(int)

            # Price position within the day's range
            range_safe = (df["High"] - df["Low"]).replace(0, np.nan)
            df["Close_Position"] = (df["Close"] - df["Low"]) / range_safe

            self.logger.info("Added price pattern features")

        except Exception as e:
            self.logger.error(f"Error adding price patterns: {str(e)}")
            raise

        return df

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers

        Args:
            data: Input dataframe

        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df = data.copy()
        initial_shape = df.shape

        try:
            # Handle missing values
            if self.config.data.fill_missing:
                method = self.config.data.missing_method
                if method == "forward":
                    df = df.ffill()  # Forward fill instead of fillna(method="ffill")
                elif method == "backward":
                    df = df.fillna(method="bfill")
                elif method == "interpolate":
                    numeric_columns = df.select_dtypes(include=[np.number]).columns
                    df[numeric_columns] = df[numeric_columns].interpolate(
                        method="linear"
                    )

                # Drop any remaining NaN values
                df = df.dropna()

            # Remove outliers using IQR method for better financial data handling
            # if self.config.data.remove_outliers:
            #     numeric_columns = df.select_dtypes(include=[np.number]).columns
            #     for col in numeric_columns:
            #         if col != self.config.data.date_column:
            #             Q1 = df[col].quantile(0.25)
            #             Q3 = df[col].quantile(0.75)
            #             IQR = Q3 - Q1
            #             lower_bound = Q1 - 1.5 * IQR
            #             upper_bound = Q3 + 1.5 * IQR

            #             outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            #             df = df[~outliers]

            self.logger.info(f"Data cleaning: {initial_shape} -> {df.shape}")

        except Exception as e:
            self.logger.error(f"Error cleaning data: {str(e)}")
            raise

        return df

    def scale_features(
        self, data: pd.DataFrame, fit: bool = True, scaler_type: str = "standard"
    ) -> pd.DataFrame:
        """
        Scale numerical features

        Args:
            data: Input dataframe
            fit: Whether to fit the scaler (True for training, False for inference)
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')

        Returns:
            pd.DataFrame: Scaled dataframe
        """
        if not self.config.model.scale_features:
            return data

        df = data.copy()

        try:
            # Get numerical columns to scale
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            # Exclude date column if present
            if self.config.data.date_column in numeric_columns:
                numeric_columns.remove(self.config.data.date_column)

            # Choose scaler
            scaler_map = {
                "standard": StandardScaler,
                "minmax": MinMaxScaler,
                "robust": RobustScaler,
            }

            if scaler_type not in scaler_map:
                raise ValueError(f"Unknown scaler type: {scaler_type}")

            # Scale each column
            for col in numeric_columns:
                if fit:
                    # Fit new scaler
                    scaler = scaler_map[scaler_type]()
                    df[col] = scaler.fit_transform(df[[col]])
                    self.scalers[col] = scaler
                else:
                    # Use existing scaler
                    if col in self.scalers:
                        df[col] = self.scalers[col].transform(df[[col]])
                    else:
                        self.logger.warning(f"No scaler found for column: {col}")

            if fit:
                self.logger.info(
                    f"Fitted {scaler_type} scalers for {len(numeric_columns)} columns"
                )

        except Exception as e:
            self.logger.error(f"Error scaling features: {str(e)}")
            raise

        return df

    def inverse_transform_target(
        self, scaled_values: np.ndarray, target_column: str
    ) -> np.ndarray:
        """
        Inverse transform scaled target values

        Args:
            scaled_values: Scaled values to inverse transform
            target_column: Name of target column

        Returns:
            np.ndarray: Original scale values
        """
        if target_column in self.scalers:
            return (
                self.scalers[target_column]
                .inverse_transform(scaled_values.reshape(-1, 1))
                .flatten()
            )
        else:
            self.logger.warning(f"No scaler found for target column: {target_column}")
            return scaled_values

    def process_data(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Complete data processing pipeline

        Args:
            data: Input dataframe
            fit: Whether to fit scalers (True for training, False for inference)

        Returns:
            pd.DataFrame: Processed dataframe
        """
        self.logger.info("Starting data processing pipeline...")

        try:
            # Store original columns
            if fit:
                self.original_columns = data.columns.tolist()

            # Add technical indicators
            df = self.add_technical_indicators(data)

            # Add cyclical features
            df = self.add_cyclical_features(df)

            # Add price patterns
            df = self.add_price_patterns(df)

            FileUtils.save_csv(df, "demo_results/processed_data.csv")

            # Clean data
            df = self.clean_data(df)

            FileUtils.save_csv(df, "demo_results/cleaned_processed_data.csv")

            # Scale features
            df = self.scale_features(df, fit=fit)

            # Store feature names
            if fit:
                feature_columns = [
                    col for col in df.columns if col != self.config.data.date_column
                ]
                self.feature_names = feature_columns

                self.logger.info(f"Processing complete. Final shape: {df.shape}")
                self.logger.info(f"Feature columns: {len(self.feature_names)}")

        except Exception as e:
            self.logger.error(f"Error in data processing pipeline: {str(e)}")
            raise

        return df

    def get_feature_importance_names(self) -> List[str]:
        """
        Get list of feature names for importance analysis

        Returns:
            List[str]: Feature names
        """
        return self.feature_names.copy()

    def select_features(
        self, df: pd.DataFrame, feature_importance_threshold: float = 0.01
    ) -> List[str]:
        """
        Select important features based on correlation and variance.

        Args:
            df: Input DataFrame
            feature_importance_threshold: Minimum variance threshold

        Returns:
            List of selected feature names
        """
        try:
            # Remove non-numeric columns and date columns
            numeric_df = df.select_dtypes(include=[np.number])
            if self.config.data.date_column in numeric_df.columns:
                numeric_df = numeric_df.drop(columns=[self.config.data.date_column])

            # Remove features with low variance
            variances = numeric_df.var()
            high_variance_features = variances[
                variances > feature_importance_threshold
            ].index.tolist()

            # Remove highly correlated features
            if len(high_variance_features) > 1:
                corr_matrix = numeric_df[high_variance_features].corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )

                # Find features with correlation higher than 0.95
                to_drop = [
                    column
                    for column in upper_triangle.columns
                    if any(upper_triangle[column] > 0.95)
                ]
                selected_features = [
                    col for col in high_variance_features if col not in to_drop
                ]
            else:
                selected_features = high_variance_features

            self.logger.info(
                f"Selected {len(selected_features)} features from {len(numeric_df.columns)} total"
            )

            return selected_features

        except Exception as e:
            self.logger.error(f"Error selecting features: {str(e)}")
            raise

    def create_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set by applying all feature engineering methods.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with comprehensive features
        """
        try:
            self.logger.info("Starting comprehensive feature engineering")

            # Add technical indicators
            data = self.add_technical_indicators(df)

            # Add cyclical features
            if self.config.data.date_column in data.columns:
                data = self.add_cyclical_features(data)

            # Add price patterns
            data = self.add_price_patterns(data)

            # Add lag features for key columns
            key_columns = ["Close", "Volume"] + [
                col for col in data.columns if "RSI" in col
            ]
            available_columns = [col for col in key_columns if col in data.columns]
            if available_columns:
                data = self.add_lag_features(data, available_columns, [1, 2, 3, 5])

            # Add rolling statistics for key columns
            if available_columns:
                data = self.add_rolling_statistics(data, available_columns, [5, 10, 20])

            # Remove rows with NaN values
            initial_shape = data.shape
            data = data.dropna()
            final_shape = data.shape

            self.logger.info(
                f"Feature engineering complete. Shape: {initial_shape} -> {final_shape}"
            )

            return data

        except Exception as e:
            self.logger.error(f"Error in comprehensive feature engineering: {str(e)}")
            raise

    def save_scalers(self, filepath: str) -> None:
        """
        Save fitted scalers to file

        Args:
            filepath: Path to save scalers
        """
        try:
            from ..utils import FileUtils

            FileUtils.save_object(self.scalers, filepath)
            self.logger.info(f"Scalers saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving scalers: {str(e)}")
            raise

    def load_scalers(self, filepath: str) -> None:
        """
        Load fitted scalers from file

        Args:
            filepath: Path to load scalers from
        """
        try:
            from ..utils import FileUtils

            self.scalers = FileUtils.load_object(filepath)
            self.logger.info(f"Scalers loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading scalers: {str(e)}")
            raise
