# data/feature_engineering.py

from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Import technical analysis library
try:
    import ta

    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

from ..interfaces.feature_engineering_interface import IFeatureEngineering
from common.logger.logger_factory import LoggerFactory


class BasicFeatureEngineering(IFeatureEngineering):
    """Basic feature engineering implementation for financial time series"""

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        add_technical_indicators: bool = True,
        add_datetime_features: bool = False,
        normalize_features: bool = True,
    ):
        self.feature_columns = feature_columns or [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        self.add_technical_indicators = add_technical_indicators
        self.add_datetime_features = add_datetime_features
        self.normalize_features = normalize_features

        self.scalers: Dict[str, StandardScaler] = {}
        self.fitted_feature_names: List[str] = []
        self.is_fitted = False

        self.logger = LoggerFactory.get_logger("BasicFeatureEngineering")

    def fit(self, data: pd.DataFrame) -> None:
        """Fit feature engineering on training data"""

        self.logger.info("Fitting feature engineering")
        self.logger.info(
            f"Input data shape: {data.shape}, Columns: {data.columns.tolist()}"
        )

        # Create features
        transformed_data = self._create_features(data)

        self.logger.info(
            f"Transformed data shape: {transformed_data.shape}, Columns: {transformed_data.columns.tolist()}"
        )

        # Store feature names
        self.fitted_feature_names = [
            col for col in transformed_data.columns if col not in ["timestamp"]
        ]

        # Fit scalers
        if self.normalize_features:
            for feature in self.fitted_feature_names:
                if feature in transformed_data.columns:
                    self.scalers[feature] = StandardScaler()
                    self.scalers[feature].fit(transformed_data[[feature]])

        self.is_fitted = True
        self.logger.info(
            f"Feature engineering fitted with {len(self.fitted_feature_names)} features"
        )

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted feature engineering"""

        if not self.is_fitted:
            raise ValueError("Feature engineering must be fitted before transform")

        # Create features
        transformed_data = self._create_features(data)

        # Apply scaling
        if self.normalize_features:
            for feature in self.fitted_feature_names:
                if feature in transformed_data.columns and feature in self.scalers:
                    transformed_data[feature] = (
                        self.scalers[feature]
                        .transform(transformed_data[[feature]])
                        .flatten()
                    )

        return transformed_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data"""

        self.fit(data)
        return self.transform(data)

    def get_feature_names(self) -> List[str]:
        """Get names of output features"""

        if not self.is_fitted:
            raise ValueError("Feature engineering must be fitted first")

        return self.fitted_feature_names.copy()

    def create_sequences(
        self,
        data: pd.DataFrame,
        context_length: int,
        prediction_length: int,
        target_column: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets for time series modeling"""

        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        feature_data = data[self.fitted_feature_names].values
        target_data = data[target_column].values

        sequences = []
        targets = []

        for i in range(len(data) - context_length - prediction_length + 1):
            # Input sequence
            seq = feature_data[i : i + context_length]
            sequences.append(seq)

            # Target sequence
            if prediction_length == 1:
                target = target_data[i + context_length]
            else:
                target = target_data[
                    i + context_length : i + context_length + prediction_length
                ]
            targets.append(target)

        sequences = np.array(sequences)
        targets = np.array(targets)

        self.logger.info(
            f"Created {len(sequences)} sequences with shape {sequences.shape}"
        )

        return sequences, targets

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features from raw data"""

        df = data.copy()

        # Ensure we have base features
        base_features = ["open", "high", "low", "close", "volume"]
        for feature in base_features:
            if feature not in df.columns:
                raise ValueError(f"Required feature '{feature}' not found in data")

        # Add technical indicators if requested
        if self.add_technical_indicators:
            df = self._add_technical_indicators(df)

        # Add datetime features if requested
        if self.add_datetime_features and "timestamp" in df.columns:
            df = self._add_datetime_features(df)

        # Select only specified features (if provided) plus any new technical indicators
        if self.feature_columns:
            available_features = [
                col for col in self.feature_columns if col in df.columns
            ]
            # Add technical indicators to selection
            tech_indicators = [
                col
                for col in df.columns
                if any(
                    col.startswith(prefix)
                    for prefix in [
                        "sma_",
                        "ema_",
                        "rsi_",
                        "bb_",
                        "macd_",
                        "returns_",
                        "volume_",
                    ]
                )
            ]
            # Add datetime features to selection
            datetime_features = [
                col
                for col in df.columns
                if any(
                    col.startswith(prefix)
                    for prefix in [
                        "hour_",
                        "day_",
                        "month_",
                        "year_",
                        "weekday_",
                        "quarter_",
                        "dayofyear_",
                    ]
                )
                or col.endswith(("_sin", "_cos"))
                or col in ["year_norm", "month", "day", "weekday", "hour", "quarter"]
            ]
            available_features.extend(tech_indicators)
            available_features.extend(datetime_features)

            # Keep timestamp if it exists
            if "timestamp" in df.columns:
                available_features.append("timestamp")

            # Remove duplicates and filter DataFrame
            available_features = list(set(available_features))
            missing_features = [
                col for col in available_features if col not in df.columns
            ]
            if missing_features:
                self.logger.warning(
                    f"Missing features will be ignored: {missing_features}"
                )

            final_features = [col for col in available_features if col in df.columns]
            df = df[final_features]

        # Drop any rows with NaN values
        df = df.dropna().reset_index(drop=True)

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using ta library or fallback to manual calculation"""

        if TA_AVAILABLE:
            return self._add_ta_indicators(df)
        else:
            self.logger.warning("ta library not available, using manual calculations")
            return self._add_manual_indicators(df)

    def _add_ta_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using ta library"""

        try:
            # Simple Moving Averages
            for window in [5, 10, 20, 50]:
                df[f"sma_{window}"] = ta.trend.sma_indicator(df["close"], window=window)

            # Exponential Moving Averages
            for window in [5, 10, 20, 50]:
                df[f"ema_{window}"] = ta.trend.ema_indicator(df["close"], window=window)

            # RSI
            df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
            df["rsi_7"] = ta.momentum.rsi(df["close"], window=7)

            # Bollinger Bands
            bb_bands = ta.volatility.BollingerBands(
                df["close"], window=20, window_dev=2
            )
            df["bb_upper"] = bb_bands.bollinger_hband()
            df["bb_middle"] = bb_bands.bollinger_mavg()
            df["bb_lower"] = bb_bands.bollinger_lband()
            df["bb_width"] = df["bb_upper"] - df["bb_lower"]
            df["bb_pctb"] = bb_bands.bollinger_pband()  # %B indicator

            # MACD
            macd = ta.trend.MACD(df["close"])
            df["macd_line"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_histogram"] = macd.macd_diff()

            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
            df["stoch_k"] = stoch.stoch()
            df["stoch_d"] = stoch.stoch_signal()

            # Price-based features
            df["returns_1"] = df["close"].pct_change()
            df["returns_5"] = df["close"].pct_change(periods=5)
            df["returns_20"] = df["close"].pct_change(periods=20)

            # High-Low features
            df["hl_ratio"] = df["high"] / df["low"]
            df["oc_ratio"] = df["open"] / df["close"]
            df["hl_pct"] = (df["high"] - df["low"]) / df["close"]

            # Volume indicators
            df["volume_sma_20"] = ta.trend.sma_indicator(df["volume"], window=20)
            df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
            df["volume_price_trend"] = ta.volume.volume_price_trend(
                df["close"], df["volume"]
            )

            # Additional momentum indicators
            df["williams_r"] = ta.momentum.williams_r(
                df["high"], df["low"], df["close"]
            )

            # Volatility indicators
            df["atr"] = ta.volatility.average_true_range(
                df["high"], df["low"], df["close"]
            )

            self.logger.info("Added technical indicators using ta library")

        except Exception as e:
            self.logger.error(f"Error with ta library indicators: {e}")
            # Fallback to manual indicators
            df = self._add_manual_indicators(df)

        return df

    def _add_manual_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators using manual calculations (fallback)"""

        try:
            # Simple Moving Averages
            for window in [5, 10, 20, 50]:
                df[f"sma_{window}"] = df["close"].rolling(window=window).mean()

            # Exponential Moving Averages
            for span in [5, 10, 20, 50]:
                df[f"ema_{span}"] = df["close"].ewm(span=span).mean()

            # RSI (simplified version)
            for period in [7, 14]:
                delta = df["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / (loss + 1e-8)  # Avoid division by zero
                df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            df["bb_middle"] = df["close"].rolling(window=20).mean()
            bb_std = df["close"].rolling(window=20).std()
            df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
            df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
            df["bb_width"] = df["bb_upper"] - df["bb_lower"]
            df["bb_pctb"] = (df["close"] - df["bb_lower"]) / (
                df["bb_upper"] - df["bb_lower"]
            )

            # MACD (simplified)
            ema_12 = df["close"].ewm(span=12).mean()
            ema_26 = df["close"].ewm(span=26).mean()
            df["macd_line"] = ema_12 - ema_26
            df["macd_signal"] = df["macd_line"].ewm(span=9).mean()
            df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

            # Stochastic Oscillator (simplified)
            low_14 = df["low"].rolling(window=14).min()
            high_14 = df["high"].rolling(window=14).max()
            df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14 + 1e-8)
            df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

            # Price changes and returns
            df["returns_1"] = df["close"].pct_change()
            df["returns_5"] = df["close"].pct_change(periods=5)
            df["returns_20"] = df["close"].pct_change(periods=20)

            # High-Low features
            df["hl_ratio"] = df["high"] / (df["low"] + 1e-8)
            df["oc_ratio"] = df["open"] / (df["close"] + 1e-8)
            df["hl_pct"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)

            # Volume indicators
            df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
            df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1e-8)

            # Williams %R
            df["williams_r"] = (
                -100 * (high_14 - df["close"]) / (high_14 - low_14 + 1e-8)
            )

            # Average True Range (ATR)
            tr1 = df["high"] - df["low"]
            tr2 = abs(df["high"] - df["close"].shift())
            tr3 = abs(df["low"] - df["close"].shift())
            tr = pd.DataFrame([tr1, tr2, tr3]).max()
            df["atr"] = tr.rolling(window=14).mean()

            self.logger.info("Added technical indicators using manual calculations")

        except Exception as e:
            self.logger.error(f"Error in manual indicators calculation: {e}")

        return df

    def _add_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add datetime-based features from timestamp column"""

        try:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            dt = df["timestamp"]

            # Basic time features (linear)
            df["year_norm"] = (dt.dt.year - dt.dt.year.min()) / (
                dt.dt.year.max() - dt.dt.year.min() + 1e-8
            )
            df["month"] = dt.dt.month
            df["day"] = dt.dt.day
            df["weekday"] = dt.dt.weekday  # 0=Monday, 6=Sunday

            # Cyclical encoding for periodic features (more robust)
            df["month_sin"] = np.sin(2 * np.pi * (dt.dt.month - 1) / 12)  # 0-11 range
            df["month_cos"] = np.cos(2 * np.pi * (dt.dt.month - 1) / 12)

            # Day of month (handle varying month lengths)
            max_day = dt.dt.days_in_month
            df["day_sin"] = np.sin(2 * np.pi * (dt.dt.day - 1) / max_day)
            df["day_cos"] = np.cos(2 * np.pi * (dt.dt.day - 1) / max_day)

            df["weekday_sin"] = np.sin(2 * np.pi * dt.dt.weekday / 7)
            df["weekday_cos"] = np.cos(2 * np.pi * dt.dt.weekday / 7)

            # Check if we have intraday data (hours)
            unique_hours = dt.dt.hour.nunique()
            if unique_hours > 1:  # More than just one hour means intraday data
                df["hour"] = dt.dt.hour
                df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
                df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)

                # Add minute features if available
                unique_minutes = dt.dt.minute.nunique()
                if unique_minutes > 1:
                    df["minute_sin"] = np.sin(2 * np.pi * dt.dt.minute / 60)
                    df["minute_cos"] = np.cos(2 * np.pi * dt.dt.minute / 60)

            # Quarter of year
            df["quarter"] = dt.dt.quarter
            df["quarter_sin"] = np.sin(2 * np.pi * (dt.dt.quarter - 1) / 4)
            df["quarter_cos"] = np.cos(2 * np.pi * (dt.dt.quarter - 1) / 4)

            # Day of year
            df["dayofyear_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
            df["dayofyear_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)

            self.logger.info(
                f"Added datetime features: year_norm, month, day, weekday, cyclical encodings"
            )

        except Exception as e:
            self.logger.error(f"Error adding datetime features: {e}")

        return df
