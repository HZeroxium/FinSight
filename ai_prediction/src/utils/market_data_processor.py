# utils/market_data_processor.py

"""
Market data processing utilities for converting raw exchange data
into standardized formats suitable for AI/ML model training.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from ..common.logger import LoggerFactory, LoggerType, LogLevel


class MarketDataProcessor:
    """Utility class for processing and standardizing market data"""

    def __init__(self, logger_name: str = "market_processor"):
        """
        Initialize MarketDataProcessor

        Args:
            logger_name: Name for the logger instance
        """
        self.logger = LoggerFactory.get_logger(
            name=logger_name,
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            use_colors=True,
        )

    def standardize_ohlcv(self, raw_ohlcv: List[List], symbol: str) -> pd.DataFrame:
        """
        Standardize OHLCV data to common format

        Args:
            raw_ohlcv: Raw OHLCV data [[timestamp, open, high, low, close, volume], ...]
            symbol: Trading symbol

        Returns:
            Standardized DataFrame with OHLCV data
        """
        try:
            if not raw_ohlcv:
                return pd.DataFrame()

            df = pd.DataFrame(
                raw_ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

            # Convert timestamp to datetime
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["symbol"] = symbol

            # Ensure numeric types
            numeric_cols = ["open", "high", "low", "close", "volume"]
            df[numeric_cols] = df[numeric_cols].astype(float)

            # Validate OHLCV data integrity
            df = self._validate_ohlcv_data(df)

            # Calculate basic price features
            df = self._add_price_features(df)

            # Add technical indicators if enough data
            if len(df) >= 20:  # Minimum required for most indicators
                df = self._add_technical_indicators(df)

            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)

            self.logger.info(f"Standardized {len(df)} OHLCV records for {symbol}")
            return df

        except Exception as e:
            self.logger.error(f"Failed to standardize OHLCV data for {symbol}: {e}")
            raise

    def _validate_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data"""
        # Remove rows with invalid data
        initial_len = len(df)

        # Remove rows where high < low or close/open are outside high/low range
        df = df[
            (df["high"] >= df["low"])
            & (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
            & (df["volume"] >= 0)
        ]

        # Remove rows with zero or negative prices
        df = df[
            (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)
        ]

        cleaned_len = len(df)
        if cleaned_len < initial_len:
            self.logger.warning(
                f"Removed {initial_len - cleaned_len} invalid OHLCV records"
            )

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features"""
        # Price changes
        df["price_change"] = df["close"].pct_change()
        df["volume_change"] = df["volume"].pct_change()

        # Price ratios
        df["high_low_ratio"] = df["high"] / df["low"]
        df["close_open_ratio"] = df["close"] / df["open"]

        # Typical price and weighted close
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["weighted_close"] = (df["high"] + df["low"] + 2 * df["close"]) / 4

        # Price range and body size
        df["price_range"] = df["high"] - df["low"]
        df["body_size"] = abs(df["close"] - df["open"])
        df["upper_shadow"] = df["high"] - np.maximum(df["open"], df["close"])
        df["lower_shadow"] = np.minimum(df["open"], df["close"]) - df["low"]

        # Relative position in range
        df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
        df["close_position"] = df["close_position"].fillna(0.5)  # Handle zero range

        # Volume features
        df["volume_price"] = df["volume"] * df["typical_price"]

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame"""
        try:
            # Import technical analysis library
            import ta

            # Moving averages
            df["sma_5"] = ta.trend.sma_indicator(df["close"], window=5)
            df["sma_10"] = ta.trend.sma_indicator(df["close"], window=10)
            df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
            df["sma_50"] = ta.trend.sma_indicator(df["close"], window=min(50, len(df)))

            df["ema_5"] = ta.trend.ema_indicator(df["close"], window=5)
            df["ema_10"] = ta.trend.ema_indicator(df["close"], window=10)
            df["ema_12"] = ta.trend.ema_indicator(df["close"], window=12)
            df["ema_26"] = ta.trend.ema_indicator(df["close"], window=min(26, len(df)))

            # MACD
            if len(df) >= 26:
                df["macd"] = ta.trend.macd_diff(df["close"])
                df["macd_signal"] = ta.trend.macd_signal(df["close"])
                df["macd_histogram"] = ta.trend.macd_diff(
                    df["close"]
                ) - ta.trend.macd_signal(df["close"])

            # RSI
            df["rsi"] = ta.momentum.rsi(df["close"], window=14)
            df["rsi_6"] = ta.momentum.rsi(df["close"], window=6)

            # Bollinger Bands
            df["bb_upper"] = ta.volatility.bollinger_hband(df["close"], window=20)
            df["bb_middle"] = ta.volatility.bollinger_mavg(df["close"], window=20)
            df["bb_lower"] = ta.volatility.bollinger_lband(df["close"], window=20)
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            df["bb_position"] = (df["close"] - df["bb_lower"]) / (
                df["bb_upper"] - df["bb_lower"]
            )

            # Stochastic Oscillator
            df["stoch_k"] = ta.momentum.stoch(df["high"], df["low"], df["close"])
            df["stoch_d"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"])

            # Williams %R
            df["williams_r"] = ta.momentum.williams_r(
                df["high"], df["low"], df["close"]
            )

            # Average True Range (ATR)
            df["atr"] = ta.volatility.average_true_range(
                df["high"], df["low"], df["close"]
            )

            # Commodity Channel Index (CCI)
            df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"])

            # Volume indicators
            df["volume_sma"] = ta.trend.sma_indicator(df["volume"], window=20)
            df["volume_ratio"] = df["volume"] / df["volume_sma"]

            # On-Balance Volume
            df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])

            # Accumulation/Distribution Line
            df["ad_line"] = ta.volume.acc_dist_index(
                df["high"], df["low"], df["close"], df["volume"]
            )

            # Money Flow Index
            df["mfi"] = ta.volume.money_flow_index(
                df["high"], df["low"], df["close"], df["volume"]
            )

            # Parabolic SAR
            df["sar"] = ta.trend.psar_up(df["high"], df["low"], df["close"])

            # Keltner Channels
            df["kc_upper"] = ta.volatility.keltner_channel_hband(
                df["high"], df["low"], df["close"]
            )
            df["kc_middle"] = ta.volatility.keltner_channel_mband(
                df["high"], df["low"], df["close"]
            )
            df["kc_lower"] = ta.volatility.keltner_channel_lband(
                df["high"], df["low"], df["close"]
            )

        except ImportError:
            self.logger.warning(
                "Technical analysis library (ta) not available. Skipping technical indicators."
            )
        except Exception as e:
            self.logger.warning(f"Error calculating technical indicators: {e}")

        return df

    def standardize_trades(self, raw_trades: List[Dict], symbol: str) -> pd.DataFrame:
        """
        Standardize trade data to common format

        Args:
            raw_trades: Raw trade data
            symbol: Trading symbol

        Returns:
            Standardized DataFrame with trade data
        """
        try:
            if not raw_trades:
                return pd.DataFrame()

            df = pd.DataFrame(raw_trades)

            # Standardize column names
            column_mapping = {
                "timestamp": "timestamp",
                "datetime": "datetime",
                "price": "price",
                "amount": "amount",
                "qty": "amount",  # Binance uses 'qty'
                "side": "side",
                "id": "trade_id",
                "orderId": "order_id",
                "time": "timestamp",
            }

            # Rename columns if they exist
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df = df.rename(columns={old_col: new_col})

            # Convert timestamp to datetime if needed
            if "timestamp" in df.columns and "datetime" not in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            elif "datetime" in df.columns and df["datetime"].dtype == "object":
                df["datetime"] = pd.to_datetime(df["datetime"])

            df["symbol"] = symbol

            # Ensure numeric types
            numeric_columns = ["price", "amount"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Calculate trade value
            if "price" in df.columns and "amount" in df.columns:
                df["value"] = df["price"] * df["amount"]

            # Add trade features
            df = self._add_trade_features(df)

            # Sort by timestamp
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp").reset_index(drop=True)
            elif "datetime" in df.columns:
                df = df.sort_values("datetime").reset_index(drop=True)

            self.logger.info(f"Standardized {len(df)} trade records for {symbol}")
            return df

        except Exception as e:
            self.logger.error(f"Failed to standardize trade data for {symbol}: {e}")
            raise

    def _add_trade_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to trade data"""
        if len(df) == 0:
            return df

        # Price momentum and changes
        if "price" in df.columns:
            df["price_change"] = df["price"].diff()
            df["price_pct_change"] = df["price"].pct_change()

        # Volume features
        if "amount" in df.columns:
            df["amount_rolling_mean"] = (
                df["amount"].rolling(window=10, min_periods=1).mean()
            )
            df["amount_vs_mean"] = df["amount"] / df["amount_rolling_mean"]

        # Trade intensity
        if "datetime" in df.columns:
            df["time_since_last"] = df["datetime"].diff().dt.total_seconds()

        # Side encoding
        if "side" in df.columns:
            df["is_buy"] = (df["side"] == "buy").astype(int)
            df["is_sell"] = (df["side"] == "sell").astype(int)

        return df

    def standardize_orderbook(
        self, raw_orderbook: Dict, symbol: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Standardize orderbook data to common format

        Args:
            raw_orderbook: Raw orderbook data with 'bids' and 'asks'
            symbol: Trading symbol

        Returns:
            Dictionary with 'bids' and 'asks' DataFrames
        """
        try:
            result = {}

            for side in ["bids", "asks"]:
                if side in raw_orderbook and raw_orderbook[side]:
                    # Check if orderbook data is in list format
                    orderbook_data = raw_orderbook[side]

                    # Handle different orderbook formats
                    if isinstance(orderbook_data, list) and len(orderbook_data) > 0:
                        # Check the structure of the first item
                        first_item = orderbook_data[0]

                        if isinstance(first_item, list):
                            # Handle different list lengths (some exchanges include timestamp)
                            if len(first_item) >= 2:
                                # Take only price and amount (first 2 columns)
                                clean_data = [
                                    [item[0], item[1]] for item in orderbook_data
                                ]
                                df = pd.DataFrame(
                                    clean_data, columns=["price", "amount"]
                                )
                            else:
                                continue  # Skip invalid data
                        else:
                            # Handle dict format
                            df = pd.DataFrame(orderbook_data)
                            if "price" not in df.columns or "amount" not in df.columns:
                                # Try common alternative column names
                                column_mapping = {
                                    "0": "price",
                                    "1": "amount",
                                    "Price": "price",
                                    "Amount": "amount",
                                    "size": "amount",
                                    "qty": "amount",
                                }
                                df = df.rename(columns=column_mapping)

                                # If still missing required columns, skip
                                if (
                                    "price" not in df.columns
                                    or "amount" not in df.columns
                                ):
                                    continue
                    else:
                        continue

                    # Ensure numeric types
                    df["price"] = pd.to_numeric(df["price"], errors="coerce")
                    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

                    # Remove invalid entries
                    df = df.dropna()
                    df = df[(df["price"] > 0) & (df["amount"] > 0)]

                    if len(df) > 0:
                        # Calculate derived features
                        df["value"] = df["price"] * df["amount"]
                        df["symbol"] = symbol
                        df["side"] = side[:-1]  # 'bid' or 'ask'
                        df["timestamp"] = pd.Timestamp.now(tz=timezone.utc)

                        # Add order book features
                        df = self._add_orderbook_features(df, side)

                        result[side] = df

            # Add spread and depth analysis if both sides exist
            if "bids" in result and "asks" in result:
                spread_analysis = self._calculate_spread_analysis(
                    result["bids"], result["asks"]
                )
                result["spread_analysis"] = spread_analysis

            total_levels = sum(
                len(df) for df in result.values() if isinstance(df, pd.DataFrame)
            )
            self.logger.info(
                f"Standardized orderbook with {total_levels} levels for {symbol}"
            )
            return result

        except Exception as e:
            self.logger.error(f"Failed to standardize orderbook data for {symbol}: {e}")
            raise

    def _add_orderbook_features(self, df: pd.DataFrame, side: str) -> pd.DataFrame:
        """Add derived features to orderbook data"""
        # Distance from best price
        if side == "bids":
            best_price = df["price"].max()
            df["distance_from_best"] = best_price - df["price"]
        else:  # asks
            best_price = df["price"].min()
            df["distance_from_best"] = df["price"] - best_price

        # Cumulative amounts
        if side == "bids":
            df = df.sort_values("price", ascending=False)
        else:
            df = df.sort_values("price", ascending=True)

        df["cumulative_amount"] = df["amount"].cumsum()
        df["cumulative_value"] = df["value"].cumsum()

        # Level index (0 is best price)
        df["level"] = range(len(df))

        return df

    def _calculate_spread_analysis(
        self, bids_df: pd.DataFrame, asks_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate spread and market depth analysis"""
        if len(bids_df) == 0 or len(asks_df) == 0:
            return {}

        best_bid = bids_df["price"].max()
        best_ask = asks_df["price"].min()

        analysis = {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": best_ask - best_bid,
            "spread_percentage": ((best_ask - best_bid) / best_bid) * 100,
            "mid_price": (best_bid + best_ask) / 2,
        }

        # Depth analysis at different levels
        depth_levels = [0.1, 0.5, 1.0, 2.0, 5.0]  # Percentage levels

        for level_pct in depth_levels:
            # Calculate price levels
            bid_level = best_bid * (1 - level_pct / 100)
            ask_level = best_ask * (1 + level_pct / 100)

            # Calculate depth
            bid_depth = bids_df[bids_df["price"] >= bid_level]["amount"].sum()
            ask_depth = asks_df[asks_df["price"] <= ask_level]["amount"].sum()

            analysis[f"bid_depth_{level_pct}pct"] = bid_depth
            analysis[f"ask_depth_{level_pct}pct"] = ask_depth
            analysis[f"total_depth_{level_pct}pct"] = bid_depth + ask_depth

        return analysis

    def standardize_ticker(self, raw_ticker: Dict, symbol: str) -> Dict[str, Any]:
        """
        Standardize ticker data to common format

        Args:
            raw_ticker: Raw ticker data
            symbol: Trading symbol

        Returns:
            Standardized ticker dictionary
        """
        try:
            # Handle different ticker formats from different exchanges
            ticker_mappings = {
                # CCXT format
                "last": ["last", "close", "lastPrice"],
                "bid": ["bid", "bidPrice"],
                "ask": ["ask", "askPrice"],
                "high": ["high", "highPrice"],
                "low": ["low", "lowPrice"],
                "open": ["open", "openPrice"],
                "close": ["close", "lastPrice"],
                "volume": ["baseVolume", "volume"],
                "quoteVolume": ["quoteVolume", "quoteVol"],
                "change": ["change", "priceChange"],
                "percentage": ["percentage", "priceChangePercent"],
                "timestamp": ["timestamp", "closeTime"],
            }

            standardized = {"symbol": symbol}

            # Map fields using fallback logic
            for std_field, possible_fields in ticker_mappings.items():
                value = None
                for field in possible_fields:
                    if field in raw_ticker and raw_ticker[field] is not None:
                        value = raw_ticker[field]
                        break

                # Convert to appropriate type
                if value is not None:
                    if std_field in ["timestamp"]:
                        standardized[std_field] = value
                    else:
                        try:
                            standardized[std_field] = (
                                float(value) if value != "" else None
                            )
                        except (ValueError, TypeError):
                            standardized[std_field] = None
                else:
                    standardized[std_field] = None

            # Calculate additional metrics
            self._add_ticker_metrics(standardized)

            # Add metadata
            standardized["processed_timestamp"] = datetime.now(timezone.utc).isoformat()
            standardized["data_source"] = raw_ticker.get("exchange_id", "unknown")

            self.logger.debug(f"Standardized ticker data for {symbol}")
            return standardized

        except Exception as e:
            self.logger.error(f"Failed to standardize ticker data for {symbol}: {e}")
            raise

    def _add_ticker_metrics(self, ticker: Dict[str, Any]) -> None:
        """Add calculated metrics to ticker data"""
        # Spread calculations
        if ticker.get("bid") and ticker.get("ask"):
            ticker["spread"] = ticker["ask"] - ticker["bid"]
            ticker["mid_price"] = (ticker["ask"] + ticker["bid"]) / 2
            ticker["spread_percentage"] = (ticker["spread"] / ticker["mid_price"]) * 100

        # Price position calculations
        if all(ticker.get(k) for k in ["high", "low", "last"]):
            ticker["price_position"] = (
                (ticker["last"] - ticker["low"]) / (ticker["high"] - ticker["low"])
            ) * 100

        # Volume ratio
        if ticker.get("volume") and ticker.get("quoteVolume"):
            # Average price based on volume
            ticker["avg_price"] = (
                ticker["quoteVolume"] / ticker["volume"]
                if ticker["volume"] > 0
                else None
            )

    def aggregate_ohlcv(self, df: pd.DataFrame, timeframe: str = "1H") -> pd.DataFrame:
        """
        Aggregate OHLCV data to different timeframes

        Args:
            df: DataFrame with OHLCV data
            timeframe: Target timeframe (e.g., '1H', '4H', '1D')

        Returns:
            Aggregated DataFrame
        """
        try:
            if "datetime" not in df.columns:
                raise ValueError("DataFrame must have 'datetime' column")

            if len(df) == 0:
                return df

            df_copy = df.copy()
            df_copy = df_copy.set_index("datetime")

            # Define aggregation rules
            agg_rules = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "timestamp": "first",
                "symbol": "first",
            }

            # Add aggregation for derived columns if they exist
            derived_columns = {
                "volume_price": "sum",
                "typical_price": "mean",
                "weighted_close": "mean",
                "price_range": "mean",
                "body_size": "mean",
            }

            for col, agg_func in derived_columns.items():
                if col in df_copy.columns:
                    agg_rules[col] = agg_func

            # Only aggregate existing columns
            agg_rules = {k: v for k, v in agg_rules.items() if k in df_copy.columns}

            aggregated = df_copy.resample(timeframe).agg(agg_rules)
            aggregated = aggregated.dropna().reset_index()

            # Recalculate percentage-based features
            if len(aggregated) > 1:
                aggregated["price_change"] = aggregated["close"].pct_change()
                aggregated["volume_change"] = aggregated["volume"].pct_change()

            self.logger.info(
                f"Aggregated {len(df)} records to {len(aggregated)} {timeframe} candles"
            )
            return aggregated

        except Exception as e:
            self.logger.error(f"Failed to aggregate OHLCV data: {e}")
            raise

    def create_ml_features(
        self, df: pd.DataFrame, lookback_periods: List[int] = None
    ) -> pd.DataFrame:
        """
        Create machine learning features from OHLCV data

        Args:
            df: DataFrame with OHLCV data
            lookback_periods: List of periods for rolling features

        Returns:
            DataFrame with ML features
        """
        if lookback_periods is None:
            lookback_periods = [5, 10, 20, 50]

        try:
            ml_df = df.copy()

            # Rolling statistics for different periods
            for period in lookback_periods:
                if len(df) >= period:
                    # Price rolling features
                    ml_df[f"close_sma_{period}"] = df["close"].rolling(period).mean()
                    ml_df[f"close_std_{period}"] = df["close"].rolling(period).std()
                    ml_df[f"close_min_{period}"] = df["close"].rolling(period).min()
                    ml_df[f"close_max_{period}"] = df["close"].rolling(period).max()

                    # Volume rolling features
                    ml_df[f"volume_sma_{period}"] = df["volume"].rolling(period).mean()
                    ml_df[f"volume_std_{period}"] = df["volume"].rolling(period).std()

                    # Return features
                    ml_df[f"return_{period}"] = df["close"].pct_change(period)
                    ml_df[f"volatility_{period}"] = (
                        df["close"].pct_change().rolling(period).std()
                    )

            # Lag features
            for lag in [1, 2, 3, 5]:
                ml_df[f"close_lag_{lag}"] = df["close"].shift(lag)
                ml_df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
                if "rsi" in df.columns:
                    ml_df[f"rsi_lag_{lag}"] = df["rsi"].shift(lag)

            # Forward-looking targets (for supervised learning)
            for horizon in [1, 3, 5, 10]:
                ml_df[f"return_forward_{horizon}"] = df["close"].pct_change(-horizon)
                ml_df[f"high_forward_{horizon}"] = df["high"].shift(-horizon)
                ml_df[f"low_forward_{horizon}"] = df["low"].shift(-horizon)

            self.logger.info(f"Created ML features with {len(ml_df.columns)} columns")
            return ml_df

        except Exception as e:
            self.logger.error(f"Failed to create ML features: {e}")
            raise

    def detect_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regimes (trending, ranging, volatile)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with regime indicators
        """
        try:
            regime_df = df.copy()

            if len(df) < 20:
                return regime_df

            # Trend indicators
            if "sma_20" in df.columns and "sma_50" in df.columns:
                regime_df["trend_direction"] = np.where(
                    df["sma_20"] > df["sma_50"], 1, -1
                )

            # Volatility regime
            if "close" in df.columns:
                returns = df["close"].pct_change()
                regime_df["volatility_20"] = returns.rolling(20).std()
                regime_df["volatility_regime"] = pd.cut(
                    regime_df["volatility_20"],
                    bins=3,
                    labels=["low_vol", "medium_vol", "high_vol"],
                )

            # Range-bound vs trending
            if "atr" in df.columns and "close" in df.columns:
                regime_df["atr_normalized"] = df["atr"] / df["close"]
                regime_df["market_regime"] = np.where(
                    regime_df["atr_normalized"] > regime_df["atr_normalized"].median(),
                    "trending",
                    "ranging",
                )

            return regime_df

        except Exception as e:
            self.logger.error(f"Failed to detect market regime: {e}")
            return df

    def validate_data_quality(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """
        Validate data quality and return quality metrics

        Args:
            df: DataFrame to validate
            data_type: Type of data (ohlcv, trades, etc.)

        Returns:
            Dictionary with quality metrics
        """
        try:
            quality_metrics = {
                "total_records": len(df),
                "missing_values": df.isnull().sum().to_dict(),
                "duplicate_records": df.duplicated().sum(),
                "data_type": data_type,
                "quality_score": 0.0,
            }

            if data_type == "ohlcv" and len(df) > 0:
                # OHLCV specific validations
                quality_metrics.update(
                    {
                        "invalid_ohlc": (
                            (df["high"] < df["low"])
                            | (df["close"] > df["high"])
                            | (df["close"] < df["low"])
                            | (df["open"] > df["high"])
                            | (df["open"] < df["low"])
                        ).sum(),
                        "zero_volume": (df["volume"] == 0).sum(),
                        "negative_prices": (
                            (df["open"] <= 0)
                            | (df["high"] <= 0)
                            | (df["low"] <= 0)
                            | (df["close"] <= 0)
                        ).sum(),
                    }
                )

            # Calculate quality score
            total_issues = (
                sum(quality_metrics["missing_values"].values())
                + quality_metrics["duplicate_records"]
                + quality_metrics.get("invalid_ohlc", 0)
                + quality_metrics.get("zero_volume", 0)
                + quality_metrics.get("negative_prices", 0)
            )

            if quality_metrics["total_records"] > 0:
                quality_metrics["quality_score"] = max(
                    0, 1 - (total_issues / quality_metrics["total_records"])
                )

            return quality_metrics

        except Exception as e:
            self.logger.error(f"Failed to validate data quality: {e}")
            return {"error": str(e)}
