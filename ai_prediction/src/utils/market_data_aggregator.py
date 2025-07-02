# utils/data_aggregator.py

"""
Data aggregator for consolidating collected market data into unified datasets.
Combines OHLCV, trades, orderbook, and ticker data for ML training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..common.logger import LoggerFactory, LoggerType, LogLevel
from .market_data_storage import MarketDataStorage


class MarketDataAggregator:
    """Aggregates market data into unified datasets for ML training"""

    def __init__(self, base_dir: str = "data", logger_name: str = "data_aggregator"):
        """
        Initialize DataAggregator

        Args:
            base_dir: Base directory containing collected data
            logger_name: Name for the logger instance
        """
        self.base_dir = Path(base_dir)
        self.logger = LoggerFactory.get_logger(
            name=logger_name,
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            use_colors=True,
        )

        # Initialize dataset storage
        self.dataset_storage = MarketDataStorage(
            base_dir=str(self.base_dir / "datasets")
        )

    def aggregate_symbol_data(
        self,
        exchange: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        intervals: List[str],
    ) -> pd.DataFrame:
        """
        Aggregate all data for a specific symbol into a unified dataset

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            intervals: List of timeframes

        Returns:
            Unified DataFrame with all market data
        """
        try:
            # Clean symbol for file naming
            clean_symbol = symbol.replace("/", "_")
            exchange_dir = self.base_dir / exchange

            # Start with OHLCV data as the base
            base_df = self._load_ohlcv_data(
                exchange_dir, clean_symbol, intervals, start_date, end_date
            )

            if base_df.empty:
                self.logger.warning(f"No OHLCV data found for {symbol}")
                return pd.DataFrame()

            # Print number of rows and columns in the base DataFrame
            self.logger.info(
                f"Loaded OHLCV data for {symbol}: {len(base_df)} rows, {len(base_df.columns)} columns"
            )

            # Add orderbook features
            base_df = self._add_orderbook_features(base_df, exchange_dir, clean_symbol)

            self.logger.info(
                f"Added orderbook features for {symbol}: {len(base_df)} rows, {len(base_df.columns)} columns"
            )

            # Add trade features
            base_df = self._add_trade_features(base_df, exchange_dir, clean_symbol)

            self.logger.info(
                f"Added trade features for {symbol}: {len(base_df)} rows, {len(base_df.columns)} columns"
            )

            # Add ticker features
            base_df = self._add_ticker_features(base_df, exchange_dir, clean_symbol)

            self.logger.info(
                f"Added ticker features for {symbol}: {len(base_df)} rows, {len(base_df.columns)} columns"
            )

            # Add metadata
            base_df["exchange"] = exchange
            base_df["symbol"] = symbol
            base_df["dataset_created"] = datetime.now().isoformat()

            self.logger.info(
                f"Aggregated dataset for {symbol}: {len(base_df)} rows, {len(base_df.columns)} features"
            )
            return base_df

        except Exception as e:
            self.logger.error(f"Failed to aggregate data for {symbol}: {e}")
            return pd.DataFrame()

    def _load_ohlcv_data(
        self,
        exchange_dir: Path,
        clean_symbol: str,
        intervals: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Load and combine OHLCV data from different timeframes"""
        all_ohlcv = []

        for interval in intervals:
            # Find processed OHLCV files for this symbol and interval
            pattern = f"*{clean_symbol}*{interval}*processed*.csv"
            ohlcv_files = list((exchange_dir / "ohlcv").glob(pattern))

            # Sort files by modification time to get the most recent one
            if ohlcv_files:
                ohlcv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            for file_path in ohlcv_files:
                try:
                    df = pd.read_csv(file_path)

                    # Convert datetime column
                    if "datetime" in df.columns:
                        df["datetime"] = pd.to_datetime(df["datetime"])
                    elif "timestamp" in df.columns:
                        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

                    # Don't filter by date range here since the collected data
                    # already represents the requested period. The issue was
                    # double-filtering which reduced the dataset size.
                    # The data collection process already ensures we get data
                    # for the requested period, so we should use all collected data.

                    if not df.empty:
                        # Add interval identifier
                        df["timeframe"] = interval

                        # Rename columns to include timeframe
                        rename_cols = {}
                        for col in ["open", "high", "low", "close", "volume"]:
                            if col in df.columns:
                                rename_cols[col] = f"{col}_{interval}"

                        df = df.rename(columns=rename_cols)
                        all_ohlcv.append(df)

                        # Log the actual date range of loaded data
                        if "datetime" in df.columns:
                            actual_start = df["datetime"].min()
                            actual_end = df["datetime"].max()
                            self.logger.info(
                                f"Loaded OHLCV data for {clean_symbol} ({interval}): "
                                f"{len(df)} rows from {actual_start} to {actual_end}"
                            )

                        # Only process the most recent file for each interval
                        break

                except Exception as e:
                    self.logger.warning(f"Error loading OHLCV file {file_path}: {e}")

        if not all_ohlcv:
            return pd.DataFrame()

        # Merge all timeframes on datetime
        base_df = all_ohlcv[0].set_index("datetime")

        for df in all_ohlcv[1:]:
            df_indexed = df.set_index("datetime")
            # Only merge overlapping columns if they don't already exist
            cols_to_merge = [
                col for col in df_indexed.columns if col not in base_df.columns
            ]
            if cols_to_merge:
                base_df = base_df.join(df_indexed[cols_to_merge], how="outer")

        return base_df.reset_index()

    def _add_orderbook_features(
        self, base_df: pd.DataFrame, exchange_dir: Path, clean_symbol: str
    ) -> pd.DataFrame:
        """Add orderbook-derived features to the base dataset"""
        try:
            # Find orderbook files
            pattern = f"*{clean_symbol}*orderbook*processed*.csv"
            orderbook_files = list((exchange_dir / "orderbook").glob(pattern))

            if not orderbook_files:
                return base_df

            # Load the most recent orderbook data
            latest_file = max(orderbook_files, key=lambda p: p.stat().st_mtime)

            try:
                ob_df = pd.read_csv(latest_file)

                if "side" in ob_df.columns:
                    # Calculate spread metrics
                    bids = ob_df[ob_df["side"] == "bid"]
                    asks = ob_df[ob_df["side"] == "ask"]

                    if not bids.empty and not asks.empty:
                        spread_metrics = {
                            "best_bid": (
                                bids["price"].max()
                                if "price" in bids.columns
                                else np.nan
                            ),
                            "best_ask": (
                                asks["price"].min()
                                if "price" in asks.columns
                                else np.nan
                            ),
                            "bid_volume_total": (
                                bids["amount"].sum()
                                if "amount" in bids.columns
                                else np.nan
                            ),
                            "ask_volume_total": (
                                asks["amount"].sum()
                                if "amount" in asks.columns
                                else np.nan
                            ),
                        }

                        # Calculate spread
                        if not np.isnan(spread_metrics["best_bid"]) and not np.isnan(
                            spread_metrics["best_ask"]
                        ):
                            spread_metrics["spread"] = (
                                spread_metrics["best_ask"] - spread_metrics["best_bid"]
                            )
                            spread_metrics["spread_pct"] = (
                                spread_metrics["spread"] / spread_metrics["best_bid"]
                            ) * 100

                        # Add to base DataFrame
                        for key, value in spread_metrics.items():
                            base_df[f"orderbook_{key}"] = value

            except Exception as e:
                self.logger.warning(
                    f"Error processing orderbook file {latest_file}: {e}"
                )

        except Exception as e:
            self.logger.warning(f"Error adding orderbook features: {e}")

        return base_df

    def _add_trade_features(
        self, base_df: pd.DataFrame, exchange_dir: Path, clean_symbol: str
    ) -> pd.DataFrame:
        """Add trade-derived features to the base dataset"""
        try:
            # Find trade files
            pattern = f"*{clean_symbol}*trades*processed*.csv"
            trade_files = list((exchange_dir / "trades").glob(pattern))

            if not trade_files:
                return base_df

            # Load the most recent trade data
            latest_file = max(trade_files, key=lambda p: p.stat().st_mtime)

            try:
                trades_df = pd.read_csv(latest_file)

                if not trades_df.empty:
                    # Calculate trade metrics
                    trade_metrics = {}

                    if "price" in trades_df.columns:
                        trade_metrics["trades_price_mean"] = trades_df["price"].mean()
                        trade_metrics["trades_price_std"] = trades_df["price"].std()

                    # Handle both 'amount' and 'qty' columns
                    amount_col = "qty" if "qty" in trades_df.columns else "amount"
                    if amount_col in trades_df.columns:
                        trade_metrics["trades_volume_mean"] = trades_df[
                            amount_col
                        ].mean()
                        trade_metrics["trades_volume_total"] = trades_df[
                            amount_col
                        ].sum()
                        trade_metrics["trades_count"] = len(trades_df)

                    if "side" in trades_df.columns:
                        # Count buy/sell trades
                        buy_trades = trades_df[
                            trades_df["side"].isin(["buy", "BUY", "b", True])
                        ]
                        sell_trades = trades_df[
                            trades_df["side"].isin(["sell", "SELL", "s", False])
                        ]

                        trade_metrics["trades_buy_count"] = len(buy_trades)
                        trade_metrics["trades_sell_count"] = len(sell_trades)
                        trade_metrics["trades_buy_ratio"] = (
                            len(buy_trades) / len(trades_df)
                            if len(trades_df) > 0
                            else 0
                        )

                    # Add to base DataFrame
                    for key, value in trade_metrics.items():
                        base_df[f"{key}"] = value

            except Exception as e:
                self.logger.warning(f"Error processing trade file {latest_file}: {e}")

        except Exception as e:
            self.logger.warning(f"Error adding trade features: {e}")

        return base_df

    def _add_ticker_features(
        self, base_df: pd.DataFrame, exchange_dir: Path, clean_symbol: str
    ) -> pd.DataFrame:
        """Add ticker-derived features to the base dataset"""
        try:
            # Find ticker files
            pattern = f"*{clean_symbol}*ticker*processed*.json"
            ticker_files = list((exchange_dir / "tickers").glob(pattern))

            if not ticker_files:
                return base_df

            # Load the most recent ticker data
            latest_file = max(ticker_files, key=lambda p: p.stat().st_mtime)

            try:
                import json

                with open(latest_file, "r") as f:
                    ticker_data = json.load(f)

                # Extract key ticker metrics
                ticker_metrics = {}
                ticker_fields = [
                    "bid",
                    "ask",
                    "last",
                    "high",
                    "low",
                    "volume",
                    "change",
                    "percentage",
                ]

                for field in ticker_fields:
                    if field in ticker_data and ticker_data[field] is not None:
                        try:
                            ticker_metrics[f"ticker_{field}"] = float(
                                ticker_data[field]
                            )
                        except (ValueError, TypeError):
                            pass

                # Add calculated metrics
                if "ticker_bid" in ticker_metrics and "ticker_ask" in ticker_metrics:
                    ticker_metrics["ticker_spread"] = (
                        ticker_metrics["ticker_ask"] - ticker_metrics["ticker_bid"]
                    )
                    ticker_metrics["ticker_mid_price"] = (
                        ticker_metrics["ticker_ask"] + ticker_metrics["ticker_bid"]
                    ) / 2

                # Add to base DataFrame
                for key, value in ticker_metrics.items():
                    base_df[key] = value

            except Exception as e:
                self.logger.warning(f"Error processing ticker file {latest_file}: {e}")

        except Exception as e:
            self.logger.warning(f"Error adding ticker features: {e}")

        return base_df

    def create_dataset(
        self,
        exchange: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        intervals: List[str],
    ) -> Optional[Path]:
        """
        Create and save a unified dataset for a symbol

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            intervals: List of timeframes

        Returns:
            Path to saved dataset file
        """
        try:
            # Aggregate data
            dataset_df = self.aggregate_symbol_data(
                exchange, symbol, start_date, end_date, intervals
            )

            if dataset_df.empty:
                self.logger.warning(f"No data to create dataset for {symbol}")
                return None

            # Create filename
            clean_symbol = symbol.replace("/", "_")
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            intervals_str = "_".join(intervals)

            filename = f"{exchange}_{clean_symbol}_{start_str}_{end_str}_{intervals_str}_dataset"

            # Save dataset
            file_path = self.dataset_storage.save_csv(
                dataset_df, filename, timestamp_suffix=False, index=False
            )

            # Save metadata
            metadata = {
                "exchange": exchange,
                "symbol": symbol,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "intervals": intervals,
                "records_count": len(dataset_df),
                "features_count": len(dataset_df.columns),
                "created_at": datetime.now().isoformat(),
                "file_path": str(file_path),
            }

            self.dataset_storage.save_json(
                metadata, f"{filename}_metadata", timestamp_suffix=False
            )

            self.logger.info(f"Created dataset for {symbol}: {file_path}")
            return file_path

        except Exception as e:
            self.logger.error(f"Failed to create dataset for {symbol}: {e}")
            return None

    def create_multiple_datasets(
        self, collection_summary: Dict[str, Any]
    ) -> Dict[str, Path]:
        """
        Create datasets for multiple symbols from collection summary

        Args:
            collection_summary: Summary from data collection

        Returns:
            Dictionary mapping symbols to dataset file paths
        """
        datasets = {}

        try:
            exchange = collection_summary.get("exchange", "unknown")
            intervals = collection_summary.get("timeframes", [])

            # Parse dates
            start_date = datetime.fromisoformat(
                collection_summary.get("start_date", datetime.now().isoformat())
            )
            end_date = datetime.fromisoformat(
                collection_summary.get("end_date", datetime.now().isoformat())
            )

            # Create dataset for each symbol
            for symbol in collection_summary.get("symbols", []):
                try:
                    # Check if we have data for this symbol
                    symbol_data = collection_summary.get("collected_data", {}).get(
                        symbol, {}
                    )
                    if not symbol_data.get("ohlcv"):
                        self.logger.warning(
                            f"No OHLCV data found for {symbol}, skipping dataset creation"
                        )
                        continue

                    dataset_path = self.create_dataset(
                        exchange=exchange,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        intervals=intervals,
                    )

                    if dataset_path:
                        datasets[symbol] = dataset_path

                except Exception as e:
                    self.logger.error(f"Failed to create dataset for {symbol}: {e}")

            self.logger.info(f"Created {len(datasets)} datasets successfully")
            return datasets

        except Exception as e:
            self.logger.error(f"Failed to create multiple datasets: {e}")
            return {}
