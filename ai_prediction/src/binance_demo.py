# binance_demo.py

"""
Comprehensive Binance API Demo using python-binance library.

This module demonstrates all major public endpoints available in the Binance API
for market data collection, suitable for AI/ML dataset creation.

Features:
- Market data fetching (OHLCV, tickers, trades, order book)
- Historical data collection
- Real-time data streaming
- Data standardization and storage
- Rate limiting and error handling

Requirements:
    pip install python-binance pandas numpy ta pyarrow

Author: Expert Software Architect
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

from .common.logger import LoggerFactory, LoggerType, LogLevel
from .utils import (
    BaseDataCollector,
    ExchangeUtils,
    retry_on_error,
    DataValidator,
    DataAggregator,
)


class BinanceDataCollector(BaseDataCollector):
    """Comprehensive Binance data collector for AI/ML datasets"""

    def __init__(
        self,
        testnet: bool = False,
        rate_limit: bool = True,
        logger_name: str = "binance_collector",
    ):
        """
        Initialize Binance data collector

        Args:
            testnet: Whether to use Binance testnet
            rate_limit: Whether to enable rate limiting
            enable_realtime: Whether to enable real-time data collection
            logger_name: Name for the logger instance
        """
        super().__init__(exchange_name="binance", logger_name=logger_name)

        self.testnet = testnet
        self.rate_limit = rate_limit
        self.client = None
        self.twm = None

        # Initialize data validator
        self.validator = DataValidator()

        # Initialize data aggregator
        self.aggregator = DataAggregator(base_dir="data")

        # Initialize Binance client
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Binance client"""
        try:
            # Initialize client without overriding API URL
            self.client = Client(
                api_key=None,  # No API key needed for public endpoints
                api_secret=None,
                testnet=self.testnet,
            )

            # Remove the problematic API_URL override
            # The client will use the correct endpoints automatically

            self.logger.info(f"Initialized Binance client (testnet: {self.testnet})")

        except Exception as e:
            self.logger.error(f"Failed to initialize Binance client: {e}")
            raise

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(BinanceAPIException, BinanceRequestException),
    )
    def get_server_time(self) -> Dict[str, Any]:
        """Get Binance server time"""
        try:
            server_time = self.client.get_server_time()
            server_time["datetime"] = pd.to_datetime(
                server_time["serverTime"], unit="ms"
            )

            self.logger.info(f"Server time: {server_time['datetime']}")
            return server_time
        except Exception as e:
            self.logger.error(f"Failed to get server time: {e}")
            # Try alternative approach if main method fails
            try:
                import requests

                response = requests.get(
                    "https://api.binance.com/api/v3/time", timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    data["datetime"] = pd.to_datetime(data["serverTime"], unit="ms")
                    self.logger.info(f"Server time (fallback): {data['datetime']}")
                    return data
                else:
                    raise Exception(
                        f"Fallback request failed with status {response.status_code}"
                    )
            except Exception as fallback_error:
                self.logger.error(f"Fallback method also failed: {fallback_error}")
                raise e

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(BinanceAPIException, BinanceRequestException),
    )
    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get exchange trading rules and symbol information"""
        if symbol:
            exchange_info = self.client.get_symbol_info(symbol)
        else:
            exchange_info = self.client.get_exchange_info()

        # Save exchange info
        filename = f"exchange_info_{symbol}" if symbol else "exchange_info"
        self.storage.save_json(exchange_info, filename, subfolder="markets")

        symbols_count = (
            len(exchange_info.get("symbols", [])) if "symbols" in exchange_info else 1
        )
        self.logger.info(f"Retrieved exchange info for {symbols_count} symbols")
        return exchange_info

    def get_all_symbols(self) -> List[str]:
        """Get all available trading symbols"""
        exchange_info = self.get_exchange_info()
        symbols = [
            s["symbol"] for s in exchange_info["symbols"] if s["status"] == "TRADING"
        ]

        # Save symbols list
        symbols_data = {
            "symbols": symbols,
            "count": len(symbols),
            "timestamp": datetime.now().isoformat(),
        }

        self.storage.save_json(symbols_data, "all_symbols", subfolder="symbols")
        self.logger.info(f"Retrieved {len(symbols)} active trading symbols")
        return symbols

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(BinanceAPIException, BinanceRequestException),
    )
    def fetch_ticker(self, symbol: str, validate: bool = True) -> Dict[str, Any]:
        """Fetch ticker for a specific symbol with optional validation"""
        try:
            # Normalize symbol format
            symbol = ExchangeUtils.normalize_symbol(symbol, "binance")

            ticker = self.client.get_ticker(symbol=symbol)
            ticker["exchange_id"] = "binance"
            ticker["fetch_timestamp"] = datetime.now().isoformat()

            # Validate data if requested
            if validate:
                validation_report = self.validator.validate_ticker_data(ticker, symbol)
                if not validation_report["is_valid"]:
                    self.logger.warning(
                        f"Ticker validation failed for {symbol}: {validation_report['issues']}"
                    )

            # Save raw data
            filename = ExchangeUtils.create_filename("binance", symbol, "ticker")
            self.storage.save_json(ticker, f"{filename}_raw", subfolder="tickers")

            # Standardize and save
            self.standardize_and_save_ticker(ticker, symbol)

            # Store in real-time storage if enabled
            if self.realtime_storage:
                ticker["symbol"] = symbol
                ticker["exchange"] = "binance"
                self.realtime_storage.store_ticker(ticker)

            self.logger.debug(f"Fetched ticker for {symbol}")
            return ticker

        except Exception as e:
            self.logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            raise

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(BinanceAPIException, BinanceRequestException),
    )
    def fetch_all_tickers(self) -> List[Dict[str, Any]]:
        """Fetch all tickers"""
        tickers = self.client.get_ticker()

        # Save raw data
        self.storage.save_json(tickers, "all_tickers_raw", subfolder="tickers")

        # Process and save individually
        for ticker in tickers:
            self.standardize_and_save_ticker(ticker, ticker["symbol"])

        self.logger.info(f"Fetched {len(tickers)} tickers")
        return tickers

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(BinanceAPIException, BinanceRequestException),
    )
    def fetch_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Fetch order book for a symbol"""
        try:
            orderbook = self.client.get_order_book(symbol=symbol, limit=limit)

            # Add metadata
            orderbook["symbol"] = symbol
            orderbook["timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()
            orderbook["limit"] = limit

            # Save raw data
            self.storage.save_json(
                orderbook, f"orderbook_{symbol}_raw", subfolder="orderbook"
            )

            # Standardize and save - fix the orderbook processing
            try:
                self.standardize_and_save_orderbook(orderbook, symbol)
            except Exception as processing_error:
                self.logger.warning(
                    f"Orderbook processing error for {symbol}: {processing_error}"
                )
                # Continue execution even if processing fails

            self.logger.debug(f"Fetched orderbook for {symbol} with {limit} levels")
            return orderbook

        except Exception as e:
            self.logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
            raise

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(BinanceAPIException, BinanceRequestException),
    )
    def fetch_trades(
        self, symbol: str, limit: int = 500, validate: bool = True
    ) -> List[Dict[str, Any]]:
        """Fetch recent trades for a symbol with validation"""
        try:
            # Normalize symbol format
            symbol = ExchangeUtils.normalize_symbol(symbol, "binance")

            trades = self.client.get_recent_trades(symbol=symbol, limit=limit)

            # Add metadata
            for trade in trades:
                trade["exchange_id"] = "binance"
                trade["symbol"] = symbol

            # Validate data if requested
            if validate and trades:
                try:
                    trades_df = pd.DataFrame(trades)
                    validation_report = self.validator.validate_trade_data(
                        trades_df, symbol
                    )
                    if not validation_report["is_valid"]:
                        self.logger.warning(
                            f"Trades validation failed for {symbol}: {validation_report['issues']}"
                        )
                except Exception as validation_error:
                    self.logger.warning(
                        f"Trade validation error for {symbol}: {validation_error}"
                    )

            # Save raw data
            filename = ExchangeUtils.create_filename("binance", symbol, "trades")
            self.storage.save_json(trades, f"{filename}_raw", subfolder="trades")

            # Standardize and save
            self.standardize_and_save_trades(trades, symbol, "recent_trades")

            # Store in real-time storage if enabled
            if self.realtime_storage:
                for trade in trades:
                    self.realtime_storage.store_trade(trade)

            self.logger.debug(f"Fetched {len(trades)} recent trades for {symbol}")
            return trades

        except Exception as e:
            self.logger.error(f"Failed to fetch trades for {symbol}: {e}")
            raise

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(BinanceAPIException, BinanceRequestException),
    )
    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
    ) -> List[List]:
        """Fetch kline/candlestick data for a symbol"""
        # Convert datetime to string if needed
        if isinstance(start_time, datetime):
            start_time = start_time.strftime("%d %b %Y %H:%M:%S")
        if isinstance(end_time, datetime):
            end_time = end_time.strftime("%d %b %Y %H:%M:%S")

        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            startTime=start_time,
            endTime=end_time,
        )

        # Convert to OHLCV format
        ohlcv_data = [
            [int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
            for k in klines
        ]

        # Save raw data
        filename = ExchangeUtils.create_filename("binance", symbol, "klines", interval)
        self.storage.save_json(klines, f"{filename}_raw", subfolder="klines")

        # Standardize and save
        self.standardize_and_save_ohlcv(ohlcv_data, symbol, interval)

        self.logger.debug(f"Fetched {len(klines)} klines for {symbol} ({interval})")
        return klines

    def fetch_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_str: str,
        end_str: Optional[str] = None,
        limit: int = 1000,
    ) -> List[List]:
        """Fetch historical klines using start/end time strings"""
        klines = self.client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_str,
            end_str=end_str,
            limit=limit,
        )

        if klines:
            # Convert to OHLCV format
            ohlcv_data = [
                [
                    int(k[0]),
                    float(k[1]),
                    float(k[2]),
                    float(k[3]),
                    float(k[4]),
                    float(k[5]),
                ]
                for k in klines
            ]

            # Standardize and save
            filename_suffix = (
                f"historical_{start_str}_{end_str}".replace(" ", "_")
                if end_str
                else f"from_{start_str}".replace(" ", "_")
            )
            self.standardize_and_save_ohlcv(
                ohlcv_data, symbol, interval, filename_suffix
            )

        self.logger.debug(
            f"Fetched {len(klines)} historical klines for {symbol} ({interval})"
        )
        return klines

    def fetch_historical_klines_chunked(
        self,
        symbol: str,
        interval: str,
        days_back: int = 30,
        chunk_size: int = 1000,
        end_date: Optional[datetime] = None,
    ) -> List[List]:
        """
        Fetch historical klines with chunking for large datasets

        Args:
            symbol: Trading symbol
            interval: Kline interval
            days_back: Number of days to go back
            chunk_size: Maximum number of klines per request
            end_date: End date for data collection (default: now)

        Returns:
            List of OHLCV data
        """
        try:
            # Use provided end_date or default to now
            actual_end_date = end_date or datetime.now()
            start_time = actual_end_date - timedelta(days=days_back)

            self.logger.info(
                f"Fetching klines for {symbol} ({interval}) from {start_time.strftime('%Y-%m-%d')} "
                f"to {actual_end_date.strftime('%Y-%m-%d')} ({days_back} days)"
            )

            all_klines = []
            current_start = start_time

            # Fetch data in chunks until we reach the end date
            while current_start < actual_end_date:
                try:
                    # Calculate chunk end time (don't exceed actual end date)
                    chunk_end = min(current_start + timedelta(days=30), actual_end_date)

                    self.logger.debug(
                        f"Fetching chunk: {current_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}"
                    )

                    # Fetch klines for this chunk
                    chunk_klines = self.client.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_str=current_start.strftime("%d %b %Y"),
                        end_str=chunk_end.strftime("%d %b %Y"),
                        limit=chunk_size,
                    )

                    if not chunk_klines:
                        self.logger.warning(
                            f"No data returned for chunk {current_start} to {chunk_end}"
                        )
                        break

                    # Add to all klines (avoid duplicates by checking timestamp)
                    if all_klines:
                        # Remove first kline if it's a duplicate (same timestamp as last kline)
                        last_timestamp = all_klines[-1][0]
                        if chunk_klines and chunk_klines[0][0] == last_timestamp:
                            chunk_klines = chunk_klines[1:]

                    all_klines.extend(chunk_klines)

                    self.logger.debug(
                        f"Fetched {len(chunk_klines)} klines for chunk, total: {len(all_klines)}"
                    )

                    # Move to next chunk
                    if chunk_klines:
                        # Use the last timestamp + 1 interval as the new start
                        last_timestamp = chunk_klines[-1][0]
                        current_start = datetime.fromtimestamp(
                            last_timestamp / 1000
                        ) + timedelta(hours=1)
                    else:
                        # If no data, move forward by chunk size
                        current_start = chunk_end

                    # Apply rate limiting between chunks
                    self.apply_rate_limiting()

                except Exception as e:
                    self.logger.error(
                        f"Error fetching chunk {current_start} to {chunk_end}: {e}"
                    )
                    # Move to next chunk even if this one failed
                    current_start = chunk_end
                    continue

            if all_klines:
                # Convert to OHLCV format
                ohlcv_data = [
                    [
                        int(k[0]),
                        float(k[1]),
                        float(k[2]),
                        float(k[3]),
                        float(k[4]),
                        float(k[5]),
                    ]
                    for k in all_klines
                ]

                # Save raw data
                filename = ExchangeUtils.create_filename(
                    "binance", symbol, "klines", interval
                )
                filename_suffix = f"historical_{start_time.strftime('%Y%m%d')}_{actual_end_date.strftime('%Y%m%d')}"
                self.storage.save_json(
                    all_klines, f"{filename}_{filename_suffix}_raw", subfolder="klines"
                )

                # Standardize and save
                self.standardize_and_save_ohlcv(
                    ohlcv_data, symbol, interval, filename_suffix
                )

            self.logger.info(
                f"Fetched {len(all_klines)} historical klines for {symbol} ({interval}) "
                f"from {start_time.strftime('%Y-%m-%d')} to {actual_end_date.strftime('%Y-%m-%d')}"
            )
            return all_klines

        except Exception as e:
            self.logger.error(
                f"Failed to fetch chunked historical klines for {symbol}: {e}"
            )
            return []

    def fetch_trades_chunked(
        self,
        symbol: str,
        total_limit: int = 5000,
        chunk_size: int = 1000,
        validate: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Fetch trades with chunking for large datasets

        Args:
            symbol: Trading symbol
            total_limit: Total number of trades to fetch
            chunk_size: Number of trades per request (max 1000 for Binance)
            validate: Whether to validate data

        Returns:
            List of trade data
        """
        try:
            symbol = ExchangeUtils.normalize_symbol(symbol, "binance")
            all_trades = []
            remaining = total_limit

            self.logger.info(
                f"Fetching {total_limit} trades for {symbol} in chunks of {chunk_size}"
            )

            while remaining > 0 and len(all_trades) < total_limit:
                current_limit = min(remaining, chunk_size)

                try:
                    # Get trades with fromId parameter for pagination if we have previous trades
                    kwargs = {"symbol": symbol, "limit": current_limit}
                    if all_trades:
                        # Use the last trade ID + 1 as starting point
                        last_id = all_trades[-1].get("id", 0)
                        kwargs["fromId"] = last_id + 1

                    trades = self.client.get_recent_trades(**kwargs)

                    if not trades:
                        self.logger.warning("No more trades available")
                        break

                    # Add metadata
                    for trade in trades:
                        trade["exchange_id"] = "binance"
                        trade["symbol"] = symbol

                    all_trades.extend(trades)
                    remaining -= len(trades)

                    self.logger.debug(
                        f"Fetched {len(trades)} trades, total: {len(all_trades)}"
                    )

                    # Apply rate limiting
                    self.apply_rate_limiting()

                    # Break if we got less than requested (end of available data)
                    if len(trades) < current_limit:
                        break

                except Exception as e:
                    self.logger.error(f"Error in trades chunk: {e}")
                    break

            # Trim to exact limit if we got more
            all_trades = all_trades[:total_limit]

            # Validate if requested
            if validate and all_trades:
                try:
                    trades_df = pd.DataFrame(all_trades)
                    validation_report = self.validator.validate_trade_data(
                        trades_df, symbol
                    )
                    if not validation_report["is_valid"]:
                        self.logger.warning(
                            f"Trades validation failed for {symbol}: {validation_report['issues']}"
                        )
                except Exception as validation_error:
                    self.logger.warning(
                        f"Trade validation error for {symbol}: {validation_error}"
                    )

            # Save data
            if all_trades:
                filename = ExchangeUtils.create_filename("binance", symbol, "trades")
                self.storage.save_json(
                    all_trades, f"{filename}_chunked_raw", subfolder="trades"
                )
                self.standardize_and_save_trades(all_trades, symbol, "chunked_trades")

            self.logger.info(f"Fetched {len(all_trades)} total trades for {symbol}")
            return all_trades

        except Exception as e:
            self.logger.error(f"Failed to fetch chunked trades for {symbol}: {e}")
            return []

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(BinanceAPIException, BinanceRequestException),
    )
    def fetch_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Fetch order book for a symbol"""
        try:
            orderbook = self.client.get_order_book(symbol=symbol, limit=limit)

            # Add metadata
            orderbook["symbol"] = symbol
            orderbook["timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()
            orderbook["limit"] = limit

            # Save raw data
            self.storage.save_json(
                orderbook, f"orderbook_{symbol}_raw", subfolder="orderbook"
            )

            # Standardize and save - fix the orderbook processing
            try:
                self.standardize_and_save_orderbook(orderbook, symbol)
            except Exception as processing_error:
                self.logger.warning(
                    f"Orderbook processing error for {symbol}: {processing_error}"
                )
                # Continue execution even if processing fails

            self.logger.debug(f"Fetched orderbook for {symbol} with {limit} levels")
            return orderbook

        except Exception as e:
            self.logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
            raise

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(BinanceAPIException, BinanceRequestException),
    )
    def fetch_trades(
        self, symbol: str, limit: int = 500, validate: bool = True
    ) -> List[Dict[str, Any]]:
        """Fetch recent trades for a symbol with validation"""
        try:
            # Normalize symbol format
            symbol = ExchangeUtils.normalize_symbol(symbol, "binance")

            trades = self.client.get_recent_trades(symbol=symbol, limit=limit)

            # Add metadata
            for trade in trades:
                trade["exchange_id"] = "binance"
                trade["symbol"] = symbol

            # Validate data if requested
            if validate and trades:
                try:
                    trades_df = pd.DataFrame(trades)
                    validation_report = self.validator.validate_trade_data(
                        trades_df, symbol
                    )
                    if not validation_report["is_valid"]:
                        self.logger.warning(
                            f"Trades validation failed for {symbol}: {validation_report['issues']}"
                        )
                except Exception as validation_error:
                    self.logger.warning(
                        f"Trade validation error for {symbol}: {validation_error}"
                    )

            # Save raw data
            filename = ExchangeUtils.create_filename("binance", symbol, "trades")
            self.storage.save_json(trades, f"{filename}_raw", subfolder="trades")

            # Standardize and save
            self.standardize_and_save_trades(trades, symbol, "recent_trades")

            # Store in real-time storage if enabled
            if self.realtime_storage:
                for trade in trades:
                    self.realtime_storage.store_trade(trade)

            self.logger.debug(f"Fetched {len(trades)} recent trades for {symbol}")
            return trades

        except Exception as e:
            self.logger.error(f"Failed to fetch trades for {symbol}: {e}")
            raise

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(BinanceAPIException, BinanceRequestException),
    )
    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
    ) -> List[List]:
        """Fetch kline/candlestick data for a symbol"""
        # Convert datetime to string if needed
        if isinstance(start_time, datetime):
            start_time = start_time.strftime("%d %b %Y %H:%M:%S")
        if isinstance(end_time, datetime):
            end_time = end_time.strftime("%d %b %Y %H:%M:%S")

        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            startTime=start_time,
            endTime=end_time,
        )

        # Convert to OHLCV format
        ohlcv_data = [
            [int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
            for k in klines
        ]

        # Save raw data
        filename = ExchangeUtils.create_filename("binance", symbol, "klines", interval)
        self.storage.save_json(klines, f"{filename}_raw", subfolder="klines")

        # Standardize and save
        self.standardize_and_save_ohlcv(ohlcv_data, symbol, interval)

        self.logger.debug(f"Fetched {len(klines)} klines for {symbol} ({interval})")
        return klines

    def fetch_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_str: str,
        end_str: Optional[str] = None,
        limit: int = 1000,
    ) -> List[List]:
        """Fetch historical klines using start/end time strings"""
        klines = self.client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_str,
            end_str=end_str,
            limit=limit,
        )

        if klines:
            # Convert to OHLCV format
            ohlcv_data = [
                [
                    int(k[0]),
                    float(k[1]),
                    float(k[2]),
                    float(k[3]),
                    float(k[4]),
                    float(k[5]),
                ]
                for k in klines
            ]

            # Standardize and save
            filename_suffix = (
                f"historical_{start_str}_{end_str}".replace(" ", "_")
                if end_str
                else f"from_{start_str}".replace(" ", "_")
            )
            self.standardize_and_save_ohlcv(
                ohlcv_data, symbol, interval, filename_suffix
            )

        self.logger.debug(
            f"Fetched {len(klines)} historical klines for {symbol} ({interval})"
        )
        return klines

    def fetch_historical_klines_chunked(
        self,
        symbol: str,
        interval: str,
        days_back: int = 30,
        chunk_size: int = 1000,
        end_date: Optional[datetime] = None,
    ) -> List[List]:
        """
        Fetch historical klines with chunking for large datasets

        Args:
            symbol: Trading symbol
            interval: Kline interval
            days_back: Number of days to go back
            chunk_size: Maximum number of klines per request
            end_date: End date for data collection (default: now)

        Returns:
            List of OHLCV data
        """
        try:
            # Use provided end_date or default to now
            actual_end_date = end_date or datetime.now()
            start_time = actual_end_date - timedelta(days=days_back)

            self.logger.info(
                f"Fetching klines for {symbol} ({interval}) from {start_time.strftime('%Y-%m-%d')} "
                f"to {actual_end_date.strftime('%Y-%m-%d')} ({days_back} days)"
            )

            all_klines = []
            current_start = start_time

            # Fetch data in chunks until we reach the end date
            while current_start < actual_end_date:
                try:
                    # Calculate chunk end time (don't exceed actual end date)
                    chunk_end = min(current_start + timedelta(days=30), actual_end_date)

                    self.logger.debug(
                        f"Fetching chunk: {current_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}"
                    )

                    # Fetch klines for this chunk
                    chunk_klines = self.client.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_str=current_start.strftime("%d %b %Y"),
                        end_str=chunk_end.strftime("%d %b %Y"),
                        limit=chunk_size,
                    )

                    if not chunk_klines:
                        self.logger.warning(
                            f"No data returned for chunk {current_start} to {chunk_end}"
                        )
                        break

                    # Add to all klines (avoid duplicates by checking timestamp)
                    if all_klines:
                        # Remove first kline if it's a duplicate (same timestamp as last kline)
                        last_timestamp = all_klines[-1][0]
                        if chunk_klines and chunk_klines[0][0] == last_timestamp:
                            chunk_klines = chunk_klines[1:]

                    all_klines.extend(chunk_klines)

                    self.logger.debug(
                        f"Fetched {len(chunk_klines)} klines for chunk, total: {len(all_klines)}"
                    )

                    # Move to next chunk
                    if chunk_klines:
                        # Use the last timestamp + 1 interval as the new start
                        last_timestamp = chunk_klines[-1][0]
                        current_start = datetime.fromtimestamp(
                            last_timestamp / 1000
                        ) + timedelta(hours=1)
                    else:
                        # If no data, move forward by chunk size
                        current_start = chunk_end

                    # Apply rate limiting between chunks
                    self.apply_rate_limiting()

                except Exception as e:
                    self.logger.error(
                        f"Error fetching chunk {current_start} to {chunk_end}: {e}"
                    )
                    # Move to next chunk even if this one failed
                    current_start = chunk_end
                    continue

            if all_klines:
                # Convert to OHLCV format
                ohlcv_data = [
                    [
                        int(k[0]),
                        float(k[1]),
                        float(k[2]),
                        float(k[3]),
                        float(k[4]),
                        float(k[5]),
                    ]
                    for k in all_klines
                ]

                # Save raw data
                filename = ExchangeUtils.create_filename(
                    "binance", symbol, "klines", interval
                )
                filename_suffix = f"historical_{start_time.strftime('%Y%m%d')}_{actual_end_date.strftime('%Y%m%d')}"
                self.storage.save_json(
                    all_klines, f"{filename}_{filename_suffix}_raw", subfolder="klines"
                )

                # Standardize and save
                self.standardize_and_save_ohlcv(
                    ohlcv_data, symbol, interval, filename_suffix
                )

            self.logger.info(
                f"Fetched {len(all_klines)} historical klines for {symbol} ({interval}) "
                f"from {start_time.strftime('%Y-%m-%d')} to {actual_end_date.strftime('%Y-%m-%d')}"
            )
            return all_klines

        except Exception as e:
            self.logger.error(
                f"Failed to fetch chunked historical klines for {symbol}: {e}"
            )
            return []

    def fetch_trades_chunked(
        self,
        symbol: str,
        total_limit: int = 5000,
        chunk_size: int = 1000,
        validate: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Fetch trades with chunking for large datasets

        Args:
            symbol: Trading symbol
            total_limit: Total number of trades to fetch
            chunk_size: Number of trades per request (max 1000 for Binance)
            validate: Whether to validate data

        Returns:
            List of trade data
        """
        try:
            symbol = ExchangeUtils.normalize_symbol(symbol, "binance")
            all_trades = []
            remaining = total_limit

            self.logger.info(
                f"Fetching {total_limit} trades for {symbol} in chunks of {chunk_size}"
            )

            while remaining > 0 and len(all_trades) < total_limit:
                current_limit = min(remaining, chunk_size)

                try:
                    # Get trades with fromId parameter for pagination if we have previous trades
                    kwargs = {"symbol": symbol, "limit": current_limit}
                    if all_trades:
                        # Use the last trade ID + 1 as starting point
                        last_id = all_trades[-1].get("id", 0)
                        kwargs["fromId"] = last_id + 1

                    trades = self.client.get_recent_trades(**kwargs)

                    if not trades:
                        self.logger.warning("No more trades available")
                        break

                    # Add metadata
                    for trade in trades:
                        trade["exchange_id"] = "binance"
                        trade["symbol"] = symbol

                    all_trades.extend(trades)
                    remaining -= len(trades)

                    self.logger.debug(
                        f"Fetched {len(trades)} trades, total: {len(all_trades)}"
                    )

                    # Apply rate limiting
                    self.apply_rate_limiting()

                    # Break if we got less than requested (end of available data)
                    if len(trades) < current_limit:
                        break

                except Exception as e:
                    self.logger.error(f"Error in trades chunk: {e}")
                    break

            # Trim to exact limit if we got more
            all_trades = all_trades[:total_limit]

            # Validate if requested
            if validate and all_trades:
                try:
                    trades_df = pd.DataFrame(all_trades)
                    validation_report = self.validator.validate_trade_data(
                        trades_df, symbol
                    )
                    if not validation_report["is_valid"]:
                        self.logger.warning(
                            f"Trades validation failed for {symbol}: {validation_report['issues']}"
                        )
                except Exception as validation_error:
                    self.logger.warning(
                        f"Trade validation error for {symbol}: {validation_error}"
                    )

            # Save data
            if all_trades:
                filename = ExchangeUtils.create_filename("binance", symbol, "trades")
                self.storage.save_json(
                    all_trades, f"{filename}_chunked_raw", subfolder="trades"
                )
                self.standardize_and_save_trades(all_trades, symbol, "chunked_trades")

            self.logger.info(f"Fetched {len(all_trades)} total trades for {symbol}")
            return all_trades

        except Exception as e:
            self.logger.error(f"Failed to fetch chunked trades for {symbol}: {e}")
            return []

    def start_realtime_streams(self, symbols: List[str]) -> None:
        """Start real-time data streams (placeholder implementation)"""
        self.logger.info(f"Real-time streaming would start for symbols: {symbols}")
        # Note: Real WebSocket implementation would be added here in production

    def stop_realtime_streams(self) -> None:
        """Stop real-time data streams (placeholder implementation)"""
        self.logger.info("Real-time streaming stopped")
        # Note: Real WebSocket cleanup would be added here in production

    def collect_symbol_data(
        self,
        symbol: str,
        intervals: List[str],
        days_back: int,
        include_trades: bool = True,
        include_orderbook: bool = True,
        end_date: Optional[datetime] = None,
        max_trades: int = 5000,
        max_orderbook_levels: int = 5000,
    ) -> Dict[str, Any]:
        """Collect data for a single symbol with configurable limits"""
        symbol_data = {"ohlcv": {}, "trades": 0, "orderbook": 0, "ticker": False}

        try:
            # Collect OHLCV for each interval with chunking
            for interval in intervals:
                try:
                    klines = self.fetch_historical_klines_chunked(
                        symbol=symbol,
                        interval=interval,
                        days_back=days_back,
                        chunk_size=1000,  # Binance limit per request
                        end_date=end_date,
                    )
                    symbol_data["ohlcv"][interval] = len(klines)

                except Exception as e:
                    error_msg = f"Failed to get {interval} klines for {symbol}: {e}"
                    self.logger.error(error_msg)
                    raise

            # Collect trades with chunking
            if include_trades and max_trades > 0:
                if max_trades <= 1000:
                    # Use single request for small amounts
                    trades = self.fetch_trades(symbol=symbol, limit=max_trades)
                else:
                    # Use chunked approach for large amounts
                    trades = self.fetch_trades_chunked(
                        symbol=symbol, total_limit=max_trades, chunk_size=1000
                    )
                symbol_data["trades"] = len(trades)

            # Collect order book (single request - Binance doesn't support historical orderbook)
            if include_orderbook:
                orderbook_limit = min(max_orderbook_levels, 5000)  # Binance max is 5000
                orderbook = self.fetch_orderbook(symbol=symbol, limit=orderbook_limit)
                symbol_data["orderbook"] = len(orderbook.get("bids", [])) + len(
                    orderbook.get("asks", [])
                )

            # Collect ticker
            ticker = self.fetch_ticker(symbol=symbol)
            symbol_data["ticker"] = True

        except Exception as e:
            self.logger.error(f"Error collecting data for {symbol}: {e}")
            raise

        return symbol_data

    def collect_comprehensive_data(
        self,
        symbols: Optional[List[str]] = None,
        intervals: Optional[List[str]] = None,
        days_back: int = 30,
        end_date: Optional[datetime] = None,
        include_trades: bool = True,
        include_orderbook: bool = True,
        enable_validation: bool = True,
        realtime_duration: int = 0,
        create_datasets: bool = True,
        max_trades_per_symbol: int = 5000,
        max_orderbook_levels: int = 1000,
    ) -> Dict[str, Any]:
        """
        Collect comprehensive data with configurable limits for each data type

        Args:
            symbols: List of symbols to collect data for (default: first 3 active symbols)
            intervals: List of intervals to collect OHLCV data for (default: ["1h", "4h"])
            days_back: Number of days to go back for historical data (default: 30)
            end_date: End date for data collection (default: now)
            include_trades: Whether to include recent trades (default: True)
            include_orderbook: Whether to include order book data (default: True)
            enable_validation: Whether to validate collected data (default: True)
            realtime_duration: Duration in seconds for real-time data collection (0 = disabled)
            create_datasets: Whether to create unified datasets after collection (default: True)
            max_trades_per_symbol: Maximum trades to fetch per symbol (default: 5000)
            max_orderbook_levels: Maximum orderbook levels to fetch per symbol (default: 1000)
        """
        # Use defaults if not specified
        symbols = symbols or self.get_default_symbols()[:3]  # Limit for demo
        intervals = intervals or ["1h", "4h"]

        # Use provided end_date or default to now
        actual_end_date = end_date or datetime.now()
        start_date = actual_end_date - timedelta(days=days_back)

        # Create collection summary
        collection_summary = self.create_collection_summary(
            symbols=symbols,
            timeframes=intervals,
            days_back=days_back,
            end_date=actual_end_date.isoformat(),
            start_date=start_date.isoformat(),
            include_trades=include_trades,
            include_orderbook=include_orderbook,
            enable_validation=enable_validation,
            realtime_duration=realtime_duration,
        )

        self.logger.info(
            f"Starting comprehensive data collection for {len(symbols)} symbols "
            f"from {start_date.strftime('%Y-%m-%d')} to {actual_end_date.strftime('%Y-%m-%d')}"
        )

        # Collect historical data for each symbol
        for symbol in symbols:
            try:
                symbol_data = self.collect_symbol_data(
                    symbol=symbol,
                    intervals=intervals,
                    days_back=days_back,
                    end_date=actual_end_date,
                    include_trades=include_trades,
                    include_orderbook=include_orderbook,
                    max_trades=max_trades_per_symbol,
                    max_orderbook_levels=max_orderbook_levels,
                )

                collection_summary["collected_data"][symbol] = symbol_data

                # Update total items
                collection_summary["total_items"] += sum(symbol_data["ohlcv"].values())
                collection_summary["total_items"] += symbol_data["trades"]
                collection_summary["total_items"] += symbol_data["orderbook"]
                if symbol_data["ticker"]:
                    collection_summary["total_items"] += 1

                self.logger.info(f"Completed historical data collection for {symbol}")

            except Exception as e:
                error_msg = f"Failed to collect data for {symbol}: {e}"
                self.logger.error(error_msg)
                collection_summary["errors"].append(error_msg)

        # Start real-time collection if requested
        if realtime_duration > 0 and self.enable_realtime:
            try:
                self.logger.info(
                    f"Starting {realtime_duration}s real-time data collection"
                )
                self.start_realtime_streams(symbols)

                # Wait for specified duration
                time.sleep(realtime_duration)

                self.stop_realtime_streams()

                # Add real-time statistics to summary
                if self.realtime_storage:
                    rt_stats = self.realtime_storage.get_statistics()
                    collection_summary["realtime_stats"] = rt_stats

            except Exception as e:
                error_msg = f"Error during real-time collection: {e}"
                self.logger.error(error_msg)
                collection_summary["errors"].append(error_msg)

        # Calculate collection statistics
        collection_stats = ExchangeUtils.calculate_collection_stats(
            collection_summary["collected_data"]
        )
        collection_summary["statistics"] = collection_stats

        # Create unified datasets if requested
        if create_datasets:
            try:
                self.logger.info("Creating unified datasets from collected data...")
                datasets = self.aggregator.create_multiple_datasets(collection_summary)

                collection_summary["datasets"] = {
                    symbol: str(path) for symbol, path in datasets.items()
                }

                self.logger.info(f"Created {len(datasets)} unified datasets")

            except Exception as e:
                error_msg = f"Failed to create datasets: {e}"
                self.logger.error(error_msg)
                collection_summary["errors"].append(error_msg)

        # Save collection summary
        self.save_collection_summary(collection_summary)

        # Log results
        self.log_collection_results(collection_summary)

        return collection_summary


def main():
    """Demonstrate Binance data collection capabilities"""
    # Initialize logger for main function
    logger = LoggerFactory.get_logger(
        name="binance_demo_main",
        logger_type=LoggerType.STANDARD,
        level=LogLevel.INFO,
        use_colors=True,
        log_file="binance_demo.log",
        file_level=LogLevel.DEBUG,
    )

    try:
        logger.info("Starting Enhanced Binance API Demo")

        # Initialize data collector with real-time capabilities
        collector = BinanceDataCollector(
            testnet=False,
            rate_limit=True,
            enable_realtime=False,  # Disable real-time for initial testing
        )

        # Test basic connectivity with better error handling
        try:
            server_time = collector.get_server_time()
            logger.info(f"Connected to Binance. Server time: {server_time['datetime']}")
        except Exception as e:
            logger.error(f"Failed to connect to Binance API: {e}")
            logger.info("This might be due to network issues or API rate limits")
            return

        # Demo data collection for selected symbols
        demo_symbols = [
            "BTCUSDT",
            # "ETHUSDT",
        ]

        demo_intervals = [
            "1h",
            # "4h",
            # "1d",
        ]

        # Configuration for data collection
        days_back = 2874  # Now this will actually fetch 1000 days of data
        max_trades_per_symbol = 10000  # Fetch up to 10k trades per symbol
        max_orderbook_levels = 1000  # Fetch up to 1k orderbook levels

        # Example: Collect data for a specific date range
        from datetime import timedelta

        end_date = datetime(2025, 6, 30)  # Example end date

        logger.info(f"Collecting comprehensive data for symbols: {demo_symbols}")
        logger.info(
            f"Date range: {(end_date - timedelta(days=days_back)).strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )
        logger.info(f"Max trades per symbol: {max_trades_per_symbol}")
        logger.info(f"Max orderbook levels: {max_orderbook_levels}")

        # Collect comprehensive data with configurable limits
        collection_summary = collector.collect_comprehensive_data(
            symbols=demo_symbols,
            intervals=demo_intervals,
            days_back=days_back,
            end_date=end_date,
            # include_trades=False,
            # include_orderbook=False,
            enable_validation=True,
            realtime_duration=0,
            create_datasets=True,
            max_trades_per_symbol=max_trades_per_symbol,
            max_orderbook_levels=max_orderbook_levels,
        )

        # Display enhanced summary
        logger.info("Enhanced Collection Summary:")
        logger.info(f"  Total items: {collection_summary['total_items']}")

        if collection_summary.get("statistics"):
            stats = collection_summary["statistics"]
            logger.info(f"  Total symbols processed: {stats.get('total_symbols', 0)}")
            logger.info(f"  OHLCV candles: {stats.get('total_ohlcv_candles', 0)}")
            logger.info(f"  Trades: {stats.get('total_trades', 0)}")

        # Display dataset information
        if collection_summary.get("datasets"):
            logger.info("Created Datasets:")
            for symbol, dataset_path in collection_summary["datasets"].items():
                logger.info(f"  {symbol}: {dataset_path}")

        if collection_summary.get("errors"):
            logger.warning(f"  Errors encountered: {len(collection_summary['errors'])}")

        # Cleanup and finalize
        collector.cleanup_and_finalize()

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
