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
    RealTimeDataStorage,
    DataValidator,
)


class BinanceDataCollector(BaseDataCollector):
    """Comprehensive Binance data collector for AI/ML datasets"""

    def __init__(
        self,
        testnet: bool = False,
        rate_limit: bool = True,
        enable_realtime: bool = False,
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
        self.enable_realtime = enable_realtime
        self.client = None
        self.twm = None

        # Initialize real-time storage if enabled
        self.realtime_storage = None
        if self.enable_realtime:
            self.realtime_storage = RealTimeDataStorage(
                base_dir="data/binance/realtime", buffer_size=1000, flush_interval=30
            )

        # Initialize data validator
        self.validator = DataValidator()

        # Initialize Binance client
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Binance client"""
        try:
            self.client = Client(
                api_key=None,  # No API key needed for public endpoints
                api_secret=None,
                testnet=self.testnet,
            )

            if self.rate_limit:
                self.client.API_URL = "https://api.binance.com/api/v3"

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
        server_time = self.client.get_server_time()
        server_time["datetime"] = pd.to_datetime(server_time["serverTime"], unit="ms")

        self.logger.info(f"Server time: {server_time['datetime']}")
        return server_time

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
            self.realtime_storage.store_ticker(ticker)

        self.logger.debug(f"Fetched ticker for {symbol}")
        return ticker

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
        orderbook = self.client.get_order_book(symbol=symbol, limit=limit)

        # Add metadata
        orderbook["symbol"] = symbol
        orderbook["timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()
        orderbook["limit"] = limit

        # Save raw data
        self.storage.save_json(
            orderbook, f"orderbook_{symbol}_raw", subfolder="orderbook"
        )

        # Standardize and save
        self.standardize_and_save_orderbook(orderbook, symbol)

        self.logger.debug(f"Fetched orderbook for {symbol} with {limit} levels")
        return orderbook

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(BinanceAPIException, BinanceRequestException),
    )
    def fetch_trades(
        self, symbol: str, limit: int = 500, validate: bool = True
    ) -> List[Dict[str, Any]]:
        """Fetch recent trades for a symbol with validation"""
        # Normalize symbol format
        symbol = ExchangeUtils.normalize_symbol(symbol, "binance")

        trades = self.client.get_recent_trades(symbol=symbol, limit=limit)

        # Add metadata
        for trade in trades:
            trade["exchange_id"] = "binance"
            trade["symbol"] = symbol

        # Validate data if requested
        if validate and trades:
            trades_df = pd.DataFrame(trades)
            validation_report = self.validator.validate_trade_data(trades_df, symbol)
            if not validation_report["is_valid"]:
                self.logger.warning(
                    f"Trades validation failed for {symbol}: {validation_report['issues']}"
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

    def collect_symbol_data(
        self,
        symbol: str,
        intervals: List[str],
        days_back: int,
        include_trades: bool = True,
        include_orderbook: bool = True,
    ) -> Dict[str, Any]:
        """Collect data for a single symbol"""
        symbol_data = {"ohlcv": {}, "trades": 0, "orderbook": 0, "ticker": False}

        try:
            # Collect OHLCV for each interval
            for interval in intervals:
                try:
                    start_time = datetime.now() - timedelta(days=days_back)
                    klines = self.fetch_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_str=start_time.strftime("%d %b %Y"),
                        limit=self.data_config.max_ohlcv_limit,
                    )
                    symbol_data["ohlcv"][interval] = len(klines)
                    self.apply_rate_limiting()

                except Exception as e:
                    error_msg = f"Failed to get {interval} klines for {symbol}: {e}"
                    self.logger.error(error_msg)
                    raise

            # Collect recent trades
            if include_trades:
                trades = self.fetch_trades(
                    symbol=symbol, limit=self.data_config.max_trades_limit
                )
                symbol_data["trades"] = len(trades)
                self.apply_rate_limiting()

            # Collect order book
            if include_orderbook:
                orderbook = self.fetch_orderbook(
                    symbol=symbol, limit=self.data_config.max_orderbook_limit
                )
                symbol_data["orderbook"] = len(orderbook.get("bids", [])) + len(
                    orderbook.get("asks", [])
                )
                self.apply_rate_limiting()

            # Collect ticker
            ticker = self.fetch_ticker(symbol=symbol)
            symbol_data["ticker"] = True
            self.apply_rate_limiting()

        except Exception as e:
            self.logger.error(f"Error collecting data for {symbol}: {e}")
            raise

        return symbol_data

    def collect_comprehensive_data(
        self,
        symbols: Optional[List[str]] = None,
        intervals: Optional[List[str]] = None,
        days_back: int = 30,
        include_trades: bool = True,
        include_orderbook: bool = True,
        enable_validation: bool = True,
        realtime_duration: int = 0,
    ) -> Dict[str, Any]:
        """
        Collect comprehensive data with optional real-time streaming

        Args:
            realtime_duration: Duration in seconds for real-time data collection (0 = disabled)
        """
        # Use defaults if not specified
        symbols = symbols or self.get_default_symbols()[:3]  # Limit for demo
        intervals = intervals or ["1h", "4h"]

        # Create collection summary
        collection_summary = self.create_collection_summary(
            symbols=symbols,
            timeframes=intervals,
            days_back=days_back,
            include_trades=include_trades,
            include_orderbook=include_orderbook,
            enable_validation=enable_validation,
            realtime_duration=realtime_duration,
        )

        self.logger.info(
            f"Starting comprehensive data collection for {len(symbols)} symbols"
        )

        # Collect historical data for each symbol
        for symbol in symbols:
            try:
                symbol_data = {
                    "ohlcv": {},
                    "trades": 0,
                    "orderbook": 0,
                    "ticker": False,
                }

                # Collect historical OHLCV for each interval
                for interval in intervals:
                    try:
                        klines = self.fetch_historical_klines_chunked(
                            symbol=symbol, interval=interval, days_back=days_back
                        )
                        symbol_data["ohlcv"][interval] = len(klines)
                        self.apply_rate_limiting()

                    except Exception as e:
                        error_msg = f"Failed to get {interval} klines for {symbol}: {e}"
                        self.logger.error(error_msg)
                        collection_summary["errors"].append(error_msg)

                # Collect recent trades
                if include_trades:
                    try:
                        trades = self.fetch_trades(
                            symbol=symbol,
                            limit=self.data_config.max_trades_limit,
                            validate=enable_validation,
                        )
                        symbol_data["trades"] = len(trades)
                        self.apply_rate_limiting()
                    except Exception as e:
                        error_msg = f"Failed to fetch trades for {symbol}: {e}"
                        self.logger.error(error_msg)
                        collection_summary["errors"].append(error_msg)

                # Collect order book
                if include_orderbook:
                    try:
                        orderbook = self.fetch_orderbook(
                            symbol=symbol, limit=self.data_config.max_orderbook_limit
                        )
                        symbol_data["orderbook"] = len(orderbook.get("bids", [])) + len(
                            orderbook.get("asks", [])
                        )
                        self.apply_rate_limiting()
                    except Exception as e:
                        error_msg = f"Failed to fetch orderbook for {symbol}: {e}"
                        self.logger.error(error_msg)
                        collection_summary["errors"].append(error_msg)

                # Collect ticker
                try:
                    ticker = self.fetch_ticker(
                        symbol=symbol, validate=enable_validation
                    )
                    symbol_data["ticker"] = True
                    self.apply_rate_limiting()
                except Exception as e:
                    error_msg = f"Failed to fetch ticker for {symbol}: {e}"
                    self.logger.error(error_msg)
                    collection_summary["errors"].append(error_msg)

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
    )

    try:
        logger.info("Starting Enhanced Binance API Demo")

        # Initialize data collector with real-time capabilities
        collector = BinanceDataCollector(
            testnet=False,
            rate_limit=True,
            enable_realtime=True,  # Enable real-time features
        )

        # Test basic connectivity
        server_time = collector.get_server_time()
        logger.info(f"Connected to Binance. Server time: {server_time['datetime']}")

        # Demo data collection for selected symbols
        demo_symbols = ["BTCUSDT", "ETHUSDT"]
        demo_intervals = ["1h", "4h"]

        logger.info(f"Collecting comprehensive data for symbols: {demo_symbols}")

        # Collect comprehensive data including real-time streaming
        collection_summary = collector.collect_comprehensive_data(
            symbols=demo_symbols,
            intervals=demo_intervals,
            days_back=7,
            include_trades=True,
            include_orderbook=True,
            enable_validation=True,
            # realtime_duration=30,  # 30 seconds of real-time data
        )

        # Display enhanced summary
        logger.info("Enhanced Collection Summary:")
        logger.info(f"  Total items: {collection_summary['total_items']}")
        logger.info(
            f"  Data quality score: {collection_summary.get('statistics', {}).get('coverage', {})}"
        )

        if "realtime_stats" in collection_summary:
            rt_stats = collection_summary["realtime_stats"]
            logger.info(f"  Real-time messages: {rt_stats['total_messages']}")
            logger.info(f"  Messages per second: {rt_stats['messages_per_second']:.2f}")

        # Cleanup and finalize
        collector.cleanup_and_finalize()

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
