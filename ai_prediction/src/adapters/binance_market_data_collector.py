# adapters/binance_market_data_collector.py

"""
Binance Market Data Collector Implementation

Implements the MarketDataCollector interface for collecting data from Binance exchange.
Focuses on OHLCV data collection using python-binance library.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

from ..interfaces.market_data_collector import MarketDataCollector
from ..interfaces.errors import CollectionError, ValidationError
from ..common.logger import LoggerFactory, LoggerType, LogLevel
from ..utils.datetime_utils import DateTimeUtils


class BinanceMarketDataCollector(MarketDataCollector):
    """Binance implementation of MarketDataCollector interface"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
    ):
        """
        Initialize Binance market data collector

        Args:
            api_key: Binance API key (optional for public endpoints)
            api_secret: Binance API secret (optional for public endpoints)
            testnet: Whether to use testnet endpoints
        """
        self.logger = LoggerFactory.get_logger(
            name="binance_collector",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
        )

        try:
            self.client = Client(
                api_key=api_key, api_secret=api_secret, testnet=testnet
            )
            self.testnet = testnet
            self._exchange_info = None

            self.logger.info(f"Initialized Binance collector (testnet: {testnet})")
        except Exception as e:
            raise CollectionError(f"Failed to initialize Binance client: {str(e)}")

    def get_available_symbols(self) -> List[str]:
        """Get all available trading symbols from Binance"""
        try:
            if not self._exchange_info:
                self._exchange_info = self.client.get_exchange_info()

            symbols = [
                symbol_info["symbol"]
                for symbol_info in self._exchange_info["symbols"]
                if symbol_info["status"] == "TRADING"
            ]

            self.logger.info(f"Retrieved {len(symbols)} available symbols")
            return symbols

        except (BinanceAPIException, BinanceRequestException) as e:
            raise CollectionError(f"Failed to fetch available symbols: {str(e)}")
        except Exception as e:
            raise CollectionError(f"Unexpected error fetching symbols: {str(e)}")

    def get_available_timeframes(self) -> List[str]:
        """Get all supported timeframes for OHLCV data"""
        # Binance supported intervals
        timeframes = [
            "1m",
            "3m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "8h",
            "12h",
            "1d",
            "3d",
            "1w",
            "1M",
        ]

        self.logger.debug(f"Available timeframes: {timeframes}")
        return timeframes

    def get_symbol_listing_date(self, symbol: str) -> Optional[str]:
        """
        Get the earliest available date for a symbol on Binance.

        Args:
            symbol: Trading symbol

        Returns:
            Earliest available date in ISO 8601 format, None if not available
        """
        try:
            # Try to get a small amount of historical data from a very early date
            # Binance will return data from the actual listing date
            test_klines = self.client.get_historical_klines(
                symbol=symbol,
                interval="1d",
                start_str="1 Jan 2010",  # Very early date
                limit=1,
            )

            if test_klines:
                # First available timestamp
                timestamp_ms = test_klines[0][0]
                timestamp_dt = DateTimeUtils.from_timestamp_ms(timestamp_ms)
                return DateTimeUtils.to_iso_string(timestamp_dt)

            return None

        except (BinanceAPIException, BinanceRequestException) as e:
            self.logger.warning(f"Could not get listing date for {symbol}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error getting listing date for {symbol}: {str(e)}"
            )
            return None

    def collect_ohlcv(
        self, symbol: str, timeframe: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Collect OHLCV data for a symbol within date range

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Time interval (e.g., '1h', '1d')
            start_date: Start date in ISO 8601 format (e.g., '2024-01-01T00:00:00Z')
            end_date: End date in ISO 8601 format (e.g., '2024-01-31T23:59:59Z')

        Returns:
            List of OHLCV records with standardized format
        """
        try:
            # Validate inputs
            self._validate_symbol(symbol)
            self._validate_timeframe(timeframe)
            start_dt, end_dt = DateTimeUtils.validate_date_range(start_date, end_date)

            self.logger.info(
                f"Collecting OHLCV data for {symbol} ({timeframe}) from {start_date} to {end_date}"
            )

            # Check if start_date is before symbol listing
            listing_date = self.get_symbol_listing_date(symbol)
            if listing_date:
                listing_dt = DateTimeUtils.to_utc_datetime(listing_date)
                if start_dt < listing_dt:
                    self.logger.info(
                        f"Adjusting start date from {start_date} to symbol listing date {listing_date}"
                    )
                    start_dt = listing_dt
                    start_date = listing_date

            # Convert to Binance format
            start_str = DateTimeUtils.format_timestamp_for_exchange(start_dt, "binance")
            end_str = DateTimeUtils.format_timestamp_for_exchange(end_dt, "binance")

            # Collect data in chunks if date range is large
            all_klines = []
            chunk_start = start_dt

            while chunk_start < end_dt:
                # Calculate chunk end (max 30 days for efficiency)
                chunk_end = min(chunk_start + timedelta(days=30), end_dt)

                try:
                    chunk_start_str = DateTimeUtils.format_timestamp_for_exchange(
                        chunk_start, "binance"
                    )
                    chunk_end_str = DateTimeUtils.format_timestamp_for_exchange(
                        chunk_end, "binance"
                    )

                    chunk_klines = self.client.get_historical_klines(
                        symbol=symbol,
                        interval=timeframe,
                        start_str=chunk_start_str,
                        end_str=chunk_end_str,
                        limit=1000,
                    )

                    if not chunk_klines:
                        break

                    all_klines.extend(chunk_klines)
                    chunk_start = chunk_end

                    # Rate limiting
                    time.sleep(0.1)

                except (BinanceAPIException, BinanceRequestException) as e:
                    self.logger.warning(
                        f"Failed to fetch chunk {chunk_start} to {chunk_end}: {e}"
                    )
                    chunk_start = chunk_end
                    continue

            # Convert to standardized format
            standardized_data = self._standardize_ohlcv_data(
                all_klines, symbol, timeframe
            )

            # Filter by exact date range (use original end_dt)
            filtered_data: List[Dict[str, Any]] = []
            original_end_dt = DateTimeUtils.to_utc_datetime(end_date)
            for record in standardized_data:
                record_dt = DateTimeUtils.to_utc_datetime(record["timestamp"])
                if start_dt <= record_dt <= original_end_dt:
                    filtered_data.append(record)

            self.logger.info(
                f"Collected {len(filtered_data)} OHLCV records for {symbol}"
            )
            return filtered_data

        except (BinanceAPIException, BinanceRequestException) as e:
            raise CollectionError(f"Binance API error collecting OHLCV data: {str(e)}")
        except Exception as e:
            raise CollectionError(f"Failed to collect OHLCV data: {str(e)}")

    def collect_trades(
        self, symbol: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """Collect trade data - placeholder implementation"""
        raise NotImplementedError(
            "Trade data collection will be implemented in next phase"
        )

    def collect_orderbook(self, symbol: str, timestamp: str) -> Dict[str, Any]:
        """Collect order book data - placeholder implementation"""
        raise NotImplementedError(
            "Order book collection will be implemented in next phase"
        )

    def collect_ticker(self, symbol: str, timestamp: str) -> Dict[str, Any]:
        """Collect ticker data - placeholder implementation"""
        raise NotImplementedError("Ticker collection will be implemented in next phase")

    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information and trading rules"""
        try:
            if not self._exchange_info:
                self._exchange_info = self.client.get_exchange_info()

            return {
                "exchange_name": "binance",
                "symbols": [s["symbol"] for s in self._exchange_info["symbols"]],
                "timeframes": self.get_available_timeframes(),
                "rate_limits": self._extract_rate_limits(self._exchange_info),
                "trading_rules": self._extract_trading_rules(self._exchange_info),
            }

        except Exception as e:
            raise CollectionError(f"Failed to get exchange info: {str(e)}")

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol is supported by Binance"""
        try:
            available_symbols = self.get_available_symbols()
            return symbol in available_symbols
        except Exception:
            return False

    def validate_timeframe(self, timeframe: str) -> bool:
        """Validate if a timeframe is supported by Binance"""
        available_timeframes = self.get_available_timeframes()
        return timeframe in available_timeframes

    def _validate_symbol(self, symbol: str) -> None:
        """Validate symbol format and availability"""
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol must be a non-empty string")

        if not self.validate_symbol(symbol):
            raise ValidationError(f"Symbol {symbol} is not available on Binance")

    def _validate_timeframe(self, timeframe: str) -> None:
        """Validate timeframe format"""
        if not self.validate_timeframe(timeframe):
            raise ValidationError(f"Timeframe {timeframe} is not supported by Binance")

    def _validate_and_parse_dates(self, start_date: str, end_date: str) -> tuple:
        """Validate and parse ISO 8601 date strings"""
        try:
            start_dt, end_dt = DateTimeUtils.validate_date_range(start_date, end_date)

            # Check if date range is reasonable (not too far in the past or future)
            now = DateTimeUtils.now_utc()
            if end_dt > now:
                raise ValidationError("End date cannot be in the future")

            return start_dt, end_dt

        except ValueError as e:
            raise ValidationError(f"Date validation failed: {str(e)}")

    def _standardize_ohlcv_data(
        self, klines: List[List], symbol: str, timeframe: str
    ) -> List[Dict[str, Any]]:
        """Convert Binance klines to standardized OHLCV format"""
        standardized = []

        for kline in klines:
            # Binance kline format: [timestamp, open, high, low, close, volume, ...]
            timestamp_ms = kline[0]
            timestamp_dt = DateTimeUtils.from_timestamp_ms(timestamp_ms)

            record = {
                "timestamp": DateTimeUtils.to_iso_string(timestamp_dt),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
                "symbol": symbol,
                "timeframe": timeframe,
            }
            standardized.append(record)

        return standardized

    def _extract_rate_limits(self, exchange_info: Dict) -> Dict[str, Any]:
        """Extract rate limit information from exchange info"""
        return {
            "rate_limit_type": exchange_info.get("rateLimitType"),
            "rate_limits": exchange_info.get("rateLimits", []),
        }

    def _extract_trading_rules(self, exchange_info: Dict) -> Dict[str, Any]:
        """Extract trading rules from exchange info"""
        return {
            "timezone": exchange_info.get("timezone"),
            "server_time": exchange_info.get("serverTime"),
            "symbols_count": len(exchange_info.get("symbols", [])),
        }
