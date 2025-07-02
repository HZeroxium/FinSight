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

    def collect_ohlcv(
        self, symbol: str, timeframe: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Collect OHLCV data for a symbol within date range

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Time interval (e.g., '1h', '1d')
            start_date: Start date in ISO 8601 format
            end_date: End date in ISO 8601 format

        Returns:
            List of OHLCV records with standardized format
        """
        try:
            # Validate inputs
            self._validate_symbol(symbol)
            self._validate_timeframe(timeframe)
            start_dt, end_dt = self._validate_and_parse_dates(start_date, end_date)

            self.logger.info(
                f"Collecting OHLCV data for {symbol} ({timeframe}) from {start_date} to {end_date}"
            )

            # Convert to Binance format
            start_str = start_dt.strftime("%d %b %Y %H:%M:%S")
            end_str = end_dt.strftime("%d %b %Y %H:%M:%S")

            # Collect data in chunks if date range is large
            all_klines = []
            chunk_start = start_dt

            while chunk_start < end_dt:
                # Calculate chunk end (max 30 days for efficiency)
                chunk_end = min(chunk_start + timedelta(days=30), end_dt)

                try:
                    chunk_klines = self.client.get_historical_klines(
                        symbol=symbol,
                        interval=timeframe,
                        start_str=chunk_start.strftime("%d %b %Y %H:%M:%S"),
                        end_str=chunk_end.strftime("%d %b %Y %H:%M:%S"),
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

            # Filter by exact date range
            filtered_data = [
                record
                for record in standardized_data
                if start_dt
                <= datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
                <= end_dt
            ]

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
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

            if start_dt >= end_dt:
                raise ValidationError("Start date must be before end date")

            # Check if date range is reasonable (not too far in the past or future)
            now = datetime.now().replace(tzinfo=start_dt.tzinfo)
            if end_dt > now:
                raise ValidationError("End date cannot be in the future")

            return start_dt, end_dt

        except ValueError as e:
            raise ValidationError(
                f"Invalid date format. Expected ISO 8601 format: {str(e)}"
            )

    def _standardize_ohlcv_data(
        self, klines: List[List], symbol: str, timeframe: str
    ) -> List[Dict[str, Any]]:
        """Convert Binance klines to standardized OHLCV format"""
        standardized = []

        for kline in klines:
            # Binance kline format: [timestamp, open, high, low, close, volume, ...]
            record = {
                "timestamp": datetime.fromtimestamp(kline[0] / 1000).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
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
