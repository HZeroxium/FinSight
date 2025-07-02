"""
Market Data Collector Interface

Defines the contract for collecting market data from various exchanges.
Focuses solely on data collection logic without storage concerns.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime


class MarketDataCollector(ABC):
    """
    Abstract interface for market data collectors.

    Implementations should handle:
    - Exchange-specific API integration
    - Rate limiting and chunking internally
    - Data standardization
    - Error handling and retries
    """

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Get all available trading symbols from the exchange.

        Returns:
            List of available trading symbols

        Raises:
            CollectionError: If unable to fetch symbols
        """
        pass

    @abstractmethod
    def get_available_timeframes(self) -> List[str]:
        """
        Get all supported timeframes for OHLCV data.

        Returns:
            List of supported timeframes (e.g., ['1m', '5m', '1h', '1d'])

        Raises:
            CollectionError: If unable to fetch timeframes
        """
        pass

    @abstractmethod
    def collect_ohlcv(
        self, symbol: str, timeframe: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Collect OHLCV (candlestick) data for a symbol within date range.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Time interval (e.g., '1h', '1d')
            start_date: Start date in ISO 8601 format (e.g., '2024-01-01T00:00:00Z')
            end_date: End date in ISO 8601 format (e.g., '2024-01-31T23:59:59Z')

        Returns:
            List of OHLCV records with standardized format:
            [
                {
                    'timestamp': '2024-01-01T00:00:00Z',
                    'open': 45000.0,
                    'high': 46000.0,
                    'low': 44000.0,
                    'close': 45500.0,
                    'volume': 1234.56,
                    'symbol': 'BTCUSDT',
                    'timeframe': '1h'
                },
                ...
            ]

        Raises:
            CollectionError: If unable to collect data
            ValidationError: If date format is invalid
        """
        pass

    @abstractmethod
    def collect_trades(
        self, symbol: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Collect trade data for a symbol within date range.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            start_date: Start date in ISO 8601 format
            end_date: End date in ISO 8601 format

        Returns:
            List of trade records with standardized format:
            [
                {
                    'timestamp': '2024-01-01T00:00:00.123Z',
                    'price': 45000.0,
                    'quantity': 0.5,
                    'side': 'buy',  # 'buy' or 'sell'
                    'trade_id': '12345',
                    'symbol': 'BTCUSDT'
                },
                ...
            ]

        Raises:
            CollectionError: If unable to collect data
            ValidationError: If date format is invalid
        """
        pass

    @abstractmethod
    def collect_orderbook(self, symbol: str, timestamp: str) -> Dict[str, Any]:
        """
        Collect order book snapshot for a symbol at specific timestamp.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timestamp: Timestamp in ISO 8601 format

        Returns:
            Order book data with standardized format:
            {
                'timestamp': '2024-01-01T00:00:00Z',
                'symbol': 'BTCUSDT',
                'bids': [
                    {'price': 44999.0, 'quantity': 1.5},
                    {'price': 44998.0, 'quantity': 2.0},
                    ...
                ],
                'asks': [
                    {'price': 45001.0, 'quantity': 1.2},
                    {'price': 45002.0, 'quantity': 0.8},
                    ...
                ]
            }

        Raises:
            CollectionError: If unable to collect data
            ValidationError: If timestamp format is invalid
        """
        pass

    @abstractmethod
    def collect_ticker(self, symbol: str, timestamp: str) -> Dict[str, Any]:
        """
        Collect ticker data for a symbol at specific timestamp.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timestamp: Timestamp in ISO 8601 format

        Returns:
            Ticker data with standardized format:
            {
                'timestamp': '2024-01-01T00:00:00Z',
                'symbol': 'BTCUSDT',
                'last_price': 45000.0,
                'bid_price': 44999.0,
                'ask_price': 45001.0,
                'volume_24h': 12345.67,
                'price_change_24h': 500.0,
                'price_change_percent_24h': 1.12,
                'high_24h': 46000.0,
                'low_24h': 44000.0
            }

        Raises:
            CollectionError: If unable to collect data
            ValidationError: If timestamp format is invalid
        """
        pass

    @abstractmethod
    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange information and trading rules.

        Returns:
            Exchange information including:
            {
                'exchange_name': 'binance',
                'symbols': [...],
                'timeframes': [...],
                'rate_limits': {...},
                'trading_rules': {...}
            }

        Raises:
            CollectionError: If unable to fetch exchange info
        """
        pass

    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is supported by the exchange.

        Args:
            symbol: Trading symbol to validate

        Returns:
            True if symbol is valid and tradable
        """
        pass

    @abstractmethod
    def validate_timeframe(self, timeframe: str) -> bool:
        """
        Validate if a timeframe is supported by the exchange.

        Args:
            timeframe: Timeframe to validate

        Returns:
            True if timeframe is supported
        """
        pass
