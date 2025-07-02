"""
Market Data Repository Interface

Defines the contract for storing and retrieving market data.
Handles CRUD operations for different types of market data.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class MarketDataRepository(ABC):
    """
    Abstract interface for market data storage and retrieval.

    Implementations should handle:
    - Data persistence (files, databases, etc.)
    - Query optimization
    - Data integrity and consistency
    - Efficient storage formats
    """

    # OHLCV Operations
    @abstractmethod
    def save_ohlcv(
        self, exchange: str, symbol: str, timeframe: str, data: List[Dict[str, Any]]
    ) -> bool:
        """
        Save OHLCV data to repository.

        Args:
            exchange: Exchange name (e.g., 'binance')
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Time interval (e.g., '1h')
            data: List of OHLCV records

        Returns:
            True if save successful

        Raises:
            RepositoryError: If save operation fails
        """
        pass

    @abstractmethod
    def get_ohlcv(
        self, exchange: str, symbol: str, timeframe: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve OHLCV data from repository.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Time interval
            start_date: Start date in ISO 8601 format
            end_date: End date in ISO 8601 format

        Returns:
            List of OHLCV records within date range

        Raises:
            RepositoryError: If retrieval fails
            ValidationError: If date format is invalid
        """
        pass

    @abstractmethod
    def delete_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> bool:
        """
        Delete OHLCV data from repository.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Time interval
            start_date: Optional start date (delete from this date)
            end_date: Optional end date (delete until this date)

        Returns:
            True if deletion successful

        Raises:
            RepositoryError: If deletion fails
        """
        pass

    # Trade Operations
    @abstractmethod
    def save_trades(
        self, exchange: str, symbol: str, data: List[Dict[str, Any]]
    ) -> bool:
        """
        Save trade data to repository.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            data: List of trade records

        Returns:
            True if save successful

        Raises:
            RepositoryError: If save operation fails
        """
        pass

    @abstractmethod
    def get_trades(
        self, exchange: str, symbol: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve trade data from repository.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            start_date: Start date in ISO 8601 format
            end_date: End date in ISO 8601 format

        Returns:
            List of trade records within date range

        Raises:
            RepositoryError: If retrieval fails
        """
        pass

    # Order Book Operations
    @abstractmethod
    def save_orderbook(self, exchange: str, symbol: str, data: Dict[str, Any]) -> bool:
        """
        Save order book snapshot to repository.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            data: Order book snapshot

        Returns:
            True if save successful

        Raises:
            RepositoryError: If save operation fails
        """
        pass

    @abstractmethod
    def get_orderbook(
        self, exchange: str, symbol: str, timestamp: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve order book snapshot from repository.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timestamp: Specific timestamp in ISO 8601 format

        Returns:
            Order book snapshot at timestamp, None if not found

        Raises:
            RepositoryError: If retrieval fails
        """
        pass

    # Ticker Operations
    @abstractmethod
    def save_ticker(self, exchange: str, symbol: str, data: Dict[str, Any]) -> bool:
        """
        Save ticker data to repository.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            data: Ticker data

        Returns:
            True if save successful

        Raises:
            RepositoryError: If save operation fails
        """
        pass

    @abstractmethod
    def get_ticker(
        self, exchange: str, symbol: str, timestamp: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve ticker data from repository.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timestamp: Specific timestamp in ISO 8601 format

        Returns:
            Ticker data at timestamp, None if not found

        Raises:
            RepositoryError: If retrieval fails
        """
        pass

    # Query Operations
    @abstractmethod
    def get_available_symbols(self, exchange: str) -> List[str]:
        """
        Get all available symbols for an exchange in repository.

        Args:
            exchange: Exchange name

        Returns:
            List of available symbols

        Raises:
            RepositoryError: If query fails
        """
        pass

    @abstractmethod
    def get_available_timeframes(self, exchange: str, symbol: str) -> List[str]:
        """
        Get all available timeframes for a symbol.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            List of available timeframes

        Raises:
            RepositoryError: If query fails
        """
        pass

    @abstractmethod
    def get_data_range(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        timeframe: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        """
        Get the date range of available data for a symbol.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            data_type: Type of data ('ohlcv', 'trades', 'orderbook', 'ticker')
            timeframe: Optional timeframe for OHLCV data

        Returns:
            Dictionary with 'start_date' and 'end_date' in ISO 8601 format,
            None if no data available

        Raises:
            RepositoryError: If query fails
        """
        pass

    @abstractmethod
    def check_data_exists(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        start_date: str,
        end_date: str,
        timeframe: Optional[str] = None,
    ) -> bool:
        """
        Check if data exists for the specified criteria.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            data_type: Type of data ('ohlcv', 'trades', 'orderbook', 'ticker')
            start_date: Start date in ISO 8601 format
            end_date: End date in ISO 8601 format
            timeframe: Optional timeframe for OHLCV data

        Returns:
            True if data exists for the entire range

        Raises:
            RepositoryError: If check fails
        """
        pass

    @abstractmethod
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about the storage backend.

        Returns:
            Storage information including:
            {
                'storage_type': 'file' | 'database',
                'location': '/path/to/data' | 'connection_string',
                'total_size': 1024,  # in bytes
                'available_exchanges': [...],
                'total_symbols': 100,
                'oldest_data': '2020-01-01T00:00:00Z',
                'newest_data': '2024-01-01T00:00:00Z'
            }

        Raises:
            RepositoryError: If unable to get storage info
        """
        pass

    # Batch Operations
    @abstractmethod
    def batch_save_ohlcv(self, data: List[Dict[str, Any]]) -> bool:
        """
        Save multiple OHLCV datasets in a batch operation.

        Args:
            data: List of dictionaries containing:
                {
                    'exchange': 'binance',
                    'symbol': 'BTCUSDT',
                    'timeframe': '1h',
                    'records': [...]
                }

        Returns:
            True if all saves successful

        Raises:
            RepositoryError: If batch save fails
        """
        pass

    @abstractmethod
    def optimize_storage(self) -> bool:
        """
        Optimize storage for better performance (e.g., compact files, rebuild indexes).

        Returns:
            True if optimization successful

        Raises:
            RepositoryError: If optimization fails
        """
        pass

