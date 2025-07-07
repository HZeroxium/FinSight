# interfaces/market_data_repository.py

"""
Market Data Repository Interface

Defines the contract for storing and retrieving market data.
Handles CRUD operations for different types of market data using proper schemas.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..schemas.ohlcv_schemas import OHLCVSchema, OHLCVBatchSchema, OHLCVQuerySchema


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
        self, exchange: str, symbol: str, timeframe: str, data: List[OHLCVSchema]
    ) -> bool:
        """
        Save OHLCV data to repository.

        Args:
            exchange: Exchange name (e.g., 'binance')
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Time interval (e.g., '1h')
            data: List of OHLCV schema instances

        Returns:
            True if save successful

        Raises:
            RepositoryError: If save operation fails
        """
        pass

    @abstractmethod
    def get_ohlcv(self, query: OHLCVQuerySchema) -> List[OHLCVSchema]:
        """
        Retrieve OHLCV data from repository.

        Args:
            query: OHLCV query schema instance

        Returns:
            List of OHLCV schema instances

        Raises:
            RepositoryError: If retrieval fails
            ValidationError: If query is invalid
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
    def batch_save_ohlcv(self, data: List[OHLCVBatchSchema]) -> bool:
        """
        Save multiple OHLCV batch schemas in a batch operation.

        Args:
            data: List of OHLCV batch schema instances

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

    # Administrative Operations
    @abstractmethod
    async def count_all_records(self) -> int:
        """
        Count total number of OHLCV records in repository.

        Returns:
            Total count of records

        Raises:
            RepositoryError: If count operation fails
        """
        pass

    @abstractmethod
    async def get_available_symbols(self) -> List[str]:
        """
        Get all available symbols across all exchanges.

        Returns:
            List of unique symbols

        Raises:
            RepositoryError: If query fails
        """
        pass

    @abstractmethod
    async def get_available_exchanges(self) -> List[str]:
        """
        Get all available exchanges in repository.

        Returns:
            List of exchange names

        Raises:
            RepositoryError: If query fails
        """
        pass

    @abstractmethod
    async def get_available_timeframes(self) -> List[str]:
        """
        Get all available timeframes across all data.

        Returns:
            List of timeframes

        Raises:
            RepositoryError: If query fails
        """
        pass

    @abstractmethod
    async def count_records(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """
        Count records for specific criteria.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Count of matching records

        Raises:
            RepositoryError: If count operation fails
        """
        pass

    @abstractmethod
    async def count_records_since(self, cutoff_date: datetime) -> int:
        """
        Count records since a specific date.

        Args:
            cutoff_date: Date to count records from

        Returns:
            Count of records since date

        Raises:
            RepositoryError: If count operation fails
        """
        pass

    @abstractmethod
    async def delete_records_before_date(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        cutoff_date: datetime,
    ) -> int:
        """
        Delete records before a specific date.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Data timeframe
            cutoff_date: Date before which to delete records

        Returns:
            Number of records deleted

        Raises:
            RepositoryError: If delete operation fails
        """
        pass
