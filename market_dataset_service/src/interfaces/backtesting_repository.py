# interfaces/backtesting_repository.py

"""
Backtesting Repository Interface

Defines the contract for backtesting data persistence operations.
Supports CRUD operations for backtest results and history.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..schemas.backtesting_schemas import (
    BacktestResult,
    BacktestHistoryItem,
    StrategyType,
)


class BacktestingRepositoryError(Exception):
    """Base exception for backtesting repository errors."""

    pass


class BacktestingRepository(ABC):
    """Abstract interface for backtesting data persistence."""

    @abstractmethod
    async def save_backtest_result(
        self,
        backtest_id: str,
        result: BacktestResult,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save backtest result to storage.

        Args:
            backtest_id: Unique identifier for the backtest
            result: Complete backtest result
            metadata: Additional metadata to store

        Returns:
            True if save was successful

        Raises:
            BacktestingRepositoryError: If save operation fails
        """
        pass

    @abstractmethod
    async def get_backtest_result(
        self,
        backtest_id: str,
        include_trades: bool = True,
        include_equity_curve: bool = True,
    ) -> Optional[BacktestResult]:
        """
        Retrieve backtest result by ID.

        Args:
            backtest_id: Unique identifier for the backtest
            include_trades: Whether to include trade details
            include_equity_curve: Whether to include equity curve data

        Returns:
            Backtest result if found, None otherwise

        Raises:
            BacktestingRepositoryError: If retrieval fails
        """
        pass

    @abstractmethod
    async def delete_backtest_result(self, backtest_id: str) -> bool:
        """
        Delete backtest result from storage.

        Args:
            backtest_id: Unique identifier for the backtest

        Returns:
            True if deletion was successful

        Raises:
            BacktestingRepositoryError: If deletion fails
        """
        pass

    @abstractmethod
    async def list_backtest_history(
        self,
        limit: int = 10,
        offset: int = 0,
        strategy_filter: Optional[str] = None,
        symbol_filter: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        sort_by: str = "executed_at",
        sort_order: str = "desc",
    ) -> List[BacktestHistoryItem]:
        """
        List backtest history with filtering and pagination.

        Args:
            limit: Maximum number of items to return
            offset: Number of items to skip
            strategy_filter: Filter by strategy name
            symbol_filter: Filter by trading symbol
            start_date: Filter by execution start date
            end_date: Filter by execution end date
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)

        Returns:
            List of backtest history items

        Raises:
            BacktestingRepositoryError: If query fails
        """
        pass

    @abstractmethod
    async def count_backtest_history(
        self,
        strategy_filter: Optional[str] = None,
        symbol_filter: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """
        Count total backtest history items matching filters.

        Args:
            strategy_filter: Filter by strategy name
            symbol_filter: Filter by trading symbol
            start_date: Filter by execution start date
            end_date: Filter by execution end date

        Returns:
            Total count of matching items

        Raises:
            BacktestingRepositoryError: If count fails
        """
        pass

    @abstractmethod
    async def backtest_exists(self, backtest_id: str) -> bool:
        """
        Check if backtest result exists.

        Args:
            backtest_id: Unique identifier for the backtest

        Returns:
            True if backtest exists

        Raises:
            BacktestingRepositoryError: If check fails
        """
        pass

    @abstractmethod
    async def get_backtest_summary(
        self, backtest_id: str
    ) -> Optional[BacktestHistoryItem]:
        """
        Get backtest summary without full result data.

        Args:
            backtest_id: Unique identifier for the backtest

        Returns:
            Backtest summary if found, None otherwise

        Raises:
            BacktestingRepositoryError: If retrieval fails
        """
        pass

    @abstractmethod
    async def cleanup_old_results(
        self, cutoff_date: datetime, dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Clean up old backtest results before specified date.

        Args:
            cutoff_date: Delete results older than this date
            dry_run: If True, only count what would be deleted

        Returns:
            Dictionary with cleanup statistics

        Raises:
            BacktestingRepositoryError: If cleanup fails
        """
        pass

    @abstractmethod
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics for backtest data.

        Returns:
            Dictionary with storage statistics

        Raises:
            BacktestingRepositoryError: If stats retrieval fails
        """
        pass

    @abstractmethod
    async def optimize_storage(self) -> bool:
        """
        Optimize storage for better performance.

        Returns:
            True if optimization was successful

        Raises:
            BacktestingRepositoryError: If optimization fails
        """
        pass
