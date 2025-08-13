# services/backtesting_data_service.py

"""
Backtesting Data Service

Provides high-level CRUD operations for backtest results and history.
Acts as an abstraction layer over the repository pattern.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from math import ceil

from ..interfaces.backtesting_repository import (
    BacktestingRepository,
    BacktestingRepositoryError,
)
from ..schemas.backtesting_schemas import (
    BacktestResult,
    BacktestHistoryItem,
    BacktestHistoryResponse,
    BacktestDeletionResponse,
    StrategyType,
)
from common.logger import LoggerFactory


class BacktestingDataServiceError(Exception):
    """Base exception for backtesting data service errors."""

    pass


class BacktestingDataService:
    """Service for managing backtesting data operations."""

    def __init__(self, repository: BacktestingRepository):
        """
        Initialize backtesting data service.

        Args:
            repository: Repository implementation for data persistence
        """
        self.repository = repository
        self.logger = LoggerFactory.get_logger(name="backtesting_data_service")

        self.logger.info("BacktestingDataService initialized")

    async def save_backtest_result(
        self,
        result: BacktestResult,
        backtest_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save backtest result with auto-generated ID.

        Args:
            result: Complete backtest result
            backtest_id: Optional custom ID, generates UUID if not provided
            metadata: Additional metadata to store

        Returns:
            Generated or provided backtest ID

        Raises:
            BacktestingDataServiceError: If save operation fails
        """
        try:
            # Generate ID if not provided
            if backtest_id is None:
                backtest_id = str(uuid.uuid4())

            # Validate result
            self._validate_backtest_result(result)

            # Save to repository
            success = await self.repository.save_backtest_result(
                backtest_id=backtest_id, result=result, metadata=metadata
            )

            if not success:
                raise BacktestingDataServiceError("Repository save operation failed")

            self.logger.info(f"Saved backtest result with ID: {backtest_id}")
            return backtest_id

        except BacktestingRepositoryError as e:
            self.logger.error(f"Repository error saving backtest result: {e}")
            raise BacktestingDataServiceError(f"Failed to save backtest result: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error saving backtest result: {e}")
            raise BacktestingDataServiceError(f"Failed to save backtest result: {e}")

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
            BacktestingDataServiceError: If retrieval fails
        """
        try:
            result = await self.repository.get_backtest_result(
                backtest_id=backtest_id,
                include_trades=include_trades,
                include_equity_curve=include_equity_curve,
            )

            if result:
                self.logger.info(f"Retrieved backtest result: {backtest_id}")
            else:
                self.logger.warning(f"Backtest result not found: {backtest_id}")

            return result

        except BacktestingRepositoryError as e:
            self.logger.error(f"Repository error retrieving backtest result: {e}")
            raise BacktestingDataServiceError(
                f"Failed to retrieve backtest result: {e}"
            )
        except Exception as e:
            self.logger.error(f"Unexpected error retrieving backtest result: {e}")
            raise BacktestingDataServiceError(
                f"Failed to retrieve backtest result: {e}"
            )

    async def delete_backtest_result(
        self, backtest_id: str
    ) -> BacktestDeletionResponse:
        """
        Delete backtest result.

        Args:
            backtest_id: Unique identifier for the backtest

        Returns:
            Deletion response with status information

        Raises:
            BacktestingDataServiceError: If deletion fails
        """
        try:
            # Check if backtest exists
            exists = await self.repository.backtest_exists(backtest_id)
            if not exists:
                return BacktestDeletionResponse(
                    success=False,
                    backtest_id=backtest_id,
                    message="Backtest result not found",
                )

            # Delete from repository
            success = await self.repository.delete_backtest_result(backtest_id)

            if success:
                self.logger.info(f"Deleted backtest result: {backtest_id}")
                return BacktestDeletionResponse(
                    success=True,
                    backtest_id=backtest_id,
                    message="Backtest result deleted successfully",
                )
            else:
                return BacktestDeletionResponse(
                    success=False,
                    backtest_id=backtest_id,
                    message="Failed to delete backtest result",
                )

        except BacktestingRepositoryError as e:
            self.logger.error(f"Repository error deleting backtest result: {e}")
            raise BacktestingDataServiceError(f"Failed to delete backtest result: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error deleting backtest result: {e}")
            raise BacktestingDataServiceError(f"Failed to delete backtest result: {e}")

    async def get_backtest_history(
        self,
        page: int = 1,
        per_page: int = 10,
        strategy_filter: Optional[str] = None,
        symbol_filter: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        sort_by: str = "executed_at",
        sort_order: str = "desc",
    ) -> BacktestHistoryResponse:
        """
        Get paginated backtest history.

        Args:
            page: Page number (1-based)
            per_page: Items per page
            strategy_filter: Filter by strategy name
            symbol_filter: Filter by trading symbol
            start_date: Filter by execution start date
            end_date: Filter by execution end date
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)

        Returns:
            Paginated backtest history response

        Raises:
            BacktestingDataServiceError: If retrieval fails
        """
        try:
            # Validate pagination parameters
            if page < 1:
                page = 1
            if per_page < 1:
                per_page = 10
            elif per_page > 100:
                per_page = 100

            # Calculate offset
            offset = (page - 1) * per_page

            # Get total count for pagination
            total_count = await self.repository.count_backtest_history(
                strategy_filter=strategy_filter,
                symbol_filter=symbol_filter,
                start_date=start_date,
                end_date=end_date,
            )

            # Get history items
            history_items = await self.repository.list_backtest_history(
                limit=per_page,
                offset=offset,
                strategy_filter=strategy_filter,
                symbol_filter=symbol_filter,
                start_date=start_date,
                end_date=end_date,
                sort_by=sort_by,
                sort_order=sort_order,
            )

            # Calculate total pages
            total_pages = ceil(total_count / per_page) if total_count > 0 else 1

            self.logger.info(
                f"Retrieved {len(history_items)} backtest history items (page {page}/{total_pages})"
            )

            return BacktestHistoryResponse(
                history=history_items,
                count=len(history_items),
                page=page,
                total_pages=total_pages,
                timestamp=datetime.utcnow(),
            )

        except BacktestingRepositoryError as e:
            self.logger.error(f"Repository error retrieving backtest history: {e}")
            raise BacktestingDataServiceError(
                f"Failed to retrieve backtest history: {e}"
            )
        except Exception as e:
            self.logger.error(f"Unexpected error retrieving backtest history: {e}")
            raise BacktestingDataServiceError(
                f"Failed to retrieve backtest history: {e}"
            )

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
            BacktestingDataServiceError: If retrieval fails
        """
        try:
            summary = await self.repository.get_backtest_summary(backtest_id)

            if summary:
                self.logger.info(f"Retrieved backtest summary: {backtest_id}")
            else:
                self.logger.warning(f"Backtest summary not found: {backtest_id}")

            return summary

        except BacktestingRepositoryError as e:
            self.logger.error(f"Repository error retrieving backtest summary: {e}")
            raise BacktestingDataServiceError(
                f"Failed to retrieve backtest summary: {e}"
            )
        except Exception as e:
            self.logger.error(f"Unexpected error retrieving backtest summary: {e}")
            raise BacktestingDataServiceError(
                f"Failed to retrieve backtest summary: {e}"
            )

    async def backtest_exists(self, backtest_id: str) -> bool:
        """
        Check if backtest result exists.

        Args:
            backtest_id: Unique identifier for the backtest

        Returns:
            True if backtest exists

        Raises:
            BacktestingDataServiceError: If check fails
        """
        try:
            exists = await self.repository.backtest_exists(backtest_id)
            return exists

        except BacktestingRepositoryError as e:
            self.logger.error(f"Repository error checking backtest existence: {e}")
            raise BacktestingDataServiceError(
                f"Failed to check backtest existence: {e}"
            )
        except Exception as e:
            self.logger.error(f"Unexpected error checking backtest existence: {e}")
            raise BacktestingDataServiceError(
                f"Failed to check backtest existence: {e}"
            )

    async def cleanup_old_results(
        self, cutoff_date: datetime, dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Clean up old backtest results.

        Args:
            cutoff_date: Delete results older than this date
            dry_run: If True, only count what would be deleted

        Returns:
            Dictionary with cleanup statistics

        Raises:
            BacktestingDataServiceError: If cleanup fails
        """
        try:
            result = await self.repository.cleanup_old_results(
                cutoff_date=cutoff_date, dry_run=dry_run
            )

            if dry_run:
                self.logger.info(
                    f"Dry run cleanup: would delete {result.get('deleted_count', 0)} results"
                )
            else:
                self.logger.info(
                    f"Cleanup completed: deleted {result.get('deleted_count', 0)} results"
                )

            return result

        except BacktestingRepositoryError as e:
            self.logger.error(f"Repository error during cleanup: {e}")
            raise BacktestingDataServiceError(f"Failed to cleanup old results: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during cleanup: {e}")
            raise BacktestingDataServiceError(f"Failed to cleanup old results: {e}")

    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage statistics

        Raises:
            BacktestingDataServiceError: If stats retrieval fails
        """
        try:
            stats = await self.repository.get_storage_stats()
            self.logger.info("Retrieved storage statistics")
            return stats

        except BacktestingRepositoryError as e:
            self.logger.error(f"Repository error getting storage stats: {e}")
            raise BacktestingDataServiceError(f"Failed to get storage stats: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting storage stats: {e}")
            raise BacktestingDataServiceError(f"Failed to get storage stats: {e}")

    async def optimize_storage(self) -> bool:
        """
        Optimize storage for better performance.

        Returns:
            True if optimization was successful

        Raises:
            BacktestingDataServiceError: If optimization fails
        """
        try:
            success = await self.repository.optimize_storage()
            if success:
                self.logger.info("Storage optimization completed successfully")
            else:
                self.logger.warning("Storage optimization failed")
            return success

        except BacktestingRepositoryError as e:
            self.logger.error(f"Repository error optimizing storage: {e}")
            raise BacktestingDataServiceError(f"Failed to optimize storage: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error optimizing storage: {e}")
            raise BacktestingDataServiceError(f"Failed to optimize storage: {e}")

    def _validate_backtest_result(self, result: BacktestResult) -> None:
        """
        Validate backtest result before saving.

        Args:
            result: Backtest result to validate

        Raises:
            BacktestingDataServiceError: If validation fails
        """
        try:
            # Check required fields
            if not result.symbol:
                raise BacktestingDataServiceError("Symbol is required")
            if not result.timeframe:
                raise BacktestingDataServiceError("Timeframe is required")
            if not result.strategy_type:
                raise BacktestingDataServiceError("Strategy type is required")

            # Check date consistency
            if result.start_date >= result.end_date:
                raise BacktestingDataServiceError("Start date must be before end date")

            # Check capital values
            if result.initial_capital <= 0:
                raise BacktestingDataServiceError("Initial capital must be positive")

            # Check execution time
            if result.execution_time_seconds < 0:
                raise BacktestingDataServiceError("Execution time cannot be negative")

            # Validate metrics
            if not result.metrics:
                raise BacktestingDataServiceError("Performance metrics are required")

            self.logger.debug(f"Backtest result validation passed for {result.symbol}")

        except Exception as e:
            self.logger.error(f"Backtest result validation failed: {e}")
            raise BacktestingDataServiceError(f"Invalid backtest result: {e}")

    async def get_strategy_performance_stats(
        self,
        strategy_type: Optional[StrategyType] = None,
        symbol: Optional[str] = None,
        days_back: int = 30,
    ) -> Dict[str, Any]:
        """
        Get performance statistics for strategies.

        Args:
            strategy_type: Filter by strategy type
            symbol: Filter by trading symbol
            days_back: Number of days back to analyze

        Returns:
            Dictionary with performance statistics

        Raises:
            BacktestingDataServiceError: If stats retrieval fails
        """
        try:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date.replace(day=end_date.day - days_back)

            # Get history items
            history_items = await self.repository.list_backtest_history(
                limit=1000,  # Large limit to get all recent results
                offset=0,
                strategy_filter=strategy_type.value if strategy_type else None,
                symbol_filter=symbol,
                start_date=start_date,
                end_date=end_date,
                sort_by="executed_at",
                sort_order="desc",
            )

            if not history_items:
                return {
                    "total_backtests": 0,
                    "strategies": {},
                    "symbols": {},
                    "period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "days": days_back,
                    },
                }

            # Calculate statistics
            stats = {
                "total_backtests": len(history_items),
                "strategies": {},
                "symbols": {},
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days_back,
                },
            }

            # Group by strategy
            for item in history_items:
                strategy_name = item.strategy_type.value
                if strategy_name not in stats["strategies"]:
                    stats["strategies"][strategy_name] = {
                        "count": 0,
                        "avg_return": 0.0,
                        "avg_sharpe": 0.0,
                        "avg_drawdown": 0.0,
                        "win_rate": 0.0,
                    }

                stats["strategies"][strategy_name]["count"] += 1
                stats["strategies"][strategy_name]["avg_return"] += item.total_return
                stats["strategies"][strategy_name]["avg_sharpe"] += (
                    item.sharpe_ratio or 0.0
                )
                stats["strategies"][strategy_name]["avg_drawdown"] += item.max_drawdown
                stats["strategies"][strategy_name]["win_rate"] += item.win_rate

            # Calculate averages
            for strategy_name, strategy_stats in stats["strategies"].items():
                count = strategy_stats["count"]
                strategy_stats["avg_return"] /= count
                strategy_stats["avg_sharpe"] /= count
                strategy_stats["avg_drawdown"] /= count
                strategy_stats["win_rate"] /= count

            # Group by symbol
            for item in history_items:
                symbol_name = item.symbol
                if symbol_name not in stats["symbols"]:
                    stats["symbols"][symbol_name] = {
                        "count": 0,
                        "avg_return": 0.0,
                        "strategies_used": set(),
                    }

                stats["symbols"][symbol_name]["count"] += 1
                stats["symbols"][symbol_name]["avg_return"] += item.total_return
                stats["symbols"][symbol_name]["strategies_used"].add(
                    item.strategy_type.value
                )

            # Calculate averages and convert sets to lists
            for symbol_name, symbol_stats in stats["symbols"].items():
                count = symbol_stats["count"]
                symbol_stats["avg_return"] /= count
                symbol_stats["strategies_used"] = list(symbol_stats["strategies_used"])

            self.logger.info(
                f"Generated performance statistics for {len(history_items)} backtests"
            )
            return stats

        except BacktestingRepositoryError as e:
            self.logger.error(f"Repository error getting performance stats: {e}")
            raise BacktestingDataServiceError(f"Failed to get performance stats: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting performance stats: {e}")
            raise BacktestingDataServiceError(f"Failed to get performance stats: {e}")
