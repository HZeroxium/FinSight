# adapters/file_backtesting_repository.py

"""
File-based Backtesting Repository Implementation

Stores backtest results and history in JSON files on the filesystem.
Provides fast local storage for development and testing.
"""

import json
import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..interfaces.backtesting_repository import (
    BacktestingRepository,
    BacktestingRepositoryError,
)
from ..schemas.backtesting_schemas import (
    BacktestResult,
    BacktestHistoryItem,
    StrategyType,
)
from ..common.logger import LoggerFactory


class FileBacktestingRepository(BacktestingRepository):
    """File-based implementation of BacktestingRepository."""

    def __init__(self, base_directory: str = "data/backtests"):
        """
        Initialize file-based backtesting repository.

        Args:
            base_directory: Base directory for storing backtest files
        """
        self.base_directory = Path(base_directory)
        self.results_directory = self.base_directory / "results"
        self.history_directory = self.base_directory / "history"
        self.logger = LoggerFactory.get_logger(name="file_backtesting_repository")

        # Create directories if they don't exist
        self._ensure_directories()

        self.logger.info(
            f"FileBacktestingRepository initialized with base directory: {base_directory}"
        )

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        try:
            self.base_directory.mkdir(parents=True, exist_ok=True)
            self.results_directory.mkdir(parents=True, exist_ok=True)
            self.history_directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise BacktestingRepositoryError(f"Failed to create directories: {e}")

    def _get_result_file_path(self, backtest_id: str) -> Path:
        """Get file path for backtest result."""
        return self.results_directory / f"{backtest_id}.json"

    def _get_history_file_path(self, backtest_id: str) -> Path:
        """Get file path for backtest history item."""
        return self.history_directory / f"{backtest_id}.json"

    def _serialize_datetime(self, obj: Any) -> Any:
        """Custom JSON serializer for datetime objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _deserialize_datetime(self, date_str: str) -> datetime:
        """Deserialize datetime from ISO format string."""
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))

    async def save_backtest_result(
        self,
        backtest_id: str,
        result: BacktestResult,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save backtest result to file."""
        try:
            # Save full result
            result_file = self._get_result_file_path(backtest_id)
            result_data = {
                "backtest_id": backtest_id,
                "result": result.model_dump(),
                "metadata": metadata or {},
                "saved_at": datetime.utcnow().isoformat(),
            }

            with open(result_file, "w") as f:
                json.dump(result_data, f, indent=2, default=self._serialize_datetime)

            # Save history summary
            history_item = BacktestHistoryItem(
                backtest_id=backtest_id,
                symbol=result.symbol,
                timeframe=result.timeframe,
                strategy_type=result.strategy_type,
                total_return=result.metrics.total_return,
                sharpe_ratio=result.metrics.sharpe_ratio,
                max_drawdown=result.metrics.max_drawdown,
                win_rate=result.metrics.win_rate,
                start_date=result.start_date,
                end_date=result.end_date,
                executed_at=datetime.utcnow(),
                execution_time_seconds=result.execution_time_seconds,
                status="completed",
            )

            history_file = self._get_history_file_path(backtest_id)
            history_data = {
                "backtest_id": backtest_id,
                "history_item": history_item.model_dump(),
                "saved_at": datetime.utcnow().isoformat(),
            }

            with open(history_file, "w") as f:
                json.dump(history_data, f, indent=2, default=self._serialize_datetime)

            self.logger.info(f"Saved backtest result: {backtest_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save backtest result {backtest_id}: {e}")
            raise BacktestingRepositoryError(f"Failed to save backtest result: {e}")

    async def get_backtest_result(
        self,
        backtest_id: str,
        include_trades: bool = True,
        include_equity_curve: bool = True,
    ) -> Optional[BacktestResult]:
        """Retrieve backtest result from file."""
        try:
            result_file = self._get_result_file_path(backtest_id)

            if not result_file.exists():
                return None

            with open(result_file, "r") as f:
                data = json.load(f)

            result_data = data["result"]

            # Filter out trades and equity curve if not requested
            if not include_trades:
                result_data["trades"] = []
            if not include_equity_curve:
                result_data["equity_curve"] = []

            # Convert datetime strings back to datetime objects
            for field in ["start_date", "end_date"]:
                if field in result_data:
                    result_data[field] = self._deserialize_datetime(result_data[field])

            # Convert trade dates
            if include_trades:
                for trade in result_data.get("trades", []):
                    trade["entry_date"] = self._deserialize_datetime(
                        trade["entry_date"]
                    )
                    if trade.get("exit_date"):
                        trade["exit_date"] = self._deserialize_datetime(
                            trade["exit_date"]
                        )

            # Convert equity curve dates
            if include_equity_curve:
                for point in result_data.get("equity_curve", []):
                    point["timestamp"] = self._deserialize_datetime(point["timestamp"])

            result = BacktestResult(**result_data)
            self.logger.info(f"Retrieved backtest result: {backtest_id}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to retrieve backtest result {backtest_id}: {e}")
            raise BacktestingRepositoryError(f"Failed to retrieve backtest result: {e}")

    async def delete_backtest_result(self, backtest_id: str) -> bool:
        """Delete backtest result files."""
        try:
            result_file = self._get_result_file_path(backtest_id)
            history_file = self._get_history_file_path(backtest_id)

            deleted = False
            if result_file.exists():
                result_file.unlink()
                deleted = True

            if history_file.exists():
                history_file.unlink()
                deleted = True

            if deleted:
                self.logger.info(f"Deleted backtest result: {backtest_id}")
            else:
                self.logger.warning(
                    f"Backtest result not found for deletion: {backtest_id}"
                )

            return deleted

        except Exception as e:
            self.logger.error(f"Failed to delete backtest result {backtest_id}: {e}")
            raise BacktestingRepositoryError(f"Failed to delete backtest result: {e}")

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
        """List backtest history from files."""
        try:
            history_items = []

            # Read all history files
            for history_file in self.history_directory.glob("*.json"):
                try:
                    with open(history_file, "r") as f:
                        data = json.load(f)

                    item_data = data["history_item"]

                    # Convert datetime strings
                    for field in ["start_date", "end_date", "executed_at"]:
                        if field in item_data:
                            item_data[field] = self._deserialize_datetime(
                                item_data[field]
                            )

                    history_item = BacktestHistoryItem(**item_data)

                    # Apply filters
                    if (
                        strategy_filter
                        and history_item.strategy_type.value != strategy_filter
                    ):
                        continue
                    if symbol_filter and history_item.symbol != symbol_filter:
                        continue
                    if start_date and history_item.executed_at < start_date:
                        continue
                    if end_date and history_item.executed_at > end_date:
                        continue

                    history_items.append(history_item)

                except Exception as e:
                    self.logger.warning(
                        f"Failed to load history file {history_file}: {e}"
                    )
                    continue

            # Sort items
            reverse = sort_order.lower() == "desc"
            if sort_by in ["executed_at", "start_date", "end_date"]:
                history_items.sort(key=lambda x: getattr(x, sort_by), reverse=reverse)
            elif sort_by in [
                "total_return",
                "sharpe_ratio",
                "max_drawdown",
                "win_rate",
            ]:
                history_items.sort(
                    key=lambda x: getattr(x, sort_by) or 0, reverse=reverse
                )
            else:
                history_items.sort(
                    key=lambda x: getattr(x, sort_by, ""), reverse=reverse
                )

            # Apply pagination
            start_idx = offset
            end_idx = offset + limit
            result = history_items[start_idx:end_idx]

            self.logger.info(f"Retrieved {len(result)} backtest history items")
            return result

        except Exception as e:
            self.logger.error(f"Failed to list backtest history: {e}")
            raise BacktestingRepositoryError(f"Failed to list backtest history: {e}")

    async def count_backtest_history(
        self,
        strategy_filter: Optional[str] = None,
        symbol_filter: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """Count backtest history items."""
        try:
            count = 0

            for history_file in self.history_directory.glob("*.json"):
                try:
                    with open(history_file, "r") as f:
                        data = json.load(f)

                    item_data = data["history_item"]

                    # Apply filters
                    if (
                        strategy_filter
                        and item_data.get("strategy_type") != strategy_filter
                    ):
                        continue
                    if symbol_filter and item_data.get("symbol") != symbol_filter:
                        continue
                    if start_date:
                        executed_at = self._deserialize_datetime(
                            item_data.get("executed_at")
                        )
                        if executed_at < start_date:
                            continue
                    if end_date:
                        executed_at = self._deserialize_datetime(
                            item_data.get("executed_at")
                        )
                        if executed_at > end_date:
                            continue

                    count += 1

                except Exception as e:
                    self.logger.warning(
                        f"Failed to count history file {history_file}: {e}"
                    )
                    continue

            return count

        except Exception as e:
            self.logger.error(f"Failed to count backtest history: {e}")
            raise BacktestingRepositoryError(f"Failed to count backtest history: {e}")

    async def backtest_exists(self, backtest_id: str) -> bool:
        """Check if backtest result exists."""
        result_file = self._get_result_file_path(backtest_id)
        return result_file.exists()

    async def get_backtest_summary(
        self, backtest_id: str
    ) -> Optional[BacktestHistoryItem]:
        """Get backtest summary from history file."""
        try:
            history_file = self._get_history_file_path(backtest_id)

            if not history_file.exists():
                return None

            with open(history_file, "r") as f:
                data = json.load(f)

            item_data = data["history_item"]

            # Convert datetime strings
            for field in ["start_date", "end_date", "executed_at"]:
                if field in item_data:
                    item_data[field] = self._deserialize_datetime(item_data[field])

            return BacktestHistoryItem(**item_data)

        except Exception as e:
            self.logger.error(f"Failed to get backtest summary {backtest_id}: {e}")
            raise BacktestingRepositoryError(f"Failed to get backtest summary: {e}")

    async def cleanup_old_results(
        self, cutoff_date: datetime, dry_run: bool = False
    ) -> Dict[str, Any]:
        """Clean up old backtest results."""
        try:
            deleted_count = 0
            total_size = 0
            errors = []

            for history_file in self.history_directory.glob("*.json"):
                try:
                    with open(history_file, "r") as f:
                        data = json.load(f)

                    item_data = data["history_item"]
                    executed_at = self._deserialize_datetime(
                        item_data.get("executed_at")
                    )

                    if executed_at < cutoff_date:
                        backtest_id = item_data["backtest_id"]

                        if not dry_run:
                            result_file = self._get_result_file_path(backtest_id)

                            # Get file sizes
                            if result_file.exists():
                                total_size += result_file.stat().st_size
                            if history_file.exists():
                                total_size += history_file.stat().st_size

                            # Delete files
                            if result_file.exists():
                                result_file.unlink()
                            if history_file.exists():
                                history_file.unlink()

                        deleted_count += 1

                except Exception as e:
                    errors.append(f"Failed to process {history_file}: {e}")
                    continue

            return {
                "deleted_count": deleted_count,
                "total_size_bytes": total_size,
                "errors": errors,
                "dry_run": dry_run,
                "cutoff_date": cutoff_date.isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to cleanup old results: {e}")
            raise BacktestingRepositoryError(f"Failed to cleanup old results: {e}")

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            result_files = list(self.results_directory.glob("*.json"))
            history_files = list(self.history_directory.glob("*.json"))

            total_size = 0
            for file_path in result_files + history_files:
                total_size += file_path.stat().st_size

            return {
                "total_results": len(result_files),
                "total_history_items": len(history_files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "storage_type": "file_system",
                "base_directory": str(self.base_directory),
                "results_directory": str(self.results_directory),
                "history_directory": str(self.history_directory),
            }

        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {e}")
            raise BacktestingRepositoryError(f"Failed to get storage stats: {e}")

    async def optimize_storage(self) -> bool:
        """Optimize storage (no-op for file system)."""
        # For file system, we could implement compression or cleanup
        # For now, just return True
        return True
