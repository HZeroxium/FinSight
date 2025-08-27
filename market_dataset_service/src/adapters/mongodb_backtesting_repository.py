# adapters/mongodb_backtesting_repository.py

"""
MongoDB Backtesting Repository Implementation

Stores backtest results and history in MongoDB collections.
Provides scalable and queryable storage for production use.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from common.logger import LoggerFactory
from motor.motor_asyncio import (AsyncIOMotorClient, AsyncIOMotorCollection,
                                 AsyncIOMotorDatabase)
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

from ..interfaces.backtesting_repository import (BacktestingRepository,
                                                 BacktestingRepositoryError)
from ..schemas.backtesting_schemas import BacktestHistoryItem, BacktestResult


class MongoDBBacktestingRepository(BacktestingRepository):
    """MongoDB implementation of BacktestingRepository."""

    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017/",
        database_name: str = "finsight_backtesting",
        results_collection: str = "backtest_results",
        history_collection: str = "backtest_history",
    ):
        """
        Initialize MongoDB backtesting repository.

        Args:
            connection_string: MongoDB connection string
            database_name: Database name
            results_collection: Collection name for backtest results
            history_collection: Collection name for backtest history
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.results_collection_name = results_collection
        self.history_collection_name = history_collection
        self.logger = LoggerFactory.get_logger(name="mongodb_backtesting_repository")

        # Initialize MongoDB client
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.results_collection: Optional[AsyncIOMotorCollection] = None
        self.history_collection: Optional[AsyncIOMotorCollection] = None

        self.logger.info(
            f"MongoDBBacktestingRepository initialized for database: {database_name}"
        )

    async def _ensure_connection(self) -> None:
        """Ensure MongoDB connection is established."""
        if self.client is None:
            try:
                self.client = AsyncIOMotorClient(self.connection_string)
                self.database = self.client[self.database_name]
                self.results_collection = self.database[self.results_collection_name]
                self.history_collection = self.database[self.history_collection_name]

                # Create indexes for better query performance
                await self._create_indexes()

                self.logger.info("MongoDB connection established")
            except Exception as e:
                self.logger.error(f"Failed to connect to MongoDB: {e}")
                raise BacktestingRepositoryError(f"Failed to connect to MongoDB: {e}")

    async def _create_indexes(self) -> None:
        """Create indexes for better query performance."""
        try:
            # Index on backtest_id for both collections
            await self.results_collection.create_index("backtest_id", unique=True)
            await self.history_collection.create_index("backtest_id", unique=True)

            # Indexes for history collection queries
            await self.history_collection.create_index("executed_at")
            await self.history_collection.create_index("strategy_type")
            await self.history_collection.create_index("symbol")
            await self.history_collection.create_index("start_date")
            await self.history_collection.create_index("end_date")

            # Compound indexes for common queries
            await self.history_collection.create_index(
                [("strategy_type", ASCENDING), ("executed_at", DESCENDING)]
            )
            await self.history_collection.create_index(
                [("symbol", ASCENDING), ("executed_at", DESCENDING)]
            )

            self.logger.info("MongoDB indexes created successfully")
        except Exception as e:
            self.logger.warning(f"Failed to create indexes: {e}")

    def _convert_backtest_result_to_dict(
        self, result: BacktestResult
    ) -> Dict[str, Any]:
        """Convert BacktestResult to MongoDB document."""
        doc = result.model_dump()

        # Convert datetime objects to ISO strings for MongoDB
        for field in ["start_date", "end_date"]:
            if field in doc and isinstance(doc[field], datetime):
                doc[field] = doc[field].isoformat()

        # Convert trade dates
        for trade in doc.get("trades", []):
            if "entry_date" in trade and isinstance(trade["entry_date"], datetime):
                trade["entry_date"] = trade["entry_date"].isoformat()
            if "exit_date" in trade and isinstance(trade["exit_date"], datetime):
                trade["exit_date"] = trade["exit_date"].isoformat()

        # Convert equity curve timestamps
        for point in doc.get("equity_curve", []):
            if "timestamp" in point and isinstance(point["timestamp"], datetime):
                point["timestamp"] = point["timestamp"].isoformat()

        return doc

    def _convert_dict_to_backtest_result(self, doc: Dict[str, Any]) -> BacktestResult:
        """Convert MongoDB document to BacktestResult."""
        # Convert datetime strings back to datetime objects
        for field in ["start_date", "end_date"]:
            if field in doc and isinstance(doc[field], str):
                doc[field] = datetime.fromisoformat(doc[field])

        # Convert trade dates
        for trade in doc.get("trades", []):
            if "entry_date" in trade and isinstance(trade["entry_date"], str):
                trade["entry_date"] = datetime.fromisoformat(trade["entry_date"])
            if "exit_date" in trade and isinstance(trade["exit_date"], str):
                trade["exit_date"] = datetime.fromisoformat(trade["exit_date"])

        # Convert equity curve timestamps
        for point in doc.get("equity_curve", []):
            if "timestamp" in point and isinstance(point["timestamp"], str):
                point["timestamp"] = datetime.fromisoformat(point["timestamp"])

        # Remove MongoDB _id field
        doc.pop("_id", None)

        return BacktestResult(**doc)

    async def save_backtest_result(
        self,
        backtest_id: str,
        result: BacktestResult,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save backtest result to MongoDB."""
        try:
            await self._ensure_connection()

            # Prepare result document
            result_doc = self._convert_backtest_result_to_dict(result)
            result_doc.update(
                {
                    "backtest_id": backtest_id,
                    "metadata": metadata or {},
                    "saved_at": datetime.utcnow().isoformat(),
                }
            )

            # Save full result
            await self.results_collection.replace_one(
                {"backtest_id": backtest_id}, result_doc, upsert=True
            )

            # Prepare history document
            history_doc = {
                "backtest_id": backtest_id,
                "symbol": result.symbol,
                "timeframe": result.timeframe,
                "strategy_type": result.strategy_type.value,
                "total_return": result.metrics.total_return,
                "sharpe_ratio": result.metrics.sharpe_ratio,
                "max_drawdown": result.metrics.max_drawdown,
                "win_rate": result.metrics.win_rate,
                "start_date": result.start_date.isoformat(),
                "end_date": result.end_date.isoformat(),
                "executed_at": datetime.utcnow().isoformat(),
                "execution_time_seconds": result.execution_time_seconds,
                "status": "completed",
                "saved_at": datetime.utcnow().isoformat(),
            }

            # Save history summary
            await self.history_collection.replace_one(
                {"backtest_id": backtest_id}, history_doc, upsert=True
            )

            self.logger.info(f"Saved backtest result to MongoDB: {backtest_id}")
            return True

        except PyMongoError as e:
            self.logger.error(
                f"MongoDB error saving backtest result {backtest_id}: {e}"
            )
            raise BacktestingRepositoryError(f"Failed to save backtest result: {e}")
        except Exception as e:
            self.logger.error(f"Failed to save backtest result {backtest_id}: {e}")
            raise BacktestingRepositoryError(f"Failed to save backtest result: {e}")

    async def get_backtest_result(
        self,
        backtest_id: str,
        include_trades: bool = True,
        include_equity_curve: bool = True,
    ) -> Optional[BacktestResult]:
        """Retrieve backtest result from MongoDB."""
        try:
            await self._ensure_connection()

            # Build projection to exclude unnecessary fields
            projection = {}
            if not include_trades:
                projection["trades"] = 0
            if not include_equity_curve:
                projection["equity_curve"] = 0

            doc = await self.results_collection.find_one(
                {"backtest_id": backtest_id}, projection if projection else None
            )

            if doc is None:
                return None

            result = self._convert_dict_to_backtest_result(doc)
            self.logger.info(f"Retrieved backtest result from MongoDB: {backtest_id}")
            return result

        except PyMongoError as e:
            self.logger.error(
                f"MongoDB error retrieving backtest result {backtest_id}: {e}"
            )
            raise BacktestingRepositoryError(f"Failed to retrieve backtest result: {e}")
        except Exception as e:
            self.logger.error(f"Failed to retrieve backtest result {backtest_id}: {e}")
            raise BacktestingRepositoryError(f"Failed to retrieve backtest result: {e}")

    async def delete_backtest_result(self, backtest_id: str) -> bool:
        """Delete backtest result from MongoDB."""
        try:
            await self._ensure_connection()

            # Delete from both collections
            result_delete = await self.results_collection.delete_one(
                {"backtest_id": backtest_id}
            )
            history_delete = await self.history_collection.delete_one(
                {"backtest_id": backtest_id}
            )

            deleted = (
                result_delete.deleted_count > 0 or history_delete.deleted_count > 0
            )

            if deleted:
                self.logger.info(f"Deleted backtest result from MongoDB: {backtest_id}")
            else:
                self.logger.warning(
                    f"Backtest result not found for deletion: {backtest_id}"
                )

            return deleted

        except PyMongoError as e:
            self.logger.error(
                f"MongoDB error deleting backtest result {backtest_id}: {e}"
            )
            raise BacktestingRepositoryError(f"Failed to delete backtest result: {e}")
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
        """List backtest history from MongoDB."""
        try:
            await self._ensure_connection()

            # Build filter query
            filter_query = {}
            if strategy_filter:
                filter_query["strategy_type"] = strategy_filter
            if symbol_filter:
                filter_query["symbol"] = symbol_filter
            if start_date:
                filter_query["executed_at"] = {"$gte": start_date.isoformat()}
            if end_date:
                if "executed_at" not in filter_query:
                    filter_query["executed_at"] = {}
                filter_query["executed_at"]["$lte"] = end_date.isoformat()

            # Build sort order
            sort_direction = DESCENDING if sort_order.lower() == "desc" else ASCENDING
            sort_query = [(sort_by, sort_direction)]

            # Execute query
            cursor = (
                self.history_collection.find(filter_query)
                .sort(sort_query)
                .skip(offset)
                .limit(limit)
            )
            documents = await cursor.to_list(length=limit)

            # Convert documents to BacktestHistoryItem
            history_items = []
            for doc in documents:
                # Convert datetime strings back to datetime objects
                for field in ["start_date", "end_date", "executed_at"]:
                    if field in doc and isinstance(doc[field], str):
                        doc[field] = datetime.fromisoformat(doc[field])

                # Remove MongoDB _id field
                doc.pop("_id", None)
                doc.pop("saved_at", None)

                history_item = BacktestHistoryItem(**doc)
                history_items.append(history_item)

            self.logger.info(
                f"Retrieved {len(history_items)} backtest history items from MongoDB"
            )
            return history_items

        except PyMongoError as e:
            self.logger.error(f"MongoDB error listing backtest history: {e}")
            raise BacktestingRepositoryError(f"Failed to list backtest history: {e}")
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
        """Count backtest history items in MongoDB."""
        try:
            await self._ensure_connection()

            # Build filter query
            filter_query = {}
            if strategy_filter:
                filter_query["strategy_type"] = strategy_filter
            if symbol_filter:
                filter_query["symbol"] = symbol_filter
            if start_date:
                filter_query["executed_at"] = {"$gte": start_date.isoformat()}
            if end_date:
                if "executed_at" not in filter_query:
                    filter_query["executed_at"] = {}
                filter_query["executed_at"]["$lte"] = end_date.isoformat()

            count = await self.history_collection.count_documents(filter_query)
            return count

        except PyMongoError as e:
            self.logger.error(f"MongoDB error counting backtest history: {e}")
            raise BacktestingRepositoryError(f"Failed to count backtest history: {e}")
        except Exception as e:
            self.logger.error(f"Failed to count backtest history: {e}")
            raise BacktestingRepositoryError(f"Failed to count backtest history: {e}")

    async def backtest_exists(self, backtest_id: str) -> bool:
        """Check if backtest result exists in MongoDB."""
        try:
            await self._ensure_connection()

            count = await self.results_collection.count_documents(
                {"backtest_id": backtest_id}
            )
            return count > 0

        except PyMongoError as e:
            self.logger.error(
                f"MongoDB error checking backtest existence {backtest_id}: {e}"
            )
            raise BacktestingRepositoryError(f"Failed to check backtest existence: {e}")
        except Exception as e:
            self.logger.error(f"Failed to check backtest existence {backtest_id}: {e}")
            raise BacktestingRepositoryError(f"Failed to check backtest existence: {e}")

    async def get_backtest_summary(
        self, backtest_id: str
    ) -> Optional[BacktestHistoryItem]:
        """Get backtest summary from MongoDB."""
        try:
            await self._ensure_connection()

            doc = await self.history_collection.find_one({"backtest_id": backtest_id})

            if doc is None:
                return None

            # Convert datetime strings back to datetime objects
            for field in ["start_date", "end_date", "executed_at"]:
                if field in doc and isinstance(doc[field], str):
                    doc[field] = datetime.fromisoformat(doc[field])

            # Remove MongoDB _id field
            doc.pop("_id", None)
            doc.pop("saved_at", None)

            return BacktestHistoryItem(**doc)

        except PyMongoError as e:
            self.logger.error(
                f"MongoDB error getting backtest summary {backtest_id}: {e}"
            )
            raise BacktestingRepositoryError(f"Failed to get backtest summary: {e}")
        except Exception as e:
            self.logger.error(f"Failed to get backtest summary {backtest_id}: {e}")
            raise BacktestingRepositoryError(f"Failed to get backtest summary: {e}")

    async def cleanup_old_results(
        self, cutoff_date: datetime, dry_run: bool = False
    ) -> Dict[str, Any]:
        """Clean up old backtest results from MongoDB."""
        try:
            await self._ensure_connection()

            cutoff_iso = cutoff_date.isoformat()
            filter_query = {"executed_at": {"$lt": cutoff_iso}}

            if dry_run:
                # Count documents that would be deleted
                results_count = await self.results_collection.count_documents(
                    filter_query
                )
                history_count = await self.history_collection.count_documents(
                    filter_query
                )

                return {
                    "deleted_count": results_count,
                    "history_deleted_count": history_count,
                    "errors": [],
                    "dry_run": True,
                    "cutoff_date": cutoff_iso,
                }
            else:
                # Delete documents
                results_delete = await self.results_collection.delete_many(filter_query)
                history_delete = await self.history_collection.delete_many(filter_query)

                return {
                    "deleted_count": results_delete.deleted_count,
                    "history_deleted_count": history_delete.deleted_count,
                    "errors": [],
                    "dry_run": False,
                    "cutoff_date": cutoff_iso,
                }

        except PyMongoError as e:
            self.logger.error(f"MongoDB error cleaning up old results: {e}")
            raise BacktestingRepositoryError(f"Failed to cleanup old results: {e}")
        except Exception as e:
            self.logger.error(f"Failed to cleanup old results: {e}")
            raise BacktestingRepositoryError(f"Failed to cleanup old results: {e}")

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics from MongoDB."""
        try:
            await self._ensure_connection()

            # Get collection stats
            results_count = await self.results_collection.count_documents({})
            history_count = await self.history_collection.count_documents({})

            # Get database stats
            db_stats = await self.database.command("dbStats")

            return {
                "total_results": results_count,
                "total_history_items": history_count,
                "total_size_bytes": db_stats.get("dataSize", 0),
                "total_size_mb": round(db_stats.get("dataSize", 0) / (1024 * 1024), 2),
                "storage_type": "mongodb",
                "database_name": self.database_name,
                "collections": {
                    "results": self.results_collection_name,
                    "history": self.history_collection_name,
                },
                "indexes": db_stats.get("indexes", 0),
                "index_size_bytes": db_stats.get("indexSize", 0),
            }

        except PyMongoError as e:
            self.logger.error(f"MongoDB error getting storage stats: {e}")
            raise BacktestingRepositoryError(f"Failed to get storage stats: {e}")
        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {e}")
            raise BacktestingRepositoryError(f"Failed to get storage stats: {e}")

    async def optimize_storage(self) -> bool:
        """Optimize storage in MongoDB."""
        try:
            await self._ensure_connection()

            # Rebuild indexes for better performance
            await self.results_collection.reindex()
            await self.history_collection.reindex()

            self.logger.info("MongoDB storage optimization completed")
            return True

        except PyMongoError as e:
            self.logger.error(f"MongoDB error optimizing storage: {e}")
            raise BacktestingRepositoryError(f"Failed to optimize storage: {e}")
        except Exception as e:
            self.logger.error(f"Failed to optimize storage: {e}")
            raise BacktestingRepositoryError(f"Failed to optimize storage: {e}")

    async def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.database = None
            self.results_collection = None
            self.history_collection = None
            self.logger.info("MongoDB connection closed")

    def __del__(self):
        """Cleanup on deletion."""
        if self.client:
            self.client.close()
