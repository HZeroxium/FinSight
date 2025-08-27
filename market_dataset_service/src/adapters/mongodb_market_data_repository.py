# adapters/mongodb_market_data_repository.py

"""
MongoDB Market Data Repository Implementation

Implements the MarketDataRepository interface using MongoDB for document-based storage.
Provides efficient storage and querying for market data with flexible schema support.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from pymongo import ASCENDING, DESCENDING, MongoClient, ReplaceOne
    from pymongo.collection import Collection
    from pymongo.errors import PyMongoError

    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

from common.logger import LoggerFactory

from ..converters.ohlcv_converter import OHLCVConverter
from ..interfaces.errors import RepositoryError
from ..interfaces.market_data_repository import MarketDataRepository
from ..models.ohlcv_models import OHLCVModelMongoDB
from ..schemas.ohlcv_schemas import (OHLCVBatchSchema, OHLCVQuerySchema,
                                     OHLCVSchema)
from ..utils.datetime_utils import DateTimeUtils


class MongoDBMarketDataRepository(MarketDataRepository):
    """MongoDB implementation of MarketDataRepository for document-based storage"""

    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017/",
        database_name: str = "finsight_market_data",
        ohlcv_collection: str = "ohlcv",
    ):
        """
        Initialize MongoDB repository

        Args:
            connection_string: MongoDB connection string
            database_name: Database name for storing data
            ohlcv_collection: Collection name for OHLCV data
        """
        if not PYMONGO_AVAILABLE:
            raise RepositoryError(
                "PyMongo is not available. Please install it: pip install pymongo"
            )

        self.logger = LoggerFactory.get_logger(name="mongodb_repository")
        self.connection_string = connection_string
        self.database_name = database_name
        self.ohlcv_collection_name = ohlcv_collection
        self.converter = OHLCVConverter()

        try:
            self.client = MongoClient(connection_string)
            self.db = self.client[database_name]
            # Test connection
            self.client.admin.command("ping")

            # Initialize OHLCV collection with proper indexing
            self._setup_ohlcv_collection()

            self.logger.info(f"Connected to MongoDB at {connection_string}")
        except Exception as e:
            self.logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise RepositoryError(f"Failed to connect to MongoDB: {str(e)}")

    def _setup_ohlcv_collection(self) -> None:
        """Setup OHLCV collection with proper indexing"""
        collection = self.db[self.ohlcv_collection_name]

        # Create compound indexes for efficient querying
        # Primary index: exchange + symbol + timeframe + timestamp (unique)
        collection.create_index(
            [
                ("exchange", ASCENDING),
                ("symbol", ASCENDING),
                ("timeframe", ASCENDING),
                ("timestamp", ASCENDING),
            ],
            unique=True,
            name="exchange_symbol_timeframe_timestamp_unique",
        )

        # Query optimization indexes
        collection.create_index(
            [("exchange", ASCENDING), ("symbol", ASCENDING), ("timeframe", ASCENDING)],
            name="exchange_symbol_timeframe",
        )

        collection.create_index([("timestamp", ASCENDING)], name="timestamp")
        collection.create_index([("exchange", ASCENDING)], name="exchange")
        collection.create_index([("symbol", ASCENDING)], name="symbol")
        collection.create_index([("timeframe", ASCENDING)], name="timeframe")

        self.logger.info(
            f"Setup OHLCV collection '{self.ohlcv_collection_name}' with proper indexing"
        )

    def _get_ohlcv_collection(self) -> Collection:
        """Get the OHLCV collection"""
        return self.db[self.ohlcv_collection_name]

    async def save_ohlcv(
        self, exchange: str, symbol: str, timeframe: str, data: List[OHLCVSchema]
    ) -> bool:
        """Save OHLCV data to MongoDB"""

        def _save_sync():
            try:
                collection = self._get_ohlcv_collection()

                # Convert schemas to MongoDB models
                mongo_models = [
                    self.converter.schema_to_mongodb_model(schema) for schema in data
                ]

                # Prepare bulk upsert operations
                operations = []
                for model in mongo_models:
                    # Create filter for unique constraint
                    filter_query = {
                        "exchange": model.exchange,
                        "symbol": model.symbol,
                        "timeframe": model.timeframe,
                        "timestamp": model.timestamp,
                    }

                    operations.append(
                        ReplaceOne(
                            filter_query,
                            model.model_dump(exclude={"id"}),
                            upsert=True,
                        )
                    )

                if operations:
                    result = collection.bulk_write(operations)
                    self.logger.info(
                        f"Saved {len(data)} OHLCV records for {exchange}:{symbol}:{timeframe}"
                    )
                    return True

                return False

            except PyMongoError as e:
                self.logger.error(f"MongoDB error saving OHLCV data: {str(e)}")
                raise RepositoryError(f"Failed to save OHLCV data: {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected error saving OHLCV data: {str(e)}")
                raise RepositoryError(f"Failed to save OHLCV data: {str(e)}")

        return await asyncio.to_thread(_save_sync)

    async def get_ohlcv(self, query: OHLCVQuerySchema) -> List[OHLCVSchema]:
        """Get OHLCV data from MongoDB"""

        def _get_sync():
            try:
                collection = self._get_ohlcv_collection()

                # Build query filter
                filter_dict = {
                    "exchange": query.exchange,
                    "symbol": query.symbol,
                    "timeframe": query.timeframe,
                }

                if query.start_date:
                    filter_dict["timestamp"] = {"$gte": query.start_date}

                if query.end_date:
                    if "timestamp" in filter_dict:
                        filter_dict["timestamp"]["$lte"] = query.end_date
                    else:
                        filter_dict["timestamp"] = {"$lte": query.end_date}

                # Execute query
                cursor = collection.find(filter_dict).sort("timestamp", ASCENDING)

                if query.limit:
                    cursor = cursor.limit(query.limit)

                # Convert to schemas
                results = []
                for doc in cursor:
                    mongo_model = OHLCVModelMongoDB(**doc)
                    schema = self.converter.mongodb_model_to_schema(mongo_model)
                    results.append(schema)

                return results

            except PyMongoError as e:
                self.logger.error(f"MongoDB error getting OHLCV data: {str(e)}")
                raise RepositoryError(f"Failed to get OHLCV data: {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected error getting OHLCV data: {str(e)}")
                raise RepositoryError(f"Failed to get OHLCV data: {str(e)}")

        return await asyncio.to_thread(_get_sync)

    async def delete_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> bool:
        """Delete OHLCV data from MongoDB"""

        def _delete_sync():
            try:
                collection = self._get_ohlcv_collection()

                # Build delete filter
                filter_dict = {
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe,
                }

                if start_date:
                    start_dt = DateTimeUtils.parse_iso_string(start_date)
                    filter_dict["timestamp"] = {"$gte": start_dt}

                if end_date:
                    end_dt = DateTimeUtils.parse_iso_string(end_date)
                    if "timestamp" in filter_dict:
                        filter_dict["timestamp"]["$lte"] = end_dt
                    else:
                        filter_dict["timestamp"] = {"$lte": end_dt}

                result = collection.delete_many(filter_dict)
                self.logger.info(
                    f"Deleted {result.deleted_count} OHLCV records for {exchange}:{symbol}:{timeframe}"
                )
                return True

            except PyMongoError as e:
                self.logger.error(f"MongoDB error deleting OHLCV data: {str(e)}")
                raise RepositoryError(f"Failed to delete OHLCV data: {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected error deleting OHLCV data: {str(e)}")
                raise RepositoryError(f"Failed to delete OHLCV data: {str(e)}")

        return await asyncio.to_thread(_delete_sync)

    # Placeholder implementations for trade/ticker/orderbook operations
    async def save_trades(
        self, exchange: str, symbol: str, data: List[Dict[str, Any]]
    ) -> bool:
        """Save trades data - placeholder implementation"""
        raise NotImplementedError("Trade saving will be implemented in next phase")

    async def get_trades(
        self, exchange: str, symbol: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """Get trades data - placeholder implementation"""
        raise NotImplementedError("Trade retrieval will be implemented in next phase")

    async def save_orderbook(
        self, exchange: str, symbol: str, data: Dict[str, Any]
    ) -> bool:
        """Save orderbook data - placeholder implementation"""
        raise NotImplementedError("Orderbook saving will be implemented in next phase")

    async def get_orderbook(
        self, exchange: str, symbol: str, timestamp: str
    ) -> Optional[Dict[str, Any]]:
        """Get orderbook data - placeholder implementation"""
        raise NotImplementedError(
            "Orderbook retrieval will be implemented in next phase"
        )

    async def save_ticker(
        self, exchange: str, symbol: str, data: Dict[str, Any]
    ) -> bool:
        """Save ticker data - placeholder implementation"""
        raise NotImplementedError("Ticker saving will be implemented in next phase")

    async def get_ticker(
        self, exchange: str, symbol: str, timestamp: str
    ) -> Optional[Dict[str, Any]]:
        """Get ticker data - placeholder implementation"""
        raise NotImplementedError("Ticker retrieval will be implemented in next phase")

    async def get_available_symbols(self, exchange: str) -> List[str]:
        """Get all available symbols for an exchange from MongoDB"""

        def _get_symbols_sync():
            try:
                collection = self._get_ohlcv_collection()

                # Use aggregation to get distinct symbols for the exchange
                pipeline = [
                    {"$match": {"exchange": exchange}},
                    {"$group": {"_id": "$symbol"}},
                    {"$sort": {"_id": 1}},
                ]

                symbols = []
                for result in collection.aggregate(pipeline):
                    symbols.append(result["_id"])

                return sorted(symbols)

            except Exception as e:
                raise RepositoryError(f"Failed to get available symbols: {str(e)}")

        return await asyncio.to_thread(_get_symbols_sync)

    async def get_available_timeframes(self, exchange: str, symbol: str) -> List[str]:
        """Get all available timeframes for a symbol from MongoDB"""

        def _get_timeframes_sync():
            try:
                collection = self._get_ohlcv_collection()

                pipeline = [
                    {"$match": {"exchange": exchange, "symbol": symbol}},
                    {"$group": {"_id": "$timeframe"}},
                    {"$sort": {"_id": 1}},
                ]

                timeframes = []
                for result in collection.aggregate(pipeline):
                    timeframes.append(result["_id"])

                return sorted(timeframes)

            except Exception as e:
                raise RepositoryError(f"Failed to get available timeframes: {str(e)}")

        return await asyncio.to_thread(_get_timeframes_sync)

    async def get_data_range(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        timeframe: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        """Get data range for symbol from MongoDB"""

        def _get_range_sync():
            try:
                if data_type == "ohlcv" and timeframe:
                    collection = self._get_ohlcv_collection()

                    # Get earliest and latest timestamps
                    earliest = collection.find_one(
                        {"exchange": exchange, "symbol": symbol},
                        sort=[("timestamp", ASCENDING)],
                    )
                    latest = collection.find_one(
                        {"exchange": exchange, "symbol": symbol},
                        sort=[("timestamp", DESCENDING)],
                    )

                    if earliest and latest:
                        return {
                            "start_date": DateTimeUtils.to_iso_string(
                                earliest["timestamp"]
                            ),
                            "end_date": DateTimeUtils.to_iso_string(
                                latest["timestamp"]
                            ),
                        }

                return None

            except Exception as e:
                raise RepositoryError(f"Failed to get data range: {str(e)}")

        return await asyncio.to_thread(_get_range_sync)

    async def check_data_exists(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        start_date: str,
        end_date: str,
        timeframe: Optional[str] = None,
    ) -> bool:
        """Check if data exists for specified criteria"""

        def _check_exists_sync():
            try:
                if data_type == "ohlcv" and timeframe:
                    collection = self._get_ohlcv_collection()

                    start_dt = DateTimeUtils.parse_iso_string(start_date)
                    end_dt = DateTimeUtils.parse_iso_string(end_date)

                    count = collection.count_documents(
                        {
                            "exchange": exchange,
                            "symbol": symbol,
                            "timestamp": {"$gte": start_dt, "$lte": end_dt},
                        }
                    )

                    return count > 0

                return False

            except Exception as e:
                raise RepositoryError(f"Failed to check data exists: {str(e)}")

        return await asyncio.to_thread(_check_exists_sync)

    async def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information for MongoDB"""

        def _get_storage_info_sync():
            try:
                # Get database stats
                stats = self.db.command("dbstats")

                # Get OHLCV collection
                collection = self._get_ohlcv_collection()

                # Count total documents
                total_docs = collection.count_documents({})

                return {
                    "storage_type": "database",
                    "location": self.connection_string,
                    "database_name": self.database_name,
                    "collection_name": self.ohlcv_collection_name,
                    "total_size": stats.get("dataSize", 0),
                    "total_documents": total_docs,
                    "available_exchanges": [],  # Will be populated separately
                    "storage_engine": "MongoDB",
                }

            except Exception as e:
                raise RepositoryError(f"Failed to get storage info: {str(e)}")

        return await asyncio.to_thread(_get_storage_info_sync)

    async def batch_save_ohlcv(self, data: List[OHLCVBatchSchema]) -> bool:
        """Batch save OHLCV data"""

        def _batch_save_sync():
            try:
                for batch in data:
                    # This will call save_ohlcv for each batch
                    asyncio.create_task(
                        self.save_ohlcv(
                            batch.exchange, batch.symbol, batch.timeframe, batch.records
                        )
                    )

                return True

            except Exception as e:
                raise RepositoryError(f"Failed to batch save OHLCV data: {str(e)}")

        return await asyncio.to_thread(_batch_save_sync)

    async def optimize_storage(self) -> bool:
        """Optimize MongoDB storage"""

        def _optimize_sync():
            try:
                # Rebuild indexes for OHLCV collection
                collection = self._get_ohlcv_collection()
                collection.reindex()

                self.logger.info("MongoDB storage optimization completed")
                return True

            except Exception as e:
                raise RepositoryError(f"Failed to optimize storage: {str(e)}")

        return await asyncio.to_thread(_optimize_sync)

    # Administrative Operations
    async def count_all_records(self) -> int:
        """Count total number of OHLCV records in repository"""

        def _count_all_sync():
            try:
                collection = self._get_ohlcv_collection()
                total_count = collection.count_documents({})
                return total_count

            except Exception as e:
                raise RepositoryError(f"Failed to count all records: {str(e)}")

        return await asyncio.to_thread(_count_all_sync)

    async def get_all_available_symbols(self) -> List[str]:
        """Get all available symbols across all exchanges"""

        def _get_all_symbols_sync():
            try:
                collection = self._get_ohlcv_collection()

                pipeline = [
                    {"$group": {"_id": "$symbol"}},
                    {"$sort": {"_id": 1}},
                ]

                symbols = []
                for result in collection.aggregate(pipeline):
                    symbols.append(result["_id"])

                return sorted(symbols)

            except Exception as e:
                raise RepositoryError(f"Failed to get all available symbols: {str(e)}")

        return await asyncio.to_thread(_get_all_symbols_sync)

    async def get_available_exchanges(self) -> List[str]:
        """Get all available exchanges in repository"""

        def _get_exchanges_sync():
            try:
                collection = self._get_ohlcv_collection()

                pipeline = [
                    {"$group": {"_id": "$exchange"}},
                    {"$sort": {"_id": 1}},
                ]

                exchanges = []
                for result in collection.aggregate(pipeline):
                    exchanges.append(result["_id"])

                return sorted(exchanges)

            except Exception as e:
                raise RepositoryError(f"Failed to get available exchanges: {str(e)}")

        return await asyncio.to_thread(_get_exchanges_sync)

    async def get_all_available_timeframes(self) -> List[str]:
        """Get all available timeframes across all data"""

        def _get_all_timeframes_sync():
            try:
                collection = self._get_ohlcv_collection()

                pipeline = [
                    {"$group": {"_id": "$timeframe"}},
                    {"$sort": {"_id": 1}},
                ]

                timeframes = []
                for result in collection.aggregate(pipeline):
                    timeframes.append(result["_id"])

                return sorted(timeframes)

            except Exception as e:
                raise RepositoryError(
                    f"Failed to get all available timeframes: {str(e)}"
                )

        return await asyncio.to_thread(_get_all_timeframes_sync)

    async def count_records(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """Count records for specific criteria"""

        def _count_records_sync():
            try:
                collection = self._get_ohlcv_collection()

                filter_dict = {
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe,
                }

                if start_date:
                    filter_dict["timestamp"] = {"$gte": start_date}
                if end_date:
                    if "timestamp" in filter_dict:
                        filter_dict["timestamp"]["$lte"] = end_date
                    else:
                        filter_dict["timestamp"] = {"$lte": end_date}

                return collection.count_documents(filter_dict)

            except Exception as e:
                raise RepositoryError(f"Failed to count records: {str(e)}")

        return await asyncio.to_thread(_count_records_sync)

    async def count_records_since(self, cutoff_date: datetime) -> int:
        """Count records since a specific date"""

        def _count_since_sync():
            try:
                collection = self._get_ohlcv_collection()
                count = collection.count_documents({"timestamp": {"$gte": cutoff_date}})
                return count

            except Exception as e:
                raise RepositoryError(f"Failed to count records since date: {str(e)}")

        return await asyncio.to_thread(_count_since_sync)

    async def delete_records_before_date(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        cutoff_date: datetime,
    ) -> int:
        """Delete records before a specific date"""

        def _delete_before_sync():
            try:
                collection = self._get_ohlcv_collection()

                result = collection.delete_many(
                    {
                        "exchange": exchange,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "timestamp": {"$lt": cutoff_date},
                    }
                )

                return result.deleted_count

            except Exception as e:
                raise RepositoryError(f"Failed to delete records before date: {str(e)}")

        return await asyncio.to_thread(_delete_before_sync)

    def close(self) -> None:
        """Close MongoDB client connection"""
        if hasattr(self, "client"):
            self.client.close()
            self.logger.info("Closed MongoDB connection")
