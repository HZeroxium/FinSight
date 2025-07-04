# adapters/mongodb_market_data_repository.py

"""
MongoDB Market Data Repository Implementation

Implements the MarketDataRepository interface using MongoDB for document-based storage.
Provides efficient storage and querying for market data with flexible schema support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

try:
    from pymongo import MongoClient, ASCENDING, DESCENDING, ReplaceOne
    from pymongo.errors import PyMongoError
    from pymongo.collection import Collection
    from pymongo.database import Database

    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

from ..interfaces.market_data_repository import MarketDataRepository
from ..interfaces.errors import RepositoryError
from ..common.logger import LoggerFactory
from ..schemas.ohlcv_schemas import OHLCVSchema, OHLCVBatchSchema, OHLCVQuerySchema
from ..models.ohlcv_models import OHLCVModelMongoDB
from ..converters.ohlcv_converter import OHLCVConverter
from ..utils.datetime_utils import DateTimeUtils


class MongoDBMarketDataRepository(MarketDataRepository):
    """MongoDB implementation of MarketDataRepository for document-based storage"""

    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017/",
        database_name: str = "finsight_market_data",
        collection_prefix: str = "ohlcv",
    ):
        """
        Initialize MongoDB repository

        Args:
            connection_string: MongoDB connection string
            database_name: Database name for storing data
            collection_prefix: Prefix for collection names
        """
        if not PYMONGO_AVAILABLE:
            raise RepositoryError(
                "PyMongo not available. Install with: pip install pymongo"
            )

        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_prefix = collection_prefix

        self.logger = LoggerFactory.get_logger(name="mongodb_repository")
        self.converter = OHLCVConverter()

        # Initialize client and database
        try:
            self.client = MongoClient(connection_string)
            self.db: Database = self.client[database_name]

            # Test connection
            self._test_connection()

            # Initialize indexes
            self._initialize_indexes()

            self.logger.info(f"Initialized MongoDB repository at {connection_string}")

        except Exception as e:
            raise RepositoryError(f"Failed to initialize MongoDB client: {str(e)}")

    def save_ohlcv(
        self, exchange: str, symbol: str, timeframe: str, data: List[OHLCVSchema]
    ) -> bool:
        """Save OHLCV data to MongoDB"""
        try:
            if not data:
                self.logger.warning(
                    f"No data to save for {exchange}/{symbol}/{timeframe}"
                )
                return True

            # Get collection for this exchange/symbol/timeframe
            collection = self._get_ohlcv_collection(exchange, symbol, timeframe)
            self._ensure_collection_indexes(collection)

            # Convert schemas to MongoDB models
            models = [self.converter.schema_to_mongodb_model(schema) for schema in data]

            # Convert to documents
            documents = []
            for model in models:
                doc = model.model_dump(by_alias=True, exclude={"id"})
                # Ensure timestamp is timezone-aware
                doc["timestamp"] = self._ensure_utc_timezone(doc["timestamp"])
                documents.append(doc)

            # Use upsert to handle duplicates
            operations: List[ReplaceOne] = []
            for doc in documents:
                filt = {
                    "timestamp": doc["timestamp"],
                    "exchange": doc["exchange"],
                    "symbol": doc["symbol"],
                    "timeframe": doc["timeframe"],
                }
                operations.append(ReplaceOne(filter=filt, replacement=doc, upsert=True))

            if operations:
                result = collection.bulk_write(operations, ordered=False)
                saved_count = result.upserted_count + result.modified_count
                self.logger.info(
                    f"Saved {saved_count} OHLCV records to MongoDB for "
                    f"{exchange}/{symbol}/{timeframe}"
                )

            return True

        except PyMongoError as e:
            raise RepositoryError(f"MongoDB error saving OHLCV data: {str(e)}")
        except Exception as e:
            raise RepositoryError(f"Failed to save OHLCV data: {str(e)}")

    def get_ohlcv(self, query: OHLCVQuerySchema) -> List[OHLCVSchema]:
        """Retrieve OHLCV data from MongoDB"""
        try:
            # Get collection for this exchange/symbol/timeframe
            collection = self._get_ohlcv_collection(
                query.exchange, query.symbol, query.timeframe
            )

            # Build MongoDB query
            mongo_query = {
                "exchange": query.exchange,
                "symbol": query.symbol,
                "timeframe": query.timeframe,
                "timestamp": {
                    "$gte": self._ensure_utc_timezone(query.start_date),
                    "$lte": self._ensure_utc_timezone(query.end_date),
                },
            }

            # Execute query with sorting and optional limit
            cursor = collection.find(mongo_query).sort("timestamp", ASCENDING)

            if query.limit:
                cursor = cursor.limit(query.limit)

            # Convert documents to models, then to schemas
            models = []
            for doc in cursor:
                try:
                    model = OHLCVModelMongoDB(**doc)
                    models.append(model)
                except Exception as e:
                    self.logger.warning(f"Skipping invalid document: {e}")
                    continue

            # Convert models to schemas
            schemas = [
                self.converter.mongodb_model_to_schema(model) for model in models
            ]

            self.logger.info(f"Retrieved {len(schemas)} OHLCV records from MongoDB")
            return schemas

        except PyMongoError as e:
            raise RepositoryError(f"MongoDB error retrieving OHLCV data: {str(e)}")
        except Exception as e:
            raise RepositoryError(f"Failed to retrieve OHLCV data: {str(e)}")

    def delete_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> bool:
        """Delete OHLCV data from MongoDB"""
        try:
            collection = self._get_ohlcv_collection(exchange, symbol, timeframe)

            # Build delete query
            delete_query = {
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": timeframe,
            }

            # Add date range if specified
            if start_date and end_date:
                start_dt, end_dt = DateTimeUtils.validate_date_range(
                    start_date, end_date
                )
                delete_query["timestamp"] = {
                    "$gte": self._ensure_utc_timezone(start_dt),
                    "$lte": self._ensure_utc_timezone(end_dt),
                }

            # Execute delete
            result = collection.delete_many(delete_query)

            self.logger.info(
                f"Deleted {result.deleted_count} OHLCV records from MongoDB "
                f"for {exchange}/{symbol}/{timeframe}"
            )
            return True

        except PyMongoError as e:
            raise RepositoryError(f"MongoDB error deleting OHLCV data: {str(e)}")
        except Exception as e:
            raise RepositoryError(f"Failed to delete OHLCV data: {str(e)}")

    # Placeholder implementations for other data types
    def save_trades(
        self, exchange: str, symbol: str, data: List[Dict[str, Any]]
    ) -> bool:
        """Save trade data - placeholder implementation"""
        raise NotImplementedError(
            "Trade data storage will be implemented in next phase"
        )

    def get_trades(
        self, exchange: str, symbol: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """Retrieve trade data - placeholder implementation"""
        raise NotImplementedError(
            "Trade data retrieval will be implemented in next phase"
        )

    def save_orderbook(self, exchange: str, symbol: str, data: Dict[str, Any]) -> bool:
        """Save order book data - placeholder implementation"""
        raise NotImplementedError(
            "Order book storage will be implemented in next phase"
        )

    def get_orderbook(
        self, exchange: str, symbol: str, timestamp: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve order book data - placeholder implementation"""
        raise NotImplementedError(
            "Order book retrieval will be implemented in next phase"
        )

    def save_ticker(self, exchange: str, symbol: str, data: Dict[str, Any]) -> bool:
        """Save ticker data - placeholder implementation"""
        raise NotImplementedError("Ticker storage will be implemented in next phase")

    def get_ticker(
        self, exchange: str, symbol: str, timestamp: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve ticker data - placeholder implementation"""
        raise NotImplementedError("Ticker retrieval will be implemented in next phase")

    def get_available_symbols(self, exchange: str) -> List[str]:
        """Get all available symbols for an exchange from MongoDB"""
        try:
            # Use aggregation to get distinct symbols across all collections
            pipeline = [
                {"$match": {"exchange": exchange}},
                {"$group": {"_id": "$symbol"}},
                {"$sort": {"_id": 1}},
            ]

            symbols = set()
            # Check all OHLCV collections
            for collection_name in self.db.list_collection_names():
                if collection_name.startswith(f"{self.collection_prefix}_"):
                    collection = self.db[collection_name]
                    for result in collection.aggregate(pipeline):
                        symbols.add(result["_id"])

            return sorted(list(symbols))

        except Exception as e:
            raise RepositoryError(f"Failed to get available symbols: {str(e)}")

    def get_available_timeframes(self, exchange: str, symbol: str) -> List[str]:
        """Get all available timeframes for a symbol from MongoDB"""
        try:
            pipeline = [
                {"$match": {"exchange": exchange, "symbol": symbol}},
                {"$group": {"_id": "$timeframe"}},
                {"$sort": {"_id": 1}},
            ]

            timeframes = set()
            # Check all OHLCV collections
            for collection_name in self.db.list_collection_names():
                if collection_name.startswith(f"{self.collection_prefix}_"):
                    collection = self.db[collection_name]
                    for result in collection.aggregate(pipeline):
                        timeframes.add(result["_id"])

            return sorted(list(timeframes))

        except Exception as e:
            raise RepositoryError(f"Failed to get available timeframes: {str(e)}")

    def get_data_range(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        timeframe: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        """Get the date range of available data from MongoDB"""
        try:
            if data_type == "ohlcv" and timeframe:
                collection = self._get_ohlcv_collection(exchange, symbol, timeframe)

                query = {
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe,
                }

                # Find min and max timestamps
                min_doc = collection.find(query).sort("timestamp", ASCENDING).limit(1)
                max_doc = collection.find(query).sort("timestamp", DESCENDING).limit(1)

                min_timestamp = None
                max_timestamp = None

                for doc in min_doc:
                    min_timestamp = doc["timestamp"]
                    break

                for doc in max_doc:
                    max_timestamp = doc["timestamp"]
                    break

                if min_timestamp and max_timestamp:
                    return {
                        "start_date": DateTimeUtils.to_iso_string(min_timestamp),
                        "end_date": DateTimeUtils.to_iso_string(max_timestamp),
                    }

            return None

        except Exception as e:
            raise RepositoryError(f"Failed to get data range: {str(e)}")

    def check_data_exists(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        start_date: str,
        end_date: str,
        timeframe: Optional[str] = None,
    ) -> bool:
        """Check if data exists for the specified criteria in MongoDB"""
        try:
            if data_type == "ohlcv" and timeframe:
                collection = self._get_ohlcv_collection(exchange, symbol, timeframe)

                start_dt, end_dt = DateTimeUtils.validate_date_range(
                    start_date, end_date
                )

                query = {
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": {
                        "$gte": self._ensure_utc_timezone(start_dt),
                        "$lte": self._ensure_utc_timezone(end_dt),
                    },
                }

                count = collection.count_documents(query, limit=1)
                return count > 0

            return False

        except Exception as e:
            raise RepositoryError(f"Failed to check data existence: {str(e)}")

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the MongoDB storage"""
        try:
            # Get database stats
            db_stats = self.db.command("dbStats")

            # Get collection information
            collections = []
            total_documents = 0

            for collection_name in self.db.list_collection_names():
                if collection_name.startswith(f"{self.collection_prefix}_"):
                    collection = self.db[collection_name]
                    count = collection.estimated_document_count()
                    total_documents += count
                    collections.append(
                        {
                            "name": collection_name,
                            "document_count": count,
                        }
                    )

            # Get available exchanges
            exchanges = set()
            for collection_info in collections:
                collection = self.db[collection_info["name"]]
                distinct_exchanges = collection.distinct("exchange")
                exchanges.update(distinct_exchanges)

            return {
                "storage_type": "database",
                "location": self.connection_string,
                "database_name": self.database_name,
                "total_size": db_stats.get("dataSize", 0),
                "total_documents": total_documents,
                "total_collections": len(collections),
                "available_exchanges": sorted(list(exchanges)),
                "collections": collections,
            }

        except Exception as e:
            raise RepositoryError(f"Failed to get storage info: {str(e)}")

    def batch_save_ohlcv(self, data: List[OHLCVBatchSchema]) -> bool:
        """Save multiple OHLCV batch schemas in a batch operation"""
        try:
            success_count = 0

            for batch in data:
                try:
                    success = self.save_ohlcv(
                        exchange=batch.exchange,
                        symbol=batch.symbol,
                        timeframe=batch.timeframe,
                        data=batch.records,
                    )
                    if success:
                        success_count += 1
                except Exception as e:
                    self.logger.error(
                        f"Failed to save batch {batch.exchange}/{batch.symbol}: {e}"
                    )

            self.logger.info(
                f"Batch save completed: {success_count}/{len(data)} batches saved"
            )
            return success_count == len(data)

        except Exception as e:
            raise RepositoryError(f"Batch save failed: {str(e)}")

    def optimize_storage(self) -> bool:
        """Optimize MongoDB storage by creating/rebuilding indexes"""
        try:
            optimized_count = 0

            # Optimize all OHLCV collections
            for collection_name in self.db.list_collection_names():
                if collection_name.startswith(f"{self.collection_prefix}_"):
                    collection = self.db[collection_name]

                    # Rebuild indexes
                    collection.reindex()
                    optimized_count += 1

            self.logger.info(
                f"Optimization completed: {optimized_count} collections optimized"
            )
            return True

        except Exception as e:
            raise RepositoryError(f"Storage optimization failed: {str(e)}")

    def _get_ohlcv_collection(
        self, exchange: str, symbol: str, timeframe: str
    ) -> Collection:
        """Get MongoDB collection for OHLCV data"""
        # Create collection name: ohlcv_binance_BTCUSDT_1d
        # collection_name = f"{self.collection_prefix}_{exchange}_{symbol}_{timeframe}"
        collection_name = self.collection_prefix
        return self.db[collection_name]

    def _initialize_indexes(self) -> None:
        """Initialize MongoDB indexes for optimal performance"""
        try:
            # We'll create indexes on collections as they're created
            # This is done in _ensure_collection_indexes method
            pass
        except Exception as e:
            self.logger.warning(f"Failed to initialize some indexes: {e}")

    def _ensure_collection_indexes(self, collection: Collection) -> None:
        """Ensure proper indexes exist on a collection"""
        try:
            # Compound index for efficient queries
            collection.create_index(
                [
                    ("exchange", ASCENDING),
                    ("symbol", ASCENDING),
                    ("timeframe", ASCENDING),
                    ("timestamp", ASCENDING),
                ],
                unique=True,
                background=True,
            )

            # Individual indexes for common queries
            collection.create_index("timestamp", background=True)
            collection.create_index("exchange", background=True)
            collection.create_index("symbol", background=True)
            collection.create_index("timeframe", background=True)

        except Exception as e:
            self.logger.warning(f"Failed to create indexes: {e}")

    def _test_connection(self) -> None:
        """Test connection to MongoDB"""
        try:
            # Simple ping to test connection
            self.client.admin.command("ping")
        except Exception as e:
            raise RepositoryError(f"MongoDB connection test failed: {str(e)}")

    def _ensure_utc_timezone(self, dt: datetime) -> datetime:
        """
        Ensure datetime has UTC timezone.

        Args:
            dt: Datetime instance

        Returns:
            UTC timezone-aware datetime
        """
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            elif dt.tzinfo != timezone.utc:
                return dt.astimezone(timezone.utc)
            return dt
        else:
            raise ValueError(f"Expected datetime object, got {type(dt)}")

    def close(self) -> None:
        """Close MongoDB client connection"""
        if hasattr(self, "client"):
            self.client.close()
            self.logger.info("Closed MongoDB connection")
