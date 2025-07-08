# adapters/influx_market_data_repository.py

"""
InfluxDB Market Data Repository Implementation

Implements the MarketDataRepository interface using InfluxDB for time-series data storage.
Provides efficient storage and querying for large volumes of market data.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    from influxdb_client.rest import ApiException

    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

from ..interfaces.market_data_repository import MarketDataRepository
from ..interfaces.errors import RepositoryError, ValidationError
from ..common.logger import LoggerFactory
from ..schemas.ohlcv_schemas import OHLCVSchema, OHLCVBatchSchema, OHLCVQuerySchema
from ..models.ohlcv_models import OHLCVModelInfluxDB
from ..converters.ohlcv_converter import OHLCVConverter
from ..utils.datetime_utils import DateTimeUtils


class InfluxMarketDataRepository(MarketDataRepository):
    """InfluxDB implementation of MarketDataRepository for time-series data"""

    def __init__(
        self,
        url: str = "http://localhost:8086",
        token: str = "",
        org: str = "finsight",
        bucket: str = "market_data",
    ):
        """
        Initialize InfluxDB repository

        Args:
            url: InfluxDB server URL
            token: Authentication token
            org: Organization name
            bucket: Bucket name for storing data
        """
        if not INFLUXDB_AVAILABLE:
            raise RepositoryError(
                "InfluxDB client is not available. Please install it: pip install influxdb-client"
            )

        self.logger = LoggerFactory.get_logger(name="influxdb_repository")
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.converter = OHLCVConverter()

        try:
            self.client = InfluxDBClient(url=url, token=token, org=org)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()

            # Test connection
            self.client.ping()
            self.logger.info(f"Connected to InfluxDB at {url}")
        except Exception as e:
            self.logger.error(f"Failed to connect to InfluxDB: {str(e)}")
            raise RepositoryError(f"Failed to connect to InfluxDB: {str(e)}")

    def _create_point_from_schema(self, schema: OHLCVSchema) -> Point:
        """Create InfluxDB Point from OHLCV schema"""
        return (
            Point("ohlcv")
            .tag("exchange", schema.exchange)
            .tag("symbol", schema.symbol)
            .tag("timeframe", schema.timeframe)
            .field("open", schema.open)
            .field("high", schema.high)
            .field("low", schema.low)
            .field("close", schema.close)
            .field("volume", schema.volume)
            .time(schema.timestamp, WritePrecision.NS)
        )

    async def save_ohlcv(
        self, exchange: str, symbol: str, timeframe: str, data: List[OHLCVSchema]
    ) -> bool:
        """Save OHLCV data to InfluxDB"""

        def _save_sync():
            try:
                if not data:
                    self.logger.warning(
                        f"No data to save for {exchange}/{symbol}/{timeframe}"
                    )
                    return False

                # Convert schemas to InfluxDB points
                points = [self._create_point_from_schema(schema) for schema in data]

                # Write points to InfluxDB
                self.write_api.write(bucket=self.bucket, org=self.org, record=points)

                self.logger.info(
                    f"Saved {len(data)} OHLCV records to InfluxDB for {exchange}:{symbol}:{timeframe}"
                )
                return True

            except ApiException as e:
                self.logger.error(f"InfluxDB API error saving OHLCV data: {str(e)}")
                raise RepositoryError(f"Failed to save OHLCV data: {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected error saving OHLCV data: {str(e)}")
                raise RepositoryError(f"Failed to save OHLCV data: {str(e)}")

        return await asyncio.to_thread(_save_sync)

    async def get_ohlcv(self, query: OHLCVQuerySchema) -> List[OHLCVSchema]:
        """Get OHLCV data from InfluxDB"""

        def _get_sync():
            try:
                # Build Flux query
                start_time = (
                    DateTimeUtils.to_iso_string(query.start_date)
                    if query.start_date
                    else "-30d"
                )
                end_time = (
                    DateTimeUtils.to_iso_string(query.end_date)
                    if query.end_date
                    else "now()"
                )

                flux_query = f"""
                from(bucket: "{self.bucket}")
                |> range(start: {start_time}, stop: {end_time})
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> filter(fn: (r) => r.exchange == "{query.exchange}")
                |> filter(fn: (r) => r.symbol == "{query.symbol}")
                |> filter(fn: (r) => r.timeframe == "{query.timeframe}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"])
                """

                if query.limit:
                    flux_query += f"|> limit(n: {query.limit})"

                # Execute query
                tables = self.query_api.query(flux_query, org=self.org)

                results = []
                for table in tables:
                    for record in table.records:
                        # Convert record to schema
                        schema = OHLCVSchema(
                            timestamp=record.get_time(),
                            open=record.get_value_by_key("open"),
                            high=record.get_value_by_key("high"),
                            low=record.get_value_by_key("low"),
                            close=record.get_value_by_key("close"),
                            volume=record.get_value_by_key("volume"),
                            exchange=record.values.get("exchange"),
                            symbol=record.values.get("symbol"),
                            timeframe=record.values.get("timeframe"),
                        )
                        results.append(schema)

                return results

            except ApiException as e:
                self.logger.error(f"InfluxDB API error getting OHLCV data: {str(e)}")
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
        """Delete OHLCV data from InfluxDB"""

        def _delete_sync():
            try:
                # Build delete query
                delete_start = start_date or "1970-01-01T00:00:00Z"
                delete_stop = end_date or "now()"

                predicate = f'_measurement="ohlcv" AND exchange="{exchange}" AND symbol="{symbol}" AND timeframe="{timeframe}"'

                delete_api = self.client.delete_api()
                delete_api.delete(
                    start=delete_start,
                    stop=delete_stop,
                    predicate=predicate,
                    bucket=self.bucket,
                    org=self.org,
                )

                self.logger.info(
                    f"Deleted OHLCV records from InfluxDB for {exchange}:{symbol}:{timeframe}"
                )
                return True

            except ApiException as e:
                self.logger.error(f"InfluxDB API error deleting OHLCV data: {str(e)}")
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
        """Get all available symbols for an exchange from InfluxDB"""

        def _get_symbols_sync():
            try:
                flux_query = f"""
                from(bucket: "{self.bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> filter(fn: (r) => r.exchange == "{exchange}")
                |> distinct(column: "symbol")
                |> sort()
                """

                tables = self.query_api.query(flux_query, org=self.org)

                symbols = []
                for table in tables:
                    for record in table.records:
                        symbol = record.values.get("symbol")
                        if symbol and symbol not in symbols:
                            symbols.append(symbol)

                return sorted(symbols)

            except Exception as e:
                raise RepositoryError(f"Failed to get available symbols: {str(e)}")

        return await asyncio.to_thread(_get_symbols_sync)

    async def get_available_timeframes(self, exchange: str, symbol: str) -> List[str]:
        """Get all available timeframes for a symbol from InfluxDB"""

        def _get_timeframes_sync():
            try:
                flux_query = f"""
                from(bucket: "{self.bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> filter(fn: (r) => r.exchange == "{exchange}")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> distinct(column: "timeframe")
                |> sort()
                """

                tables = self.query_api.query(flux_query, org=self.org)

                timeframes = []
                for table in tables:
                    for record in table.records:
                        timeframe = record.values.get("timeframe")
                        if timeframe and timeframe not in timeframes:
                            timeframes.append(timeframe)

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
        """Get data range for symbol from InfluxDB"""

        def _get_range_sync():
            try:
                if data_type == "ohlcv" and timeframe:
                    flux_query = f"""
                    from(bucket: "{self.bucket}")
                    |> range(start: -10y)
                    |> filter(fn: (r) => r._measurement == "ohlcv")
                    |> filter(fn: (r) => r.exchange == "{exchange}")
                    |> filter(fn: (r) => r.symbol == "{symbol}")
                    |> filter(fn: (r) => r.timeframe == "{timeframe}")
                    |> group(columns: ["_measurement"])
                    |> min()
                    |> yield(name: "min")
                    
                    from(bucket: "{self.bucket}")
                    |> range(start: -10y)
                    |> filter(fn: (r) => r._measurement == "ohlcv")
                    |> filter(fn: (r) => r.exchange == "{exchange}")
                    |> filter(fn: (r) => r.symbol == "{symbol}")
                    |> filter(fn: (r) => r.timeframe == "{timeframe}")
                    |> group(columns: ["_measurement"])
                    |> max()
                    |> yield(name: "max")
                    """

                    tables = self.query_api.query(flux_query, org=self.org)

                    min_time = None
                    max_time = None

                    for table in tables:
                        for record in table.records:
                            if record.get_result() == "min" and not min_time:
                                min_time = record.get_time()
                            elif record.get_result() == "max" and not max_time:
                                max_time = record.get_time()

                    if min_time and max_time:
                        return {
                            "start_date": DateTimeUtils.to_iso_string(min_time),
                            "end_date": DateTimeUtils.to_iso_string(max_time),
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
                    flux_query = f"""
                    from(bucket: "{self.bucket}")
                    |> range(start: {start_date}, stop: {end_date})
                    |> filter(fn: (r) => r._measurement == "ohlcv")
                    |> filter(fn: (r) => r.exchange == "{exchange}")
                    |> filter(fn: (r) => r.symbol == "{symbol}")
                    |> filter(fn: (r) => r.timeframe == "{timeframe}")
                    |> count()
                    """

                    tables = self.query_api.query(flux_query, org=self.org)

                    for table in tables:
                        for record in table.records:
                            count = record.get_value()
                            return count > 0

                return False

            except Exception as e:
                raise RepositoryError(f"Failed to check data exists: {str(e)}")

        return await asyncio.to_thread(_check_exists_sync)

    async def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information for InfluxDB"""

        def _get_storage_info_sync():
            try:
                # Get bucket info
                buckets_api = self.client.buckets_api()
                buckets = buckets_api.find_buckets()

                bucket_info = None
                for bucket in buckets.buckets:
                    if bucket.name == self.bucket:
                        bucket_info = bucket
                        break

                return {
                    "storage_type": "database",
                    "location": self.url,
                    "bucket": self.bucket,
                    "organization": self.org,
                    "bucket_id": bucket_info.id if bucket_info else None,
                    "storage_engine": "InfluxDB",
                }

            except Exception as e:
                raise RepositoryError(f"Failed to get storage info: {str(e)}")

        return await asyncio.to_thread(_get_storage_info_sync)

    async def batch_save_ohlcv(self, data: List[OHLCVBatchSchema]) -> bool:
        """Batch save OHLCV data"""

        def _batch_save_sync():
            try:
                for batch in data:
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
        """Optimize InfluxDB storage"""

        def _optimize_sync():
            try:
                # InfluxDB automatically optimizes storage, but we can trigger compaction
                self.logger.info("InfluxDB storage optimization completed (automatic)")
                return True

            except Exception as e:
                raise RepositoryError(f"Failed to optimize storage: {str(e)}")

        return await asyncio.to_thread(_optimize_sync)

    # Administrative Operations
    async def count_all_records(self) -> int:
        """Count total number of OHLCV records in repository"""

        def _count_all_sync():
            try:
                flux_query = f"""
                from(bucket: "{self.bucket}")
                |> range(start: -10y)
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> count()
                |> sum()
                """

                tables = self.query_api.query(flux_query, org=self.org)

                total_count = 0
                for table in tables:
                    for record in table.records:
                        total_count += record.get_value()

                return total_count

            except Exception as e:
                raise RepositoryError(f"Failed to count all records: {str(e)}")

        return await asyncio.to_thread(_count_all_sync)

    async def get_all_available_symbols(self) -> List[str]:
        """Get all available symbols across all exchanges"""

        def _get_all_symbols_sync():
            try:
                flux_query = f"""
                from(bucket: "{self.bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> distinct(column: "symbol")
                |> sort()
                """

                tables = self.query_api.query(flux_query, org=self.org)

                symbols = []
                for table in tables:
                    for record in table.records:
                        symbol = record.values.get("symbol")
                        if symbol and symbol not in symbols:
                            symbols.append(symbol)

                return sorted(symbols)

            except Exception as e:
                raise RepositoryError(f"Failed to get all available symbols: {str(e)}")

        return await asyncio.to_thread(_get_all_symbols_sync)

    async def get_available_exchanges(self) -> List[str]:
        """Get all available exchanges in repository"""

        def _get_exchanges_sync():
            try:
                flux_query = f"""
                from(bucket: "{self.bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> distinct(column: "exchange")
                |> sort()
                """

                tables = self.query_api.query(flux_query, org=self.org)

                exchanges = []
                for table in tables:
                    for record in table.records:
                        exchange = record.values.get("exchange")
                        if exchange and exchange not in exchanges:
                            exchanges.append(exchange)

                return sorted(exchanges)

            except Exception as e:
                raise RepositoryError(f"Failed to get available exchanges: {str(e)}")

        return await asyncio.to_thread(_get_exchanges_sync)

    async def get_all_available_timeframes(self) -> List[str]:
        """Get all available timeframes across all data"""

        def _get_all_timeframes_sync():
            try:
                flux_query = f"""
                from(bucket: "{self.bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> distinct(column: "timeframe")
                |> sort()
                """

                tables = self.query_api.query(flux_query, org=self.org)

                timeframes = []
                for table in tables:
                    for record in table.records:
                        timeframe = record.values.get("timeframe")
                        if timeframe and timeframe not in timeframes:
                            timeframes.append(timeframe)

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
                start_str = (
                    DateTimeUtils.to_iso_string(start_date) if start_date else "-10y"
                )
                end_str = DateTimeUtils.to_iso_string(end_date) if end_date else "now()"

                flux_query = f"""
                from(bucket: "{self.bucket}")
                |> range(start: {start_str}, stop: {end_str})
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> filter(fn: (r) => r.exchange == "{exchange}")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r.timeframe == "{timeframe}")
                |> count()
                |> sum()
                """

                tables = self.query_api.query(flux_query, org=self.org)

                count = 0
                for table in tables:
                    for record in table.records:
                        count += record.get_value()

                return count

            except Exception as e:
                raise RepositoryError(f"Failed to count records: {str(e)}")

        return await asyncio.to_thread(_count_records_sync)

    async def count_records_since(self, cutoff_date: datetime) -> int:
        """Count records since a specific date"""

        def _count_since_sync():
            try:
                cutoff_str = DateTimeUtils.to_iso_string(cutoff_date)

                flux_query = f"""
                from(bucket: "{self.bucket}")
                |> range(start: {cutoff_str})
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> count()
                |> sum()
                """

                tables = self.query_api.query(flux_query, org=self.org)

                count = 0
                for table in tables:
                    for record in table.records:
                        count += record.get_value()

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
                # First count records to be deleted
                count_query = f"""
                from(bucket: "{self.bucket}")
                |> range(start: -10y, stop: {DateTimeUtils.to_iso_string(cutoff_date)})
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> filter(fn: (r) => r.exchange == "{exchange}")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r.timeframe == "{timeframe}")
                |> count()
                |> sum()
                """

                tables = self.query_api.query(count_query, org=self.org)

                deleted_count = 0
                for table in tables:
                    for record in table.records:
                        deleted_count += record.get_value()

                # Delete records
                predicate = f'_measurement="ohlcv" AND exchange="{exchange}" AND symbol="{symbol}" AND timeframe="{timeframe}"'

                delete_api = self.client.delete_api()
                delete_api.delete(
                    start="1970-01-01T00:00:00Z",
                    stop=DateTimeUtils.to_iso_string(cutoff_date),
                    predicate=predicate,
                    bucket=self.bucket,
                    org=self.org,
                )

                return deleted_count

            except Exception as e:
                raise RepositoryError(f"Failed to delete records before date: {str(e)}")

        return await asyncio.to_thread(_delete_before_sync)

    def close(self) -> None:
        """Close InfluxDB client connection"""
        if hasattr(self, "client"):
            self.client.close()
            self.logger.info("Closed InfluxDB connection")
