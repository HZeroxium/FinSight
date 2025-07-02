# adapters/influx_market_data_repository.py

"""
InfluxDB Market Data Repository Implementation

Implements the MarketDataRepository interface using InfluxDB for time-series data storage.
Provides efficient storage and querying for large volumes of market data.
"""

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
from ..common.logger import LoggerFactory, LoggerType, LogLevel


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
                "InfluxDB client not available. Install with: pip install influxdb-client"
            )

        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket

        self.logger = LoggerFactory.get_logger(
            name="influx_repository",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
        )

        # Initialize client
        try:
            self.client = InfluxDBClient(url=url, token=token, org=org)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()

            # Test connection
            self._test_connection()

            self.logger.info(f"Initialized InfluxDB repository at {url}")

        except Exception as e:
            raise RepositoryError(f"Failed to initialize InfluxDB client: {str(e)}")

    def save_ohlcv(
        self, exchange: str, symbol: str, timeframe: str, data: List[Dict[str, Any]]
    ) -> bool:
        """Save OHLCV data to InfluxDB"""
        try:
            if not data:
                self.logger.warning(
                    f"No data to save for {exchange}/{symbol}/{timeframe}"
                )
                return True

            # Validate data format
            self._validate_ohlcv_data(data)

            # Convert to InfluxDB points
            points = []
            for record in data:
                timestamp = self._parse_timestamp(record["timestamp"])

                point = (
                    Point("ohlcv")
                    .tag("exchange", exchange)
                    .tag("symbol", symbol)
                    .tag("timeframe", timeframe)
                    .field("open", float(record["open"]))
                    .field("high", float(record["high"]))
                    .field("low", float(record["low"]))
                    .field("close", float(record["close"]))
                    .field("volume", float(record["volume"]))
                    .time(timestamp, WritePrecision.S)
                )
                points.append(point)

            # Write to InfluxDB
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)

            self.logger.info(
                f"Saved {len(points)} OHLCV records to InfluxDB for {exchange}/{symbol}/{timeframe}"
            )
            return True

        except ApiException as e:
            raise RepositoryError(f"InfluxDB API error saving OHLCV data: {str(e)}")
        except Exception as e:
            raise RepositoryError(f"Failed to save OHLCV data: {str(e)}")

    def get_ohlcv(
        self, exchange: str, symbol: str, timeframe: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """Retrieve OHLCV data from InfluxDB"""
        try:
            # Validate and parse dates
            start_dt, end_dt = self._parse_dates(start_date, end_date)

            # Build Flux query
            query = f"""
                from(bucket: "{self.bucket}")
                |> range(start: {start_dt.isoformat()}, stop: {end_dt.isoformat()})
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> filter(fn: (r) => r.exchange == "{exchange}")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r.timeframe == "{timeframe}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"])
            """

            # Execute query
            result = self.query_api.query(org=self.org, query=query)

            # Convert to standardized format
            records = []
            for table in result:
                for record in table.records:
                    data_record = {
                        "timestamp": record.get_time().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "open": float(record.values.get("open", 0)),
                        "high": float(record.values.get("high", 0)),
                        "low": float(record.values.get("low", 0)),
                        "close": float(record.values.get("close", 0)),
                        "volume": float(record.values.get("volume", 0)),
                        "symbol": symbol,
                        "timeframe": timeframe,
                    }
                    records.append(data_record)

            self.logger.info(f"Retrieved {len(records)} OHLCV records from InfluxDB")
            return records

        except ApiException as e:
            raise RepositoryError(f"InfluxDB API error retrieving OHLCV data: {str(e)}")
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
        """Delete OHLCV data from InfluxDB"""
        try:
            # Build delete predicate
            predicate = f'_measurement="ohlcv" AND exchange="{exchange}" AND symbol="{symbol}" AND timeframe="{timeframe}"'

            # Parse date range if provided
            if start_date and end_date:
                start_dt, end_dt = self._parse_dates(start_date, end_date)
                start_rfc = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                end_rfc = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                # Delete all data for this series
                start_rfc = "1970-01-01T00:00:00Z"
                end_rfc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Execute delete
            delete_api = self.client.delete_api()
            delete_api.delete(
                start=start_rfc,
                stop=end_rfc,
                predicate=predicate,
                bucket=self.bucket,
                org=self.org,
            )

            self.logger.info(
                f"Deleted OHLCV data for {exchange}/{symbol}/{timeframe} from {start_rfc} to {end_rfc}"
            )
            return True

        except ApiException as e:
            raise RepositoryError(f"InfluxDB API error deleting OHLCV data: {str(e)}")
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
        """Get all available symbols for an exchange from InfluxDB"""
        try:
            query = f"""
                from(bucket: "{self.bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> filter(fn: (r) => r.exchange == "{exchange}")
                |> distinct(column: "symbol")
                |> keep(columns: ["symbol"])
            """

            result = self.query_api.query(org=self.org, query=query)

            symbols = set()
            for table in result:
                for record in table.records:
                    symbol = record.values.get("symbol")
                    if symbol:
                        symbols.add(symbol)

            return sorted(list(symbols))

        except Exception as e:
            raise RepositoryError(f"Failed to get available symbols: {str(e)}")

    def get_available_timeframes(self, exchange: str, symbol: str) -> List[str]:
        """Get all available timeframes for a symbol from InfluxDB"""
        try:
            query = f"""
                from(bucket: "{self.bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> filter(fn: (r) => r.exchange == "{exchange}")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> distinct(column: "timeframe")
                |> keep(columns: ["timeframe"])
            """

            result = self.query_api.query(org=self.org, query=query)

            timeframes = set()
            for table in result:
                for record in table.records:
                    timeframe = record.values.get("timeframe")
                    if timeframe:
                        timeframes.add(timeframe)

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
        """Get the date range of available data from InfluxDB"""
        try:
            if data_type == "ohlcv" and timeframe:
                query = f"""
                    from(bucket: "{self.bucket}")
                    |> range(start: -10y)
                    |> filter(fn: (r) => r._measurement == "ohlcv")
                    |> filter(fn: (r) => r.exchange == "{exchange}")
                    |> filter(fn: (r) => r.symbol == "{symbol}")
                    |> filter(fn: (r) => r.timeframe == "{timeframe}")
                    |> keep(columns: ["_time"])
                    |> min()
                """

                min_result = self.query_api.query(org=self.org, query=query)

                query_max = query.replace("|> min()", "|> max()")
                max_result = self.query_api.query(org=self.org, query=query_max)

                min_time = None
                max_time = None

                for table in min_result:
                    for record in table.records:
                        min_time = record.get_time()
                        break

                for table in max_result:
                    for record in table.records:
                        max_time = record.get_time()
                        break

                if min_time and max_time:
                    return {
                        "start_date": min_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "end_date": max_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        """Check if data exists for the specified criteria in InfluxDB"""
        try:
            if data_type == "ohlcv" and timeframe:
                start_dt, end_dt = self._parse_dates(start_date, end_date)

                query = f"""
                    from(bucket: "{self.bucket}")
                    |> range(start: {start_dt.isoformat()}, stop: {end_dt.isoformat()})
                    |> filter(fn: (r) => r._measurement == "ohlcv")
                    |> filter(fn: (r) => r.exchange == "{exchange}")
                    |> filter(fn: (r) => r.symbol == "{symbol}")
                    |> filter(fn: (r) => r.timeframe == "{timeframe}")
                    |> count()
                """

                result = self.query_api.query(org=self.org, query=query)

                for table in result:
                    for record in table.records:
                        count = record.get_value()
                        return count > 0

            return False

        except Exception as e:
            raise RepositoryError(f"Failed to check data existence: {str(e)}")

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the InfluxDB storage"""
        try:
            # Get bucket info
            buckets_api = self.client.buckets_api()
            bucket_info = buckets_api.find_bucket_by_name(self.bucket)

            # Get basic statistics
            query = f"""
                from(bucket: "{self.bucket}")
                |> range(start: -1y)
                |> group()
                |> count()
            """

            result = self.query_api.query(org=self.org, query=query)
            total_points = 0

            for table in result:
                for record in table.records:
                    total_points = record.get_value()
                    break

            # Get available exchanges
            exchanges_query = f"""
                from(bucket: "{self.bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> distinct(column: "exchange")
                |> keep(columns: ["exchange"])
            """

            exchanges_result = self.query_api.query(org=self.org, query=exchanges_query)
            exchanges = []

            for table in exchanges_result:
                for record in table.records:
                    exchange = record.values.get("exchange")
                    if exchange:
                        exchanges.append(exchange)

            return {
                "storage_type": "database",
                "location": self.url,
                "bucket_id": bucket_info.id if bucket_info else None,
                "organization": self.org,
                "total_points": total_points,
                "available_exchanges": sorted(exchanges),
                "total_symbols": len(exchanges) * 10,  # Approximate
            }

        except Exception as e:
            raise RepositoryError(f"Failed to get storage info: {str(e)}")

    def batch_save_ohlcv(self, data: List[Dict[str, Any]]) -> bool:
        """Save multiple OHLCV datasets in a batch operation"""
        try:
            all_points = []

            for dataset in data:
                exchange = dataset["exchange"]
                symbol = dataset["symbol"]
                timeframe = dataset["timeframe"]
                records = dataset["records"]

                # Convert each record to InfluxDB point
                for record in records:
                    timestamp = self._parse_timestamp(record["timestamp"])

                    point = (
                        Point("ohlcv")
                        .tag("exchange", exchange)
                        .tag("symbol", symbol)
                        .tag("timeframe", timeframe)
                        .field("open", float(record["open"]))
                        .field("high", float(record["high"]))
                        .field("low", float(record["low"]))
                        .field("close", float(record["close"]))
                        .field("volume", float(record["volume"]))
                        .time(timestamp, WritePrecision.S)
                    )
                    all_points.append(point)

            # Write all points in batch
            self.write_api.write(bucket=self.bucket, org=self.org, record=all_points)

            self.logger.info(
                f"Batch saved {len(all_points)} OHLCV points from {len(data)} datasets"
            )
            return True

        except Exception as e:
            raise RepositoryError(f"Batch save failed: {str(e)}")

    def optimize_storage(self) -> bool:
        """Optimize InfluxDB storage (placeholder - InfluxDB handles optimization internally)"""
        try:
            # InfluxDB handles optimization internally through compaction
            # This could trigger manual compaction if needed
            self.logger.info("InfluxDB handles storage optimization automatically")
            return True

        except Exception as e:
            raise RepositoryError(f"Storage optimization failed: {str(e)}")

    def _test_connection(self) -> None:
        """Test connection to InfluxDB"""
        try:
            # Test with a simple query
            health = self.client.health()
            if health.status != "pass":
                raise RepositoryError(f"InfluxDB health check failed: {health.status}")
        except Exception as e:
            raise RepositoryError(f"InfluxDB connection test failed: {str(e)}")

    def _validate_ohlcv_data(self, data: List[Dict[str, Any]]) -> None:
        """Validate OHLCV data format"""
        required_fields = ["timestamp", "open", "high", "low", "close", "volume"]

        for record in data:
            for field in required_fields:
                if field not in record:
                    raise ValidationError(f"Missing required field: {field}")

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime object"""
        try:
            # Handle both with and without timezone
            if timestamp_str.endswith("Z"):
                return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(timestamp_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
        except ValueError as e:
            raise ValidationError(f"Invalid timestamp format: {str(e)}")

    def _parse_dates(self, start_date: str, end_date: str) -> tuple:
        """Parse and validate ISO 8601 date strings"""
        try:
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

            if start_dt >= end_dt:
                raise ValidationError("Start date must be before end date")

            return start_dt, end_dt

        except ValueError as e:
            raise ValidationError(
                f"Invalid date format. Expected ISO 8601 format: {str(e)}"
            )

    def close(self) -> None:
        """Close InfluxDB client connection"""
        if hasattr(self, "client"):
            self.client.close()
            self.logger.info("Closed InfluxDB connection")
