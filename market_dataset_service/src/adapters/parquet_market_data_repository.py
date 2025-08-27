# adapters/parquet_market_data_repository.py

"""
Parquet Market Data Repository

High-performance repository implementation using Parquet format for data storage.
Supports both local file system and object storage backends via StorageClient.
"""

import asyncio
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from common.logger import LoggerFactory, LoggerType, LogLevel
from pyarrow import Table

from ..interfaces.errors import RepositoryError
from ..interfaces.market_data_repository import MarketDataRepository
from ..schemas.ohlcv_schemas import (OHLCVBatchSchema, OHLCVQuerySchema,
                                     OHLCVSchema)
from ..utils.datetime_utils import DateTimeUtils
from ..utils.storage_client import StorageClient, StorageClientError


class ParquetMarketDataRepository(MarketDataRepository):
    """
    Parquet-based market data repository with object storage support.

    Features:
    - High-performance columnar storage using Parquet format
    - Partitioned storage by exchange/symbol/timeframe/date
    - Object storage backend support (S3, MinIO, etc.)
    - Efficient querying and data compression
    - Schema evolution support
    """

    def __init__(
        self,
        storage_client: Optional[StorageClient] = None,
        base_path: str = "data/market_data",
        local_cache_dir: str = "cache/parquet",
        partition_scheme: str = "exchange/symbol/timeframe/year/month",
        compression: str = "snappy",
        use_object_storage: bool = False,
        cache_enabled: bool = True,
        max_cache_size_mb: int = 1024,
    ):
        """
        Initialize Parquet market data repository.

        Args:
            storage_client: Object storage client for remote storage
            base_path: Base path for data storage
            local_cache_dir: Local cache directory for temporary files
            partition_scheme: Partitioning scheme for data organization
            compression: Parquet compression algorithm
            use_object_storage: Whether to use object storage backend
            cache_enabled: Whether to enable local caching
            max_cache_size_mb: Maximum cache size in MB
        """
        self.storage_client = storage_client
        self.base_path = Path(base_path)
        self.local_cache_dir = Path(local_cache_dir)
        self.partition_scheme = partition_scheme
        self.compression = compression
        self.use_object_storage = use_object_storage
        self.cache_enabled = cache_enabled
        self.max_cache_size_mb = max_cache_size_mb

        # Initialize logger
        self.logger = LoggerFactory.get_logger(
            name="parquet-repository",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file="logs/parquet_repository.log",
        )

        # Create local directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        # Validate configuration
        if use_object_storage and not storage_client:
            raise RepositoryError("StorageClient required when use_object_storage=True")

        self.logger.info(
            f"Parquet repository initialized - "
            f"base_path: {self.base_path}, "
            f"object_storage: {self.use_object_storage}, "
            f"compression: {self.compression}"
        )

    def _get_partition_path(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        data_type: str = "ohlcv",
    ) -> str:
        """Generate partition path based on scheme."""
        year = timestamp.year
        month = timestamp.month
        day = timestamp.day

        if self.partition_scheme == "exchange/symbol/timeframe/year/month":
            return f"{data_type}/{exchange}/{symbol}/{timeframe}/{year:04d}/{month:02d}"
        elif self.partition_scheme == "exchange/symbol/timeframe/year/month/day":
            return f"{data_type}/{exchange}/{symbol}/{timeframe}/{year:04d}/{month:02d}/{day:02d}"
        else:
            return f"{data_type}/{exchange}/{symbol}/{timeframe}/{year:04d}/{month:02d}"

    def _get_file_name(self, timestamp: datetime, data_type: str = "ohlcv") -> str:
        """Generate file name for the data."""
        date_str = timestamp.strftime("%Y%m%d")
        return f"{data_type}_{date_str}.parquet"

    def _schemas_to_dataframe(self, schemas: List[OHLCVSchema]) -> pd.DataFrame:
        """Convert OHLCV schemas to pandas DataFrame."""
        data = []
        for schema in schemas:
            data.append(
                {
                    "timestamp": schema.timestamp,
                    "open": schema.open,
                    "high": schema.high,
                    "low": schema.low,
                    "close": schema.close,
                    "volume": schema.volume,
                }
            )

        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _dataframe_to_schemas(self, df: pd.DataFrame) -> List[OHLCVSchema]:
        """Convert pandas DataFrame to OHLCV schemas."""
        schemas = []
        for _, row in df.iterrows():
            schemas.append(
                OHLCVSchema(
                    timestamp=row["timestamp"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
        return schemas

    async def _upload_to_storage(self, local_path: Path, object_key: str) -> bool:
        """Upload file to object storage."""
        if not self.use_object_storage or not self.storage_client:
            return True

        try:
            return await self.storage_client.upload_file(
                local_file_path=local_path,
                object_key=object_key,
                content_type="application/octet-stream",
            )
        except StorageClientError as e:
            self.logger.error(f"Failed to upload to storage: {e}")
            raise RepositoryError(f"Storage upload failed: {e}")

    async def _download_from_storage(self, object_key: str, local_path: Path) -> bool:
        """Download file from object storage."""
        if not self.use_object_storage or not self.storage_client:
            return local_path.exists()

        try:
            if await self.storage_client.object_exists(object_key):
                return await self.storage_client.download_file(
                    object_key=object_key,
                    local_file_path=local_path,
                )
            return False
        except StorageClientError as e:
            self.logger.error(f"Failed to download from storage: {e}")
            return False

    async def save_ohlcv(
        self, exchange: str, symbol: str, timeframe: str, data: List[OHLCVSchema]
    ) -> bool:
        """Save OHLCV data to Parquet format."""
        try:
            if not data:
                self.logger.warning(
                    f"No data to save for {exchange}/{symbol}/{timeframe}"
                )
                return True

            # Convert to DataFrame
            df = self._schemas_to_dataframe(data)
            if df.empty:
                return True

            # Group data by date for partitioning
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date

            success_count = 0
            total_partitions = 0

            for date, group_df in df.groupby("date"):
                total_partitions += 1
                timestamp = datetime.combine(date, datetime.min.time())

                # Generate paths
                partition_path = self._get_partition_path(
                    exchange, symbol, timeframe, timestamp, "ohlcv"
                )
                file_name = self._get_file_name(timestamp, "ohlcv")

                # Local file path
                local_file_path = self.base_path / partition_path / file_name
                local_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Object storage path
                object_key = f"{partition_path}/{file_name}"

                try:
                    # Remove the temporary date column
                    save_df = group_df.drop(columns=["date"])

                    # Convert to PyArrow table for better type control
                    table = pa.Table.from_pandas(save_df, preserve_index=False)

                    # Write to Parquet
                    pq.write_table(
                        table,
                        local_file_path,
                        compression=self.compression,
                        write_statistics=True,
                        use_dictionary=True,
                    )

                    # Upload to object storage if enabled
                    if self.use_object_storage:
                        await self._upload_to_storage(local_file_path, object_key)

                    success_count += 1
                    self.logger.debug(
                        f"Saved {len(group_df)} records for {exchange}/{symbol}/{timeframe} on {date}"
                    )

                except Exception as e:
                    self.logger.error(
                        f"Failed to save partition {exchange}/{symbol}/{timeframe}/{date}: {e}"
                    )
                    continue

            if success_count == total_partitions:
                self.logger.info(
                    f"Successfully saved {len(data)} OHLCV records for {exchange}/{symbol}/{timeframe} "
                    f"across {total_partitions} partitions"
                )
                return True
            else:
                self.logger.warning(
                    f"Partial save success: {success_count}/{total_partitions} partitions saved"
                )
                return success_count > 0

        except Exception as e:
            self.logger.error(f"Error saving OHLCV data: {e}")
            raise RepositoryError(f"Failed to save OHLCV data: {e}")

    async def get_ohlcv(self, query: OHLCVQuerySchema) -> List[OHLCVSchema]:
        """Retrieve OHLCV data from Parquet files."""
        try:
            # Generate date range for partitions
            start_date = query.start_date.date()
            end_date = query.end_date.date()

            all_data = []
            current_date = start_date

            while current_date <= end_date:
                timestamp = datetime.combine(current_date, datetime.min.time())

                # Generate paths
                partition_path = self._get_partition_path(
                    query.exchange, query.symbol, query.timeframe, timestamp, "ohlcv"
                )
                file_name = self._get_file_name(timestamp, "ohlcv")

                # Local file path
                local_file_path = self.base_path / partition_path / file_name
                object_key = f"{partition_path}/{file_name}"

                try:
                    # Try to download from storage if not exists locally
                    if not local_file_path.exists() and self.use_object_storage:
                        await self._download_from_storage(object_key, local_file_path)

                    # Read Parquet file if exists
                    if local_file_path.exists():
                        df = pd.read_parquet(local_file_path)

                        # Filter by timestamp range
                        mask = (df["timestamp"] >= query.start_date) & (
                            df["timestamp"] <= query.end_date
                        )
                        filtered_df = df[mask]

                        if not filtered_df.empty:
                            all_data.append(filtered_df)
                            self.logger.debug(
                                f"Loaded {len(filtered_df)} records from {local_file_path}"
                            )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to read partition {local_file_path}: {e}"
                    )
                    continue

                current_date += timedelta(days=1)

            # Combine all data
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df = combined_df.sort_values("timestamp")

                # Apply limit if specified
                if query.limit and len(combined_df) > query.limit:
                    combined_df = combined_df.head(query.limit)

                schemas = self._dataframe_to_schemas(combined_df)
                self.logger.info(
                    f"Retrieved {len(schemas)} OHLCV records for "
                    f"{query.exchange}/{query.symbol}/{query.timeframe}"
                )
                return schemas
            else:
                self.logger.info(
                    f"No OHLCV data found for {query.exchange}/{query.symbol}/{query.timeframe}"
                )
                return []

        except Exception as e:
            self.logger.error(f"Error retrieving OHLCV data: {e}")
            raise RepositoryError(f"Failed to retrieve OHLCV data: {e}")

    async def delete_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> bool:
        """Delete OHLCV data from repository."""
        try:
            if start_date and end_date:
                # Parse date range
                start_dt = DateTimeUtils.parse_iso_string(start_date)
                end_dt = DateTimeUtils.parse_iso_string(end_date)

                # Delete specific date range
                current_date = start_dt.date()
                end_date_dt = end_dt.date()
                deleted_count = 0

                while current_date <= end_date_dt:
                    timestamp = datetime.combine(current_date, datetime.min.time())

                    # Generate paths
                    partition_path = self._get_partition_path(
                        exchange, symbol, timeframe, timestamp, "ohlcv"
                    )
                    file_name = self._get_file_name(timestamp, "ohlcv")

                    # Local file path
                    local_file_path = self.base_path / partition_path / file_name
                    object_key = f"{partition_path}/{file_name}"

                    # Delete local file
                    if local_file_path.exists():
                        local_file_path.unlink()
                        deleted_count += 1

                    # Delete from object storage
                    if self.use_object_storage and self.storage_client:
                        try:
                            await self.storage_client.delete_object(object_key)
                        except StorageClientError:
                            pass  # Continue if object doesn't exist

                    current_date += timedelta(days=1)

                self.logger.info(
                    f"Deleted {deleted_count} OHLCV files for {exchange}/{symbol}/{timeframe}"
                )
                return deleted_count > 0
            else:
                # Delete all data for symbol/timeframe
                pattern_path = self.base_path / "ohlcv" / exchange / symbol / timeframe
                deleted_count = 0

                if pattern_path.exists():
                    # Delete local files
                    for file_path in pattern_path.rglob("*.parquet"):
                        file_path.unlink()
                        deleted_count += 1

                    # Clean up empty directories
                    shutil.rmtree(pattern_path, ignore_errors=True)

                # Delete from object storage
                if self.use_object_storage and self.storage_client:
                    prefix = f"ohlcv/{exchange}/{symbol}/{timeframe}/"
                    try:
                        objects = await self.storage_client.list_objects(prefix=prefix)
                        for obj in objects:
                            await self.storage_client.delete_object(obj["key"])
                    except StorageClientError:
                        pass

                self.logger.info(
                    f"Deleted all OHLCV data for {exchange}/{symbol}/{timeframe}"
                )
                return True

        except Exception as e:
            self.logger.error(f"Error deleting OHLCV data: {e}")
            raise RepositoryError(f"Failed to delete OHLCV data: {e}")

    # Trade operations (simplified implementation)
    async def save_trades(
        self, exchange: str, symbol: str, data: List[Dict[str, Any]]
    ) -> bool:
        """Save trade data (simplified implementation)."""
        try:
            if not data:
                return True

            df = pd.DataFrame(data)
            if df.empty:
                return True

            # Use current date for partitioning
            timestamp = datetime.now()
            partition_path = self._get_partition_path(
                exchange, symbol, "trades", timestamp, "trades"
            )
            file_name = self._get_file_name(timestamp, "trades")

            local_file_path = self.base_path / partition_path / file_name
            local_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to Parquet
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, local_file_path, compression=self.compression)

            # Upload to storage if enabled
            if self.use_object_storage:
                object_key = f"{partition_path}/{file_name}"
                await self._upload_to_storage(local_file_path, object_key)

            self.logger.info(f"Saved {len(data)} trade records for {exchange}/{symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving trade data: {e}")
            raise RepositoryError(f"Failed to save trade data: {e}")

    async def get_trades(
        self, exchange: str, symbol: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """Retrieve trade data (simplified implementation)."""
        try:
            # Implementation similar to get_ohlcv but for trades
            # For brevity, returning empty list
            self.logger.info(
                f"Trade retrieval not fully implemented for {exchange}/{symbol}"
            )
            return []
        except Exception as e:
            self.logger.error(f"Error retrieving trade data: {e}")
            return []

    # Order book operations (placeholder implementations)
    async def save_orderbook(
        self, exchange: str, symbol: str, data: Dict[str, Any]
    ) -> bool:
        """Save order book data (placeholder)."""
        self.logger.debug(f"Order book save not implemented for {exchange}/{symbol}")
        return True

    async def get_orderbook(
        self, exchange: str, symbol: str, timestamp: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve order book data (placeholder)."""
        self.logger.debug(
            f"Order book retrieval not implemented for {exchange}/{symbol}"
        )
        return None

    # Ticker operations (placeholder implementations)
    async def save_ticker(
        self, exchange: str, symbol: str, data: Dict[str, Any]
    ) -> bool:
        """Save ticker data (placeholder)."""
        self.logger.debug(f"Ticker save not implemented for {exchange}/{symbol}")
        return True

    async def get_ticker(
        self, exchange: str, symbol: str, timestamp: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve ticker data (placeholder)."""
        self.logger.debug(f"Ticker retrieval not implemented for {exchange}/{symbol}")
        return None

    # Query operations
    async def get_available_symbols(self, exchange: str) -> List[str]:
        """Get all available symbols for an exchange."""
        try:
            symbols = set()
            ohlcv_path = self.base_path / "ohlcv" / exchange

            if ohlcv_path.exists():
                for symbol_dir in ohlcv_path.iterdir():
                    if symbol_dir.is_dir():
                        symbols.add(symbol_dir.name)

            # Also check object storage if enabled
            if self.use_object_storage and self.storage_client:
                try:
                    prefix = f"ohlcv/{exchange}/"
                    objects = await self.storage_client.list_objects(prefix=prefix)
                    for obj in objects:
                        # Extract symbol from path: ohlcv/exchange/symbol/...
                        parts = obj["key"].split("/")
                        if len(parts) >= 3:
                            symbols.add(parts[2])
                except StorageClientError:
                    pass

            symbol_list = sorted(list(symbols))
            self.logger.info(f"Found {len(symbol_list)} symbols for {exchange}")
            return symbol_list

        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []

    async def get_available_timeframes(self, exchange: str, symbol: str) -> List[str]:
        """Get all available timeframes for a symbol."""
        try:
            timeframes = set()
            symbol_path = self.base_path / "ohlcv" / exchange / symbol

            if symbol_path.exists():
                for tf_dir in symbol_path.iterdir():
                    if tf_dir.is_dir():
                        timeframes.add(tf_dir.name)

            # Also check object storage if enabled
            if self.use_object_storage and self.storage_client:
                try:
                    prefix = f"ohlcv/{exchange}/{symbol}/"
                    objects = await self.storage_client.list_objects(prefix=prefix)
                    for obj in objects:
                        # Extract timeframe from path: ohlcv/exchange/symbol/timeframe/...
                        parts = obj["key"].split("/")
                        if len(parts) >= 4:
                            timeframes.add(parts[3])
                except StorageClientError:
                    pass

            tf_list = sorted(list(timeframes))
            self.logger.info(f"Found {len(tf_list)} timeframes for {exchange}/{symbol}")
            return tf_list

        except Exception as e:
            self.logger.error(f"Error getting available timeframes: {e}")
            return []

    async def get_data_range(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        timeframe: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        """Get the date range of available data."""
        try:
            if data_type == "ohlcv" and timeframe:
                # Scan all partition directories to find date range
                base_dir = self.base_path / "ohlcv" / exchange / symbol / timeframe
                dates = []

                if base_dir.exists():
                    # Scan local files
                    for year_dir in base_dir.iterdir():
                        if year_dir.is_dir() and year_dir.name.isdigit():
                            for month_dir in year_dir.iterdir():
                                if month_dir.is_dir() and month_dir.name.isdigit():
                                    for file_path in month_dir.glob("*.parquet"):
                                        # Extract date from filename
                                        filename = file_path.stem
                                        if "_" in filename:
                                            date_part = filename.split("_")[-1]
                                            try:
                                                date_obj = datetime.strptime(
                                                    date_part, "%Y%m%d"
                                                )
                                                dates.append(date_obj)
                                            except ValueError:
                                                continue

                # Also check object storage
                if self.use_object_storage and self.storage_client:
                    try:
                        prefix = f"ohlcv/{exchange}/{symbol}/{timeframe}/"
                        objects = await self.storage_client.list_objects(prefix=prefix)
                        for obj in objects:
                            # Extract date from object key
                            filename = obj["key"].split("/")[-1]
                            if filename.endswith(".parquet") and "_" in filename:
                                date_part = filename.split("_")[-1].replace(
                                    ".parquet", ""
                                )
                                try:
                                    date_obj = datetime.strptime(date_part, "%Y%m%d")
                                    dates.append(date_obj)
                                except ValueError:
                                    continue
                    except StorageClientError:
                        pass

                if dates:
                    min_date = min(dates)
                    max_date = max(dates)
                    return {
                        "start_date": DateTimeUtils.to_iso_string(min_date),
                        "end_date": DateTimeUtils.to_iso_string(max_date),
                    }

            return None

        except Exception as e:
            self.logger.error(f"Error getting data range: {e}")
            return None

    async def check_data_exists(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        start_date: str,
        end_date: str,
        timeframe: Optional[str] = None,
    ) -> bool:
        """Check if data exists for the specified criteria."""
        try:
            if data_type == "ohlcv" and timeframe:
                # Check if any data exists in the date range
                start_dt = DateTimeUtils.parse_iso_string(start_date)
                end_dt = DateTimeUtils.parse_iso_string(end_date)

                current_date = start_dt.date()
                end_date_dt = end_dt.date()

                while current_date <= end_date_dt:
                    timestamp = datetime.combine(current_date, datetime.min.time())
                    partition_path = self._get_partition_path(
                        exchange, symbol, timeframe, timestamp, "ohlcv"
                    )
                    file_name = self._get_file_name(timestamp, "ohlcv")

                    local_file_path = self.base_path / partition_path / file_name

                    # Check local file
                    if local_file_path.exists():
                        return True

                    # Check object storage
                    if self.use_object_storage and self.storage_client:
                        object_key = f"{partition_path}/{file_name}"
                        if await self.storage_client.object_exists(object_key):
                            return True

                    current_date += timedelta(days=1)

            return False

        except Exception as e:
            self.logger.error(f"Error checking data existence: {e}")
            return False

    async def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage backend."""
        try:
            info = {
                "storage_type": "parquet",
                "location": str(self.base_path),
                "compression": self.compression,
                "partition_scheme": self.partition_scheme,
                "use_object_storage": self.use_object_storage,
                "cache_enabled": self.cache_enabled,
            }

            # Add object storage info if enabled
            if self.use_object_storage and self.storage_client:
                storage_info = await self.storage_client.get_storage_info()
                info["object_storage"] = storage_info

            # Calculate local storage size
            total_size = 0
            file_count = 0
            for file_path in self.base_path.rglob("*.parquet"):
                total_size += file_path.stat().st_size
                file_count += 1

            info.update(
                {
                    "total_size": total_size,
                    "total_files": file_count,
                    "cache_size": self._get_cache_size(),
                }
            )

            return info

        except Exception as e:
            self.logger.error(f"Error getting storage info: {e}")
            return {"storage_type": "parquet", "error": str(e)}

    def _get_cache_size(self) -> int:
        """Get current cache size in bytes."""
        try:
            total_size = 0
            for file_path in self.local_cache_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception:
            return 0

    # Batch operations
    async def batch_save_ohlcv(self, data: List[OHLCVBatchSchema]) -> bool:
        """Save multiple OHLCV batch schemas."""
        try:
            success_count = 0
            for batch in data:
                if await self.save_ohlcv(
                    batch.exchange, batch.symbol, batch.timeframe, batch.records
                ):
                    success_count += 1

            self.logger.info(
                f"Batch save completed: {success_count}/{len(data)} batches successful"
            )
            return success_count == len(data)

        except Exception as e:
            self.logger.error(f"Error in batch save: {e}")
            raise RepositoryError(f"Batch save failed: {e}")

    async def optimize_storage(self) -> bool:
        """Optimize storage for better performance."""
        try:
            # Clear old cache files
            if self.cache_enabled:
                cache_size = self._get_cache_size()
                if cache_size > self.max_cache_size_mb * 1024 * 1024:
                    # Remove oldest cache files
                    cache_files = list(self.local_cache_dir.rglob("*"))
                    cache_files.sort(key=lambda x: x.stat().st_mtime)

                    removed_size = 0
                    target_size = (
                        self.max_cache_size_mb * 1024 * 1024 * 0.7
                    )  # Clean to 70%

                    for file_path in cache_files:
                        if file_path.is_file():
                            size = file_path.stat().st_size
                            file_path.unlink()
                            removed_size += size

                            if cache_size - removed_size <= target_size:
                                break

                    self.logger.info(f"Cache cleanup: removed {removed_size} bytes")

            self.logger.info("Storage optimization completed")
            return True

        except Exception as e:
            self.logger.error(f"Error optimizing storage: {e}")
            return False

    # Administrative operations
    async def count_all_records(self) -> int:
        """Count total number of OHLCV records in repository."""
        try:
            total_count = 0

            # Count local files
            for file_path in self.base_path.rglob("ohlcv_*.parquet"):
                try:
                    parquet_file = pq.ParquetFile(file_path)
                    total_count += parquet_file.metadata.num_rows
                except Exception:
                    continue

            self.logger.info(f"Total OHLCV records: {total_count}")
            return total_count

        except Exception as e:
            self.logger.error(f"Error counting records: {e}")
            return 0

    async def get_all_available_symbols(self) -> List[str]:
        """Get all available symbols across all exchanges."""
        try:
            all_symbols = set()

            # Scan local files
            ohlcv_path = self.base_path / "ohlcv"
            if ohlcv_path.exists():
                for exchange_dir in ohlcv_path.iterdir():
                    if exchange_dir.is_dir():
                        for symbol_dir in exchange_dir.iterdir():
                            if symbol_dir.is_dir():
                                all_symbols.add(symbol_dir.name)

            # Also check object storage
            if self.use_object_storage and self.storage_client:
                try:
                    objects = await self.storage_client.list_objects(prefix="ohlcv/")
                    for obj in objects:
                        parts = obj["key"].split("/")
                        if len(parts) >= 3:
                            all_symbols.add(parts[2])
                except StorageClientError:
                    pass

            symbol_list = sorted(list(all_symbols))
            self.logger.info(
                f"Found {len(symbol_list)} unique symbols across all exchanges"
            )
            return symbol_list

        except Exception as e:
            self.logger.error(f"Error getting all symbols: {e}")
            return []

    async def get_available_exchanges(self) -> List[str]:
        """Get all available exchanges in repository."""
        try:
            exchanges = set()

            # Scan local files
            ohlcv_path = self.base_path / "ohlcv"
            if ohlcv_path.exists():
                for exchange_dir in ohlcv_path.iterdir():
                    if exchange_dir.is_dir():
                        exchanges.add(exchange_dir.name)

            # Also check object storage
            if self.use_object_storage and self.storage_client:
                try:
                    objects = await self.storage_client.list_objects(prefix="ohlcv/")
                    for obj in objects:
                        parts = obj["key"].split("/")
                        if len(parts) >= 2:
                            exchanges.add(parts[1])
                except StorageClientError:
                    pass

            exchange_list = sorted(list(exchanges))
            self.logger.info(f"Found {len(exchange_list)} exchanges")
            return exchange_list

        except Exception as e:
            self.logger.error(f"Error getting available exchanges: {e}")
            return []

    async def get_all_available_timeframes(self) -> List[str]:
        """Get all available timeframes across all data."""
        try:
            timeframes = set()

            # Scan local files
            ohlcv_path = self.base_path / "ohlcv"
            if ohlcv_path.exists():
                for exchange_dir in ohlcv_path.iterdir():
                    if exchange_dir.is_dir():
                        for symbol_dir in exchange_dir.iterdir():
                            if symbol_dir.is_dir():
                                for tf_dir in symbol_dir.iterdir():
                                    if tf_dir.is_dir():
                                        timeframes.add(tf_dir.name)

            # Also check object storage
            if self.use_object_storage and self.storage_client:
                try:
                    objects = await self.storage_client.list_objects(prefix="ohlcv/")
                    for obj in objects:
                        parts = obj["key"].split("/")
                        if len(parts) >= 4:
                            timeframes.add(parts[3])
                except StorageClientError:
                    pass

            tf_list = sorted(list(timeframes))
            self.logger.info(f"Found {len(tf_list)} unique timeframes")
            return tf_list

        except Exception as e:
            self.logger.error(f"Error getting all timeframes: {e}")
            return []

    async def count_records(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """Count records for specific criteria."""
        try:
            if start_date and end_date:
                # Count records in date range
                query = OHLCVQuerySchema(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                )
                data = await self.get_ohlcv(query)
                return len(data)
            else:
                # Count all records for symbol/timeframe
                total_count = 0
                pattern_path = self.base_path / "ohlcv" / exchange / symbol / timeframe

                if pattern_path.exists():
                    for file_path in pattern_path.rglob("*.parquet"):
                        try:
                            parquet_file = pq.ParquetFile(file_path)
                            total_count += parquet_file.metadata.num_rows
                        except Exception:
                            continue

                return total_count

        except Exception as e:
            self.logger.error(f"Error counting records: {e}")
            return 0

    async def count_records_since(self, cutoff_date: datetime) -> int:
        """Count records since a specific date."""
        try:
            total_count = 0
            cutoff_date_obj = cutoff_date.date()

            # Scan all parquet files
            for file_path in self.base_path.rglob("ohlcv_*.parquet"):
                # Extract date from filename
                filename = file_path.stem
                if "_" in filename:
                    date_part = filename.split("_")[-1]
                    try:
                        file_date = datetime.strptime(date_part, "%Y%m%d").date()
                        if file_date >= cutoff_date_obj:
                            parquet_file = pq.ParquetFile(file_path)
                            total_count += parquet_file.metadata.num_rows
                    except (ValueError, Exception):
                        continue

            return total_count

        except Exception as e:
            self.logger.error(f"Error counting records since date: {e}")
            return 0

    async def delete_records_before_date(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        cutoff_date: datetime,
    ) -> int:
        """Delete records before a specific date."""
        try:
            deleted_count = 0
            cutoff_date_obj = cutoff_date.date()

            pattern_path = self.base_path / "ohlcv" / exchange / symbol / timeframe

            if pattern_path.exists():
                for file_path in pattern_path.rglob("*.parquet"):
                    # Extract date from filename
                    filename = file_path.stem
                    if "_" in filename:
                        date_part = filename.split("_")[-1]
                        try:
                            file_date = datetime.strptime(date_part, "%Y%m%d").date()
                            if file_date < cutoff_date_obj:
                                # Count records before deletion
                                try:
                                    parquet_file = pq.ParquetFile(file_path)
                                    deleted_count += parquet_file.metadata.num_rows
                                except Exception:
                                    pass

                                # Delete file
                                file_path.unlink()

                                # Delete from object storage if enabled
                                if self.use_object_storage and self.storage_client:
                                    relative_path = file_path.relative_to(
                                        self.base_path
                                    )
                                    object_key = str(relative_path).replace("\\", "/")
                                    try:
                                        await self.storage_client.delete_object(
                                            object_key
                                        )
                                    except StorageClientError:
                                        pass
                        except (ValueError, Exception):
                            continue

            self.logger.info(f"Deleted {deleted_count} records before {cutoff_date}")
            return deleted_count

        except Exception as e:
            self.logger.error(f"Error deleting records before date: {e}")
            return 0
