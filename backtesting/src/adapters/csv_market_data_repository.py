# adapters/csv_market_data_repository.py

"""
CSV Market Data Repository Implementation

Implements the MarketDataRepository interface using CSV files for storage.
Provides efficient file-based storage and querying for market data.
"""

import asyncio
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..interfaces.market_data_repository import MarketDataRepository
from ..interfaces.errors import RepositoryError
from common.logger import LoggerFactory
from ..schemas.ohlcv_schemas import OHLCVSchema, OHLCVBatchSchema, OHLCVQuerySchema
from ..models.ohlcv_models import OHLCVModelCSV
from ..converters.ohlcv_converter import OHLCVConverter
from ..utils.datetime_utils import DateTimeUtils


class CSVMarketDataRepository(MarketDataRepository):
    """CSV file implementation of MarketDataRepository for file-based storage"""

    def __init__(self, base_directory: str = "data/market_data"):
        """
        Initialize CSV repository.

        Args:
            base_directory: Base directory for storing CSV files
        """
        self.base_directory = Path(base_directory)
        self.logger = LoggerFactory.get_logger(name="csv_repository")
        self.converter = OHLCVConverter()

        # Create base directory if it doesn't exist
        self.base_directory.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialized CSV repository at {self.base_directory}")

    def _get_file_path(self, exchange: str, symbol: str, timeframe: str) -> Path:
        """Get file path for specific exchange/symbol/timeframe using hierarchical structure"""
        # Create hierarchical directory structure: {base_directory}/{exchange}/{symbol}/{timeframe}.csv
        exchange_dir = self.base_directory / exchange
        symbol_dir = exchange_dir / symbol
        filename = f"{timeframe}.csv"
        return symbol_dir / filename

    def _ensure_file_exists(self, file_path: Path) -> None:
        """Ensure CSV file exists with proper headers and directory structure"""
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not file_path.exists():
            # Create file with headers
            headers = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "symbol",
                "exchange",
                "timeframe",
            ]

            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    # OHLCV Operations
    async def save_ohlcv(
        self, exchange: str, symbol: str, timeframe: str, data: List[OHLCVSchema]
    ) -> bool:
        """Save OHLCV data to CSV file"""

        def _save_sync():
            try:
                file_path = self._get_file_path(exchange, symbol, timeframe)
                self._ensure_file_exists(file_path)

                # Read existing data to avoid duplicates
                existing_timestamps = set()
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            existing_timestamps.add(row["timestamp"])

                # Filter out existing records
                new_records = []
                for schema in data:
                    timestamp_str = DateTimeUtils.to_iso_string(schema.timestamp)
                    if timestamp_str not in existing_timestamps:
                        csv_model = self.converter.schema_to_csv_model(schema)
                        new_records.append(csv_model)

                if not new_records:
                    self.logger.info("No new records to save")
                    return True

                # Append new records
                with open(file_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "timestamp",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "symbol",
                            "exchange",
                            "timeframe",
                        ],
                    )

                    for record in new_records:
                        writer.writerow(
                            {
                                "timestamp": DateTimeUtils.to_iso_string(
                                    record.timestamp
                                ),
                                "open": record.open,
                                "high": record.high,
                                "low": record.low,
                                "close": record.close,
                                "volume": record.volume,
                                "symbol": record.symbol,
                                "exchange": record.exchange,
                                "timeframe": record.timeframe,
                            }
                        )

                self.logger.info(
                    f"Saved {len(new_records)} OHLCV records to {file_path}"
                )
                return True

            except Exception as e:
                raise RepositoryError(f"Failed to save OHLCV data: {str(e)}")

        return await asyncio.to_thread(_save_sync)

    async def get_ohlcv(self, query: OHLCVQuerySchema) -> List[OHLCVSchema]:
        """Get OHLCV data from CSV file"""

        def _get_sync():
            try:
                file_path = self._get_file_path(
                    query.exchange, query.symbol, query.timeframe
                )

                if not file_path.exists():
                    return []

                records = []
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)

                    for row in reader:
                        timestamp = DateTimeUtils.parse_iso_string(row["timestamp"])

                        # Apply date filters
                        if query.start_date and timestamp < query.start_date:
                            continue
                        if query.end_date and timestamp > query.end_date:
                            continue

                        schema = OHLCVSchema(
                            timestamp=timestamp,
                            open=float(row["open"]),
                            high=float(row["high"]),
                            low=float(row["low"]),
                            close=float(row["close"]),
                            volume=float(row["volume"]),
                            symbol=row["symbol"],
                            exchange=row["exchange"],
                            timeframe=row["timeframe"],
                        )
                        records.append(schema)

                # Apply limit
                if query.limit and len(records) > query.limit:
                    records = records[: query.limit]

                return records

            except Exception as e:
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
        """Delete OHLCV data from CSV file"""

        def _delete_sync():
            try:
                file_path = self._get_file_path(exchange, symbol, timeframe)

                if not file_path.exists():
                    return True

                # Read all records
                remaining_records = []
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)

                    for row in reader:
                        timestamp = DateTimeUtils.parse_iso_string(row["timestamp"])

                        # Check if record should be deleted
                        should_delete = True
                        if start_date:
                            start_dt = DateTimeUtils.parse_iso_string(start_date)
                            if timestamp < start_dt:
                                should_delete = False
                        if end_date:
                            end_dt = DateTimeUtils.parse_iso_string(end_date)
                            if timestamp > end_dt:
                                should_delete = False

                        if not should_delete:
                            remaining_records.append(row)

                # Rewrite file with remaining records
                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    if remaining_records:
                        fieldnames = remaining_records[0].keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(remaining_records)
                    else:
                        # Write just headers if no records remain
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                                "symbol",
                                "exchange",
                                "timeframe",
                            ]
                        )

                return True

            except Exception as e:
                raise RepositoryError(f"Failed to delete OHLCV data: {str(e)}")

        return await asyncio.to_thread(_delete_sync)

    # Trade Operations (placeholder implementations)
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

    # Order Book Operations (placeholder implementations)
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

    # Ticker Operations (placeholder implementations)
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

    # Query Operations
    async def get_available_symbols(self, exchange: str) -> List[str]:
        """Get all available symbols for an exchange from CSV files"""

        def _get_symbols_sync():
            try:
                symbols = set()

                # Scan the exchange directory for symbol subdirectories
                exchange_dir = self.base_directory / exchange
                if exchange_dir.exists() and exchange_dir.is_dir():
                    for symbol_dir in exchange_dir.iterdir():
                        if symbol_dir.is_dir():
                            symbols.add(symbol_dir.name)

                return sorted(list(symbols))

            except Exception as e:
                raise RepositoryError(f"Failed to get available symbols: {str(e)}")

        return await asyncio.to_thread(_get_symbols_sync)

    async def get_available_timeframes(self, exchange: str, symbol: str) -> List[str]:
        """Get all available timeframes for a symbol from CSV files"""

        def _get_timeframes_sync():
            try:
                timeframes = set()

                # Scan the symbol directory for timeframe CSV files
                symbol_dir = self.base_directory / exchange / symbol
                if symbol_dir.exists() and symbol_dir.is_dir():
                    for file_path in symbol_dir.glob("*.csv"):
                        # Filename is the timeframe (e.g., "1h.csv")
                        timeframe = file_path.stem
                        timeframes.add(timeframe)

                return sorted(list(timeframes))

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
        """Get data range for symbol from CSV files"""

        def _get_range_sync():
            try:
                if data_type != "ohlcv" or not timeframe:
                    return None

                file_path = self._get_file_path(exchange, symbol, timeframe)

                if not file_path.exists():
                    return None

                min_timestamp = None
                max_timestamp = None

                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)

                    for row in reader:
                        timestamp = DateTimeUtils.parse_iso_string(row["timestamp"])

                        if min_timestamp is None or timestamp < min_timestamp:
                            min_timestamp = timestamp
                        if max_timestamp is None or timestamp > max_timestamp:
                            max_timestamp = timestamp

                if min_timestamp and max_timestamp:
                    return {
                        "start_date": DateTimeUtils.to_iso_string(min_timestamp),
                        "end_date": DateTimeUtils.to_iso_string(max_timestamp),
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
                if data_type != "ohlcv" or not timeframe:
                    return False

                file_path = self._get_file_path(exchange, symbol, timeframe)

                if not file_path.exists():
                    return False

                start_dt = DateTimeUtils.parse_iso_string(start_date)
                end_dt = DateTimeUtils.parse_iso_string(end_date)

                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)

                    for row in reader:
                        timestamp = DateTimeUtils.parse_iso_string(row["timestamp"])

                        if start_dt <= timestamp <= end_dt:
                            return True

                return False

            except Exception as e:
                raise RepositoryError(f"Failed to check data exists: {str(e)}")

        return await asyncio.to_thread(_check_exists_sync)

    async def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information for CSV files"""

        def _get_info_sync():
            try:
                total_size = 0
                file_count = 0
                exchanges = set()
                symbols = set()
                timeframes = set()

                # Scan hierarchical directory structure
                for exchange_dir in self.base_directory.iterdir():
                    if exchange_dir.is_dir():
                        exchange = exchange_dir.name
                        exchanges.add(exchange)

                        for symbol_dir in exchange_dir.iterdir():
                            if symbol_dir.is_dir():
                                symbol = symbol_dir.name
                                symbols.add(symbol)

                                for csv_file in symbol_dir.glob("*.csv"):
                                    total_size += csv_file.stat().st_size
                                    file_count += 1

                                    timeframe = csv_file.stem
                                    timeframes.add(timeframe)

                return {
                    "storage_type": "file",
                    "location": str(self.base_directory.absolute()),
                    "total_size": total_size,
                    "file_count": file_count,
                    "available_exchanges": sorted(list(exchanges)),
                    "total_symbols": len(symbols),
                    "total_timeframes": len(timeframes),
                }

            except Exception as e:
                raise RepositoryError(f"Failed to get storage info: {str(e)}")

        return await asyncio.to_thread(_get_info_sync)

    # Batch Operations
    async def batch_save_ohlcv(self, data: List[OHLCVBatchSchema]) -> bool:
        """Batch save OHLCV data"""

        def _batch_save_sync():
            try:
                success_count = 0

                for batch in data:
                    try:
                        # Group by exchange/symbol/timeframe
                        file_groups = {}
                        for schema in batch.data:
                            key = (schema.exchange, schema.symbol, schema.timeframe)
                            if key not in file_groups:
                                file_groups[key] = []
                            file_groups[key].append(schema)

                        # Save each group
                        for (
                            exchange,
                            symbol,
                            timeframe,
                        ), schemas in file_groups.items():
                            file_path = self._get_file_path(exchange, symbol, timeframe)
                            self._ensure_file_exists(file_path)

                            # Convert and save
                            csv_models = [
                                self.converter.schema_to_csv_model(schema)
                                for schema in schemas
                            ]

                            with open(
                                file_path, "a", newline="", encoding="utf-8"
                            ) as f:
                                writer = csv.DictWriter(
                                    f,
                                    fieldnames=[
                                        "timestamp",
                                        "open",
                                        "high",
                                        "low",
                                        "close",
                                        "volume",
                                        "symbol",
                                        "exchange",
                                        "timeframe",
                                    ],
                                )

                                for model in csv_models:
                                    writer.writerow(
                                        {
                                            "timestamp": DateTimeUtils.to_iso_string(
                                                model.timestamp
                                            ),
                                            "open": model.open,
                                            "high": model.high,
                                            "low": model.low,
                                            "close": model.close,
                                            "volume": model.volume,
                                            "symbol": model.symbol,
                                            "exchange": model.exchange,
                                            "timeframe": model.timeframe,
                                        }
                                    )

                        success_count += 1

                    except Exception as e:
                        self.logger.error(f"Failed to save batch: {str(e)}")

                return success_count == len(data)

            except Exception as e:
                raise RepositoryError(f"Failed to batch save OHLCV data: {str(e)}")

        return await asyncio.to_thread(_batch_save_sync)

    async def optimize_storage(self) -> bool:
        """Optimize CSV storage (sort files by timestamp)"""

        def _optimize_sync():
            try:
                optimized_count = 0

                # Scan hierarchical directory structure
                for exchange_dir in self.base_directory.iterdir():
                    if exchange_dir.is_dir():
                        for symbol_dir in exchange_dir.iterdir():
                            if symbol_dir.is_dir():
                                for csv_file in symbol_dir.glob("*.csv"):
                                    try:
                                        # Read all records
                                        records = []
                                        with open(csv_file, "r", encoding="utf-8") as f:
                                            reader = csv.DictReader(f)
                                            records = list(reader)

                                        if not records:
                                            continue

                                        # Sort by timestamp
                                        records.sort(key=lambda x: x["timestamp"])

                                        # Rewrite file
                                        with open(
                                            csv_file, "w", newline="", encoding="utf-8"
                                        ) as f:
                                            if records:
                                                writer = csv.DictWriter(
                                                    f, fieldnames=records[0].keys()
                                                )
                                                writer.writeheader()
                                                writer.writerows(records)

                                        optimized_count += 1

                                    except Exception as e:
                                        self.logger.error(
                                            f"Failed to optimize {csv_file}: {str(e)}"
                                        )

                self.logger.info(f"Optimized {optimized_count} CSV files")
                return True

            except Exception as e:
                raise RepositoryError(f"Failed to optimize storage: {str(e)}")

        return await asyncio.to_thread(_optimize_sync)

    # Administrative Operations
    async def count_all_records(self) -> int:
        """Count total number of OHLCV records in repository"""

        def _count_all_sync():
            try:
                total_count = 0

                # Scan hierarchical directory structure
                for exchange_dir in self.base_directory.iterdir():
                    if exchange_dir.is_dir():
                        for symbol_dir in exchange_dir.iterdir():
                            if symbol_dir.is_dir():
                                for csv_file in symbol_dir.glob("*.csv"):
                                    with open(csv_file, "r", encoding="utf-8") as f:
                                        reader = csv.reader(f)
                                        # Skip header
                                        next(reader, None)
                                        # Count remaining rows
                                        count = sum(1 for _ in reader)
                                        total_count += count

                return total_count

            except Exception as e:
                raise RepositoryError(f"Failed to count all records: {str(e)}")

        return await asyncio.to_thread(_count_all_sync)

    async def get_all_available_symbols(self) -> List[str]:
        """Get all available symbols across all exchanges"""

        def _get_all_symbols_sync():
            try:
                symbols = set()

                # Scan hierarchical directory structure
                for exchange_dir in self.base_directory.iterdir():
                    if exchange_dir.is_dir():
                        for symbol_dir in exchange_dir.iterdir():
                            if symbol_dir.is_dir():
                                symbol = symbol_dir.name
                                symbols.add(symbol)

                return sorted(list(symbols))

            except Exception as e:
                raise RepositoryError(f"Failed to get all available symbols: {str(e)}")

        return await asyncio.to_thread(_get_all_symbols_sync)

    async def get_available_exchanges(self) -> List[str]:
        """Get all available exchanges in repository"""

        def _get_exchanges_sync():
            try:
                exchanges = set()

                # Scan hierarchical directory structure
                for exchange_dir in self.base_directory.iterdir():
                    if exchange_dir.is_dir():
                        exchange = exchange_dir.name
                        exchanges.add(exchange)

                return sorted(list(exchanges))

            except Exception as e:
                raise RepositoryError(f"Failed to get available exchanges: {str(e)}")

        return await asyncio.to_thread(_get_exchanges_sync)

    async def get_all_available_timeframes(self) -> List[str]:
        """Get all available timeframes across all data"""

        def _get_all_timeframes_sync():
            try:
                timeframes = set()

                # Scan hierarchical directory structure
                for exchange_dir in self.base_directory.iterdir():
                    if exchange_dir.is_dir():
                        for symbol_dir in exchange_dir.iterdir():
                            if symbol_dir.is_dir():
                                for csv_file in symbol_dir.glob("*.csv"):
                                    timeframe = csv_file.stem
                                    timeframes.add(timeframe)

                return sorted(list(timeframes))

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
                file_path = self._get_file_path(exchange, symbol, timeframe)

                if not file_path.exists():
                    return 0

                count = 0
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)

                    for row in reader:
                        timestamp = DateTimeUtils.parse_iso_string(row["timestamp"])

                        # Apply date filters
                        if start_date and timestamp < start_date:
                            continue
                        if end_date and timestamp > end_date:
                            continue

                        count += 1

                return count

            except Exception as e:
                raise RepositoryError(f"Failed to count records: {str(e)}")

        return await asyncio.to_thread(_count_records_sync)

    async def count_records_since(self, cutoff_date: datetime) -> int:
        """Count records since a specific date"""

        def _count_since_sync():
            try:
                total_count = 0

                # Scan hierarchical directory structure
                for exchange_dir in self.base_directory.iterdir():
                    if exchange_dir.is_dir():
                        for symbol_dir in exchange_dir.iterdir():
                            if symbol_dir.is_dir():
                                for csv_file in symbol_dir.glob("*.csv"):
                                    with open(csv_file, "r", encoding="utf-8") as f:
                                        reader = csv.DictReader(f)

                                        for row in reader:
                                            timestamp = DateTimeUtils.parse_iso_string(
                                                row["timestamp"]
                                            )
                                            if timestamp >= cutoff_date:
                                                total_count += 1

                return total_count

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
                file_path = self._get_file_path(exchange, symbol, timeframe)

                if not file_path.exists():
                    return 0

                deleted_count = 0
                remaining_records = []

                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)

                    for row in reader:
                        timestamp = DateTimeUtils.parse_iso_string(row["timestamp"])

                        if timestamp < cutoff_date:
                            deleted_count += 1
                        else:
                            remaining_records.append(row)

                # Rewrite file with remaining records
                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    if remaining_records:
                        fieldnames = remaining_records[0].keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(remaining_records)
                    else:
                        # Write just headers if no records remain
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                                "symbol",
                                "exchange",
                                "timeframe",
                            ]
                        )

                return deleted_count

            except Exception as e:
                raise RepositoryError(f"Failed to delete records before date: {str(e)}")

        return await asyncio.to_thread(_delete_before_sync)

    def close(self) -> None:
        """Close CSV repository (no-op for file-based storage)"""
        self.logger.info("Closed CSV repository")
