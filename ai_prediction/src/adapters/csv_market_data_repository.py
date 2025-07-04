# adapters/csv_market_data_repository.py

"""
CSV Market Data Repository Implementation

Implements the MarketDataRepository interface using CSV files for data storage.
Provides file-based persistence for market data with proper organization.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from ..interfaces.market_data_repository import MarketDataRepository
from ..interfaces.errors import RepositoryError, ValidationError
from ..common.logger import LoggerFactory
from ..utils.datetime_utils import DateTimeUtils
from ..schemas.ohlcv_schemas import OHLCVSchema, OHLCVBatchSchema, OHLCVQuerySchema
from ..models.ohlcv_models import OHLCVModelCSV
from ..converters.ohlcv_converter import OHLCVConverter


class CSVMarketDataRepository(MarketDataRepository):
    """CSV file-based implementation of MarketDataRepository"""

    def __init__(self, base_directory: str = "data"):
        """
        Initialize CSV repository with base directory

        Args:
            base_directory: Base directory for storing CSV files
        """
        self.base_directory = Path(base_directory)
        self.logger = LoggerFactory.get_logger(name="csv_repository")
        self.converter = OHLCVConverter()

        # Create base directory structure
        self._initialize_directory_structure()

        self.logger.info(f"Initialized CSV repository at {self.base_directory}")

    def save_ohlcv(
        self, exchange: str, symbol: str, timeframe: str, data: List[OHLCVSchema]
    ) -> bool:
        """Save OHLCV data to CSV file"""
        try:
            if not data:
                self.logger.warning(
                    f"No data to save for {exchange}/{symbol}/{timeframe}"
                )
                return True

            # Convert schemas to CSV models
            models = [self.converter.schema_to_csv_model(schema) for schema in data]

            # Validate models
            for model in models:
                self._validate_ohlcv_model(model)

            # Convert models to DataFrame
            df_data = []
            for model in models:
                model_dict = model.model_dump()
                # Ensure timestamp is properly formatted
                model_dict["timestamp"] = self._ensure_utc_timezone(model.timestamp)
                df_data.append(model_dict)

            df = pd.DataFrame(df_data)

            # Convert timestamp to datetime and set as index
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp").sort_index()

            # Get file path
            file_path = self._get_ohlcv_file_path(exchange, symbol, timeframe)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file exists and merge data
            if file_path.exists():
                existing_df = pd.read_csv(
                    file_path, index_col="timestamp", parse_dates=True
                )
                # Ensure existing data has UTC timezone
                if existing_df.index.tz is None:
                    existing_df.index = existing_df.index.tz_localize("UTC")
                elif existing_df.index.tz != df.index.tz:
                    existing_df.index = existing_df.index.tz_convert("UTC")

                # Merge and remove duplicates
                combined_df = pd.concat([existing_df, df])
                combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
                combined_df = combined_df.sort_index()

                df = combined_df

            # Save to CSV
            df.to_csv(file_path, index=True)

            self.logger.info(f"Saved {len(df)} OHLCV records to {file_path}")
            return True

        except Exception as e:
            raise RepositoryError(f"Failed to save OHLCV data: {str(e)}")

    def get_ohlcv(self, query: OHLCVQuerySchema) -> List[OHLCVSchema]:
        """Retrieve OHLCV data from CSV file"""
        try:
            # Get file path
            file_path = self._get_ohlcv_file_path(
                query.exchange, query.symbol, query.timeframe
            )

            if not file_path.exists():
                self.logger.warning(f"No data file found at {file_path}")
                return []

            # Read CSV
            df = pd.read_csv(file_path, index_col="timestamp", parse_dates=True)

            # Ensure timezone-aware timestamps
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            # Filter by date range
            mask = (df.index >= query.start_date) & (df.index <= query.end_date)
            filtered_df = df.loc[mask]

            # Apply limit if specified
            if query.limit:
                filtered_df = filtered_df.head(query.limit)

            # Convert to CSV models, then to schemas
            models = []
            for timestamp, row in filtered_df.iterrows():
                model = OHLCVModelCSV(
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
                models.append(model)

            # Convert models to schemas
            schemas = [self.converter.csv_model_to_schema(model) for model in models]

            self.logger.info(f"Retrieved {len(schemas)} OHLCV records from {file_path}")
            return schemas

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
        """Delete OHLCV data from CSV file"""
        try:
            file_path = self._get_ohlcv_file_path(exchange, symbol, timeframe)

            if not file_path.exists():
                self.logger.warning(f"No data file found at {file_path}")
                return True

            # If no date range specified, delete entire file
            if not start_date and not end_date:
                file_path.unlink()
                self.logger.info(f"Deleted entire file {file_path}")
                return True

            # Parse dates and filter data
            start_dt, end_dt = DateTimeUtils.validate_date_range(start_date, end_date)

            df = pd.read_csv(file_path, index_col="timestamp", parse_dates=True)

            # Ensure timezone-aware timestamps
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            # Remove data in specified range
            mask = ~((df.index >= start_dt) & (df.index <= end_dt))
            filtered_df = df.loc[mask]

            if filtered_df.empty:
                # Delete file if no data remains
                file_path.unlink()
                self.logger.info(f"Deleted entire file {file_path} (no data remaining)")
            else:
                # Save remaining data
                filtered_df.to_csv(file_path, index=True)
                self.logger.info(
                    f"Deleted data from {start_date} to {end_date} in {file_path}"
                )

            return True

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
        """Get all available symbols for an exchange"""
        try:
            exchange_dir = self.base_directory / exchange / "ohlcv"

            if not exchange_dir.exists():
                return []

            symbols = set()
            for file_path in exchange_dir.rglob("*.csv"):
                # Extract symbol from file path structure
                # Expected: exchange/ohlcv/symbol/timeframe.csv
                if len(file_path.parts) >= 4:
                    symbol = file_path.parts[-2]
                    symbols.add(symbol)

            return sorted(list(symbols))

        except Exception as e:
            raise RepositoryError(f"Failed to get available symbols: {str(e)}")

    def get_available_timeframes(self, exchange: str, symbol: str) -> List[str]:
        """Get all available timeframes for a symbol"""
        try:
            symbol_dir = self.base_directory / exchange / "ohlcv" / symbol

            if not symbol_dir.exists():
                return []

            timeframes = []
            for file_path in symbol_dir.glob("*.csv"):
                timeframe = file_path.stem
                timeframes.append(timeframe)

            return sorted(timeframes)

        except Exception as e:
            raise RepositoryError(f"Failed to get available timeframes: {str(e)}")

    def get_data_range(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        timeframe: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        """Get the date range of available data"""
        try:
            if data_type == "ohlcv" and timeframe:
                file_path = self._get_ohlcv_file_path(exchange, symbol, timeframe)

                if not file_path.exists():
                    return None

                df = pd.read_csv(file_path, index_col="timestamp", parse_dates=True)

                if df.empty:
                    return None

                # Ensure timezone-aware timestamps
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")

                return {
                    "start_date": DateTimeUtils.to_iso_string(df.index.min()),
                    "end_date": DateTimeUtils.to_iso_string(df.index.max()),
                }
            else:
                raise NotImplementedError(f"Data type {data_type} not implemented yet")

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
        """Check if data exists for the specified criteria"""
        try:
            if data_type == "ohlcv" and timeframe:
                data_range = self.get_data_range(exchange, symbol, data_type, timeframe)

                if not data_range:
                    return False

                start_dt, end_dt = DateTimeUtils.validate_date_range(
                    start_date, end_date
                )
                range_start = DateTimeUtils.to_utc_datetime(data_range["start_date"])
                range_end = DateTimeUtils.to_utc_datetime(data_range["end_date"])

                return range_start <= start_dt and end_dt <= range_end
            else:
                return False

        except Exception as e:
            raise RepositoryError(f"Failed to check data existence: {str(e)}")

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage backend"""
        try:
            total_size = 0
            total_files = 0
            exchanges = set()
            total_symbols = 0
            oldest_date = None
            newest_date = None

            # Walk through all files
            for file_path in self.base_directory.rglob("*.csv"):
                # Calculate size
                total_size += file_path.stat().st_size
                total_files += 1

                # Extract exchange from path
                if len(file_path.parts) >= 3:
                    exchange = (
                        file_path.parts[-4]
                        if len(file_path.parts) >= 4
                        else file_path.parts[-3]
                    )
                    exchanges.add(exchange)

                # Get date range from file
                try:
                    df = pd.read_csv(file_path, index_col="timestamp", parse_dates=True)
                    if not df.empty:
                        # Ensure timezone-aware timestamps
                        if df.index.tz is None:
                            df.index = df.index.tz_localize("UTC")
                        else:
                            df.index = df.index.tz_convert("UTC")

                        file_oldest = df.index.min()
                        file_newest = df.index.max()

                        if oldest_date is None or file_oldest < oldest_date:
                            oldest_date = file_oldest
                        if newest_date is None or file_newest > newest_date:
                            newest_date = file_newest
                except:
                    continue

            # Count unique symbols
            for exchange in exchanges:
                symbols = self.get_available_symbols(exchange)
                total_symbols += len(symbols)

            return {
                "storage_type": "file",
                "location": str(self.base_directory.absolute()),
                "total_size": total_size,
                "total_files": total_files,
                "available_exchanges": sorted(list(exchanges)),
                "total_symbols": total_symbols,
                "oldest_data": (
                    DateTimeUtils.to_iso_string(oldest_date) if oldest_date else None
                ),
                "newest_data": (
                    DateTimeUtils.to_iso_string(newest_date) if newest_date else None
                ),
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
        """Optimize storage by consolidating and cleaning up files"""
        try:
            # For CSV storage, optimization could involve:
            # 1. Removing duplicate entries
            # 2. Sorting data by timestamp
            # 3. Compressing old files

            optimized_count = 0

            for file_path in self.base_directory.rglob("*.csv"):
                try:
                    df = pd.read_csv(file_path, index_col="timestamp", parse_dates=True)

                    # Remove duplicates and sort
                    original_count = len(df)
                    df = df[~df.index.duplicated(keep="last")].sort_index()

                    if len(df) != original_count:
                        df.to_csv(file_path, index=True)
                        optimized_count += 1
                        self.logger.info(
                            f"Optimized {file_path}: removed {original_count - len(df)} duplicates"
                        )

                except Exception as e:
                    self.logger.warning(f"Failed to optimize {file_path}: {e}")

            self.logger.info(
                f"Optimization completed: {optimized_count} files optimized"
            )
            return True

        except Exception as e:
            raise RepositoryError(f"Storage optimization failed: {str(e)}")

    def _initialize_directory_structure(self) -> None:
        """Initialize the directory structure for CSV storage"""
        self.base_directory.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different data types
        data_types = ["ohlcv", "trades", "orderbook", "tickers"]
        for data_type in data_types:
            (self.base_directory / data_type).mkdir(exist_ok=True)

    def _get_ohlcv_file_path(self, exchange: str, symbol: str, timeframe: str) -> Path:
        """Get the file path for OHLCV data"""
        return self.base_directory / exchange / "ohlcv" / symbol / f"{timeframe}.csv"

    def _validate_ohlcv_model(self, model: OHLCVModelCSV) -> None:
        """Validate OHLCV model data"""
        try:
            # The model validation is already handled by Pydantic
            # This method is kept for potential additional business logic validation
            pass
        except Exception as e:
            raise ValidationError(f"Model validation failed: {str(e)}")

    def _ensure_utc_timezone(self, dt: datetime) -> datetime:
        """
        Ensure datetime has UTC timezone.

        Args:
            dt: Datetime instance

        Returns:
            UTC timezone-aware datetime
        """
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        elif dt.tzinfo != timezone.utc:
            return dt.astimezone(timezone.utc)
        return dt
