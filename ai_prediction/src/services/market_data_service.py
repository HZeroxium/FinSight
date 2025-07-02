# services/market_data_service.py

"""
Market Data Service

Provides business logic layer for market data operations.
Acts as an abstraction over MarketDataRepository implementations.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from ..interfaces.market_data_repository import MarketDataRepository
from ..interfaces.errors import ValidationError
from ..common.logger import LoggerFactory
from ..utils.datetime_utils import DateTimeUtils


class MarketDataService:
    """
    Service layer for market data operations.

    Provides high-level business logic for OHLCV data management,
    including data validation, gap detection, and batch operations.
    """

    def __init__(self, repository: MarketDataRepository):
        """
        Initialize market data service.

        Args:
            repository: MarketDataRepository implementation
        """
        self.repository = repository
        self.logger = LoggerFactory.get_logger(
            name="market_data_service",
        )

        self.logger.info("Market Data Service initialized")

    def save_ohlcv_data(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        data: List[Dict[str, Any]],
        validate: bool = True,
    ) -> bool:
        """
        Save OHLCV data with validation and business logic.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Time interval
            data: List of OHLCV records
            validate: Whether to validate data before saving

        Returns:
            True if save successful

        Raises:
            ValidationError: If data validation fails
            RepositoryError: If save operation fails
        """
        try:
            if not data:
                self.logger.warning(
                    f"No data provided for {exchange}/{symbol}/{timeframe}"
                )
                return True

            # Validate data format if requested
            if validate:
                self._validate_ohlcv_data(data)

            # Sort data by timestamp to ensure proper ordering
            sorted_data = sorted(
                data, key=lambda x: DateTimeUtils.to_utc_datetime(x["timestamp"])
            )

            # Save to repository
            success = self.repository.save_ohlcv(
                exchange, symbol, timeframe, sorted_data
            )

            if success:
                self.logger.info(
                    f"Saved {len(sorted_data)} OHLCV records for {exchange}/{symbol}/{timeframe}"
                )
            else:
                self.logger.error(
                    f"Failed to save OHLCV data for {exchange}/{symbol}/{timeframe}"
                )

            return success

        except Exception as e:
            self.logger.error(f"Error saving OHLCV data: {str(e)}")
            raise

    def get_ohlcv_data(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve OHLCV data for specified date range.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Time interval
            start_date: Start date in ISO 8601 format
            end_date: End date in ISO 8601 format

        Returns:
            List of OHLCV records

        Raises:
            ValidationError: If date format is invalid
            RepositoryError: If retrieval fails
        """
        try:
            # Validate date format
            DateTimeUtils.validate_date_range(start_date, end_date)

            # Retrieve data from repository
            data = self.repository.get_ohlcv(
                exchange, symbol, timeframe, start_date, end_date
            )

            self.logger.info(
                f"Retrieved {len(data)} OHLCV records for {exchange}/{symbol}/{timeframe} "
                f"from {start_date} to {end_date}"
            )

            return data

        except Exception as e:
            self.logger.error(f"Error retrieving OHLCV data: {str(e)}")
            raise

    def get_latest_ohlcv_timestamp(
        self, exchange: str, symbol: str, timeframe: str
    ) -> Optional[str]:
        """
        Get the timestamp of the latest OHLCV record.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Time interval

        Returns:
            Latest timestamp in ISO 8601 format, None if no data exists
        """
        try:
            data_range = self.repository.get_data_range(
                exchange, symbol, "ohlcv", timeframe
            )

            if data_range:
                return data_range["end_date"]

            return None

        except Exception as e:
            self.logger.error(f"Error getting latest timestamp: {str(e)}")
            return None

    def get_data_gaps(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> List[Tuple[str, str]]:
        """
        Identify gaps in OHLCV data within specified date range.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Time interval
            start_date: Start date in ISO 8601 format
            end_date: End date in ISO 8601 format

        Returns:
            List of (gap_start, gap_end) tuples in ISO 8601 format
        """
        try:
            # Get existing data
            data = self.get_ohlcv_data(
                exchange, symbol, timeframe, start_date, end_date
            )

            if not data:
                # If no data exists, entire range is a gap
                return [(start_date, end_date)]

            # Convert timeframe to timedelta
            interval = DateTimeUtils.parse_timeframe_to_timedelta(timeframe)
            if not interval:
                self.logger.warning(f"Could not parse timeframe: {timeframe}")
                return []

            # Parse dates
            start_dt, end_dt = DateTimeUtils.validate_date_range(start_date, end_date)

            # Create expected timestamps
            expected_timestamps = []
            current = start_dt
            while current <= end_dt:
                expected_timestamps.append(current)
                current += interval

            # Get existing timestamps
            existing_timestamps = set()
            for record in data:
                ts = DateTimeUtils.to_utc_datetime(record["timestamp"])
                existing_timestamps.add(ts)

            # Find gaps
            gaps = []
            gap_start = None

            for expected_ts in expected_timestamps:
                if expected_ts not in existing_timestamps:
                    if gap_start is None:
                        gap_start = expected_ts
                else:
                    if gap_start is not None:
                        # End of gap - use previous timestamp as gap end
                        gap_end = expected_ts - interval
                        # Only add gap if it's valid (start <= end)
                        if gap_start <= gap_end:
                            gaps.append(
                                (
                                    DateTimeUtils.to_iso_string(gap_start),
                                    DateTimeUtils.to_iso_string(gap_end),
                                )
                            )
                        gap_start = None

            # Handle gap at the end
            if gap_start is not None:
                gaps.append(
                    (
                        DateTimeUtils.to_iso_string(gap_start),
                        DateTimeUtils.to_iso_string(end_dt),
                    )
                )

            if gaps:
                self.logger.info(
                    f"Found {len(gaps)} data gaps for {exchange}/{symbol}/{timeframe}"
                )

            return gaps

        except Exception as e:
            self.logger.error(f"Error detecting data gaps: {str(e)}")
            return []

    def delete_ohlcv_data(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> bool:
        """
        Delete OHLCV data for specified criteria.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Time interval
            start_date: Optional start date in ISO 8601 format
            end_date: Optional end date in ISO 8601 format

        Returns:
            True if deletion successful
        """
        try:
            # Validate date range if provided
            if start_date and end_date:
                DateTimeUtils.validate_date_range(start_date, end_date)

            success = self.repository.delete_ohlcv(
                exchange, symbol, timeframe, start_date, end_date
            )

            if success:
                date_info = (
                    f" from {start_date} to {end_date}"
                    if start_date and end_date
                    else ""
                )
                self.logger.info(
                    f"Deleted OHLCV data for {exchange}/{symbol}/{timeframe}{date_info}"
                )

            return success

        except Exception as e:
            self.logger.error(f"Error deleting OHLCV data: {str(e)}")
            raise

    def get_available_symbols(self, exchange: str) -> List[str]:
        """Get all available symbols for an exchange."""
        try:
            symbols = self.repository.get_available_symbols(exchange)
            self.logger.info(f"Found {len(symbols)} symbols for {exchange}")
            return symbols

        except Exception as e:
            self.logger.error(f"Error getting available symbols: {str(e)}")
            return []

    def get_available_timeframes(self, exchange: str, symbol: str) -> List[str]:
        """Get all available timeframes for a symbol."""
        try:
            timeframes = self.repository.get_available_timeframes(exchange, symbol)
            self.logger.info(
                f"Found {len(timeframes)} timeframes for {exchange}/{symbol}"
            )
            return timeframes

        except Exception as e:
            self.logger.error(f"Error getting available timeframes: {str(e)}")
            return []

    def check_data_exists(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> bool:
        """Check if complete data exists for the specified range."""
        try:
            return self.repository.check_data_exists(
                exchange, symbol, "ohlcv", start_date, end_date, timeframe
            )

        except Exception as e:
            self.logger.error(f"Error checking data existence: {str(e)}")
            return False

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage backend."""
        try:
            return self.repository.get_storage_info()

        except Exception as e:
            self.logger.error(f"Error getting storage info: {str(e)}")
            return {}

    def batch_save_ohlcv(self, datasets: List[Dict[str, Any]]) -> bool:
        """
        Save multiple OHLCV datasets in batch.

        Args:
            datasets: List of dataset dictionaries with exchange, symbol, timeframe, and records

        Returns:
            True if all saves successful
        """
        try:
            # Validate all datasets first
            for dataset in datasets:
                required_fields = ["exchange", "symbol", "timeframe", "records"]
                for field in required_fields:
                    if field not in dataset:
                        raise ValidationError(f"Missing required field: {field}")

                if dataset["records"]:
                    self._validate_ohlcv_data(dataset["records"])

            # Use repository batch operation
            success = self.repository.batch_save_ohlcv(datasets)

            if success:
                total_records = sum(len(d["records"]) for d in datasets)
                self.logger.info(
                    f"Batch saved {len(datasets)} datasets with {total_records} total records"
                )

            return success

        except Exception as e:
            self.logger.error(f"Error in batch save: {str(e)}")
            raise

    def _validate_ohlcv_data(self, data: List[Dict[str, Any]]) -> None:
        """Validate OHLCV data format and values."""
        required_fields = ["timestamp", "open", "high", "low", "close", "volume"]

        for i, record in enumerate(data):
            # Check required fields
            for field in required_fields:
                if field not in record:
                    raise ValidationError(
                        f"Record {i}: Missing required field '{field}'"
                    )

            # Validate timestamp format
            try:
                DateTimeUtils.to_utc_datetime(record["timestamp"])
            except ValueError as e:
                raise ValidationError(f"Record {i}: Invalid timestamp format: {e}")

            # Validate numeric values
            try:
                open_price = float(record["open"])
                high_price = float(record["high"])
                low_price = float(record["low"])
                close_price = float(record["close"])
                volume = float(record["volume"])

                # Basic OHLC validation
                if not (low_price <= open_price <= high_price):
                    raise ValidationError(
                        f"Record {i}: Open price not within high/low range"
                    )

                if not (low_price <= close_price <= high_price):
                    raise ValidationError(
                        f"Record {i}: Close price not within high/low range"
                    )

                if volume < 0:
                    raise ValidationError(f"Record {i}: Volume cannot be negative")

            except (ValueError, TypeError) as e:
                raise ValidationError(f"Record {i}: Invalid numeric value - {str(e)}")
