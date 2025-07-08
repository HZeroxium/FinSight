# services/market_data_service.py

"""
Market Data Service

Provides business logic layer for market data operations.
Acts as an abstraction over MarketDataRepository implementations.
"""

from typing import Dict, List, Any, Optional, Tuple

from ..interfaces.market_data_repository import MarketDataRepository
from ..interfaces.errors import ValidationError
from ..common.logger import LoggerFactory
from ..utils.datetime_utils import DateTimeUtils
from ..schemas.ohlcv_schemas import (
    OHLCVSchema,
    OHLCVBatchSchema,
    OHLCVQuerySchema,
    OHLCVResponseSchema,
    OHLCVStatsSchema,
)
from ..converters.ohlcv_converter import OHLCVConverter


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
        self.logger = LoggerFactory.get_logger(name="market_data_service")
        self.converter = OHLCVConverter()

        self.logger.info("Market Data Service initialized")

    async def save_ohlcv_data(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        data: List[OHLCVSchema],
        validate: bool = True,
    ) -> bool:
        """
        Save OHLCV data with validation and business logic.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Time interval
            data: List of OHLCV schema instances
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
                self._validate_ohlcv_schemas(data)

            # Sort data by timestamp to ensure proper ordering
            sorted_data = sorted(data, key=lambda x: x.timestamp)

            # Save to repository (repository handles model conversion internally)
            success = await self.repository.save_ohlcv(
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

    async def save_ohlcv_batch(
        self, batch: OHLCVBatchSchema, validate: bool = True
    ) -> bool:
        """
        Save OHLCV data using batch schema.

        Args:
            batch: OHLCV batch schema instance
            validate: Whether to validate data before saving

        Returns:
            True if save successful

        Raises:
            ValidationError: If data validation fails
            RepositoryError: If save operation fails
        """
        try:
            if validate:
                self._validate_ohlcv_schemas(batch.records)

            # Sort records by timestamp
            batch.records = sorted(batch.records, key=lambda x: x.timestamp)

            success = await self.repository.batch_save_ohlcv([batch])

            if success:
                self.logger.info(
                    f"Saved batch of {len(batch.records)} OHLCV records for {batch.exchange}/{batch.symbol}/{batch.timeframe}"
                )

            return success

        except Exception as e:
            self.logger.error(f"Error saving OHLCV batch: {str(e)}")
            raise

    async def get_ohlcv_data(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        limit: Optional[int] = None,
    ) -> OHLCVResponseSchema:
        """
        Retrieve OHLCV data for specified date range.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Time interval
            start_date: Start date in ISO 8601 format
            end_date: End date in ISO 8601 format
            limit: Optional limit on number of records

        Returns:
            OHLCV response schema instance

        Raises:
            ValidationError: If date format is invalid
            RepositoryError: If retrieval fails
        """
        try:
            # Validate date format
            start_dt, end_dt = DateTimeUtils.validate_date_range(start_date, end_date)

            # Create query schema
            query = OHLCVQuerySchema(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_dt,
                end_date=end_dt,
                limit=limit,
            )

            # Retrieve data from repository (repository returns schemas)
            data = await self.repository.get_ohlcv(query)

            # Apply limit if specified (repository might not have applied it)
            if limit and len(data) > limit:
                data = data[:limit]
                has_more = True
            else:
                has_more = False

            self.logger.info(
                f"Retrieved {len(data)} OHLCV records for {exchange}/{symbol}/{timeframe} "
                f"from {start_date} to {end_date}"
            )

            return OHLCVResponseSchema(
                data=data,
                count=len(data),
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_dt,
                end_date=end_dt,
                has_more=has_more,
            )

        except Exception as e:
            self.logger.error(f"Error retrieving OHLCV data: {str(e)}")
            raise

    async def get_latest_ohlcv_timestamp(
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
            data_range = await self.repository.get_data_range(
                exchange, symbol, "ohlcv", timeframe
            )

            if data_range:
                return data_range["end_date"]

            return None

        except Exception as e:
            self.logger.error(f"Error getting latest timestamp: {str(e)}")
            return None

    async def get_data_gaps(
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
            response = await self.get_ohlcv_data(
                exchange, symbol, timeframe, start_date, end_date
            )
            data = response.data

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
                existing_timestamps.add(record.timestamp.replace(tzinfo=None))

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

    async def delete_ohlcv_data(
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

            success = await self.repository.delete_ohlcv(
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

    async def get_available_symbols(self, exchange: str) -> List[str]:
        """Get all available symbols for an exchange."""
        try:
            symbols = await self.repository.get_available_symbols(exchange)
            self.logger.info(f"Found {len(symbols)} symbols for {exchange}")
            return symbols

        except Exception as e:
            self.logger.error(f"Error getting available symbols: {str(e)}")
            return []

    async def get_available_timeframes(self, exchange: str, symbol: str) -> List[str]:
        """Get all available timeframes for a symbol."""
        try:
            timeframes = await self.repository.get_available_timeframes(
                exchange, symbol
            )
            self.logger.info(
                f"Found {len(timeframes)} timeframes for {exchange}/{symbol}"
            )
            return timeframes

        except Exception as e:
            self.logger.error(f"Error getting available timeframes: {str(e)}")
            return []

    async def check_data_exists(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> bool:
        """Check if complete data exists for the specified range."""
        try:
            return await self.repository.check_data_exists(
                exchange, symbol, "ohlcv", start_date, end_date, timeframe
            )

        except Exception as e:
            self.logger.error(f"Error checking data existence: {str(e)}")
            return False

    async def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage backend."""
        try:
            return await self.repository.get_storage_info()

        except Exception as e:
            self.logger.error(f"Error getting storage info: {str(e)}")
            return {}

    async def batch_save_ohlcv(self, batches: List[OHLCVBatchSchema]) -> bool:
        """
        Save multiple OHLCV batch schemas.

        Args:
            batches: List of OHLCV batch schema instances

        Returns:
            True if all saves successful
        """
        try:
            # Validate all batches first
            for batch in batches:
                self._validate_ohlcv_schemas(batch.records)

            # Use repository batch operation
            success = await self.repository.batch_save_ohlcv(batches)

            if success:
                total_records = sum(len(batch.records) for batch in batches)
                self.logger.info(
                    f"Batch saved {len(batches)} batches with {total_records} total records"
                )

            return success

        except Exception as e:
            self.logger.error(f"Error in batch save: {str(e)}")
            raise

    async def get_ohlcv_stats(
        self, exchange: str, symbol: str, timeframe: str
    ) -> Optional[OHLCVStatsSchema]:
        """
        Get statistics for OHLCV data.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Time interval

        Returns:
            OHLCV statistics schema instance or None if no data
        """
        try:
            data_range = await self.repository.get_data_range(
                exchange, symbol, "ohlcv", timeframe
            )

            if not data_range:
                return None

            # Get all data to calculate statistics
            response = await self.get_ohlcv_data(
                exchange,
                symbol,
                timeframe,
                data_range["start_date"],
                data_range["end_date"],
            )

            if not response.data:
                return None

            # Calculate statistics
            prices = []
            volumes = []
            for record in response.data:
                prices.extend([record.open, record.high, record.low, record.close])
                volumes.append(record.volume)

            return OHLCVStatsSchema(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                total_records=len(response.data),
                date_range={
                    "start": DateTimeUtils.parse_iso_string(data_range["start_date"]),
                    "end": DateTimeUtils.parse_iso_string(data_range["end_date"]),
                },
                price_range={
                    "min": min(prices) if prices else 0.0,
                    "max": max(prices) if prices else 0.0,
                },
                volume_stats={
                    "min": min(volumes) if volumes else 0.0,
                    "max": max(volumes) if volumes else 0.0,
                    "avg": sum(volumes) / len(volumes) if volumes else 0.0,
                },
            )

        except Exception as e:
            self.logger.error(f"Error getting OHLCV stats: {str(e)}")
            return None

    async def get_available_exchanges(self) -> List[str]:
        """Get all available exchanges in the repository."""
        try:
            exchanges = await self.repository.get_available_exchanges()
            self.logger.info(f"Found {len(exchanges)} exchanges")
            return exchanges

        except Exception as e:
            self.logger.error(f"Error getting available exchanges: {str(e)}")
            return []

    def _validate_ohlcv_schemas(self, schemas: List[OHLCVSchema]) -> None:
        """
        Validate OHLCV schemas.

        Args:
            schemas: List of OHLCV schema instances

        Raises:
            ValidationError: If validation fails
        """
        for i, schema in enumerate(schemas):
            try:
                # Pydantic validation is already done during schema creation
                # Additional business logic validation can be added here
                if schema.high < max(schema.open, schema.close):
                    raise ValidationError(
                        f"Record {i}: High price must be >= max(open, close)"
                    )
                if schema.low > min(schema.open, schema.close):
                    raise ValidationError(
                        f"Record {i}: Low price must be <= min(open, close)"
                    )
            except Exception as e:
                raise ValidationError(
                    f"Schema validation failed for record {i}: {str(e)}"
                )
