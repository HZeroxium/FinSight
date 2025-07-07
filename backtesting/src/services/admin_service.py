# services/admin_service.py

"""
Admin Service

Provides administrative operations for the backtesting system,
including data management, server statistics, and maintenance operations.
"""

from typing import Dict, Any
from datetime import datetime

from ..interfaces.market_data_repository import MarketDataRepository
from ..services.market_data_service import MarketDataService
from ..services.market_data_collector_service import MarketDataCollectorService
from ..converters.timeframe_converter import TimeFrameConverter
from ..common.logger import LoggerFactory
from ..schemas.admin_schemas import (
    AdminStatsResponse,
    DataEnsureRequest,
    DataEnsureResponse,
    TimeframeConvertRequest,
    TimeframeConvertResponse,
    SystemHealthResponse,
)
from ..utils.datetime_utils import DateTimeUtils


class AdminService:
    """
    Service layer for administrative operations.

    Provides high-level business logic for system administration,
    data management, and server maintenance operations.
    """

    def __init__(
        self,
        market_data_service: MarketDataService,
        collector_service: MarketDataCollectorService,
        repository: MarketDataRepository,
    ):
        """
        Initialize admin service.

        Args:
            market_data_service: Market data service instance
            collector_service: Market data collector service instance
            repository: Market data repository instance
        """
        self.market_data_service = market_data_service
        self.collector_service = collector_service
        self.repository = repository
        self.timeframe_converter = TimeFrameConverter()
        self.logger = LoggerFactory.get_logger(name="admin_service")

        self.logger.info("Admin Service initialized")

    async def get_system_stats(self) -> AdminStatsResponse:
        """
        Get comprehensive system statistics.

        Returns:
            AdminStatsResponse with system metrics
        """
        try:
            self.logger.info("Gathering system statistics")

            # Get data statistics from repository
            total_records = await self.repository.count_all_records()

            # Get unique symbols and exchanges
            symbols = await self.repository.get_available_symbols()
            exchanges = await self.repository.get_available_exchanges()
            timeframes = await self.repository.get_available_timeframes()

            # Get storage info
            storage_info = await self._get_storage_info()

            # System uptime (simplified - in production you'd track this properly)
            uptime_seconds = 0  # Would implement proper uptime tracking

            stats = AdminStatsResponse(
                total_records=total_records,
                unique_symbols=len(symbols),
                unique_exchanges=len(exchanges),
                available_timeframes=timeframes,
                storage_info=storage_info,
                uptime_seconds=uptime_seconds,
                server_timestamp=datetime.utcnow(),
                symbols=symbols,
                exchanges=exchanges,
            )

            self.logger.info(f"System statistics gathered: {total_records} records")
            return stats

        except Exception as e:
            self.logger.error(f"Failed to gather system statistics: {e}")
            raise

    async def get_system_health(self) -> SystemHealthResponse:
        """
        Get system health status.

        Returns:
            SystemHealthResponse with health metrics
        """
        try:
            self.logger.info("Checking system health")

            # Check repository connectivity
            repository_healthy = await self._check_repository_health()

            # Check data freshness
            data_fresh = await self._check_data_freshness()

            # Check system resources
            memory_usage = await self._get_memory_usage()
            disk_usage = await self._get_disk_usage()

            # Overall health status
            overall_healthy = (
                repository_healthy
                and data_fresh
                and memory_usage < 0.9
                and disk_usage < 0.9
            )

            health = SystemHealthResponse(
                status="healthy" if overall_healthy else "degraded",
                repository_connected=repository_healthy,
                data_fresh=data_fresh,
                memory_usage_percent=memory_usage * 100,
                disk_usage_percent=disk_usage * 100,
                checks_timestamp=datetime.utcnow(),
            )

            self.logger.info(f"System health check completed: {health.status}")
            return health

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return SystemHealthResponse(
                status="unhealthy",
                repository_connected=False,
                data_fresh=False,
                memory_usage_percent=0,
                disk_usage_percent=0,
                checks_timestamp=datetime.utcnow(),
                error_message=str(e),
            )

    async def ensure_data_available(
        self, request: DataEnsureRequest
    ) -> DataEnsureResponse:
        """
        Ensure OHLCV data is available for specified parameters.

        Fetches data if missing or validates existing data.

        Args:
            request: Data ensure request parameters

        Returns:
            DataEnsureResponse with operation results
        """
        try:
            self.logger.info(
                f"Ensuring data availability for {request.symbol} "
                f"on {request.exchange} ({request.timeframe})"
            )

            # Check if data already exists
            data_exists = self.market_data_service.check_data_exists(
                exchange=request.exchange,
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=request.start_date.isoformat(),
                end_date=request.end_date.isoformat(),
            )

            # Determine if we need to fetch data
            needs_fetch = not data_exists or request.force_refresh

            if needs_fetch:
                # Collect missing data
                collection_result = (
                    await self.collector_service.collect_and_store_ohlcv(
                        exchange=request.exchange,
                        symbol=request.symbol,
                        timeframe=request.timeframe,
                        start_date=request.start_date.isoformat(),
                        end_date=request.end_date.isoformat(),
                    )
                )

                # Get final statistics
                final_stats = self.market_data_service.get_ohlcv_stats(
                    exchange=request.exchange,
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                )

                response = DataEnsureResponse(
                    success=collection_result["success"],
                    data_was_missing=True,
                    records_fetched=collection_result.get("records_collected", 0),
                    records_saved=collection_result.get("records_collected", 0),
                    data_statistics=final_stats,
                    operation_timestamp=DateTimeUtils.now_utc(),
                    error_message=collection_result.get("error"),
                )
            else:
                # Data exists and no refresh requested
                existing_stats = self.market_data_service.get_ohlcv_stats(
                    exchange=request.exchange,
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                )

                response = DataEnsureResponse(
                    success=True,
                    data_was_missing=False,
                    records_fetched=0,
                    records_saved=0,
                    data_statistics=existing_stats,
                    operation_timestamp=DateTimeUtils.now_utc(),
                )

            self.logger.info(
                f"Data ensure operation completed: "
                f"fetched={response.records_fetched}, saved={response.records_saved}"
            )
            return response

        except Exception as e:
            self.logger.error(f"Data ensure operation failed: {e}")
            return DataEnsureResponse(
                success=False,
                data_was_missing=False,
                records_fetched=0,
                records_saved=0,
                operation_timestamp=DateTimeUtils.now_utc(),
                error_message=str(e),
            )

    async def convert_timeframe_data(
        self, request: TimeframeConvertRequest
    ) -> TimeframeConvertResponse:
        """
        Convert OHLCV data to different timeframe.

        Args:
            request: Timeframe conversion request parameters

        Returns:
            TimeframeConvertResponse with conversion results
        """
        try:
            self.logger.info(
                f"Converting timeframe data from {request.source_timeframe} "
                f"to {request.target_timeframe} for {request.symbol}"
            )

            # Fetch source data
            source_data = self.market_data_service.get_ohlcv_data(
                exchange=request.exchange,
                symbol=request.symbol,
                timeframe=request.source_timeframe,
                start_date=request.start_date.isoformat(),
                end_date=request.end_date.isoformat(),
            )

            if not source_data.data:
                return TimeframeConvertResponse(
                    success=False,
                    source_records=0,
                    converted_records=0,
                    saved_records=0,
                    operation_timestamp=DateTimeUtils.now_utc(),
                    error_message="No source data found for conversion",
                )

            # Convert data
            converted_data = self.timeframe_converter.convert_ohlcv_data(
                data=source_data.data,
                target_timeframe=request.target_timeframe,
            )

            # Save converted data if requested
            saved_records = 0
            if request.save_converted and converted_data:
                # Update record metadata for converted data
                for record in converted_data:
                    record.timeframe = request.target_timeframe

                success = self.market_data_service.save_ohlcv_data(
                    exchange=request.exchange,
                    symbol=request.symbol,
                    timeframe=request.target_timeframe,
                    data=converted_data,
                )

                if success:
                    saved_records = len(converted_data)

            response = TimeframeConvertResponse(
                success=True,
                source_records=len(source_data.data),
                converted_records=len(converted_data),
                saved_records=saved_records,
                operation_timestamp=DateTimeUtils.now_utc(),
                converted_data=converted_data if not request.save_converted else None,
            )

            self.logger.info(
                f"Timeframe conversion completed: "
                f"source={response.source_records}, "
                f"converted={response.converted_records}, "
                f"saved={response.saved_records}"
            )
            return response

        except Exception as e:
            self.logger.error(f"Timeframe conversion failed: {e}")
            return TimeframeConvertResponse(
                success=False,
                source_records=0,
                converted_records=0,
                saved_records=0,
                operation_timestamp=DateTimeUtils.now_utc(),
                error_message=str(e),
            )

    async def cleanup_old_data(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        cutoff_date: datetime,
    ) -> Dict[str, Any]:
        """
        Clean up old data before specified date.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Data timeframe
            cutoff_date: Date before which to delete data

        Returns:
            Cleanup operation results
        """
        try:
            self.logger.info(
                f"Cleaning up data before {cutoff_date} for "
                f"{symbol} on {exchange} ({timeframe})"
            )

            # Get count before deletion
            count_before = await self.repository.count_records(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                end_date=cutoff_date,
            )

            # Perform cleanup
            deleted_count = await self.repository.delete_records_before_date(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                cutoff_date=cutoff_date,
            )

            result = {
                "success": True,
                "records_before": count_before,
                "records_deleted": deleted_count,
                "cutoff_date": cutoff_date.isoformat(),
                "operation_timestamp": DateTimeUtils.now_utc(),
            }

            self.logger.info(f"Cleanup completed: deleted {deleted_count} records")
            return result

        except Exception as e:
            self.logger.error(f"Cleanup operation failed: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "operation_timestamp": DateTimeUtils.now_utc(),
            }

    # Private helper methods

    async def _get_storage_info(self) -> Dict[str, Any]:
        """Get storage utilization information."""
        try:
            # This would be implemented based on the storage backend
            return {
                "type": "mongodb",  # or csv, influxdb based on implementation
                "size_mb": 0,  # Would calculate actual size
                "available_space_mb": 1000000,  # Would check actual space
            }
        except Exception:
            return {"type": "unknown", "size_mb": 0, "available_space_mb": 0}

    async def _check_repository_health(self) -> bool:
        """Check if repository is healthy and accessible."""
        try:
            # Simple connectivity test
            await self.repository.count_all_records()
            return True
        except Exception:
            return False

    async def _check_data_freshness(self) -> bool:
        """Check if data is fresh (has recent updates)."""
        try:
            # Check if we have data from the last 24 hours
            recent_cutoff = DateTimeUtils.get_datetime_hours_ago(24)
            recent_count = await self.repository.count_records_since(recent_cutoff)
            return recent_count > 0
        except Exception:
            return False

    async def _get_memory_usage(self) -> float:
        """Get current memory usage as percentage."""
        try:
            import psutil

            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            return 0.0  # psutil not available

    async def _get_disk_usage(self) -> float:
        """Get current disk usage as percentage."""
        try:
            import psutil

            return psutil.disk_usage("/").percent / 100.0
        except ImportError:
            return 0.0  # psutil not available
