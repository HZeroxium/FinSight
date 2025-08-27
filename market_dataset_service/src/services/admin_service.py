# services/admin_service.py

"""
Admin Service

Provides administrative operations for the backtesting system,
including data management, server statistics, and maintenance operations.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List

from common.logger import LoggerFactory

from ..converters.timeframe_converter import TimeFrameConverter
from ..interfaces.market_data_repository import MarketDataRepository
from ..schemas.admin_schemas import (AdminStatsResponse, DataEnsureRequest,
                                     DataEnsureResponse, QuickPipelineResponse,
                                     QuickSymbolPipelineResult,
                                     QuickUploadResult, SystemHealthResponse,
                                     TimeframeConvertRequest,
                                     TimeframeConvertResponse)
from ..schemas.enums import Exchange, RepositoryType, TimeFrame
from ..schemas.job_schemas import ManualJobRequest
from ..services.market_data_collector_service import MarketDataCollectorService
from ..services.market_data_service import MarketDataService
from ..utils.datetime_utils import DateTimeUtils

if TYPE_CHECKING:  # avoid runtime imports to prevent circular dependencies
    from ..misc.timeframe_load_convert_save import \
        CrossRepositoryTimeFramePipeline
    from ..services.market_data_job_service import \
        MarketDataJobManagementService
    from ..services.market_data_storage_service import MarketDataStorageService


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
        storage_service: "MarketDataStorageService",
        market_data_job_service: "MarketDataJobManagementService",
        cross_repository_pipeline: "CrossRepositoryTimeFramePipeline",
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
        self.storage_service = storage_service
        self.market_data_job_service = market_data_job_service
        self.cross_repository_pipeline = cross_repository_pipeline
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
                server_timestamp=datetime.now(timezone.utc),
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
                checks_timestamp=datetime.now(timezone.utc),
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
                checks_timestamp=datetime.now(timezone.utc),
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
            data_exists = await self.market_data_service.check_data_exists(
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
                final_stats = await self.market_data_service.get_ohlcv_stats(
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
                existing_stats = await self.market_data_service.get_ohlcv_stats(
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
            source_data = await self.market_data_service.get_ohlcv_data(
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

                success = await self.market_data_service.save_ohlcv_data(
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

    async def run_quick_collect_convert_upload_pipeline(
        self,
        config: Dict[str, Any],
    ) -> QuickPipelineResponse:
        """Run a streamlined pipeline: collect(1h) → convert(timeframes) → upload.

        Args:
            config: Configuration loaded from JSON file specifying symbols, formats, timeframes, and date range.

        Returns:
            QuickPipelineResponse: Aggregated results across all symbols.
        """
        started_at = DateTimeUtils.now_utc()
        symbol_results: List[QuickSymbolPipelineResult] = []

        exchange: str = config.get("exchange", Exchange.BINANCE.value)
        symbols: List[str] = config.get("symbols", [])
        source_timeframe: str = config.get("source_timeframe", TimeFrame.HOUR_1.value)
        target_timeframes: List[str] = config.get(
            "target_timeframes", [TimeFrame.DAY_1.value]
        )
        source_format: str = config.get("source_format", RepositoryType.CSV.value)
        target_format: str = config.get("target_format", RepositoryType.PARQUET.value)
        start_date: str = config.get("start_date")
        end_date: str = config.get("end_date")
        overwrite_existing: bool = bool(config.get("overwrite_existing", False))

        self.logger.info(
            f"Quick pipeline started for {len(symbols)} symbols: source={source_format}, target={target_format}, "
            f"src_tf={source_timeframe}, tgt_tfs={target_timeframes}, range={start_date}..{end_date}"
        )

        # Step 1: Collect data (reuse MarketDataJobManagementService manual job)
        try:
            job_service = self.market_data_job_service
            manual_req = ManualJobRequest(
                symbols=symbols,
                timeframes=[source_timeframe],
                max_lookback_days=config.get("max_lookback_days", 30),
                exchange=exchange,
            )
            job_result = await job_service.run_manual_job(manual_req)
            collection_status = job_result.status
            collected_map = {
                sym: job_result.records_collected
                for sym in (job_result.symbols or symbols)
            }
        except Exception as e:
            self.logger.error(f"Collection step failed: {e}")
            collection_status = "failed"
            collected_map = {sym: 0 for sym in symbols}

        # Step 2 & 3 per symbol: convert timeframes via pipeline and then upload
        storage_service = self.storage_service
        pipeline = self.cross_repository_pipeline

        # Configure repositories for pipeline according to formats
        try:
            if source_format == target_format:
                if source_format == RepositoryType.CSV.value:
                    from ..adapters.csv_market_data_repository import \
                        CSVMarketDataRepository

                    pipeline.source_repository = CSVMarketDataRepository()
                    pipeline.target_repository = CSVMarketDataRepository()
                elif source_format == RepositoryType.PARQUET.value:
                    from ..adapters.parquet_market_data_repository import \
                        ParquetMarketDataRepository

                    pipeline.source_repository = ParquetMarketDataRepository()
                    pipeline.target_repository = ParquetMarketDataRepository()
                else:
                    raise ValueError(f"Unsupported format: {source_format}")
            else:
                if (
                    source_format == RepositoryType.CSV.value
                    and target_format == RepositoryType.PARQUET.value
                ):
                    from ..adapters.csv_market_data_repository import \
                        CSVMarketDataRepository
                    from ..adapters.parquet_market_data_repository import \
                        ParquetMarketDataRepository

                    pipeline.source_repository = CSVMarketDataRepository()
                    pipeline.target_repository = ParquetMarketDataRepository()
                elif (
                    source_format == RepositoryType.PARQUET.value
                    and target_format == RepositoryType.CSV.value
                ):
                    from ..adapters.csv_market_data_repository import \
                        CSVMarketDataRepository
                    from ..adapters.parquet_market_data_repository import \
                        ParquetMarketDataRepository

                    pipeline.source_repository = ParquetMarketDataRepository()
                    pipeline.target_repository = CSVMarketDataRepository()
                else:
                    raise ValueError(
                        f"Unsupported format combination: {source_format} -> {target_format}"
                    )
            pipeline.target_timeframes = target_timeframes
        except Exception as e:
            self.logger.error(f"Failed to configure conversion pipeline: {e}")

        for symbol in symbols:
            per_symbol_errors: List[Dict[str, Any]] = []
            upload_results: List[QuickUploadResult] = []
            converted_ok: List[str] = []

            # Conversion step
            try:
                # Determine actual date range if not provided
                actual_start = start_date
                actual_end = end_date
                if not start_date or not end_date:
                    # infer from source repository
                    try:
                        data_range = await pipeline.source_repository.get_data_range(
                            exchange=exchange,
                            symbol=symbol,
                            data_type="ohlcv",
                            timeframe=source_timeframe,
                        )
                        if data_range:
                            actual_start = start_date or data_range.get("start_date")
                            actual_end = end_date or data_range.get("end_date")
                    except Exception as e:
                        self.logger.warning(f"Range detection failed for {symbol}: {e}")
                conv_result = await pipeline.run_cross_repository_pipeline(
                    symbols=[symbol],
                    exchange=exchange,
                    start_date=actual_start or start_date,
                    end_date=actual_end or end_date,
                    overwrite_existing=overwrite_existing,
                )
                if conv_result.get("errors"):
                    per_symbol_errors.extend(conv_result["errors"])
                converted_ok = target_timeframes
                conversion_status = "completed"
            except Exception as e:
                conversion_status = "failed"
                per_symbol_errors.append({"stage": "convert", "error": str(e)})

            # Upload step: for each timeframe, upload dataset via storage service
            for tf in converted_ok:
                try:
                    # Ensure the data exists in target repository format; convert_dataset_format does upload when upload_result=True
                    up = await storage_service.convert_dataset_format(
                        exchange=exchange,
                        symbol=symbol,
                        timeframe=tf,
                        start_date=start_date
                        or (actual_start if "actual_start" in locals() else None),
                        end_date=end_date
                        or (actual_end if "actual_end" in locals() else None),
                        source_format=target_format,  # already converted to target_format above
                        target_format=target_format,
                        upload_result=True,
                        target_timeframes=None,
                        overwrite_existing=overwrite_existing,
                    )

                    upload_results.append(
                        QuickUploadResult(
                            symbol=symbol,
                            timeframe=tf,
                            success=bool(up.get("success", False)),
                            object_key=up.get("object_key"),
                            message=up.get("message"),
                        )
                    )
                except Exception as e:
                    upload_results.append(
                        QuickUploadResult(
                            symbol=symbol,
                            timeframe=tf,
                            success=False,
                            object_key=None,
                            message=str(e),
                        )
                    )
                    per_symbol_errors.append(
                        {"stage": "upload", "timeframe": tf, "error": str(e)}
                    )

            symbol_results.append(
                QuickSymbolPipelineResult(
                    symbol=symbol,
                    collection_status=collection_status,
                    collection_records=int(collected_map.get(symbol, 0) or 0),
                    conversion_status=conversion_status,
                    converted_timeframes=converted_ok,
                    upload_results=upload_results,
                    errors=per_symbol_errors,
                )
            )

        finished_at = DateTimeUtils.now_utc()
        duration = (finished_at - started_at).total_seconds()
        any_errors = (
            any(r.errors for r in symbol_results) or collection_status != "completed"
        )

        return QuickPipelineResponse(
            exchange=exchange,
            symbols=symbols,
            source_timeframe=source_timeframe,
            target_timeframes=target_timeframes,
            source_format=source_format,
            target_format=target_format,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration,
            results_by_symbol=symbol_results,
            success=not any_errors,
            message=(
                "Pipeline completed"
                if not any_errors
                else "Pipeline completed with errors"
            ),
            metadata={"overwrite_existing": overwrite_existing},
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
