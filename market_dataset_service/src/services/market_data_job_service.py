# services/market_data_job_service.py

"""
Market Data Job Management Service

Service layer for managing market data collection jobs.
Acts as a facade over MarketDataJobService to provide REST API management interface.
"""

import os
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

from ..schemas.job_schemas import (
    JobStatus,
    MarketDataJobConfigModel,
    JobStatsModel,
    JobStatusResponse,
    JobStartRequest,
    JobStopRequest,
    ManualJobRequest,
    ManualJobResponse,
    JobConfigUpdateRequest,
    JobOperationResponse,
    DataCollectionJobRequest,
    DataCollectionJobResponse,
    HealthCheckResponse,
)
from ..market_data_job import MarketDataJobService, JobConfig
from ..core.config import settings
from common.logger import LoggerFactory, LoggerType, LogLevel


class MarketDataJobManagementService:
    """
    Management service for market data collection jobs.

    Provides a REST API interface over the existing MarketDataJobService,
    similar to how news_crawler JobManagementService works.
    """

    def __init__(
        self,
        config_file: str = "market_data_job_config.json",
        pid_file: str = "market_data_job.pid",
        log_file: str = "logs/market_data_job.log",
    ):
        """
        Initialize market data job management service.

        Args:
            config_file: Path to job configuration file
            pid_file: Path to PID file
            log_file: Path to log file
        """
        self.config_file = config_file
        self.pid_file = pid_file
        self.log_file = log_file

        # Initialize logger
        self.logger = LoggerFactory.get_logger(
            name="md-job-management",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file="logs/market_data_job_management.log",
        )

        # Lazy-initialized job service
        self._job_service: Optional[MarketDataJobService] = None

        self.logger.info("Market Data Job Management Service initialized")

    def _get_job_service(self) -> MarketDataJobService:
        """Get or create the underlying MarketDataJobService."""
        if self._job_service is None:
            self._job_service = MarketDataJobService(
                config_file=self.config_file,
                pid_file=self.pid_file,
                log_file=self.log_file,
            )
        return self._job_service

    def _convert_to_dataclass_config(
        self, config: MarketDataJobConfigModel
    ) -> JobConfig:
        """Convert Pydantic config to dataclass JobConfig."""
        return JobConfig(
            cron_schedule=config.cron_schedule,
            timezone=config.timezone,
            exchange=config.exchange,
            max_lookback_days=config.max_lookback_days,
            update_existing=config.update_existing,
            max_concurrent_symbols=config.max_concurrent_symbols,
            repository_type=config.repository_type,
            repository_config=config.repository_config,
            max_symbols_per_run=config.max_symbols_per_run,
            max_timeframes_per_run=config.max_timeframes_per_run,
            priority_symbols=config.priority_symbols,
            priority_timeframes=config.priority_timeframes,
            max_retries=config.max_retries,
            retry_delay_minutes=config.retry_delay_minutes,
            skip_failed_symbols=config.skip_failed_symbols,
            enable_notifications=config.enable_notifications,
            notification_webhook=config.notification_webhook,
            notify_on_success=config.notify_on_success,
            notify_on_error=config.notify_on_error,
        )

    def _convert_from_dataclass_config(
        self, config: JobConfig
    ) -> MarketDataJobConfigModel:
        """Convert dataclass JobConfig to Pydantic config."""
        return MarketDataJobConfigModel(
            cron_schedule=config.cron_schedule,
            timezone=config.timezone,
            exchange=config.exchange,
            max_lookback_days=config.max_lookback_days,
            update_existing=config.update_existing,
            max_concurrent_symbols=config.max_concurrent_symbols,
            repository_type=config.repository_type,
            repository_config=config.repository_config,
            max_symbols_per_run=config.max_symbols_per_run,
            max_timeframes_per_run=config.max_timeframes_per_run,
            priority_symbols=config.priority_symbols,
            priority_timeframes=config.priority_timeframes,
            max_retries=config.max_retries,
            retry_delay_minutes=config.retry_delay_minutes,
            skip_failed_symbols=config.skip_failed_symbols,
            enable_notifications=config.enable_notifications,
            notification_webhook=config.notification_webhook,
            notify_on_success=config.notify_on_success,
            notify_on_error=config.notify_on_error,
        )

    def _determine_job_status(
        self, is_running: bool, scheduler_running: bool
    ) -> JobStatus:
        """Determine job status from internal state."""
        if is_running and scheduler_running:
            return JobStatus.RUNNING
        elif is_running and not scheduler_running:
            return JobStatus.DEGRADED
        elif not is_running:
            return JobStatus.STOPPED
        else:
            return JobStatus.ERROR

    async def get_status(self) -> JobStatusResponse:
        """
        Get current job service status.

        Returns:
            JobStatusResponse: Current status information
        """
        try:
            job_service = self._get_job_service()
            status_dict = job_service.get_status()

            # Extract status information
            is_running = status_dict.get("is_running", False)
            scheduler_running = status_dict.get("scheduler_running", False)
            pid = status_dict.get("pid")
            next_run_str = status_dict.get("next_run")

            # Parse next run time
            next_run = None
            if next_run_str:
                try:
                    if isinstance(next_run_str, str):
                        next_run = datetime.fromisoformat(
                            next_run_str.replace("Z", "+00:00")
                        )
                    elif isinstance(next_run_str, datetime):
                        next_run = next_run_str
                except (ValueError, AttributeError):
                    pass

            # Calculate uptime
            uptime_seconds = None
            start_time = status_dict.get("start_time")
            if start_time and is_running:
                try:
                    if isinstance(start_time, str):
                        start_dt = datetime.fromisoformat(
                            start_time.replace("Z", "+00:00")
                        )
                    elif isinstance(start_time, datetime):
                        start_dt = start_time
                    else:
                        start_dt = None

                    if start_dt:
                        uptime_seconds = int(
                            (
                                datetime.now() - start_dt.replace(tzinfo=None)
                            ).total_seconds()
                        )
                except (ValueError, AttributeError):
                    pass

            # Create job stats
            stats = JobStatsModel(
                total_jobs=status_dict.get("total_jobs", 0),
                successful_jobs=status_dict.get("successful_jobs", 0),
                failed_jobs=status_dict.get("failed_jobs", 0),
                last_run=status_dict.get("last_run"),
                last_success=status_dict.get("last_success"),
                last_error=status_dict.get("last_error"),
                symbols_processed_last_run=status_dict.get(
                    "symbols_processed_last_run", 0
                ),
                records_collected_last_run=status_dict.get(
                    "records_collected_last_run", 0
                ),
            )

            # Determine overall status
            job_status = self._determine_job_status(is_running, scheduler_running)

            # Create health status
            health_status = {
                "service_running": is_running,
                "scheduler_active": scheduler_running,
                "config_file_exists": os.path.exists(self.config_file),
                "pid_file_exists": os.path.exists(self.pid_file),
                "log_file_exists": os.path.exists(self.log_file),
                "next_run_scheduled": next_run is not None,
            }

            return JobStatusResponse(
                status=job_status,
                is_running=is_running,
                pid=pid,
                pid_file=self.pid_file,
                config_file=self.config_file,
                log_file=self.log_file,
                scheduler_running=scheduler_running,
                next_run=next_run,
                uptime_seconds=uptime_seconds,
                stats=stats,
                health_status=health_status,
            )

        except Exception as e:
            self.logger.error(f"Error getting job status: {e}")
            return JobStatusResponse(
                status=JobStatus.ERROR,
                is_running=False,
                scheduler_running=False,
                stats=JobStatsModel(),
                health_status={"error": str(e)},
            )

    async def start_job(self, request: JobStartRequest) -> JobOperationResponse:
        """
        Start the market data job service.

        Args:
            request: Job start request

        Returns:
            JobOperationResponse: Operation result
        """
        try:
            self.logger.info(
                f"Starting market data job service (force_restart={request.force_restart})"
            )

            job_service = self._get_job_service()

            # Check if already running
            status_dict = job_service.get_status()
            is_running = status_dict.get("is_running", False)

            if is_running and not request.force_restart:
                return JobOperationResponse(
                    success=False,
                    message="Job service is already running. Use force_restart=true to restart.",
                    status=JobStatus.RUNNING,
                )

            # Stop if running and force restart is requested
            if is_running and request.force_restart:
                self.logger.info("Force restart requested - stopping current service")
                await job_service.stop()
                # Give some time for graceful shutdown
                await asyncio.sleep(2)

            # Update configuration if provided
            if request.config:
                dataclass_config = self._convert_to_dataclass_config(request.config)
                job_service.save_config(dataclass_config)
                self.logger.info("Updated job configuration")

            # Start the service
            await job_service.start()

            # Verify it started successfully
            await asyncio.sleep(1)  # Give time to start
            status_dict = job_service.get_status()
            is_running = status_dict.get("is_running", False)

            if is_running:
                current_status = self._determine_job_status(
                    is_running, status_dict.get("scheduler_running", False)
                )
                return JobOperationResponse(
                    success=True,
                    message="Market data job service started successfully",
                    status=current_status,
                    details={"pid": status_dict.get("pid")},
                )
            else:
                return JobOperationResponse(
                    success=False,
                    message="Failed to start job service - service not running after start attempt",
                    status=JobStatus.ERROR,
                )

        except Exception as e:
            self.logger.error(f"Error starting job service: {e}")
            return JobOperationResponse(
                success=False,
                message=f"Failed to start job service: {str(e)}",
                status=JobStatus.ERROR,
            )

    async def stop_job(self, request: JobStopRequest) -> JobOperationResponse:
        """
        Stop the market data job service.

        Args:
            request: Job stop request

        Returns:
            JobOperationResponse: Operation result
        """
        try:
            self.logger.info(
                f"Stopping market data job service (graceful={request.graceful})"
            )

            job_service = self._get_job_service()

            # Check if running
            status_dict = job_service.get_status()
            is_running = status_dict.get("is_running", False)

            if not is_running:
                return JobOperationResponse(
                    success=True,
                    message="Job service is already stopped",
                    status=JobStatus.STOPPED,
                )

            # Stop the service
            if request.graceful:
                await job_service.stop()
            else:
                # Force stop implementation would go here
                # For now, use the same stop method
                await job_service.stop()

            # Verify it stopped
            await asyncio.sleep(1)  # Give time to stop
            status_dict = job_service.get_status()
            is_running = status_dict.get("is_running", False)

            if not is_running:
                return JobOperationResponse(
                    success=True,
                    message="Market data job service stopped successfully",
                    status=JobStatus.STOPPED,
                )
            else:
                return JobOperationResponse(
                    success=False,
                    message="Failed to stop job service - service still running",
                    status=JobStatus.ERROR,
                )

        except Exception as e:
            self.logger.error(f"Error stopping job service: {e}")
            return JobOperationResponse(
                success=False,
                message=f"Failed to stop job service: {str(e)}",
                status=JobStatus.ERROR,
            )

    async def run_manual_job(self, request: ManualJobRequest) -> ManualJobResponse:
        """
        Run a manual data collection job.

        Args:
            request: Manual job request

        Returns:
            ManualJobResponse: Job execution result
        """
        try:
            job_id = str(uuid.uuid4())
            start_time = datetime.now()

            self.logger.info(
                f"Starting manual job {job_id} with {request.symbols or 'default'} symbols"
            )

            job_service = self._get_job_service()

            # Load current config to get defaults
            current_config = job_service.load_config()

            # Use provided values or defaults from config
            symbols = request.symbols or current_config.priority_symbols
            timeframes = request.timeframes or current_config.priority_timeframes
            max_lookback_days = (
                request.max_lookback_days or current_config.max_lookback_days
            )
            exchange = request.exchange or current_config.exchange

            # Execute manual job
            result = await job_service.run_manual_job(
                symbols=symbols,
                timeframes=timeframes,
                max_lookback_days=max_lookback_days,
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Parse results
            status = result.get("status", "completed")
            symbols_processed = len(result.get("processed_symbols", []))
            records_collected = result.get("total_records_collected", 0)
            errors = result.get("errors", [])

            return ManualJobResponse(
                status=status,
                job_id=job_id,
                symbols=symbols,
                timeframes=timeframes,
                exchange=exchange,
                max_lookback_days=max_lookback_days,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                records_collected=records_collected,
                symbols_processed=symbols_processed,
                errors=errors,
                results=result,
                async_execution=request.async_execution,
            )

        except Exception as e:
            self.logger.error(f"Error running manual job: {e}")
            return ManualJobResponse(
                status="failed",
                job_id=job_id if "job_id" in locals() else str(uuid.uuid4()),
                symbols=request.symbols or [],
                timeframes=request.timeframes or [],
                exchange=request.exchange or "binance",
                max_lookback_days=request.max_lookback_days or 30,
                start_time=start_time if "start_time" in locals() else datetime.now(),
                errors=[{"error": str(e), "timestamp": datetime.now().isoformat()}],
                async_execution=request.async_execution,
            )

    async def get_config(self) -> MarketDataJobConfigModel:
        """
        Get current job configuration.

        Returns:
            MarketDataJobConfigModel: Current configuration
        """
        try:
            job_service = self._get_job_service()
            dataclass_config = job_service.load_config()
            return self._convert_from_dataclass_config(dataclass_config)

        except Exception as e:
            self.logger.error(f"Error getting job config: {e}")
            # Return default config
            return MarketDataJobConfigModel()

    async def update_config(
        self, request: JobConfigUpdateRequest
    ) -> JobOperationResponse:
        """
        Update job configuration.

        Args:
            request: Configuration update request

        Returns:
            JobOperationResponse: Operation result
        """
        try:
            self.logger.info("Updating job configuration")

            job_service = self._get_job_service()

            # Load current config
            current_config = job_service.load_config()

            # Update only provided fields
            if request.cron_schedule is not None:
                current_config.cron_schedule = request.cron_schedule
            if request.timezone is not None:
                current_config.timezone = request.timezone
            if request.exchange is not None:
                current_config.exchange = request.exchange
            if request.max_lookback_days is not None:
                current_config.max_lookback_days = request.max_lookback_days
            if request.update_existing is not None:
                current_config.update_existing = request.update_existing
            if request.max_concurrent_symbols is not None:
                current_config.max_concurrent_symbols = request.max_concurrent_symbols
            if request.repository_type is not None:
                current_config.repository_type = request.repository_type
            if request.repository_config is not None:
                current_config.repository_config = request.repository_config
            if request.max_symbols_per_run is not None:
                current_config.max_symbols_per_run = request.max_symbols_per_run
            if request.max_timeframes_per_run is not None:
                current_config.max_timeframes_per_run = request.max_timeframes_per_run
            if request.priority_symbols is not None:
                current_config.priority_symbols = request.priority_symbols
            if request.priority_timeframes is not None:
                current_config.priority_timeframes = request.priority_timeframes
            if request.max_retries is not None:
                current_config.max_retries = request.max_retries
            if request.retry_delay_minutes is not None:
                current_config.retry_delay_minutes = request.retry_delay_minutes
            if request.enable_notifications is not None:
                current_config.enable_notifications = request.enable_notifications
            if request.notification_webhook is not None:
                current_config.notification_webhook = request.notification_webhook

            # Save updated config
            job_service.save_config(current_config)

            return JobOperationResponse(
                success=True,
                message="Job configuration updated successfully",
                details={"config_file": self.config_file},
            )

        except Exception as e:
            self.logger.error(f"Error updating job config: {e}")
            return JobOperationResponse(
                success=False,
                message=f"Failed to update job configuration: {str(e)}",
            )

    async def get_stats(self) -> JobStatsModel:
        """
        Get job execution statistics.

        Returns:
            JobStatsModel: Job statistics
        """
        try:
            job_service = self._get_job_service()
            status_dict = job_service.get_status()

            return JobStatsModel(
                total_jobs=status_dict.get("total_jobs", 0),
                successful_jobs=status_dict.get("successful_jobs", 0),
                failed_jobs=status_dict.get("failed_jobs", 0),
                last_run=status_dict.get("last_run"),
                last_success=status_dict.get("last_success"),
                last_error=status_dict.get("last_error"),
                symbols_processed_last_run=status_dict.get(
                    "symbols_processed_last_run", 0
                ),
                records_collected_last_run=status_dict.get(
                    "records_collected_last_run", 0
                ),
            )

        except Exception as e:
            self.logger.error(f"Error getting job stats: {e}")
            return JobStatsModel()

    async def health_check(self) -> HealthCheckResponse:
        """
        Perform comprehensive health check.

        Returns:
            HealthCheckResponse: Health check result
        """
        try:
            job_service = self._get_job_service()
            status_dict = job_service.get_status()

            is_running = status_dict.get("is_running", False)
            scheduler_running = status_dict.get("scheduler_running", False)

            # Check component health
            components = {
                "job_service": "healthy" if is_running else "stopped",
                "scheduler": "healthy" if scheduler_running else "stopped",
                "config_file": (
                    "healthy" if os.path.exists(self.config_file) else "missing"
                ),
                "log_file": "healthy" if os.path.exists(self.log_file) else "missing",
            }

            # Overall health status
            if is_running and scheduler_running:
                overall_status = "healthy"
            elif is_running:
                overall_status = "degraded"
            else:
                overall_status = "unhealthy"

            # Calculate uptime
            uptime_seconds = None
            start_time = status_dict.get("start_time")
            if start_time and is_running:
                try:
                    if isinstance(start_time, str):
                        start_dt = datetime.fromisoformat(
                            start_time.replace("Z", "+00:00")
                        )
                    elif isinstance(start_time, datetime):
                        start_dt = start_time
                    else:
                        start_dt = None

                    if start_dt:
                        uptime_seconds = int(
                            (
                                datetime.now() - start_dt.replace(tzinfo=None)
                            ).total_seconds()
                        )
                except (ValueError, AttributeError):
                    pass

            # Collect metrics
            metrics = {
                "total_jobs": status_dict.get("total_jobs", 0),
                "successful_jobs": status_dict.get("successful_jobs", 0),
                "failed_jobs": status_dict.get("failed_jobs", 0),
                "last_run": status_dict.get("last_run"),
                "next_run": status_dict.get("next_run"),
            }

            return HealthCheckResponse(
                status=overall_status,
                components=components,
                dependencies={
                    "config_file": self.config_file,
                    "pid_file": self.pid_file,
                    "log_file": self.log_file,
                },
                metrics=metrics,
                uptime_seconds=uptime_seconds,
            )

        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return HealthCheckResponse(
                status="unhealthy",
                components={"error": str(e)},
                dependencies={},
                metrics={},
            )

    async def run_data_collection_job(
        self, request: DataCollectionJobRequest
    ) -> DataCollectionJobResponse:
        """
        Run a specific data collection job.

        Args:
            request: Data collection job request

        Returns:
            DataCollectionJobResponse: Job execution result
        """
        try:
            job_id = str(uuid.uuid4())
            start_time = datetime.now()

            self.logger.info(f"Starting data collection job {job_id}")

            job_service = self._get_job_service()

            # Execute the collection job
            result = await job_service.run_manual_job(
                symbols=request.symbols,
                timeframes=request.timeframes,
                max_lookback_days=request.start_date
                and request.end_date
                and 365
                or 30,  # Simplified
            )

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # Parse results
            processed_symbols = result.get("processed_symbols", [])
            failed_symbols = []
            for symbol in request.symbols:
                if symbol not in processed_symbols:
                    failed_symbols.append(symbol)

            return DataCollectionJobResponse(
                job_id=job_id,
                status=result.get("status", "completed"),
                symbols_requested=request.symbols,
                symbols_processed=processed_symbols,
                symbols_failed=failed_symbols,
                timeframes_processed=request.timeframes,
                total_records_collected=result.get("total_records_collected", 0),
                execution_time_seconds=execution_time,
                errors=result.get("errors", []),
                repository_info={
                    "repository_type": request.repository_type,
                    "exchange": request.exchange,
                },
            )

        except Exception as e:
            self.logger.error(f"Error running data collection job: {e}")
            return DataCollectionJobResponse(
                job_id=job_id if "job_id" in locals() else str(uuid.uuid4()),
                status="failed",
                symbols_requested=request.symbols,
                symbols_processed=[],
                symbols_failed=request.symbols,
                timeframes_processed=[],
                total_records_collected=0,
                errors=[{"error": str(e), "timestamp": datetime.now().isoformat()}],
            )
