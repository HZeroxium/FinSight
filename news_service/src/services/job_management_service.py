# services/job_service.py

"""
Job management service providing business logic for cron job REST API.
Wraps the NewsCrawlerJobService and provides clean API for job operations.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from common.logger import LoggerFactory, LoggerType, LogLevel

from ..core.config import settings
from ..schemas.job_schemas import (JobConfigModel, JobConfigUpdateRequest,
                                   JobOperationResponse, JobStartRequest,
                                   JobStatsModel, JobStatus, JobStatusResponse,
                                   JobStopRequest, ManualJobRequest,
                                   ManualJobResponse)
from ..services.news_crawler_job_service import JobConfig as DataclassJobConfig
from ..services.news_crawler_job_service import NewsCrawlerJobService
from ..utils.cache_utils import invalidate_all_news_cache


class JobManagementService:
    """
    Job management service providing business logic layer for cron job operations.
    Acts as a facade over the NewsCrawlerJobService for REST API integration.
    """

    def __init__(
        self,
        mongo_url: str = None,
        database_name: str = None,
        config_file: str = None,
        pid_file: str = None,
        log_file: str = None,
        job_service_factory=None,  # Factory function for creating NewsCrawlerJobService
    ):
        """
        Initialize job management service.

        Args:
            mongo_url: MongoDB connection URL
            database_name: Database name for storing news
            config_file: Job configuration file path
            pid_file: Process ID file path
            log_file: Log file path
            job_service_factory: Factory function to create NewsCrawlerJobService
        """
        self.mongo_url = mongo_url or settings.mongodb_url
        self.database_name = database_name or settings.mongodb_database
        self.config_file = Path(config_file or settings.cron_job_config_file)
        self.pid_file = Path(pid_file or settings.cron_job_pid_file)
        self.log_file = log_file or settings.cron_job_log_file
        self.job_service_factory = job_service_factory

        # Initialize logger
        self.logger = LoggerFactory.get_logger(
            name="job-management-service",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file=f"{settings.log_file_path}job_management_service.log",
        )

        # Internal job service - lazy initialization
        self._job_service: Optional[NewsCrawlerJobService] = None
        self._initialized = False

    def _get_job_service(self) -> NewsCrawlerJobService:
        """Get or create the internal job service instance."""
        if self._job_service is None:
            if self.job_service_factory:
                # Use factory function (dependency injection)
                self._job_service = self.job_service_factory()
            else:
                # Fallback to direct instantiation (avoid circular import)
                from ..services.news_crawler_job_service import \
                    NewsCrawlerJobService

                self._job_service = NewsCrawlerJobService(
                    mongo_url=self.mongo_url,
                    database_name=self.database_name,
                    config_file=str(self.config_file),
                    pid_file=str(self.pid_file),
                    log_file=self.log_file,
                )
        return self._job_service

    def _convert_to_dataclass_config(
        self, config: JobConfigModel
    ) -> DataclassJobConfig:
        """Convert Pydantic model to dataclass for internal service."""
        return DataclassJobConfig(
            sources=config.sources,
            collector_preferences=config.collector_preferences,
            max_items_per_source=config.max_items_per_source,
            enable_fallback=config.enable_fallback,
            schedule=config.schedule,
            config_overrides=config.config_overrides,
            notification=config.notification,
        )

    def _convert_from_dataclass_config(
        self, config: DataclassJobConfig
    ) -> JobConfigModel:
        """Convert dataclass to Pydantic model for API responses."""
        return JobConfigModel(
            sources=config.sources,
            collector_preferences=config.collector_preferences,
            max_items_per_source=config.max_items_per_source,
            enable_fallback=config.enable_fallback,
            schedule=config.schedule,
            config_overrides=config.config_overrides,
            notification=config.notification,
        )

    def _determine_job_status(
        self, is_running: bool, scheduler_running: bool
    ) -> JobStatus:
        """Determine the current job status based on service state."""
        if not is_running:
            return JobStatus.STOPPED
        elif is_running and scheduler_running:
            return JobStatus.RUNNING
        else:
            return JobStatus.DEGRADED

    async def get_status(self) -> JobStatusResponse:
        """
        Get current job service status.

        Returns:
            JobStatusResponse: Current job status and statistics
        """
        try:
            job_service = self._get_job_service()

            # Initialize if not already done
            if not self._initialized:
                try:
                    await job_service.initialize()
                    self._initialized = True
                except Exception as e:
                    self.logger.warning(
                        f"Failed to initialize job service for status: {e}"
                    )

            # Get raw status from job service
            raw_status = job_service.get_status()

            # Convert to our API model
            status = self._determine_job_status(
                raw_status["is_running"], raw_status["scheduler_running"]
            )

            # Convert stats
            stats = JobStatsModel(**raw_status["stats"])

            return JobStatusResponse(
                status=status,
                is_running=raw_status["is_running"],
                pid=raw_status.get("pid"),
                pid_file=raw_status.get("pid_file"),
                config_file=raw_status.get("config_file"),
                log_file=raw_status.get("log_file"),
                scheduler_running=raw_status["scheduler_running"],
                next_run=raw_status.get("next_run"),
                stats=stats,
            )

        except Exception as e:
            self.logger.error(f"Failed to get job status: {e}")
            return JobStatusResponse(
                status=JobStatus.ERROR,
                is_running=False,
                scheduler_running=False,
                stats=JobStatsModel(),
            )

    async def start_job(self, request: JobStartRequest) -> JobOperationResponse:
        """
        Start the job service with optional configuration.

        Args:
            request: Job start request with optional configuration

        Returns:
            JobOperationResponse: Operation result
        """
        try:
            job_service = self._get_job_service()

            # Check if already running
            if job_service.is_running and not request.force_restart:
                return JobOperationResponse(
                    success=False,
                    message="Job service is already running. Use force_restart=true to restart.",
                    status=JobStatus.RUNNING,
                )

            # Stop if force restart requested
            if job_service.is_running and request.force_restart:
                self.logger.info(
                    "Force restart requested, stopping current job service"
                )
                await job_service.stop()
                await asyncio.sleep(1)  # Give it time to stop

            # Update configuration if provided
            if request.config:
                dataclass_config = self._convert_to_dataclass_config(request.config)
                job_service.save_config(dataclass_config)
                self.logger.info("Job configuration updated from request")

            # Start the job service
            self.logger.info("Starting job service...")

            # Initialize if not already done
            if not self._initialized:
                await job_service.initialize()
                self._initialized = True

            # Start in background task to avoid blocking
            start_task = asyncio.create_task(job_service.start())

            # Give it a moment to start
            await asyncio.sleep(2)

            # Check if it started successfully
            if job_service.is_running:
                return JobOperationResponse(
                    success=True,
                    message="Job service started successfully",
                    status=JobStatus.RUNNING,
                    details={
                        "scheduler_running": (
                            job_service.scheduler.running
                            if job_service.scheduler
                            else False
                        )
                    },
                )
            else:
                return JobOperationResponse(
                    success=False,
                    message="Job service failed to start properly",
                    status=JobStatus.ERROR,
                )

        except Exception as e:
            self.logger.error(f"Failed to start job service: {e}")
            return JobOperationResponse(
                success=False,
                message=f"Failed to start job service: {str(e)}",
                status=JobStatus.ERROR,
            )

    async def stop_job(self, request: JobStopRequest) -> JobOperationResponse:
        """
        Stop the job service.

        Args:
            request: Job stop request with graceful shutdown option

        Returns:
            JobOperationResponse: Operation result
        """
        try:
            job_service = self._get_job_service()

            if not job_service.is_running:
                return JobOperationResponse(
                    success=True,
                    message="Job service is already stopped",
                    status=JobStatus.STOPPED,
                )

            self.logger.info(f"Stopping job service (graceful={request.graceful})")
            await job_service.stop()

            # Give it time to stop
            await asyncio.sleep(1)

            return JobOperationResponse(
                success=True,
                message="Job service stopped successfully",
                status=JobStatus.STOPPED,
            )

        except Exception as e:
            self.logger.error(f"Failed to stop job service: {e}")
            return JobOperationResponse(
                success=False,
                message=f"Failed to stop job service: {str(e)}",
                status=JobStatus.ERROR,
            )

    async def run_manual_job(self, request: ManualJobRequest) -> ManualJobResponse:
        """
        Run a manual job with specified parameters.

        Args:
            request: Manual job request with sources and limits

        Returns:
            ManualJobResponse: Job execution result
        """
        job_id = f"manual_{int(datetime.now(timezone.utc).timestamp())}"
        start_time = datetime.now(timezone.utc)

        try:
            job_service = self._get_job_service()

            # Initialize if not already done
            if not self._initialized:
                await job_service.initialize()
                self._initialized = True

            self.logger.info(f"Starting manual job {job_id}")

            # Execute manual job
            result = await job_service.run_manual_job(
                sources=request.sources,
                max_items=request.max_items_per_source,
            )

            # Invalidate cache if new items were stored
            try:
                total_stored = result.get("results", {}).get("total_items_stored", 0)
                if total_stored and total_stored > 0:
                    await asyncio.sleep(
                        max(0, int(settings.cache_invalidation_delay_seconds))
                    )
                    success = await invalidate_all_news_cache()
                    if success:
                        self.logger.info(
                            f"[CACHE] INVALIDATE_ALL after manual job {job_id} (stored={total_stored})"
                        )
                    else:
                        self.logger.warning(
                            f"[CACHE] INVALIDATE_ALL had no entries after manual job {job_id}"
                        )
                else:
                    self.logger.debug(
                        f"[CACHE] Skip invalidate: no new stored items for manual job {job_id}"
                    )
            except Exception as cache_error:
                self.logger.error(
                    f"[CACHE] Failed to invalidate cache after manual job {job_id}: {cache_error}"
                )

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            return ManualJobResponse(
                status="completed",
                job_id=job_id,
                sources=result["sources"],
                max_items_per_source=result["max_items"],
                start_time=start_time.isoformat(),
                duration=duration,
                results=result,
            )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            self.logger.error(f"Manual job {job_id} failed: {e}")

            return ManualJobResponse(
                status="failed",
                job_id=job_id,
                sources=request.sources or [],
                max_items_per_source=request.max_items_per_source or 0,
                start_time=start_time.isoformat(),
                duration=duration,
                results={"error": str(e)},
            )

    async def get_config(self) -> JobConfigModel:
        """
        Get current job configuration.

        Returns:
            JobConfigModel: Current job configuration
        """
        try:
            job_service = self._get_job_service()
            dataclass_config = job_service.load_config()
            return self._convert_from_dataclass_config(dataclass_config)

        except Exception as e:
            self.logger.error(f"Failed to get job configuration: {e}")
            # Return default configuration
            return JobConfigModel()

    async def update_config(
        self, request: JobConfigUpdateRequest
    ) -> JobOperationResponse:
        """
        Update job configuration with provided values.

        Args:
            request: Configuration update request

        Returns:
            JobOperationResponse: Operation result
        """
        try:
            job_service = self._get_job_service()

            # Load current configuration
            current_config = job_service.load_config()

            # Update only provided fields
            if request.sources is not None:
                current_config.sources = request.sources
            if request.collector_preferences is not None:
                current_config.collector_preferences = request.collector_preferences
            if request.max_items_per_source is not None:
                current_config.max_items_per_source = request.max_items_per_source
            if request.enable_fallback is not None:
                current_config.enable_fallback = request.enable_fallback
            if request.schedule is not None:
                current_config.schedule = request.schedule
            if request.config_overrides is not None:
                current_config.config_overrides = request.config_overrides
            if request.notification is not None:
                current_config.notification = request.notification

            # Save updated configuration
            job_service.save_config(current_config)

            # If job is running, we need to reschedule with new config
            restart_needed = False
            if job_service.is_running and request.schedule is not None:
                restart_needed = True
                self.logger.info("Schedule updated, rescheduling job")
                await job_service.schedule_job(current_config)

            message = "Configuration updated successfully"
            if restart_needed:
                message += " and job rescheduled"

            return JobOperationResponse(
                success=True,
                message=message,
                details={"restart_needed": restart_needed},
            )

        except Exception as e:
            self.logger.error(f"Failed to update job configuration: {e}")
            return JobOperationResponse(
                success=False,
                message=f"Failed to update configuration: {str(e)}",
            )

    async def get_stats(self) -> JobStatsModel:
        """
        Get job execution statistics.

        Returns:
            JobStatsModel: Job execution statistics
        """
        try:
            job_service = self._get_job_service()

            # Initialize if not already done
            if not self._initialized:
                try:
                    await job_service.initialize()
                    self._initialized = True
                except Exception as e:
                    self.logger.warning(
                        f"Failed to initialize job service for stats: {e}"
                    )

            raw_stats = job_service.job_stats
            return JobStatsModel(**raw_stats)

        except Exception as e:
            self.logger.error(f"Failed to get job statistics: {e}")
            return JobStatsModel()

    async def health_check(self) -> bool:
        """
        Perform health check for job management service.

        Returns:
            bool: True if service is healthy
        """
        try:
            status = await self.get_status()
            return status.status in [JobStatus.RUNNING, JobStatus.STOPPED]
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    async def invalidate_cache_after_job_completion(self, job_result: dict) -> None:
        """
        Invalidate cache after job completion if job was successful.

        Args:
            job_result: Job execution result
        """
        try:
            # Check if job was successful and collected items
            if (
                job_result.get("success", False)
                and job_result.get("total_items_collected", 0) > 0
            ):
                self.logger.info("Job completed successfully, invalidating cache")

                # Add delay before cache invalidation to ensure data is committed
                await asyncio.sleep(settings.cache_invalidation_delay_seconds)

                await invalidate_all_news_cache()
                self.logger.info("Cache invalidated after successful job completion")
            else:
                self.logger.debug(
                    "Job did not collect new items, skipping cache invalidation"
                )

        except Exception as e:
            self.logger.error(f"Failed to invalidate cache after job completion: {e}")
