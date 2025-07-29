# repositories/redis_training_job_repository.py

"""
Redis-based repository implementation for managing training job persistence and retrieval
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

import redis.asyncio as redis
from redis.asyncio import Redis

from ..schemas.training_schemas import (
    TrainingJobInfo,
    TrainingJobFilter,
    TrainingProgressUpdate,
    TrainingJobPriority,
)
from ..core.constants import TrainingJobStatus, StorageConstants, TrainingConstants
from common.logger.logger_factory import LoggerFactory
from ..core.config import get_settings
from ..interfaces.training_job_repository_interface import (
    TrainingJobRepositoryInterface,
    TrainingJobRepositoryError,
)


class RedisTrainingJobRepository(TrainingJobRepositoryInterface):
    """Redis-based repository for persisting and retrieving training job information"""

    def __init__(self, redis_url: Optional[str] = None):
        self.logger = LoggerFactory.get_logger("RedisTrainingJobRepository")
        self.settings = get_settings()

        # Redis configuration
        self.redis_url = redis_url or self.settings.redis_url
        self.redis_client: Optional[Redis] = None

        # Key patterns
        self.job_key_prefix = StorageConstants.REDIS_JOB_HASH_PREFIX
        self.active_jobs_set = StorageConstants.REDIS_ACTIVE_JOBS_SET
        self.progress_key_prefix = StorageConstants.REDIS_PROGRESS_PREFIX
        self.stats_key = StorageConstants.REDIS_STATS_KEY

        # Initialize cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_initialized = False

    async def initialize(self) -> None:
        """
        Initialize async components (call this when event loop is running)
        """
        if self._is_initialized:
            return

        try:
            # Create Redis connection
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )

            # Test connection
            await self.redis_client.ping()

            # Start cleanup task
            self._start_cleanup_task()

            self._is_initialized = True
            self.logger.info("RedisTrainingJobRepository initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize RedisTrainingJobRepository: {e}")
            raise TrainingJobRepositoryError(f"Redis initialization failed: {e}")

    async def _ensure_initialized(self) -> None:
        """Ensure repository is initialized before use"""
        if not self._is_initialized:
            await self.initialize()

    async def create_job(self, job_info: TrainingJobInfo) -> bool:
        """
        Create a new training job entry

        Args:
            job_info: Training job information

        Returns:
            bool: True if job was created successfully
        """
        await self._ensure_initialized()

        try:
            # Check if job already exists
            job_key = f"{self.job_key_prefix}{job_info.job_id}"
            if await self.redis_client.exists(job_key):
                self.logger.warning(f"Job {job_info.job_id} already exists")
                return False

            # Serialize job data
            job_data = self._serialize_job(job_info)

            # Create pipeline for atomic operation
            async with self.redis_client.pipeline() as pipe:
                # Store job as hash
                pipe.hset(job_key, mapping=job_data)

                # Set TTL based on status
                ttl = self._get_ttl_for_status(job_info.status)
                if ttl:
                    pipe.expire(job_key, ttl)

                # Add to active jobs if not terminal
                if job_info.status in TrainingJobStatus.get_active_statuses():
                    pipe.sadd(self.active_jobs_set, job_info.job_id)

                # Execute pipeline
                await pipe.execute()

            self.logger.info(f"Created job {job_info.job_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create job {job_info.job_id}: {e}")
            return False

    async def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a training job

        Args:
            job_id: Job identifier
            updates: Dictionary of fields to update

        Returns:
            bool: True if job was updated successfully
        """
        await self._ensure_initialized()

        try:
            job_key = f"{self.job_key_prefix}{job_id}"

            # Check if job exists
            if not await self.redis_client.exists(job_key):
                self.logger.warning(f"Job {job_id} not found for update")
                return False

            # Get current job data
            current_data = await self.redis_client.hgetall(job_key)
            if not current_data:
                self.logger.warning(f"Job {job_id} data is empty")
                return False

            # Deserialize current job
            try:
                current_job = self._deserialize_job(current_data)
            except Exception as e:
                self.logger.error(f"Failed to deserialize job data: {e}")
                return False

            # Apply updates
            for key, value in updates.items():
                if hasattr(current_job, key):
                    setattr(current_job, key, value)

            # Update timestamps
            if "status" in updates:
                if updates["status"] in [TrainingJobStatus.TRAINING]:
                    current_job.started_at = datetime.now(timezone.utc)
                elif updates["status"] in TrainingJobStatus.get_terminal_statuses():
                    current_job.completed_at = datetime.now(timezone.utc)

            # Serialize updated job
            updated_data = self._serialize_job(current_job)

            # Create pipeline for atomic update
            async with self.redis_client.pipeline() as pipe:
                # Update job hash
                pipe.hset(job_key, mapping=updated_data)

                # Update TTL based on new status
                ttl = self._get_ttl_for_status(current_job.status)
                if ttl:
                    pipe.expire(job_key, ttl)

                # Update active jobs set
                if current_job.status in TrainingJobStatus.get_active_statuses():
                    pipe.sadd(self.active_jobs_set, job_id)
                else:
                    pipe.srem(self.active_jobs_set, job_id)

                # Execute pipeline
                await pipe.execute()

            self.logger.debug(f"Updated job {job_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update job {job_id}: {e}")
            return False

    async def get_job(self, job_id: str) -> Optional[TrainingJobInfo]:
        """
        Get a training job by ID

        Args:
            job_id: Job identifier

        Returns:
            Optional[TrainingJobInfo]: Job information if found
        """
        try:
            await self._ensure_initialized()

            job_key = f"{self.job_key_prefix}{job_id}"
            job_data = await self.redis_client.hgetall(job_key)

            if not job_data:
                return None

            return self._deserialize_job(job_data)

        except Exception as e:
            self.logger.error(f"Failed to get job {job_id}: {e}")
            return None

    async def list_jobs(
        self, filter_criteria: Optional[TrainingJobFilter] = None
    ) -> List[TrainingJobInfo]:
        """
        List training jobs with optional filtering

        Args:
            filter_criteria: Optional filter criteria

        Returns:
            List[TrainingJobInfo]: List of matching jobs
        """
        try:
            # Get all job keys
            pattern = f"{self.job_key_prefix}*"
            job_keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                job_keys.append(key)

            if not job_keys:
                return []

            # Get all jobs
            jobs = []
            for job_key in job_keys:
                job_data = await self.redis_client.hgetall(job_key)
                if job_data:
                    try:
                        job = self._deserialize_job(job_data)
                        jobs.append(job)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to deserialize job from {job_key}: {e}"
                        )
                        continue

            if not filter_criteria:
                return sorted(jobs, key=lambda x: x.created_at, reverse=True)

            # Apply filters
            filtered_jobs = await self._apply_filters(jobs, filter_criteria)
            filtered_jobs = await self._apply_sorting(filtered_jobs, filter_criteria)
            filtered_jobs = await self._apply_pagination(filtered_jobs, filter_criteria)

            return filtered_jobs

        except Exception as e:
            self.logger.error(f"Failed to list jobs: {e}")
            return []

    async def delete_job(self, job_id: str) -> bool:
        """
        Delete a training job

        Args:
            job_id: Job identifier

        Returns:
            bool: True if job was deleted successfully
        """
        try:
            await self._ensure_initialized()

            job_key = f"{self.job_key_prefix}{job_id}"
            progress_key = f"{self.progress_key_prefix}{job_id}"

            # Create pipeline for atomic deletion
            async with self.redis_client.pipeline() as pipe:
                # Delete job data
                pipe.delete(job_key)
                # Delete progress data
                pipe.delete(progress_key)
                # Remove from active jobs set
                pipe.srem(self.active_jobs_set, job_id)
                # Execute pipeline
                results = await pipe.execute()

            # Check if job was actually deleted
            deleted = results[0] > 0
            if deleted:
                self.logger.info(f"Deleted job {job_id}")
            else:
                self.logger.warning(f"Job {job_id} not found for deletion")

            return deleted

        except Exception as e:
            self.logger.error(f"Failed to delete job {job_id}: {e}")
            return False

    async def get_active_jobs(self) -> List[TrainingJobInfo]:
        """Get all active (non-terminal) training jobs"""
        try:
            await self._ensure_initialized()

            # Get active job IDs from set
            active_job_ids = await self.redis_client.smembers(self.active_jobs_set)

            if not active_job_ids:
                return []

            # Get job data for active jobs
            jobs = []
            for job_id in active_job_ids:
                job = await self.get_job(job_id)
                if job and job.status in TrainingJobStatus.get_active_statuses():
                    jobs.append(job)
                elif job and job.status in TrainingJobStatus.get_terminal_statuses():
                    # Clean up active set if job is now terminal
                    await self.redis_client.srem(self.active_jobs_set, job_id)

            return sorted(jobs, key=lambda x: x.created_at, reverse=True)

        except Exception as e:
            self.logger.error(f"Failed to get active jobs: {e}")
            return []

    async def get_jobs_by_status(
        self, status: TrainingJobStatus
    ) -> List[TrainingJobInfo]:
        """Get all jobs with specific status"""
        try:
            all_jobs = await self.list_jobs()
            return [job for job in all_jobs if job.status == status]

        except Exception as e:
            self.logger.error(f"Failed to get jobs by status {status}: {e}")
            return []

    async def update_progress(
        self, job_id: str, progress_update: TrainingProgressUpdate
    ) -> bool:
        """
        Update training job progress

        Args:
            job_id: Job identifier
            progress_update: Progress update information

        Returns:
            bool: True if progress was updated successfully
        """
        try:
            await self._ensure_initialized()

            # Update the main job with progress
            job_updates = {
                "progress": progress_update.progress,
                "current_stage": progress_update.current_stage,
            }

            # Update job
            job_updated = await self.update_job(job_id, job_updates)

            # Store progress update separately with TTL
            progress_key = f"{self.progress_key_prefix}{job_id}"
            progress_data = {
                "progress": progress_update.progress,
                "current_stage": progress_update.current_stage or "",
                "message": progress_update.message or "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            try:
                await self.redis_client.hset(progress_key, mapping=progress_data)
                await self.redis_client.expire(
                    progress_key, StorageConstants.PROGRESS_TTL
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to save progress update for job {job_id}: {e}"
                )

            return job_updated

        except Exception as e:
            self.logger.error(f"Failed to update progress for job {job_id}: {e}")
            return False

    async def get_job_statistics(self) -> Dict[str, Any]:
        """Get training job statistics"""
        try:
            await self._ensure_initialized()

            # Check cache first
            cached_stats = await self.redis_client.hgetall(self.stats_key)
            if cached_stats and "timestamp" in cached_stats:
                cache_time = datetime.fromisoformat(cached_stats["timestamp"])
                if (
                    datetime.now(timezone.utc) - cache_time
                ).total_seconds() < 300:  # 5 min cache
                    return {
                        k: int(v) if k != "timestamp" else v
                        for k, v in cached_stats.items()
                    }

            # Calculate fresh statistics
            all_jobs = await self.list_jobs()

            stats = {
                "total_jobs": len(all_jobs),
                "active_jobs": len(
                    [
                        j
                        for j in all_jobs
                        if j.status in TrainingJobStatus.get_active_statuses()
                    ]
                ),
                "completed_jobs": len(
                    [j for j in all_jobs if j.status == TrainingJobStatus.COMPLETED]
                ),
                "failed_jobs": len(
                    [j for j in all_jobs if j.status == TrainingJobStatus.FAILED]
                ),
                "cancelled_jobs": len(
                    [j for j in all_jobs if j.status == TrainingJobStatus.CANCELLED]
                ),
                "pending_jobs": len(
                    [j for j in all_jobs if j.status == TrainingJobStatus.PENDING]
                ),
                "queued_jobs": len(
                    [j for j in all_jobs if j.status == TrainingJobStatus.QUEUED]
                ),
                "training_jobs": len(
                    [j for j in all_jobs if j.status == TrainingJobStatus.TRAINING]
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Cache statistics
            await self.redis_client.hset(self.stats_key, mapping=stats)
            await self.redis_client.expire(
                self.stats_key, TrainingConstants.STATS_CACHE_TTL
            )

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get job statistics: {e}")
            return {}

    async def cleanup_old_jobs(self) -> int:
        """
        Clean up old completed and failed jobs

        Returns:
            int: Number of jobs cleaned up
        """
        try:
            await self._ensure_initialized()

            cleanup_count = 0
            cutoff_time = datetime.now(timezone.utc)

            # Get all job keys
            pattern = f"{self.job_key_prefix}*"
            job_keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                job_keys.append(key)

            for job_key in job_keys:
                try:
                    job_data = await self.redis_client.hgetall(job_key)
                    if not job_data:
                        continue

                    job = self._deserialize_job(job_data)

                    # Check if job should be cleaned up
                    should_cleanup = False

                    if job.status == TrainingJobStatus.COMPLETED and job.completed_at:
                        age = (cutoff_time - job.completed_at).total_seconds()
                        if age > StorageConstants.COMPLETED_JOB_TTL:
                            should_cleanup = True

                    elif job.status == TrainingJobStatus.FAILED and job.completed_at:
                        age = (cutoff_time - job.completed_at).total_seconds()
                        if age > StorageConstants.FAILED_JOB_TTL:
                            should_cleanup = True

                    elif job.created_at:
                        age = (cutoff_time - job.created_at).total_seconds()
                        if age > StorageConstants.JOB_STORAGE_TTL:
                            should_cleanup = True

                    if should_cleanup:
                        job_id = job_key.replace(self.job_key_prefix, "")
                        if await self.delete_job(job_id):
                            cleanup_count += 1

                except Exception as e:
                    self.logger.warning(
                        f"Failed to process job {job_key} during cleanup: {e}"
                    )
                    continue

            if cleanup_count > 0:
                self.logger.info(f"Cleaned up {cleanup_count} old jobs")

            return cleanup_count

        except Exception as e:
            self.logger.error(f"Failed to cleanup old jobs: {e}")
            return 0

    # Private methods

    def _serialize_job(self, job_info: TrainingJobInfo) -> Dict[str, str]:
        """Serialize job info for Redis storage"""
        try:
            # Convert to dict using Pydantic's model_dump
            job_dict = job_info.model_dump()

            # Convert datetime objects to ISO strings
            for key, value in job_dict.items():
                if isinstance(value, datetime):
                    job_dict[key] = value.isoformat()
                elif value is None:
                    job_dict[key] = ""
                elif isinstance(value, (dict, list)):
                    job_dict[key] = json.dumps(value)
                elif isinstance(value, (TrainingJobStatus, TrainingJobPriority)):
                    job_dict[key] = value.value
                else:
                    job_dict[key] = str(value)

            return job_dict

        except Exception as e:
            self.logger.error(f"Failed to serialize job: {e}")
            raise

    def _deserialize_job(self, job_data: Dict[str, str]) -> TrainingJobInfo:
        """Deserialize job info from Redis storage"""
        try:
            # Convert Redis hash data back to proper types
            converted_data = {}

            for key, value in job_data.items():
                if not value or value == "":
                    converted_data[key] = None
                elif key in ["created_at", "started_at", "completed_at"]:
                    try:
                        converted_data[key] = (
                            datetime.fromisoformat(value) if value else None
                        )
                    except ValueError:
                        converted_data[key] = None
                elif key in [
                    "config",
                    "training_metrics",
                    "validation_metrics",
                    "tags",
                ]:
                    try:
                        converted_data[key] = json.loads(value) if value else None
                    except json.JSONDecodeError:
                        converted_data[key] = None
                elif key == "progress":
                    try:
                        converted_data[key] = float(value)
                    except ValueError:
                        converted_data[key] = 0.0
                elif key in ["duration_seconds", "estimated_remaining_seconds"]:
                    try:
                        converted_data[key] = float(value) if value else None
                    except ValueError:
                        converted_data[key] = None
                elif key == "model_size_bytes":
                    try:
                        converted_data[key] = int(value) if value else None
                    except ValueError:
                        converted_data[key] = None
                elif key == "status":
                    # Ensure status is the enum value, not string representation
                    if value.startswith("TrainingJobStatus."):
                        value = value.replace("TrainingJobStatus.", "")
                    converted_data[key] = TrainingJobStatus(value)
                elif key == "priority":
                    # Ensure priority is the enum value, not string representation
                    if value.startswith("TrainingJobPriority."):
                        value = value.replace("TrainingJobPriority.", "")
                    converted_data[key] = TrainingJobPriority(value)
                else:
                    converted_data[key] = value

            # Create TrainingJobInfo instance
            return TrainingJobInfo(**converted_data)

        except Exception as e:
            self.logger.error(f"Failed to deserialize job data: {e}")
            raise

    def _get_ttl_for_status(self, status: TrainingJobStatus) -> Optional[int]:
        """Get TTL in seconds based on job status"""
        if status == TrainingJobStatus.COMPLETED:
            return StorageConstants.COMPLETED_JOB_TTL
        elif status == TrainingJobStatus.FAILED:
            return StorageConstants.FAILED_JOB_TTL
        elif status in TrainingJobStatus.get_active_statuses():
            return None  # No TTL for active jobs
        else:
            return StorageConstants.JOB_STORAGE_TTL

    async def _apply_filters(
        self, jobs: List[TrainingJobInfo], filter_criteria: TrainingJobFilter
    ) -> List[TrainingJobInfo]:
        """Apply filter criteria to job list"""
        filtered_jobs = jobs

        # Status filters
        if filter_criteria.statuses:
            filtered_jobs = [
                job for job in filtered_jobs if job.status in filter_criteria.statuses
            ]

        if filter_criteria.exclude_statuses:
            filtered_jobs = [
                job
                for job in filtered_jobs
                if job.status not in filter_criteria.exclude_statuses
            ]

        # Time filters
        if filter_criteria.created_after:
            filtered_jobs = [
                job
                for job in filtered_jobs
                if job.created_at >= filter_criteria.created_after
            ]

        if filter_criteria.created_before:
            filtered_jobs = [
                job
                for job in filtered_jobs
                if job.created_at <= filter_criteria.created_before
            ]

        # Model filters
        if filter_criteria.symbols:
            filtered_jobs = [
                job for job in filtered_jobs if job.symbol in filter_criteria.symbols
            ]

        if filter_criteria.model_types:
            filtered_jobs = [
                job
                for job in filtered_jobs
                if job.model_type in filter_criteria.model_types
            ]

        # Tag filters
        if filter_criteria.tags:
            filtered_jobs = [
                job
                for job in filtered_jobs
                if job.tags
                and all(job.tags.get(k) == v for k, v in filter_criteria.tags.items())
            ]

        return filtered_jobs

    async def _apply_sorting(
        self, jobs: List[TrainingJobInfo], filter_criteria: TrainingJobFilter
    ) -> List[TrainingJobInfo]:
        """Apply sorting to job list"""
        try:
            sort_key = filter_criteria.sort_by or "created_at"
            reverse = filter_criteria.sort_order == "desc"

            if sort_key == "created_at":
                return sorted(jobs, key=lambda x: x.created_at, reverse=reverse)
            elif sort_key == "progress":
                return sorted(jobs, key=lambda x: x.progress, reverse=reverse)
            elif sort_key == "status":
                return sorted(jobs, key=lambda x: x.status.value, reverse=reverse)
            else:
                return sorted(jobs, key=lambda x: x.created_at, reverse=True)

        except Exception as e:
            self.logger.warning(f"Failed to apply sorting: {e}")
            return jobs

    async def _apply_pagination(
        self, jobs: List[TrainingJobInfo], filter_criteria: TrainingJobFilter
    ) -> List[TrainingJobInfo]:
        """Apply pagination to job list"""
        start_idx = filter_criteria.offset or 0
        end_idx = start_idx + (filter_criteria.limit or len(jobs))
        return jobs[start_idx:end_idx]

    def _start_cleanup_task(self) -> None:
        """Start background cleanup task"""

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(TrainingConstants.CLEANUP_INTERVAL)
                    await self.cleanup_old_jobs()
                except Exception as e:
                    self.logger.error(f"Error in cleanup loop: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def shutdown(self) -> None:
        """
        Shutdown the repository and cleanup resources
        """
        try:
            # Cancel cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Close Redis connection
            if self.redis_client:
                await self.redis_client.aclose()

            self._is_initialized = False
            self.logger.info("RedisTrainingJobRepository shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during repository shutdown: {e}")
