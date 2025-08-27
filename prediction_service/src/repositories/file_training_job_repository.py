# repositories/file_training_job_repository.py

"""
JSON file-based repository implementation for managing training job persistence and retrieval
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from common.logger.logger_factory import LoggerFactory

from ..core.config import get_settings
from ..core.constants import (StorageConstants, TrainingConstants,
                              TrainingJobStatus)
from ..interfaces.training_job_repository_interface import \
    TrainingJobRepositoryInterface
from ..schemas.training_schemas import (TrainingJobFilter, TrainingJobInfo,
                                        TrainingProgressUpdate)


class FileTrainingJobRepository(TrainingJobRepositoryInterface):
    """Repository for persisting and retrieving training job information"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger("FileTrainingJobRepository")
        self.settings = get_settings()

        # Use jobs_dir from settings (which respects environment variables)
        self.jobs_dir = self.settings.jobs_dir
        self.jobs_dir.mkdir(exist_ok=True, parents=True)

        # Create subdirectories
        (self.jobs_dir / "active").mkdir(exist_ok=True)
        (self.jobs_dir / "completed").mkdir(exist_ok=True)
        (self.jobs_dir / "failed").mkdir(exist_ok=True)
        (self.jobs_dir / "progress").mkdir(exist_ok=True)

        # In-memory cache for active jobs
        self._job_cache: Dict[str, TrainingJobInfo] = {}
        self._cache_lock: Optional[asyncio.Lock] = None

        # Initialize cleanup task (will be started later)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_initialized = False

    async def initialize(self) -> None:
        """
        Initialize async components (call this when event loop is running)
        """
        if self._is_initialized:
            return

        try:
            # Initialize async lock
            self._cache_lock = asyncio.Lock()

            # Start cleanup task
            self._start_cleanup_task()

            self._is_initialized = True
            self.logger.info("TrainingJobRepository initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize TrainingJobRepository: {e}")
            raise

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
            assert self._cache_lock is not None, "Repository not properly initialized"
            async with self._cache_lock:
                # Check if job already exists
                if job_info.job_id in self._job_cache:
                    self.logger.warning(f"Job {job_info.job_id} already exists")
                    return False

                # Add to cache
                self._job_cache[job_info.job_id] = job_info

                # Persist to disk
                await self._save_job_to_disk(job_info)

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
            assert self._cache_lock is not None, "Repository not properly initialized"
            async with self._cache_lock:
                if job_id not in self._job_cache:
                    # Try to load from disk
                    job_info = await self._load_job_from_disk(job_id)
                    if not job_info:
                        self.logger.warning(f"Job {job_id} not found for update")
                        return False
                    self._job_cache[job_id] = job_info

                # Update job info
                job_info = self._job_cache[job_id]
                for key, value in updates.items():
                    if hasattr(job_info, key):
                        setattr(job_info, key, value)
                    else:
                        self.logger.warning(f"Unknown field {key} in job update")

                # Update cache
                self._job_cache[job_id] = job_info

                # Persist to disk
                await self._save_job_to_disk(job_info)

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

            assert self._cache_lock is not None, "Repository not properly initialized"
            async with self._cache_lock:
                # Check cache first
                if job_id in self._job_cache:
                    return self._job_cache[job_id]

                # Try to load from disk
                job_info = await self._load_job_from_disk(job_id)
                if job_info:
                    self._job_cache[job_id] = job_info

                return job_info

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
            # Get all jobs
            all_jobs = await self._get_all_jobs()

            if not filter_criteria:
                return all_jobs

            # Apply filters
            filtered_jobs = await self._apply_filters(all_jobs, filter_criteria)

            # Apply sorting and pagination
            sorted_jobs = await self._apply_sorting(filtered_jobs, filter_criteria)
            paginated_jobs = await self._apply_pagination(sorted_jobs, filter_criteria)

            return paginated_jobs

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

            assert self._cache_lock is not None, "Repository not properly initialized"
            async with self._cache_lock:
                # Remove from cache
                self._job_cache.pop(job_id, None)

                # Remove from disk
                await self._delete_job_from_disk(job_id)

                self.logger.info(f"Deleted job {job_id}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to delete job {job_id}: {e}")
            return False

    async def get_active_jobs(self) -> List[TrainingJobInfo]:
        """Get all active (non-terminal) training jobs"""
        try:
            all_jobs = await self._get_all_jobs()
            active_jobs = [
                job
                for job in all_jobs
                if TrainingJobStatus.is_active(
                    job.status.value if hasattr(job.status, "value") else job.status
                )
            ]
            return active_jobs

        except Exception as e:
            self.logger.error(f"Failed to get active jobs: {e}")
            return []

    async def get_jobs_by_status(
        self, status: TrainingJobStatus
    ) -> List[TrainingJobInfo]:
        """Get all jobs with specific status"""
        try:
            all_jobs = await self._get_all_jobs()
            status_jobs = [job for job in all_jobs if job.status == status]
            return status_jobs

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
            # Update main job info
            updates = {
                "status": progress_update.status,
                "progress": progress_update.progress,
                "current_stage": progress_update.current_stage,
            }

            # Add optional fields if present
            if progress_update.estimated_remaining_seconds is not None:
                updates["estimated_remaining_seconds"] = (
                    progress_update.estimated_remaining_seconds
                )

            success = await self.update_job(job_id, updates)

            if success:
                # Save detailed progress to separate file
                await self._save_progress_update(job_id, progress_update)

            return success

        except Exception as e:
            self.logger.error(f"Failed to update progress for job {job_id}: {e}")
            return False

    async def get_job_statistics(self) -> Dict[str, Any]:
        """Get training job statistics"""
        try:
            all_jobs = await self._get_all_jobs()

            stats = {
                "total_jobs": len(all_jobs),
                "active_jobs": 0,
                "completed_jobs": 0,
                "failed_jobs": 0,
                "cancelled_jobs": 0,
                "by_model_type": {},
                "by_status": {},
                "average_duration_seconds": None,
                "success_rate": 0.0,
            }

            # Count by status
            durations = []
            for job in all_jobs:
                status = job.status.value
                stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

                if TrainingJobStatus.is_active(
                    job.status.value if hasattr(job.status, "value") else job.status
                ):
                    stats["active_jobs"] += 1
                elif job.status == TrainingJobStatus.COMPLETED:
                    stats["completed_jobs"] += 1
                    if job.duration_seconds:
                        durations.append(job.duration_seconds)
                elif job.status == TrainingJobStatus.FAILED:
                    stats["failed_jobs"] += 1
                elif job.status == TrainingJobStatus.CANCELLED:
                    stats["cancelled_jobs"] += 1

                # Count by model type
                model_type = job.model_type
                stats["by_model_type"][model_type] = (
                    stats["by_model_type"].get(model_type, 0) + 1
                )

            # Calculate averages
            if durations:
                stats["average_duration_seconds"] = sum(durations) / len(durations)

            # Calculate success rate
            if stats["total_jobs"] > 0:
                stats["success_rate"] = stats["completed_jobs"] / stats["total_jobs"]

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
            cleaned_count = 0
            current_time = datetime.now(timezone.utc)

            all_jobs = await self._get_all_jobs()

            for job in all_jobs:
                should_cleanup = False

                # Check if job should be cleaned up based on status and age
                if job.status == TrainingJobStatus.COMPLETED:
                    if job.completed_at:
                        age = current_time - job.completed_at
                        if age.total_seconds() > StorageConstants.COMPLETED_JOB_TTL:
                            should_cleanup = True

                elif job.status == TrainingJobStatus.FAILED:
                    if job.completed_at:
                        age = current_time - job.completed_at
                        if age.total_seconds() > StorageConstants.FAILED_JOB_TTL:
                            should_cleanup = True

                # Clean up old jobs regardless of status
                age_since_created = current_time - job.created_at
                if age_since_created.total_seconds() > StorageConstants.JOB_STORAGE_TTL:
                    should_cleanup = True

                if should_cleanup:
                    await self.delete_job(job.job_id)
                    cleaned_count += 1

            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old jobs")

            return cleaned_count

        except Exception as e:
            self.logger.error(f"Failed to cleanup old jobs: {e}")
            return 0

    # Private methods

    async def _save_job_to_disk(self, job_info: TrainingJobInfo) -> None:
        """Save job to disk in appropriate directory"""
        try:
            # Determine directory based on status
            status_value = (
                job_info.status.value
                if hasattr(job_info.status, "value")
                else job_info.status
            )
            if TrainingJobStatus.is_active(status_value):
                job_dir = self.jobs_dir / "active"
            elif job_info.status == TrainingJobStatus.COMPLETED:
                job_dir = self.jobs_dir / "completed"
            else:
                job_dir = self.jobs_dir / "failed"

            job_file = job_dir / f"{job_info.job_id}.json"

            # Convert to dict for JSON serialization
            job_dict = job_info.model_dump()

            # Handle datetime serialization
            for key, value in job_dict.items():
                if isinstance(value, datetime):
                    job_dict[key] = value.isoformat()

            # Write to file
            with open(job_file, "w") as f:
                json.dump(job_dict, f, indent=2, default=str)

            # Remove from other directories if status changed
            for other_dir in ["active", "completed", "failed"]:
                if other_dir != job_dir.name:
                    other_file = self.jobs_dir / other_dir / f"{job_info.job_id}.json"
                    if other_file.exists():
                        other_file.unlink()

        except Exception as e:
            self.logger.error(f"Failed to save job {job_info.job_id} to disk: {e}")
            raise

    async def _load_job_from_disk(self, job_id: str) -> Optional[TrainingJobInfo]:
        """Load job from disk"""
        try:
            # Check all directories
            for subdir in ["active", "completed", "failed"]:
                job_file = self.jobs_dir / subdir / f"{job_id}.json"
                if job_file.exists():
                    with open(job_file, "r") as f:
                        job_dict = json.load(f)

                    # Handle datetime deserialization
                    for key in ["created_at", "started_at", "completed_at"]:
                        if job_dict.get(key):
                            job_dict[key] = datetime.fromisoformat(job_dict[key])

                    return TrainingJobInfo(**job_dict)

            return None

        except Exception as e:
            self.logger.error(f"Failed to load job {job_id} from disk: {e}")
            return None

    async def _delete_job_from_disk(self, job_id: str) -> None:
        """Delete job files from disk"""
        try:
            # Remove from all directories
            for subdir in ["active", "completed", "failed"]:
                job_file = self.jobs_dir / subdir / f"{job_id}.json"
                if job_file.exists():
                    job_file.unlink()

            # Remove progress file
            progress_file = self.jobs_dir / "progress" / f"{job_id}_progress.json"
            if progress_file.exists():
                progress_file.unlink()

        except Exception as e:
            self.logger.error(f"Failed to delete job {job_id} from disk: {e}")
            raise

    async def _get_all_jobs(self) -> List[TrainingJobInfo]:
        """Get all jobs from cache and disk"""
        try:
            await self._ensure_initialized()

            all_jobs = []

            # Get from cache first
            assert self._cache_lock is not None, "Repository not properly initialized"
            async with self._cache_lock:
                all_jobs.extend(self._job_cache.values())

            # Get cached job IDs
            cached_ids = {job.job_id for job in all_jobs}

            # Load any jobs from disk that aren't in cache
            for subdir in ["active", "completed", "failed"]:
                subdir_path = self.jobs_dir / subdir
                for job_file in subdir_path.glob("*.json"):
                    job_id = job_file.stem
                    if job_id not in cached_ids:
                        job_info = await self._load_job_from_disk(job_id)
                        if job_info:
                            all_jobs.append(job_info)
                            # Add to cache
                            async with self._cache_lock:
                                self._job_cache[job_id] = job_info

            return all_jobs

        except Exception as e:
            self.logger.error(f"Failed to get all jobs: {e}")
            return []

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
            model_type_values = [mt.value for mt in filter_criteria.model_types]
            filtered_jobs = [
                job for job in filtered_jobs if job.model_type in model_type_values
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
            sort_key = filter_criteria.sort_by
            reverse = filter_criteria.sort_order == "desc"

            if sort_key == "created_at":
                return sorted(jobs, key=lambda x: x.created_at, reverse=reverse)
            elif sort_key == "completed_at":
                return sorted(
                    jobs, key=lambda x: x.completed_at or datetime.min, reverse=reverse
                )
            elif sort_key == "progress":
                return sorted(jobs, key=lambda x: x.progress, reverse=reverse)
            elif sort_key == "duration_seconds":
                return sorted(
                    jobs, key=lambda x: x.duration_seconds or 0, reverse=reverse
                )
            else:
                return jobs

        except Exception as e:
            self.logger.warning(f"Failed to apply sorting: {e}")
            return jobs

    async def _apply_pagination(
        self, jobs: List[TrainingJobInfo], filter_criteria: TrainingJobFilter
    ) -> List[TrainingJobInfo]:
        """Apply pagination to job list"""
        start_idx = filter_criteria.offset
        end_idx = start_idx + filter_criteria.limit
        return jobs[start_idx:end_idx]

    async def _save_progress_update(
        self, job_id: str, progress_update: TrainingProgressUpdate
    ) -> None:
        """Save detailed progress update to separate file"""
        try:
            progress_file = self.jobs_dir / "progress" / f"{job_id}_progress.json"

            # Load existing progress history
            progress_history = []
            if progress_file.exists():
                with open(progress_file, "r") as f:
                    progress_history = json.load(f)

            # Add new progress update
            progress_dict = progress_update.model_dump()
            progress_dict["timestamp"] = progress_update.timestamp.isoformat()
            progress_history.append(progress_dict)

            # Keep only last 100 updates
            if len(progress_history) > 100:
                progress_history = progress_history[-100:]

            # Save back to file
            with open(progress_file, "w") as f:
                json.dump(progress_history, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save progress update for job {job_id}: {e}")

    def _start_cleanup_task(self) -> None:
        """Start background cleanup task"""

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(TrainingConstants.CLEANUP_INTERVAL)
                    await self.cleanup_old_jobs()
                except Exception as e:
                    self.logger.error(f"Error in cleanup task: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def shutdown(self) -> None:
        """Shutdown the repository and cleanup tasks"""
        try:
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            self.logger.info("Training job repository shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during repository shutdown: {e}")
