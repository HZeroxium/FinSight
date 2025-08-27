# repositories/training_job_facade.py

"""
Training Job Repository Facade

Provides dependency injection and abstraction for training job repositories.
Supports switching between different repository implementations (file-based, Redis, etc.)
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from common.logger.logger_factory import LoggerFactory

from ..core.config import get_settings
from ..core.constants import TrainingJobStatus
from ..interfaces.training_job_repository_interface import (
    TrainingJobRepositoryError, TrainingJobRepositoryInterface)
from ..schemas.training_schemas import (TrainingJobFilter, TrainingJobInfo,
                                        TrainingProgressUpdate)
from .file_training_job_repository import FileTrainingJobRepository
from .redis_training_job_repository import RedisTrainingJobRepository


class RepositoryType(Enum):
    """Available repository implementation types"""

    FILE_BASED = "file"
    REDIS = "redis"


class TrainingJobFacade:
    """
    Facade for training job repositories providing dependency injection and abstraction
    """

    def __init__(
        self, repository_type: Optional[RepositoryType] = None, **repository_kwargs
    ):
        self.logger = LoggerFactory.get_logger("TrainingJobFacade")
        self.settings = get_settings()

        # Determine repository type
        if repository_type is None:
            # Try to get from settings, default to file-based
            repo_type_str = getattr(
                self.settings, "training_job_repository_type", "file"
            )
            repository_type = (
                RepositoryType.FILE_BASED
                if repo_type_str == "file"
                else RepositoryType.REDIS
            )

        self.repository_type = repository_type
        self.repository: Optional[TrainingJobRepositoryInterface] = None
        self.repository_kwargs = repository_kwargs
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the facade and underlying repository"""
        if self._is_initialized:
            return

        try:
            # Create the appropriate repository implementation
            if self.repository_type == RepositoryType.FILE_BASED:
                self.repository = FileTrainingJobRepository()
                self.logger.info("Initialized with file-based repository")
            elif self.repository_type == RepositoryType.REDIS:
                redis_url = self.repository_kwargs.get("redis_url")
                self.repository = RedisTrainingJobRepository(redis_url=redis_url)
                self.logger.info("Initialized with Redis repository")
            else:
                raise TrainingJobRepositoryError(
                    f"Unknown repository type: {self.repository_type}"
                )

            # Initialize the repository
            await self.repository.initialize()

            self._is_initialized = True
            self.logger.info(
                f"TrainingJobFacade initialized with {self.repository_type.value} repository"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize TrainingJobFacade: {e}")
            raise TrainingJobRepositoryError(f"Failed to initialize facade: {e}")

    async def _ensure_initialized(self) -> None:
        """Ensure facade is initialized before use"""
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
        return await self.repository.create_job(job_info)

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
        return await self.repository.update_job(job_id, updates)

    async def get_job(self, job_id: str) -> Optional[TrainingJobInfo]:
        """
        Get a training job by ID

        Args:
            job_id: Job identifier

        Returns:
            Optional[TrainingJobInfo]: Job information if found
        """
        await self._ensure_initialized()
        return await self.repository.get_job(job_id)

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
        await self._ensure_initialized()
        return await self.repository.list_jobs(filter_criteria)

    async def delete_job(self, job_id: str) -> bool:
        """
        Delete a training job

        Args:
            job_id: Job identifier

        Returns:
            bool: True if job was deleted successfully
        """
        await self._ensure_initialized()
        return await self.repository.delete_job(job_id)

    async def get_active_jobs(self) -> List[TrainingJobInfo]:
        """
        Get all active (non-terminal) training jobs

        Returns:
            List[TrainingJobInfo]: List of active jobs
        """
        await self._ensure_initialized()
        return await self.repository.get_active_jobs()

    async def get_jobs_by_status(
        self, status: TrainingJobStatus
    ) -> List[TrainingJobInfo]:
        """
        Get all jobs with specific status

        Args:
            status: Job status to filter by

        Returns:
            List[TrainingJobInfo]: List of jobs with specified status
        """
        await self._ensure_initialized()
        return await self.repository.get_jobs_by_status(status)

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
        await self._ensure_initialized()
        return await self.repository.update_progress(job_id, progress_update)

    async def get_job_statistics(self) -> Dict[str, Any]:
        """
        Get training job statistics

        Returns:
            Dict[str, Any]: Dictionary with job statistics
        """
        await self._ensure_initialized()
        return await self.repository.get_job_statistics()

    async def cleanup_old_jobs(self) -> int:
        """
        Clean up old completed and failed jobs

        Returns:
            int: Number of jobs cleaned up
        """
        await self._ensure_initialized()
        return await self.repository.cleanup_old_jobs()

    async def shutdown(self) -> None:
        """
        Shutdown the facade and underlying repository
        """
        try:
            if self.repository and self._is_initialized:
                await self.repository.shutdown()

            self._is_initialized = False
            self.logger.info("TrainingJobFacade shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during facade shutdown: {e}")
            raise TrainingJobRepositoryError(f"Failed to shutdown facade: {e}")

    def get_repository_type(self) -> RepositoryType:
        """
        Get the current repository type

        Returns:
            RepositoryType: Current repository implementation type
        """
        return self.repository_type

    def get_repository_info(self) -> Dict[str, Any]:
        """
        Get information about the current repository

        Returns:
            Dict[str, Any]: Repository information
        """
        return {
            "type": self.repository_type.value,
            "class": self.repository.__class__.__name__ if self.repository else None,
            "initialized": self._is_initialized,
            "settings": {
                "repository_kwargs": self.repository_kwargs,
            },
        }

    async def switch_repository(
        self,
        new_repository_type: RepositoryType,
        migrate_data: bool = False,
        **new_repository_kwargs,
    ) -> bool:
        """
        Switch to a different repository implementation

        Args:
            new_repository_type: New repository type to switch to
            migrate_data: Whether to migrate existing data to new repository
            **new_repository_kwargs: Additional arguments for new repository

        Returns:
            bool: True if switch was successful
        """
        try:
            if new_repository_type == self.repository_type:
                self.logger.info("Already using the requested repository type")
                return True

            old_repository = self.repository
            old_data = []

            # If migrating data, get all existing jobs
            if migrate_data and old_repository and self._is_initialized:
                try:
                    old_data = await old_repository.list_jobs()
                    self.logger.info(f"Retrieved {len(old_data)} jobs for migration")
                except Exception as e:
                    self.logger.warning(f"Failed to retrieve data for migration: {e}")
                    migrate_data = False

            # Shutdown old repository
            if old_repository and self._is_initialized:
                await old_repository.shutdown()

            # Create new repository
            self.repository_type = new_repository_type
            self.repository_kwargs = new_repository_kwargs
            self._is_initialized = False

            # Initialize new repository
            await self.initialize()

            # Migrate data if requested
            if migrate_data and old_data:
                migration_count = 0
                for job in old_data:
                    try:
                        if await self.repository.create_job(job):
                            migration_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to migrate job {job.job_id}: {e}")

                self.logger.info(
                    f"Successfully migrated {migration_count}/{len(old_data)} jobs"
                )

            self.logger.info(
                f"Successfully switched to {new_repository_type.value} repository"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to switch repository: {e}")
            raise TrainingJobRepositoryError(f"Failed to switch repository: {e}")

    @classmethod
    async def create_file_based_facade(cls, **kwargs) -> "TrainingJobFacade":
        """
        Create a facade with file-based repository

        Args:
            **kwargs: Additional arguments for the repository

        Returns:
            TrainingJobFacade: Initialized facade with file-based repository
        """
        facade = cls(RepositoryType.FILE_BASED, **kwargs)
        await facade.initialize()
        return facade

    @classmethod
    async def create_redis_facade(
        cls, redis_url: Optional[str] = None, **kwargs
    ) -> "TrainingJobFacade":
        """
        Create a facade with Redis repository

        Args:
            redis_url: Redis connection URL
            **kwargs: Additional arguments for the repository

        Returns:
            TrainingJobFacade: Initialized facade with Redis repository
        """
        facade = cls(RepositoryType.REDIS, redis_url=redis_url, **kwargs)
        await facade.initialize()
        return facade

    @classmethod
    async def create_from_settings(cls) -> "TrainingJobFacade":
        """
        Create a facade based on application settings

        Returns:
            TrainingJobFacade: Initialized facade with repository type from settings
        """
        facade = cls()  # Will use settings to determine repository type
        await facade.initialize()
        return facade


# Convenience functions for dependency injection

_global_facade: Optional[TrainingJobFacade] = None


async def get_training_job_facade() -> TrainingJobFacade:
    """
    Get or create the global training job facade instance

    Returns:
        TrainingJobFacade: Global facade instance
    """
    global _global_facade

    if _global_facade is None or not _global_facade._is_initialized:
        _global_facade = await TrainingJobFacade.create_from_settings()

    return _global_facade


async def set_global_training_job_facade(facade: TrainingJobFacade) -> None:
    """
    Set the global training job facade instance

    Args:
        facade: The facade instance to set as global
    """
    global _global_facade

    # Shutdown old facade if it exists
    if _global_facade and _global_facade._is_initialized:
        await _global_facade.shutdown()

    _global_facade = facade


async def shutdown_global_facade() -> None:
    """Shutdown the global facade instance"""
    global _global_facade

    if _global_facade and _global_facade._is_initialized:
        await _global_facade.shutdown()
        _global_facade = None
