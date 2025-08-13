# interfaces/training_job_repository_interface.py

"""
Training Job Repository Interface

Defines the contract for training job persistence operations.
Supports CRUD operations for training jobs with async operations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ..schemas.training_schemas import (
    TrainingJobInfo,
    TrainingJobFilter,
    TrainingProgressUpdate,
)
from ..core.constants import TrainingJobStatus


class TrainingJobRepositoryError(Exception):
    """Base exception for training job repository errors."""

    pass


class TrainingJobRepositoryInterface(ABC):
    """Abstract interface for training job data persistence."""

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the repository (call this when event loop is running).

        Raises:
            TrainingJobRepositoryError: If initialization fails
        """
        pass

    @abstractmethod
    async def create_job(self, job_info: TrainingJobInfo) -> bool:
        """
        Create a new training job entry.

        Args:
            job_info: Training job information

        Returns:
            True if job was created successfully

        Raises:
            TrainingJobRepositoryError: If creation fails
        """
        pass

    @abstractmethod
    async def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a training job.

        Args:
            job_id: Job identifier
            updates: Dictionary of fields to update

        Returns:
            True if job was updated successfully

        Raises:
            TrainingJobRepositoryError: If update fails
        """
        pass

    @abstractmethod
    async def get_job(self, job_id: str) -> Optional[TrainingJobInfo]:
        """
        Get a training job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job information if found, None otherwise

        Raises:
            TrainingJobRepositoryError: If retrieval fails
        """
        pass

    @abstractmethod
    async def list_jobs(
        self, filter_criteria: Optional[TrainingJobFilter] = None
    ) -> List[TrainingJobInfo]:
        """
        List training jobs with optional filtering.

        Args:
            filter_criteria: Optional filter criteria

        Returns:
            List of matching jobs

        Raises:
            TrainingJobRepositoryError: If listing fails
        """
        pass

    @abstractmethod
    async def delete_job(self, job_id: str) -> bool:
        """
        Delete a training job.

        Args:
            job_id: Job identifier

        Returns:
            True if job was deleted successfully

        Raises:
            TrainingJobRepositoryError: If deletion fails
        """
        pass

    @abstractmethod
    async def get_active_jobs(self) -> List[TrainingJobInfo]:
        """
        Get all active (non-terminal) training jobs.

        Returns:
            List of active jobs

        Raises:
            TrainingJobRepositoryError: If retrieval fails
        """
        pass

    @abstractmethod
    async def get_jobs_by_status(
        self, status: TrainingJobStatus
    ) -> List[TrainingJobInfo]:
        """
        Get all jobs with specific status.

        Args:
            status: Job status to filter by

        Returns:
            List of jobs with specified status

        Raises:
            TrainingJobRepositoryError: If retrieval fails
        """
        pass

    @abstractmethod
    async def update_progress(
        self, job_id: str, progress_update: TrainingProgressUpdate
    ) -> bool:
        """
        Update training job progress.

        Args:
            job_id: Job identifier
            progress_update: Progress update information

        Returns:
            True if progress was updated successfully

        Raises:
            TrainingJobRepositoryError: If update fails
        """
        pass

    @abstractmethod
    async def get_job_statistics(self) -> Dict[str, Any]:
        """
        Get training job statistics.

        Returns:
            Dictionary with job statistics

        Raises:
            TrainingJobRepositoryError: If statistics retrieval fails
        """
        pass

    @abstractmethod
    async def cleanup_old_jobs(self) -> int:
        """
        Clean up old completed and failed jobs.

        Returns:
            Number of jobs cleaned up

        Raises:
            TrainingJobRepositoryError: If cleanup fails
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the repository and cleanup resources.

        Raises:
            TrainingJobRepositoryError: If shutdown fails
        """
        pass
