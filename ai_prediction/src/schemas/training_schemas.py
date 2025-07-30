# schemas/training_schemas.py

"""
Enhanced training schemas with async support
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum

from .model_schemas import ModelConfig
from .enums import ModelType, TimeFrame
from .base_schemas import BaseResponse
from ..core.constants import TrainingJobStatus


class TrainingJobPriority(str, Enum):
    """Training job priority levels"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class AsyncTrainingRequest(BaseModel):
    """Enhanced request schema for asynchronous model training"""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    # Core parameters
    symbol: str = Field(..., min_length=1, description="Trading symbol (e.g., BTCUSDT)")
    timeframe: TimeFrame = Field(TimeFrame.DAY_1, description="Data timeframe")
    model_type: ModelType = Field(
        ModelType.PATCHTST, description="Type of model to train"
    )

    # Training configuration
    config: ModelConfig = Field(
        default_factory=ModelConfig, description="Model configuration"
    )

    # Async-specific parameters
    priority: TrainingJobPriority = Field(
        TrainingJobPriority.NORMAL, description="Training job priority"
    )
    timeout_seconds: Optional[int] = Field(
        None, ge=60, le=7200, description="Training timeout in seconds (max 2 hours)"
    )
    notification_webhook: Optional[str] = Field(
        None, description="Webhook URL for training completion notifications"
    )
    tags: Optional[Dict[str, str]] = Field(
        None, description="Custom tags for job identification"
    )

    # Data configuration
    force_retrain: bool = Field(
        False, description="Force retraining even if model exists"
    )
    save_intermediate_checkpoints: bool = Field(
        True, description="Save intermediate model checkpoints during training"
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate trading symbol format"""
        if not v or len(v.strip()) < 3:
            raise ValueError("Symbol must be at least 3 characters long")
        return v.upper().strip()

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """Validate tags format"""
        if v is None:
            return v

        if len(v) > 10:
            raise ValueError("Maximum 10 tags allowed")

        for key, value in v.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Tags must be string key-value pairs")
            if len(key) > 50 or len(value) > 200:
                raise ValueError("Tag key max 50 chars, value max 200 chars")

        return v


class TrainingJobInfo(BaseModel):
    """Training job information schema"""

    model_config = ConfigDict(validate_assignment=True)

    # Job identification
    job_id: str = Field(..., description="Unique job identifier")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Data timeframe")
    model_type: str = Field(..., description="Model type")

    # Job status
    status: TrainingJobStatus = Field(..., description="Current job status")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Training progress (0-1)")

    # Timestamps
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Training start timestamp")
    completed_at: Optional[datetime] = Field(
        None, description="Training completion timestamp"
    )

    # Configuration
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Training configuration"
    )
    priority: TrainingJobPriority = Field(
        TrainingJobPriority.NORMAL, description="Job priority"
    )

    # Results and metrics
    current_stage: Optional[str] = Field(None, description="Current training stage")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_code: Optional[str] = Field(None, description="Error code if failed")

    # Performance metrics
    training_metrics: Optional[Dict[str, float]] = Field(
        None, description="Training performance metrics"
    )
    validation_metrics: Optional[Dict[str, float]] = Field(
        None, description="Validation performance metrics"
    )

    # Resource usage
    duration_seconds: Optional[float] = Field(
        None, description="Training duration in seconds"
    )
    estimated_remaining_seconds: Optional[float] = Field(
        None, description="Estimated remaining time in seconds"
    )

    # Output information
    model_path: Optional[str] = Field(None, description="Path to saved model")
    model_size_bytes: Optional[int] = Field(
        None, description="Model file size in bytes"
    )

    # Additional metadata
    tags: Optional[Dict[str, str]] = Field(None, description="Custom job tags")
    worker_id: Optional[str] = Field(None, description="Worker that processed the job")

    @field_validator("model_path", mode="before")
    def convert_path_to_str(cls, v):
        if isinstance(v, Path):
            return str(v)
        return v


class AsyncTrainingResponse(BaseResponse):
    """Response schema for asynchronous training requests"""

    model_config = ConfigDict(validate_assignment=True)

    job_id: Optional[str] = Field(None, description="Training job identifier")
    status: Optional[TrainingJobStatus] = Field(None, description="Initial job status")
    estimated_duration_seconds: Optional[int] = Field(
        None, description="Estimated training duration"
    )
    queue_position: Optional[int] = Field(
        None, description="Position in training queue"
    )
    status_endpoint: Optional[str] = Field(
        None, description="Endpoint to check job status"
    )


class TrainingJobStatusResponse(BaseResponse):
    """Response schema for training job status requests"""

    model_config = ConfigDict(validate_assignment=True)

    job: Optional[TrainingJobInfo] = Field(None, description="Training job information")


class TrainingJobListResponse(BaseResponse):
    """Response schema for listing training jobs"""

    model_config = ConfigDict(validate_assignment=True)

    jobs: List[TrainingJobInfo] = Field(
        default_factory=list, description="List of training jobs"
    )
    total_count: int = Field(0, description="Total number of jobs")
    active_count: int = Field(0, description="Number of active jobs")
    completed_count: int = Field(0, description="Number of completed jobs")
    failed_count: int = Field(0, description="Number of failed jobs")


class TrainingJobCancelRequest(BaseModel):
    """Request schema for cancelling training jobs"""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    reason: Optional[str] = Field(
        None, max_length=500, description="Cancellation reason"
    )
    force: bool = Field(
        False, description="Force cancellation even if in critical stage"
    )


class TrainingJobCancelResponse(BaseResponse):
    """Response schema for training job cancellation"""

    model_config = ConfigDict(validate_assignment=True)

    job_id: str = Field(..., description="Cancelled job identifier")
    was_running: bool = Field(..., description="Whether job was actively running")
    cleanup_completed: bool = Field(..., description="Whether cleanup was completed")


class TrainingQueueInfo(BaseModel):
    """Information about the training queue"""

    model_config = ConfigDict(validate_assignment=True)

    # Queue statistics
    total_jobs: int = Field(0, description="Total jobs in queue")
    pending_jobs: int = Field(0, description="Jobs waiting to start")
    running_jobs: int = Field(0, description="Currently running jobs")
    max_concurrent: int = Field(0, description="Maximum concurrent jobs allowed")

    # Queue health
    is_healthy: bool = Field(True, description="Whether queue is operating normally")
    last_processed_at: Optional[datetime] = Field(
        None, description="Last job processing time"
    )

    # Resource information
    available_workers: int = Field(0, description="Available worker slots")
    total_workers: int = Field(0, description="Total worker slots")

    # Performance metrics
    average_training_time_seconds: Optional[float] = Field(
        None, description="Average training time"
    )
    queue_throughput_per_hour: Optional[float] = Field(
        None, description="Jobs completed per hour"
    )


class TrainingQueueResponse(BaseResponse):
    """Response schema for training queue information"""

    model_config = ConfigDict(validate_assignment=True)

    queue_info: TrainingQueueInfo = Field(..., description="Training queue information")


class TrainingProgressUpdate(BaseModel):
    """Schema for training progress updates"""

    model_config = ConfigDict(validate_assignment=True)

    job_id: str = Field(..., description="Job identifier")
    status: TrainingJobStatus = Field(..., description="Current status")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Progress percentage")
    current_stage: str = Field(..., description="Current training stage")

    # Optional metrics
    current_epoch: Optional[int] = Field(None, description="Current training epoch")
    total_epochs: Optional[int] = Field(None, description="Total training epochs")
    current_loss: Optional[float] = Field(None, description="Current training loss")
    validation_loss: Optional[float] = Field(
        None, description="Current validation loss"
    )

    # Time estimates
    elapsed_seconds: float = Field(0.0, description="Elapsed training time")
    estimated_remaining_seconds: Optional[float] = Field(
        None, description="Estimated remaining time"
    )

    # Resource usage
    memory_usage_mb: Optional[float] = Field(None, description="Current memory usage")
    cpu_usage_percent: Optional[float] = Field(None, description="Current CPU usage")

    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Update timestamp"
    )


class TrainingJobFilter(BaseModel):
    """Filter criteria for training job queries"""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    # Status filters
    statuses: Optional[List[TrainingJobStatus]] = Field(
        None, description="Filter by statuses"
    )
    exclude_statuses: Optional[List[TrainingJobStatus]] = Field(
        None, description="Exclude specific statuses"
    )

    # Time filters
    created_after: Optional[datetime] = Field(
        None, description="Created after timestamp"
    )
    created_before: Optional[datetime] = Field(
        None, description="Created before timestamp"
    )
    completed_after: Optional[datetime] = Field(
        None, description="Completed after timestamp"
    )
    completed_before: Optional[datetime] = Field(
        None, description="Completed before timestamp"
    )

    # Model filters
    symbols: Optional[List[str]] = Field(None, description="Filter by trading symbols")
    timeframes: Optional[List[TimeFrame]] = Field(
        None, description="Filter by timeframes"
    )
    model_types: Optional[List[ModelType]] = Field(
        None, description="Filter by model types"
    )

    # Tag filters
    tags: Optional[Dict[str, str]] = Field(None, description="Filter by tags")

    # Pagination
    offset: int = Field(0, ge=0, description="Result offset")
    limit: int = Field(50, ge=1, le=1000, description="Result limit")

    # Sorting
    sort_by: str = Field("created_at", description="Sort field")
    sort_order: str = Field("desc", pattern="^(asc|desc)$", description="Sort order")


class BackgroundTaskHealth(BaseModel):
    """Background task system health information"""

    model_config = ConfigDict(validate_assignment=True)

    # System status
    is_healthy: bool = Field(..., description="Overall system health")
    active_workers: int = Field(..., description="Number of active workers")
    total_workers: int = Field(..., description="Total number of workers")

    # Queue status
    queue_size: int = Field(..., description="Current queue size")
    max_queue_size: int = Field(..., description="Maximum queue size")

    # Performance metrics
    jobs_processed_last_hour: int = Field(0, description="Jobs processed in last hour")
    average_job_duration_seconds: Optional[float] = Field(
        None, description="Average job duration"
    )

    # Resource usage
    memory_usage_percent: Optional[float] = Field(None, description="Memory usage")
    cpu_usage_percent: Optional[float] = Field(None, description="CPU usage")
    disk_usage_percent: Optional[float] = Field(None, description="Disk usage")

    # Error tracking
    errors_last_hour: int = Field(0, description="Errors in last hour")
    last_error_message: Optional[str] = Field(None, description="Last error message")
    last_health_check: datetime = Field(..., description="Last health check timestamp")


class BackgroundTaskHealthResponse(BaseResponse):
    """Response schema for background task health check"""

    model_config = ConfigDict(validate_assignment=True)

    health: BackgroundTaskHealth = Field(
        ..., description="Background task health information"
    )
