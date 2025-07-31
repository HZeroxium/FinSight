# schemas/job_schemas.py

"""
Pydantic schemas for job management API endpoints.
Handles request/response models for cron job management.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from enum import Enum

from .news_schemas import NewsSource


class JobStatus(str, Enum):
    """Job service status enumeration"""

    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    DEGRADED = "degraded"


class JobConfigModel(BaseModel):
    """Configuration model for news crawler job"""

    sources: List[str] = Field(
        default_factory=lambda: [source.value for source in NewsSource],
        description="List of news sources to crawl",
    )
    collector_preferences: Dict[str, str] = Field(
        default_factory=dict, description="Preferred collector type for each source"
    )
    max_items_per_source: int = Field(
        default=50, ge=1, le=1000, description="Maximum items to collect per source"
    )
    enable_fallback: bool = Field(
        default=True, description="Enable fallback to alternative collectors"
    )
    schedule: str = Field(
        default="0 */2 * * *",
        description="Cron schedule expression (5 fields: minute hour day month day_of_week)",
    )
    config_overrides: Dict[str, Any] = Field(
        default_factory=dict, description="Additional configuration overrides"
    )
    notification: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False, "webhook_url": None, "email": None},
        description="Notification settings",
    )

    @field_validator("schedule")
    @classmethod
    def validate_cron_schedule(cls, v: str) -> str:
        """Validate cron schedule format"""
        if not v or not isinstance(v, str):
            raise ValueError("Schedule must be a valid string")

        parts = v.strip().split()
        if len(parts) != 5:
            raise ValueError(
                "Cron schedule must have exactly 5 fields: minute hour day month day_of_week"
            )

        return v.strip()

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: List[str]) -> List[str]:
        """Validate news sources"""
        if not v:
            raise ValueError("At least one news source must be specified")

        valid_sources = [source.value for source in NewsSource]
        for source in v:
            if source not in valid_sources:
                raise ValueError(
                    f"Invalid news source: {source}. Valid sources: {valid_sources}"
                )

        return v


class JobStatsModel(BaseModel):
    """Job statistics model"""

    total_jobs: int = Field(default=0, description="Total number of jobs executed")
    successful_jobs: int = Field(default=0, description="Number of successful jobs")
    failed_jobs: int = Field(default=0, description="Number of failed jobs")
    last_run: Optional[str] = Field(
        None, description="Last job run timestamp (ISO format)"
    )
    last_success: Optional[str] = Field(
        None, description="Last successful job timestamp (ISO format)"
    )
    last_error: Optional[Dict[str, Any]] = Field(
        None, description="Last error information"
    )


class JobStatusResponse(BaseModel):
    """Job service status response"""

    service: str = Field(default="news-crawler-job", description="Service name")
    version: str = Field(default="1.0.0", description="Service version")
    status: JobStatus = Field(..., description="Current job service status")
    is_running: bool = Field(..., description="Whether the job service is running")
    pid: Optional[int] = Field(None, description="Process ID")
    pid_file: Optional[str] = Field(None, description="PID file path")
    config_file: Optional[str] = Field(None, description="Configuration file path")
    log_file: Optional[str] = Field(None, description="Log file path")
    scheduler_running: bool = Field(
        default=False, description="Whether the scheduler is running"
    )
    next_run: Optional[str] = Field(
        None, description="Next scheduled run time (ISO format)"
    )
    stats: JobStatsModel = Field(
        default_factory=JobStatsModel, description="Job statistics"
    )


class JobStartRequest(BaseModel):
    """Request to start the job service"""

    config: Optional[JobConfigModel] = Field(
        None, description="Job configuration (uses existing config if not provided)"
    )
    force_restart: bool = Field(
        default=False, description="Force restart if already running"
    )


class JobStopRequest(BaseModel):
    """Request to stop the job service"""

    graceful: bool = Field(
        default=True,
        description="Whether to stop gracefully (wait for current job to finish)",
    )


class ManualJobRequest(BaseModel):
    """Request to run a manual job"""

    sources: Optional[List[str]] = Field(
        None,
        description="Specific sources to crawl (uses config default if not provided)",
    )
    max_items_per_source: Optional[int] = Field(
        None,
        ge=1,
        le=1000,
        description="Maximum items per source (uses config default if not provided)",
    )
    config_overrides: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Temporary configuration overrides for this job",
    )

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate news sources if provided"""
        if v is None:
            return v

        if not v:
            raise ValueError("Sources list cannot be empty if provided")

        valid_sources = [source.value for source in NewsSource]
        for source in v:
            if source not in valid_sources:
                raise ValueError(
                    f"Invalid news source: {source}. Valid sources: {valid_sources}"
                )

        return v


class ManualJobResponse(BaseModel):
    """Response from manual job execution"""

    status: str = Field(..., description="Job execution status")
    job_id: str = Field(..., description="Unique job identifier")
    sources: List[str] = Field(..., description="Sources that were crawled")
    max_items_per_source: int = Field(..., description="Max items per source used")
    start_time: str = Field(..., description="Job start time (ISO format)")
    duration: Optional[float] = Field(None, description="Job duration in seconds")
    results: Optional[Dict[str, Any]] = Field(None, description="Job execution results")


class JobConfigUpdateRequest(BaseModel):
    """Request to update job configuration"""

    sources: Optional[List[str]] = Field(
        None, description="List of news sources to crawl"
    )
    collector_preferences: Optional[Dict[str, str]] = Field(
        None, description="Preferred collector type for each source"
    )
    max_items_per_source: Optional[int] = Field(
        None, ge=1, le=1000, description="Maximum items to collect per source"
    )
    enable_fallback: Optional[bool] = Field(
        None, description="Enable fallback to alternative collectors"
    )
    schedule: Optional[str] = Field(None, description="Cron schedule expression")
    config_overrides: Optional[Dict[str, Any]] = Field(
        None, description="Additional configuration overrides"
    )
    notification: Optional[Dict[str, Any]] = Field(
        None, description="Notification settings"
    )

    @field_validator("schedule")
    @classmethod
    def validate_cron_schedule(cls, v: Optional[str]) -> Optional[str]:
        """Validate cron schedule format if provided"""
        if v is None:
            return v

        if not isinstance(v, str):
            raise ValueError("Schedule must be a valid string")

        parts = v.strip().split()
        if len(parts) != 5:
            raise ValueError(
                "Cron schedule must have exactly 5 fields: minute hour day month day_of_week"
            )

        return v.strip()

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate news sources if provided"""
        if v is None:
            return v

        if not v:
            raise ValueError("Sources list cannot be empty if provided")

        valid_sources = [source.value for source in NewsSource]
        for source in v:
            if source not in valid_sources:
                raise ValueError(
                    f"Invalid news source: {source}. Valid sources: {valid_sources}"
                )

        return v


class JobOperationResponse(BaseModel):
    """Generic response for job operations"""

    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Operation result message")
    status: Optional[JobStatus] = Field(
        None, description="Current job status after operation"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Operation timestamp (ISO format)",
    )
    details: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional operation details"
    )
