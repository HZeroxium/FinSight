# schemas/job_schemas.py

"""
Market Data Job Schemas

Pydantic schemas for market data job management operations.
Replaces dataclass-based configurations with modern Pydantic v2 models.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator

from .enums import Exchange, TimeFrame, RepositoryType


class JobStatus(str, Enum):
    """Market data job service status enumeration."""

    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    DEGRADED = "degraded"
    STARTING = "starting"
    STOPPING = "stopping"


class JobPriority(str, Enum):
    """Job execution priority levels."""

    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class NotificationLevel(str, Enum):
    """Notification level settings."""

    ALL = "all"
    ERRORS_ONLY = "errors_only"
    NONE = "none"


class MarketDataJobConfigModel(BaseModel):
    """
    Configuration model for market data collection job.

    Replaces the dataclass JobConfig with Pydantic BaseModel v2.
    """

    # Job scheduling
    cron_schedule: str = Field(
        default="*/15 * * * *",
        description="Cron schedule expression (5 fields: minute hour day month day_of_week)",
    )
    timezone: str = Field(default="UTC", description="Timezone for job scheduling")

    # Collection parameters
    exchange: str = Field(
        default=Exchange.BINANCE.value, description="Exchange to collect data from"
    )
    max_lookback_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Maximum days to look back for data collection",
    )
    update_existing: bool = Field(
        default=True, description="Whether to update existing data"
    )
    max_concurrent_symbols: int = Field(
        default=5, ge=1, le=20, description="Maximum concurrent symbol processing"
    )

    # Repository configuration
    repository_type: str = Field(
        default=RepositoryType.MONGODB.value,
        description="Repository type for data storage",
    )
    repository_config: Dict[str, Any] = Field(
        default_factory=dict, description="Additional repository configuration"
    )

    # Limits and filtering
    max_symbols_per_run: int = Field(
        default=50, ge=1, le=500, description="Maximum symbols to process per run"
    )
    max_timeframes_per_run: int = Field(
        default=10, ge=1, le=50, description="Maximum timeframes to process per run"
    )
    priority_symbols: List[str] = Field(
        default_factory=lambda: ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        description="Priority symbols to process first",
    )
    priority_timeframes: List[str] = Field(
        default_factory=lambda: ["1h", "4h", "1d"],
        description="Priority timeframes to process first",
    )

    # Error handling
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed operations",
    )
    retry_delay_minutes: int = Field(
        default=5, ge=1, le=60, description="Delay between retry attempts in minutes"
    )
    skip_failed_symbols: bool = Field(
        default=True, description="Whether to skip failed symbols and continue"
    )

    # Notifications
    enable_notifications: bool = Field(
        default=False, description="Enable job completion notifications"
    )
    notification_webhook: Optional[str] = Field(
        default=None, description="Webhook URL for notifications"
    )
    notification_level: NotificationLevel = Field(
        default=NotificationLevel.ERRORS_ONLY, description="Notification level setting"
    )
    notify_on_success: bool = Field(
        default=False, description="Send notifications on successful completion"
    )
    notify_on_error: bool = Field(
        default=True, description="Send notifications on errors"
    )

    # Performance tuning
    batch_size: int = Field(
        default=1000, ge=100, le=10000, description="Batch size for data processing"
    )
    request_delay_ms: int = Field(
        default=100,
        ge=0,
        le=5000,
        description="Delay between API requests in milliseconds",
    )
    connection_timeout_seconds: int = Field(
        default=30, ge=5, le=300, description="Connection timeout for API requests"
    )

    @field_validator("cron_schedule")
    @classmethod
    def validate_cron_schedule(cls, v: str) -> str:
        """Validate cron schedule format."""
        if not v or not isinstance(v, str):
            raise ValueError("Schedule must be a valid string")

        parts = v.strip().split()
        if len(parts) != 5:
            raise ValueError(
                "Cron schedule must have exactly 5 fields: minute hour day month day_of_week"
            )

        return v.strip()

    @field_validator("exchange")
    @classmethod
    def validate_exchange(cls, v: str) -> str:
        """Validate exchange name."""
        valid_exchanges = [e.value for e in Exchange]
        if v not in valid_exchanges:
            raise ValueError(
                f"Invalid exchange: {v}. Valid exchanges: {valid_exchanges}"
            )
        return v

    @field_validator("repository_type")
    @classmethod
    def validate_repository_type(cls, v: str) -> str:
        """Validate repository type."""
        valid_types = [t.value for t in RepositoryType]
        if v not in valid_types:
            raise ValueError(
                f"Invalid repository type: {v}. Valid types: {valid_types}"
            )
        return v

    @field_validator("priority_symbols")
    @classmethod
    def validate_priority_symbols(cls, v: List[str]) -> List[str]:
        """Validate priority symbols list."""
        if not v:
            return ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # Default fallback

        # Basic symbol format validation
        for symbol in v:
            if not isinstance(symbol, str) or len(symbol) < 6:
                raise ValueError(f"Invalid symbol format: {symbol}")

        return v

    @field_validator("priority_timeframes")
    @classmethod
    def validate_priority_timeframes(cls, v: List[str]) -> List[str]:
        """Validate priority timeframes list."""
        if not v:
            return ["1h", "4h", "1d"]  # Default fallback

        valid_timeframes = [tf.value for tf in TimeFrame]
        for timeframe in v:
            if timeframe not in valid_timeframes:
                raise ValueError(
                    f"Invalid timeframe: {timeframe}. Valid: {valid_timeframes}"
                )

        return v


class JobStatsModel(BaseModel):
    """Market data job execution statistics."""

    total_jobs: int = Field(default=0, description="Total number of jobs executed")
    successful_jobs: int = Field(default=0, description="Number of successful jobs")
    failed_jobs: int = Field(default=0, description="Number of failed jobs")
    last_run: Optional[datetime] = Field(
        default=None, description="Last job run timestamp"
    )
    last_success: Optional[datetime] = Field(
        default=None, description="Last successful job timestamp"
    )
    last_error: Optional[Dict[str, Any]] = Field(
        default=None, description="Last error details"
    )
    average_runtime_seconds: Optional[float] = Field(
        default=None, description="Average job runtime in seconds"
    )
    symbols_processed_last_run: int = Field(
        default=0, description="Symbols processed in last run"
    )
    records_collected_last_run: int = Field(
        default=0, description="Records collected in last run"
    )
    data_collection_rate_per_hour: Optional[float] = Field(
        default=None, description="Data collection rate (records per hour)"
    )


class JobStatusResponse(BaseModel):
    """Market data job service status response."""

    service: str = Field(default="market-data-job", description="Service name")
    version: str = Field(default="1.0.0", description="Service version")
    status: JobStatus = Field(..., description="Current job service status")
    is_running: bool = Field(..., description="Whether the job service is running")
    pid: Optional[int] = Field(default=None, description="Process ID")
    pid_file: Optional[str] = Field(default=None, description="PID file path")
    config_file: Optional[str] = Field(
        default=None, description="Configuration file path"
    )
    log_file: Optional[str] = Field(default=None, description="Log file path")
    scheduler_running: bool = Field(
        default=False, description="Whether the scheduler is running"
    )
    next_run: Optional[datetime] = Field(
        default=None, description="Next scheduled run time"
    )
    uptime_seconds: Optional[int] = Field(
        default=None, description="Service uptime in seconds"
    )
    stats: JobStatsModel = Field(
        default_factory=JobStatsModel, description="Job execution statistics"
    )
    health_status: Dict[str, Any] = Field(
        default_factory=dict, description="Detailed health information"
    )


class JobStartRequest(BaseModel):
    """Request to start the market data job service."""

    config: Optional[MarketDataJobConfigModel] = Field(
        default=None,
        description="Job configuration (uses existing config if not provided)",
    )
    force_restart: bool = Field(
        default=False, description="Force restart if service is already running"
    )
    background_mode: bool = Field(
        default=True, description="Run service in background mode"
    )


class JobStopRequest(BaseModel):
    """Request to stop the market data job service."""

    graceful: bool = Field(default=True, description="Perform graceful shutdown")
    timeout_seconds: int = Field(
        default=30, ge=5, le=300, description="Timeout for graceful shutdown"
    )
    force_kill: bool = Field(
        default=False, description="Force kill if graceful shutdown fails"
    )


class ManualJobRequest(BaseModel):
    """Request to run a manual data collection job."""

    symbols: Optional[List[str]] = Field(
        default=None,
        description="Specific symbols to collect (uses config default if not provided)",
    )
    timeframes: Optional[List[str]] = Field(
        default=None,
        description="Specific timeframes to collect (uses config default if not provided)",
    )
    max_lookback_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=365,
        description="Maximum lookback days (uses config default if not provided)",
    )
    exchange: Optional[str] = Field(
        default=None,
        description="Exchange to collect from (uses config default if not provided)",
    )
    repository_type: Optional[str] = Field(
        default=None,
        description="Repository type for storage (uses config default if not provided)",
    )
    job_priority: JobPriority = Field(
        default=JobPriority.NORMAL, description="Job execution priority"
    )
    async_execution: bool = Field(
        default=True, description="Execute job asynchronously"
    )

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate symbols list."""
        if v is not None:
            if not v:
                raise ValueError("Symbols list cannot be empty if provided")

            for symbol in v:
                if not isinstance(symbol, str) or len(symbol) < 6:
                    raise ValueError(f"Invalid symbol format: {symbol}")

        return v

    @field_validator("timeframes")
    @classmethod
    def validate_timeframes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate timeframes list."""
        if v is not None:
            if not v:
                raise ValueError("Timeframes list cannot be empty if provided")

            valid_timeframes = [tf.value for tf in TimeFrame]
            for timeframe in v:
                if timeframe not in valid_timeframes:
                    raise ValueError(
                        f"Invalid timeframe: {timeframe}. Valid: {valid_timeframes}"
                    )

        return v

    @field_validator("exchange")
    @classmethod
    def validate_exchange(cls, v: Optional[str]) -> Optional[str]:
        """Validate exchange name."""
        if v is not None:
            valid_exchanges = [e.value for e in Exchange]
            if v not in valid_exchanges:
                raise ValueError(
                    f"Invalid exchange: {v}. Valid exchanges: {valid_exchanges}"
                )
        return v


class ManualJobResponse(BaseModel):
    """Response from manual data collection job execution."""

    status: str = Field(..., description="Job execution status")
    job_id: str = Field(..., description="Unique job identifier")
    symbols: List[str] = Field(..., description="Symbols that were processed")
    timeframes: List[str] = Field(..., description="Timeframes that were processed")
    exchange: str = Field(..., description="Exchange used for data collection")
    max_lookback_days: int = Field(..., description="Max lookback days used")
    start_time: datetime = Field(..., description="Job start time")
    end_time: Optional[datetime] = Field(default=None, description="Job end time")
    duration_seconds: Optional[float] = Field(
        default=None, description="Job duration in seconds"
    )
    records_collected: int = Field(default=0, description="Total records collected")
    symbols_processed: int = Field(default=0, description="Number of symbols processed")
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of errors encountered"
    )
    results: Optional[Dict[str, Any]] = Field(
        default=None, description="Detailed job execution results"
    )
    async_execution: bool = Field(
        default=True, description="Whether job was executed asynchronously"
    )


class JobConfigUpdateRequest(BaseModel):
    """Request to update market data job configuration."""

    cron_schedule: Optional[str] = Field(
        default=None, description="Cron schedule expression"
    )
    timezone: Optional[str] = Field(
        default=None, description="Timezone for job scheduling"
    )
    exchange: Optional[str] = Field(
        default=None, description="Exchange to collect data from"
    )
    max_lookback_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=365,
        description="Maximum days to look back for data collection",
    )
    update_existing: Optional[bool] = Field(
        default=None, description="Whether to update existing data"
    )
    max_concurrent_symbols: Optional[int] = Field(
        default=None, ge=1, le=20, description="Maximum concurrent symbol processing"
    )
    repository_type: Optional[str] = Field(
        default=None, description="Repository type for data storage"
    )
    repository_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional repository configuration"
    )
    max_symbols_per_run: Optional[int] = Field(
        default=None, ge=1, le=500, description="Maximum symbols to process per run"
    )
    max_timeframes_per_run: Optional[int] = Field(
        default=None, ge=1, le=50, description="Maximum timeframes to process per run"
    )
    priority_symbols: Optional[List[str]] = Field(
        default=None, description="Priority symbols to process first"
    )
    priority_timeframes: Optional[List[str]] = Field(
        default=None, description="Priority timeframes to process first"
    )
    max_retries: Optional[int] = Field(
        default=None,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed operations",
    )
    retry_delay_minutes: Optional[int] = Field(
        default=None, ge=1, le=60, description="Delay between retry attempts in minutes"
    )
    enable_notifications: Optional[bool] = Field(
        default=None, description="Enable job completion notifications"
    )
    notification_webhook: Optional[str] = Field(
        default=None, description="Webhook URL for notifications"
    )
    notification_level: Optional[NotificationLevel] = Field(
        default=None, description="Notification level setting"
    )
    batch_size: Optional[int] = Field(
        default=None, ge=100, le=10000, description="Batch size for data processing"
    )
    request_delay_ms: Optional[int] = Field(
        default=None,
        ge=0,
        le=5000,
        description="Delay between API requests in milliseconds",
    )

    @field_validator("cron_schedule")
    @classmethod
    def validate_cron_schedule(cls, v: Optional[str]) -> Optional[str]:
        """Validate cron schedule format."""
        if v is not None:
            if not v or not isinstance(v, str):
                raise ValueError("Schedule must be a valid string")

            parts = v.strip().split()
            if len(parts) != 5:
                raise ValueError(
                    "Cron schedule must have exactly 5 fields: minute hour day month day_of_week"
                )

            return v.strip()
        return v

    @field_validator("exchange")
    @classmethod
    def validate_exchange(cls, v: Optional[str]) -> Optional[str]:
        """Validate exchange name."""
        if v is not None:
            valid_exchanges = [e.value for e in Exchange]
            if v not in valid_exchanges:
                raise ValueError(
                    f"Invalid exchange: {v}. Valid exchanges: {valid_exchanges}"
                )
        return v

    @field_validator("repository_type")
    @classmethod
    def validate_repository_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate repository type."""
        if v is not None:
            valid_types = [t.value for t in RepositoryType]
            if v not in valid_types:
                raise ValueError(
                    f"Invalid repository type: {v}. Valid types: {valid_types}"
                )
        return v

    @field_validator("priority_symbols")
    @classmethod
    def validate_priority_symbols(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate priority symbols list."""
        if v is not None:
            if not v:
                raise ValueError("Priority symbols list cannot be empty if provided")

            for symbol in v:
                if not isinstance(symbol, str) or len(symbol) < 6:
                    raise ValueError(f"Invalid symbol format: {symbol}")

        return v

    @field_validator("priority_timeframes")
    @classmethod
    def validate_priority_timeframes(
        cls, v: Optional[List[str]]
    ) -> Optional[List[str]]:
        """Validate priority timeframes list."""
        if v is not None:
            if not v:
                raise ValueError("Priority timeframes list cannot be empty if provided")

            valid_timeframes = [tf.value for tf in TimeFrame]
            for timeframe in v:
                if timeframe not in valid_timeframes:
                    raise ValueError(
                        f"Invalid timeframe: {timeframe}. Valid: {valid_timeframes}"
                    )

        return v


class JobOperationResponse(BaseModel):
    """Generic response for job management operations."""

    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Operation result message")
    status: Optional[JobStatus] = Field(
        default=None, description="Current job status after operation"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Operation timestamp"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional operation details"
    )
    job_id: Optional[str] = Field(default=None, description="Related job identifier")


class DataCollectionJobRequest(BaseModel):
    """Request for data collection job execution."""

    symbols: List[str] = Field(..., description="List of symbols to collect")
    timeframes: List[str] = Field(..., description="List of timeframes to collect")
    start_date: Optional[str] = Field(
        default=None, description="Start date for data collection (ISO format)"
    )
    end_date: Optional[str] = Field(
        default=None, description="End date for data collection (ISO format)"
    )
    exchange: str = Field(
        default=Exchange.BINANCE.value, description="Exchange to collect from"
    )
    repository_type: str = Field(
        default=RepositoryType.MONGODB.value, description="Repository type for storage"
    )
    force_update: bool = Field(default=False, description="Force update existing data")
    batch_size: int = Field(
        default=1000, ge=100, le=10000, description="Batch size for processing"
    )

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: List[str]) -> List[str]:
        """Validate symbols list."""
        if not v:
            raise ValueError("Symbols list cannot be empty")

        for symbol in v:
            if not isinstance(symbol, str) or len(symbol) < 6:
                raise ValueError(f"Invalid symbol format: {symbol}")

        return v

    @field_validator("timeframes")
    @classmethod
    def validate_timeframes(cls, v: List[str]) -> List[str]:
        """Validate timeframes list."""
        if not v:
            raise ValueError("Timeframes list cannot be empty")

        valid_timeframes = [tf.value for tf in TimeFrame]
        for timeframe in v:
            if timeframe not in valid_timeframes:
                raise ValueError(
                    f"Invalid timeframe: {timeframe}. Valid: {valid_timeframes}"
                )

        return v


class DataCollectionJobResponse(BaseModel):
    """Response from data collection job execution."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job execution status")
    symbols_requested: List[str] = Field(..., description="Symbols that were requested")
    symbols_processed: List[str] = Field(
        default_factory=list, description="Symbols that were successfully processed"
    )
    symbols_failed: List[str] = Field(
        default_factory=list, description="Symbols that failed to process"
    )
    timeframes_processed: List[str] = Field(
        default_factory=list, description="Timeframes that were processed"
    )
    total_records_collected: int = Field(
        default=0, description="Total number of records collected"
    )
    execution_time_seconds: Optional[float] = Field(
        default=None, description="Job execution time in seconds"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of errors encountered during execution"
    )
    repository_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Information about the repository used"
    )


class HealthCheckResponse(BaseModel):
    """Health check response for the job service."""

    service: str = Field(default="market-data-job", description="Service name")
    status: str = Field(..., description="Overall health status")
    version: str = Field(default="1.0.0", description="Service version")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Health check timestamp"
    )
    components: Dict[str, str] = Field(
        default_factory=dict, description="Component health status"
    )
    dependencies: Dict[str, Any] = Field(
        default_factory=dict, description="Dependency health information"
    )
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Service metrics")
    uptime_seconds: Optional[int] = Field(
        default=None, description="Service uptime in seconds"
    )
