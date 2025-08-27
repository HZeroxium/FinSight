# schemas/admin_schemas.py

"""
Admin API schemas for request/response DTOs.

These schemas handle administrative operations including
data management, system statistics, and maintenance operations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .enums import Exchange, RepositoryType, Symbol, TimeFrame
from .ohlcv_schemas import OHLCVSchema, OHLCVStatsSchema


class DataEnsureRequest(BaseModel):
    """Request schema for ensuring data availability."""

    exchange: str = Field(
        default=Exchange.BINANCE.value, description="Exchange name (e.g., binance)"
    )
    symbol: str = Field(
        default=Symbol.BTCUSDT.value, description="Trading symbol (e.g., BTCUSDT)"
    )
    timeframe: str = Field(
        default=TimeFrame.HOUR_1.value, description="Timeframe (e.g., 1h, 1d)"
    )
    start_date: datetime = Field(
        default=datetime.now().strftime("%Y-%m-%d"),
        description="Start date for data range",
    )
    end_date: datetime = Field(
        default=datetime.now().strftime("%Y-%m-%d"),
        description="End date for data range",
    )
    force_refresh: bool = Field(
        default=False, description="Force refresh even if data exists"
    )

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat() + "Z"})


class DataEnsureResponse(BaseModel):
    """Response schema for data ensure operations."""

    success: bool = Field(..., description="Whether operation was successful")
    data_was_missing: bool = Field(..., description="Whether data was missing")
    records_fetched: int = Field(..., description="Number of records fetched")
    records_saved: int = Field(..., description="Number of records saved")
    data_statistics: Optional[OHLCVStatsSchema] = Field(
        default=None, description="Data statistics after operation"
    )
    operation_timestamp: datetime = Field(..., description="When operation completed")
    error_message: Optional[str] = Field(
        default=None, description="Error message if operation failed"
    )

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat() + "Z"})


class TimeframeConvertRequest(BaseModel):
    """Request schema for timeframe conversion."""

    exchange: str = Field(
        default=Exchange.BINANCE.value, description="Exchange name (e.g., binance)"
    )
    symbol: str = Field(
        default=Symbol.BTCUSDT.value, description="Trading symbol (e.g., BTCUSDT)"
    )
    source_timeframe: str = Field(
        default=TimeFrame.HOUR_1.value, description="Source timeframe (e.g., 1h)"
    )
    target_timeframe: str = Field(
        default=TimeFrame.DAY_1.value, description="Target timeframe (e.g., 1d)"
    )
    start_date: datetime = Field(
        default=datetime.now().strftime("%Y-%m-%d"),
        description="Start date for conversion",
    )
    end_date: datetime = Field(
        default=datetime.now().strftime("%Y-%m-%d"),
        description="End date for conversion",
    )
    save_converted: bool = Field(
        default=True, description="Whether to save converted data"
    )

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat() + "Z"})


class TimeframeConvertResponse(BaseModel):
    """Response schema for timeframe conversion."""

    success: bool = Field(..., description="Whether conversion was successful")
    source_records: int = Field(..., description="Number of source records")
    converted_records: int = Field(..., description="Number of converted records")
    saved_records: int = Field(..., description="Number of saved records")
    operation_timestamp: datetime = Field(..., description="When operation completed")
    converted_data: Optional[List[OHLCVSchema]] = Field(
        default=None, description="Converted data (if not saved)"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if conversion failed"
    )

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat() + "Z"})


class AdminStatsResponse(BaseModel):
    """Response schema for admin statistics."""

    total_records: int = Field(..., description="Total OHLCV records in system")
    unique_symbols: int = Field(..., description="Number of unique symbols")
    unique_exchanges: int = Field(..., description="Number of unique exchanges")
    available_timeframes: List[str] = Field(..., description="Available timeframes")
    symbols: List[str] = Field(..., description="List of available symbols")
    exchanges: List[str] = Field(..., description="List of available exchanges")
    storage_info: Dict[str, Any] = Field(..., description="Storage utilization info")
    uptime_seconds: int = Field(..., description="Server uptime in seconds")
    server_timestamp: datetime = Field(..., description="Current server timestamp")

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat() + "Z"})


class SystemHealthResponse(BaseModel):
    """Response schema for system health checks."""

    status: str = Field(..., description="Overall health status")
    repository_connected: bool = Field(..., description="Repository connectivity")
    data_fresh: bool = Field(..., description="Whether data is fresh")
    memory_usage_percent: float = Field(..., description="Memory usage percentage")
    disk_usage_percent: float = Field(..., description="Disk usage percentage")
    checks_timestamp: datetime = Field(..., description="When checks were performed")
    error_message: Optional[str] = Field(
        default=None, description="Error message if unhealthy"
    )

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat() + "Z"})


class CleanupRequest(BaseModel):
    """Request schema for data cleanup operations."""

    exchange: str = Field(default=Exchange.BINANCE.value, description="Exchange name")
    symbol: str = Field(default=Symbol.BTCUSDT.value, description="Trading symbol")
    timeframe: str = Field(default=TimeFrame.HOUR_1.value, description="Data timeframe")
    cutoff_date: datetime = Field(
        default=datetime.now().strftime("%Y-%m-%d"),
        description="Delete data before this date",
    )
    confirm: bool = Field(
        default=False, description="Confirmation flag for destructive operation"
    )

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat() + "Z"})


class CleanupResponse(BaseModel):
    """Response schema for cleanup operations."""

    success: bool = Field(..., description="Whether cleanup was successful")
    records_before: int = Field(..., description="Records count before cleanup")
    records_deleted: int = Field(..., description="Number of records deleted")
    cutoff_date: datetime = Field(..., description="Cutoff date used")
    operation_timestamp: datetime = Field(..., description="When operation completed")
    error_message: Optional[str] = Field(
        default=None, description="Error message if cleanup failed"
    )

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat() + "Z"})


class APIKeyValidationRequest(BaseModel):
    """Request schema for API key validation."""

    api_key: str = Field(..., description="API key to validate")


class APIKeyValidationResponse(BaseModel):
    """Response schema for API key validation."""

    valid: bool = Field(..., description="Whether API key is valid")
    expires_at: Optional[datetime] = Field(
        default=None, description="When API key expires"
    )
    permissions: List[str] = Field(
        default_factory=list, description="API key permissions"
    )

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat() + "Z"})


class QuickUploadResult(BaseModel):
    """Result of a single upload operation for a symbol and timeframe."""

    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe uploaded")
    success: bool = Field(..., description="Whether the upload succeeded")
    object_key: Optional[str] = Field(default=None, description="Uploaded object key")
    message: Optional[str] = Field(default=None, description="Status message")


class QuickSymbolPipelineResult(BaseModel):
    """Aggregated pipeline results for a single symbol."""

    symbol: str = Field(..., description="Trading symbol")
    collection_status: str = Field(..., description="Collection step status")
    collection_records: int = Field(default=0, description="Records collected in step")
    conversion_status: str = Field(..., description="Conversion step status")
    converted_timeframes: List[str] = Field(
        default_factory=list, description="Converted target timeframes"
    )
    upload_results: List[QuickUploadResult] = Field(
        default_factory=list, description="Per-timeframe upload results"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Errors encountered for this symbol"
    )


class QuickPipelineResponse(BaseModel):
    """Response schema for the quick collect → convert → upload pipeline."""

    exchange: str = Field(..., description="Exchange used for operations")
    symbols: List[str] = Field(..., description="Symbols processed")
    source_timeframe: str = Field(..., description="Source timeframe collected")
    target_timeframes: List[str] = Field(..., description="Target timeframes converted")
    source_format: str = Field(..., description="Source repository format")
    target_format: str = Field(..., description="Target repository format for upload")
    started_at: datetime = Field(..., description="Pipeline start time")
    finished_at: datetime = Field(..., description="Pipeline finish time")
    duration_seconds: float = Field(
        ..., description="Total pipeline duration in seconds"
    )
    results_by_symbol: List[QuickSymbolPipelineResult] = Field(
        default_factory=list, description="Detailed results per symbol"
    )
    success: bool = Field(..., description="Overall pipeline success status")
    message: str = Field(..., description="Summary message")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
