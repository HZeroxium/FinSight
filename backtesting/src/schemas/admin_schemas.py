# schemas/admin_schemas.py

"""
Admin API schemas for request/response DTOs.

These schemas handle administrative operations including
data management, system statistics, and maintenance operations.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from .ohlcv_schemas import OHLCVSchema, OHLCVStatsSchema


class DataEnsureRequest(BaseModel):
    """Request schema for ensuring data availability."""

    exchange: str = Field(..., description="Exchange name (e.g., binance)")
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    timeframe: str = Field(..., description="Timeframe (e.g., 1h, 1d)")
    start_date: datetime = Field(..., description="Start date for data range")
    end_date: datetime = Field(..., description="End date for data range")
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

    exchange: str = Field(..., description="Exchange name (e.g., binance)")
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    source_timeframe: str = Field(..., description="Source timeframe (e.g., 1h)")
    target_timeframe: str = Field(..., description="Target timeframe (e.g., 1d)")
    start_date: datetime = Field(..., description="Start date for conversion")
    end_date: datetime = Field(..., description="End date for conversion")
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

    exchange: str = Field(..., description="Exchange name")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Data timeframe")
    cutoff_date: datetime = Field(..., description="Delete data before this date")
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
