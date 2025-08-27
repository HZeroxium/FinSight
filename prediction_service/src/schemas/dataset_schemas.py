# schemas/dataset_schemas.py

"""
Dataset Management Schemas for AI Prediction Module

Pydantic schemas for dataset management operations including:
- Dataset discovery and listing
- Dataset availability checking
- Dataset download and caching
- Cache invalidation and management
- Dataset metadata and statistics
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .base_schemas import BaseResponse
from .enums import (CryptoSymbol, DataFormat, Exchange, StorageProviderType,
                    TimeFrame)

# ===== Dataset Discovery and Listing =====


class DatasetInfo(BaseModel):
    """Information about a dataset in storage or cache."""

    model_config = {"validate_assignment": True}

    # Dataset identification
    exchange: str = Field(..., description="Exchange name (e.g., 'binance')")
    symbol: str = Field(..., description="Trading symbol (e.g., 'BTCUSDT')")
    timeframe: str = Field(..., description="Data timeframe (e.g., '1h')")

    # Storage information
    object_key: Optional[str] = Field(None, description="Object storage key")
    format_type: Optional[str] = Field(None, description="Data format (csv, parquet)")
    date: Optional[str] = Field(None, description="Dataset date")
    filename: Optional[str] = Field(None, description="Dataset filename")

    # Metadata
    size_bytes: Optional[int] = Field(None, description="Dataset size in bytes")
    last_modified: Optional[datetime] = Field(
        None, description="Last modification time"
    )
    etag: Optional[str] = Field(None, description="ETag for versioning")

    # Availability status
    is_archived: bool = Field(
        False, description="Whether dataset is compressed/archived"
    )
    is_cached: bool = Field(False, description="Whether dataset is cached locally")
    cache_age_hours: Optional[float] = Field(None, description="Cache age in hours")

    # Data statistics
    record_count: Optional[int] = Field(
        None, description="Number of records in dataset"
    )
    date_range_start: Optional[datetime] = Field(None, description="Dataset start date")
    date_range_end: Optional[datetime] = Field(None, description="Dataset end date")


class DatasetListRequest(BaseModel):
    """Request schema for listing available datasets."""

    model_config = {"validate_assignment": True}

    # Filters
    exchange_filter: Optional[str] = Field(
        default=Exchange.BINANCE.value, description="Filter by exchange"
    )
    symbol_filter: Optional[str] = Field(
        default=CryptoSymbol.BTCUSDT.value, description="Filter by trading symbol"
    )
    timeframe_filter: Optional[str] = Field(
        default=TimeFrame.HOUR_1.value, description="Filter by timeframe"
    )
    format_filter: Optional[str] = Field(
        default=DataFormat.CSV.value, description="Filter by data format"
    )

    # Search options
    prefix: Optional[str] = Field(None, description="Object key prefix to search")
    include_cached: bool = Field(True, description="Include cached datasets")
    include_cloud: bool = Field(True, description="Include cloud datasets")

    # Pagination
    limit: int = Field(100, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Result offset for pagination")

    # Sorting
    sort_by: str = Field("last_modified", description="Sort field")
    sort_order: str = Field("desc", pattern="^(asc|desc)$", description="Sort order")


class DatasetListResponse(BaseResponse):
    """Response schema for dataset listing."""

    model_config = {"validate_assignment": True}

    datasets: List[DatasetInfo] = Field(..., description="List of available datasets")
    total_count: Optional[int] = Field(..., description="Total number of datasets")
    filtered_count: Optional[int] = Field(
        ..., description="Number of datasets after filtering"
    )

    # Summary statistics
    total_size_bytes: Optional[int] = Field(0, description="Total size of all datasets")
    unique_exchanges: List[str] = Field(
        default_factory=list, description="Available exchanges"
    )
    unique_symbols: List[str] = Field(
        default_factory=list, description="Available symbols"
    )
    unique_timeframes: List[str] = Field(
        default_factory=list, description="Available timeframes"
    )

    # Pagination info
    has_more: bool = Field(False, description="Whether more results are available")
    next_offset: Optional[int] = Field(None, description="Next offset for pagination")


# ===== Dataset Availability Checking =====


class DatasetAvailabilityRequest(BaseModel):
    """Request schema for checking dataset availability."""

    model_config = {"validate_assignment": True}

    symbol: str = Field(
        default=CryptoSymbol.BTCUSDT.value, description="Trading symbol to check"
    )
    timeframe: TimeFrame = Field(
        default=TimeFrame.HOUR_1.value, description="Data timeframe to check"
    )
    exchange: Optional[str] = Field(
        default=Exchange.BINANCE.value,
        description="Exchange to check (defaults to binance)",
    )

    # Check options
    check_cloud: bool = Field(True, description="Check cloud storage availability")
    check_cache: bool = Field(True, description="Check local cache availability")
    check_local: bool = Field(True, description="Check local file availability")


class DatasetAvailabilityResponse(BaseResponse):
    """Response schema for dataset availability check."""

    model_config = {"validate_assignment": True}

    # Availability status
    exists: bool = Field(..., description="Whether dataset exists")
    available_sources: List[str] = Field(
        default_factory=list, description="Available data sources"
    )

    # Source details
    cloud_available: bool = Field(False, description="Available in cloud storage")
    cache_available: bool = Field(False, description="Available in local cache")
    local_available: bool = Field(False, description="Available as local file")

    # Dataset information
    dataset_info: Optional[DatasetInfo] = Field(
        None, description="Dataset information if available"
    )

    # Cache information
    cache_age_hours: Optional[float] = Field(None, description="Cache age if cached")
    cache_expires_in_hours: Optional[float] = Field(
        None, description="Hours until cache expires"
    )

    # Recommendations
    recommended_action: Optional[str] = Field(
        None, description="Recommended action for user"
    )


# ===== Dataset Download and Caching =====


class DatasetDownloadRequest(BaseModel):
    """Request schema for downloading datasets."""

    model_config = {"validate_assignment": True}

    # Dataset identification
    symbol: str = Field(
        default=CryptoSymbol.BTCUSDT.value, description="Trading symbol"
    )
    timeframe: TimeFrame = Field(
        default=TimeFrame.HOUR_1.value, description="Data timeframe"
    )
    exchange: Optional[str] = Field(
        default=Exchange.BINANCE.value, description="Exchange name"
    )

    # Download options
    target_format: Optional[str] = Field(
        None, description="Target format (csv, parquet)"
    )
    force_download: bool = Field(False, description="Force download even if cached")
    update_cache: bool = Field(True, description="Update local cache after download")
    update_local: bool = Field(False, description="Update local storage after download")

    # Object key (optional, auto-generated if not provided)
    object_key: Optional[str] = Field(
        None, description="Specific object key to download"
    )

    # Cache options
    cache_ttl_hours: Optional[int] = Field(
        None, description="Custom cache TTL in hours"
    )


class DatasetDownloadResponse(BaseResponse):
    """Response schema for dataset download operations."""

    model_config = {"validate_assignment": True}

    # Download result
    download_id: str = Field(..., description="Unique download identifier")
    success: bool = Field(..., description="Whether download was successful")

    # Dataset information
    dataset_info: Optional[DatasetInfo] = Field(
        None, description="Downloaded dataset information"
    )

    # File information
    local_path: Optional[str] = Field(None, description="Local file path")
    file_size_bytes: Optional[int] = Field(None, description="Downloaded file size")

    # Cache information
    cached_path: Optional[str] = Field(None, description="Cache file path")
    cache_ttl_hours: Optional[int] = Field(None, description="Cache TTL in hours")

    # Performance metrics
    download_duration_seconds: Optional[float] = Field(
        None, description="Download duration"
    )
    download_speed_mbps: Optional[float] = Field(
        None, description="Download speed in MB/s"
    )

    # Status information
    status: str = Field(..., description="Download status")
    error_message: Optional[str] = Field(None, description="Error message if failed")


# ===== Cache Management =====


class CacheInfo(BaseModel):
    """Information about cached datasets."""

    model_config = {"validate_assignment": True}

    # Cache identification
    cache_key: str = Field(..., description="Cache key")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Data timeframe")

    # File information
    file_path: str = Field(..., description="Cache file path")
    file_size_bytes: int = Field(..., description="Cache file size")

    # Cache metadata
    created_at: datetime = Field(..., description="Cache creation time")
    last_accessed: datetime = Field(..., description="Last access time")
    access_count: int = Field(0, description="Number of times accessed")

    # TTL information
    ttl_hours: int = Field(..., description="Cache TTL in hours")
    expires_at: datetime = Field(..., description="Cache expiration time")
    is_expired: bool = Field(..., description="Whether cache is expired")

    # Source information
    source_object_key: Optional[str] = Field(None, description="Source object key")
    source_exchange: Optional[str] = Field(None, description="Source exchange")


class CacheListRequest(BaseModel):
    """Request schema for listing cached datasets."""

    model_config = {"validate_assignment": True}

    # Filters
    symbol_filter: Optional[str] = Field(
        default=CryptoSymbol.BTCUSDT.value, description="Filter by symbol"
    )
    timeframe_filter: Optional[str] = Field(
        default=TimeFrame.HOUR_1.value, description="Filter by timeframe"
    )
    expired_only: bool = Field(False, description="Show only expired caches")
    valid_only: bool = Field(False, description="Show only valid caches")

    # Pagination
    limit: int = Field(100, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Result offset for pagination")


class CacheListResponse(BaseResponse):
    """Response schema for cache listing."""

    model_config = {"validate_assignment": True}

    cached_datasets: List[CacheInfo] = Field(..., description="List of cached datasets")
    total_count: Optional[int] = Field(
        ..., description="Total number of cached datasets"
    )

    # Cache statistics
    total_cache_size_bytes: Optional[int] = Field(0, description="Total cache size")
    expired_count: Optional[int] = Field(0, description="Number of expired caches")
    valid_count: Optional[int] = Field(0, description="Number of valid caches")

    # Storage information
    cache_directory: str = Field(..., description="Cache directory path")
    available_space_bytes: Optional[int] = Field(
        None, description="Available disk space"
    )


class CacheInvalidateRequest(BaseModel):
    """Request schema for invalidating caches."""

    model_config = {"validate_assignment": True}

    # Invalidation options
    symbol: Optional[str] = Field(
        default=CryptoSymbol.BTCUSDT.value,
        description="Invalidate caches for specific symbol",
    )
    timeframe: Optional[TimeFrame] = Field(
        default=TimeFrame.HOUR_1.value,
        description="Invalidate caches for specific timeframe",
    )
    expired_only: bool = Field(False, description="Invalidate only expired caches")
    all_caches: bool = Field(False, description="Invalidate all caches")

    # Confirmation
    confirm_invalidation: bool = Field(
        False, description="Confirmation flag for bulk operations"
    )


class CacheInvalidateResponse(BaseResponse):
    """Response schema for cache invalidation."""

    model_config = {"validate_assignment": True}

    # Invalidation results
    invalidated_count: int = Field(..., description="Number of caches invalidated")
    freed_space_bytes: int = Field(..., description="Freed disk space in bytes")

    # Details
    invalidated_caches: List[str] = Field(
        default_factory=list, description="List of invalidated cache keys"
    )
    failed_invalidations: List[str] = Field(
        default_factory=list, description="List of failed invalidations"
    )

    # Status
    status: str = Field(..., description="Invalidation status")


# ===== Dataset Statistics and Health =====


class DatasetStatistics(BaseModel):
    """Comprehensive dataset statistics."""

    model_config = {"validate_assignment": True}

    # Storage statistics
    total_datasets: int = Field(..., description="Total number of datasets")
    total_size_bytes: int = Field(..., description="Total size of all datasets")
    total_records: int = Field(..., description="Total number of records")

    # Format distribution
    datasets_by_format: Dict[str, int] = Field(
        default_factory=dict, description="Dataset count by format"
    )
    size_by_format: Dict[str, int] = Field(
        default_factory=dict, description="Size by format"
    )

    # Exchange distribution
    datasets_by_exchange: Dict[str, int] = Field(
        default_factory=dict, description="Dataset count by exchange"
    )
    size_by_exchange: Dict[str, int] = Field(
        default_factory=dict, description="Size by exchange"
    )

    # Symbol distribution
    datasets_by_symbol: Dict[str, int] = Field(
        default_factory=dict, description="Dataset count by symbol"
    )
    size_by_symbol: Dict[str, int] = Field(
        default_factory=dict, description="Size by symbol"
    )

    # Timeframe distribution
    datasets_by_timeframe: Dict[str, int] = Field(
        default_factory=dict, description="Dataset count by timeframe"
    )
    size_by_timeframe: Dict[str, int] = Field(
        default_factory=dict, description="Size by timeframe"
    )

    # Cache statistics
    cached_datasets: int = Field(0, description="Number of cached datasets")
    cache_size_bytes: int = Field(0, description="Total cache size")
    cache_hit_rate: Optional[float] = Field(
        None, description="Cache hit rate percentage"
    )

    # Health metrics
    oldest_dataset_age_days: Optional[float] = Field(
        None, description="Age of oldest dataset"
    )
    newest_dataset_age_hours: Optional[float] = Field(
        None, description="Age of newest dataset"
    )
    data_freshness_score: Optional[float] = Field(
        None, description="Data freshness score (0-100)"
    )

    # Timestamps
    last_updated: datetime = Field(..., description="Last statistics update time")
    collection_duration_seconds: Optional[float] = Field(
        None, description="Statistics collection duration"
    )


class DatasetHealthCheck(BaseModel):
    """Dataset system health information."""

    model_config = {"validate_assignment": True}

    # Overall health
    is_healthy: bool = Field(..., description="Overall system health status")
    health_score: float = Field(
        ..., ge=0.0, le=100.0, description="Health score (0-100)"
    )

    # Component health
    cloud_storage_healthy: bool = Field(..., description="Cloud storage health")
    local_cache_healthy: bool = Field(..., description="Local cache health")
    local_storage_healthy: bool = Field(..., description="Local storage health")

    # Performance metrics
    average_download_time_seconds: Optional[float] = Field(
        None, description="Average download time"
    )
    cache_hit_rate: Optional[float] = Field(None, description="Cache hit rate")
    storage_response_time_ms: Optional[float] = Field(
        None, description="Storage response time"
    )

    # Error tracking
    errors_last_hour: int = Field(0, description="Errors in last hour")
    last_error_message: Optional[str] = Field(None, description="Last error message")
    error_rate_percentage: Optional[float] = Field(
        None, description="Error rate percentage"
    )

    # Resource usage
    disk_usage_percentage: Optional[float] = Field(
        None, description="Disk usage percentage"
    )
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")

    # Timestamps
    last_health_check: datetime = Field(..., description="Last health check time")
    next_health_check: datetime = Field(..., description="Next scheduled health check")


# ===== Bulk Operations =====


class BulkDatasetOperation(BaseModel):
    """Schema for bulk dataset operations."""

    model_config = {"validate_assignment": True}

    # Operation type
    operation_type: str = Field(
        ..., description="Type of operation (download, invalidate, check)"
    )

    # Target datasets
    symbols: List[str] = Field(..., description="List of symbols to operate on")
    timeframes: List[TimeFrame] = Field(
        ..., description="List of timeframes to operate on"
    )
    exchanges: Optional[List[str]] = Field(
        None, description="List of exchanges (defaults to binance)"
    )

    # Operation options
    force_operation: bool = Field(
        False, description="Force operation even if not needed"
    )
    parallel_execution: bool = Field(True, description="Execute operations in parallel")
    max_concurrent: int = Field(
        5, ge=1, le=20, description="Maximum concurrent operations"
    )

    # Notification
    notify_on_completion: bool = Field(
        False, description="Send notification on completion"
    )


class BulkOperationResponse(BaseResponse):
    """Response schema for bulk operations."""

    model_config = {"validate_assignment": True}

    # Operation results
    operation_id: str = Field(..., description="Unique operation identifier")
    total_operations: int = Field(..., description="Total number of operations")
    successful_operations: int = Field(
        ..., description="Number of successful operations"
    )
    failed_operations: int = Field(..., description="Number of failed operations")

    # Performance metrics
    total_duration_seconds: float = Field(..., description="Total operation duration")
    average_duration_per_operation: float = Field(
        ..., description="Average duration per operation"
    )

    # Detailed results
    operation_results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detailed operation results"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Operation errors"
    )

    # Status
    status: str = Field(..., description="Overall operation status")
    completion_percentage: float = Field(
        ..., ge=0.0, le=100.0, description="Completion percentage"
    )
