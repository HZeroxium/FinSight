# schemas/storage_schema.py

"""
Market Data Storage Router

RESTful endpoints for managing market data storage operations including:
- Object storage management (upload, download, list, delete)
- Dataset format conversion (CSV ↔ Parquet)
- Bulk data operations and archiving
- Storage statistics and monitoring
- User-friendly shortcut endpoints for common operations

Based on the storage service layer and cross-repository pipeline logic.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from ..schemas.enums import Exchange, TimeFrame, CryptoSymbol, RepositoryType


# Request/Response schemas with user-friendly defaults
class DatasetUploadRequest(BaseModel):
    """Request schema for dataset upload operations with smart defaults."""

    exchange: Optional[str] = Field(
        default=Exchange.BINANCE.value,
        description="Exchange name (defaults to binance)",
    )
    symbol: Optional[str] = Field(
        default=CryptoSymbol.BTCUSDT.value,
        description="Trading symbol (defaults to BTCUSDT)",
    )
    timeframe: Optional[str] = Field(
        default=TimeFrame.HOUR_1.value, description="Time interval (defaults to 1h)"
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date in ISO format (defaults to dataset's first date)",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date in ISO format (defaults to dataset's last date)",
    )
    source_format: Optional[str] = Field(
        default=RepositoryType.CSV.value,
        description="Source data format (csv or parquet, defaults to csv)",
    )
    target_format: Optional[str] = Field(
        default=RepositoryType.PARQUET.value,
        description="Target format for upload (csv or parquet, defaults to parquet)",
    )
    compress: Optional[bool] = Field(
        default=True, description="Whether to compress the dataset (defaults to True)"
    )
    include_metadata: Optional[bool] = Field(
        default=True, description="Whether to include metadata files (defaults to True)"
    )


class DatasetDownloadRequest(BaseModel):
    """Request schema for dataset download operations with smart defaults."""

    object_key: Optional[str] = Field(
        default=None,
        description="Object storage key for the dataset (auto-generated if not provided)",
    )
    symbol: Optional[str] = Field(
        default=CryptoSymbol.BTCUSDT.value,
        description="Trading symbol for auto-generating object key (defaults to BTCUSDT)",
    )
    timeframe: Optional[str] = Field(
        default=TimeFrame.HOUR_1.value,
        description="Timeframe for auto-generating object key (defaults to 1h)",
    )
    exchange: Optional[str] = Field(
        default=Exchange.BINANCE.value,
        description="Exchange for auto-generating object key (defaults to binance)",
    )
    extract_archive: Optional[bool] = Field(
        default=False,
        description="Whether to extract archive files (defaults to False)",
    )
    target_directory: Optional[str] = Field(
        default=None, description="Target directory for extraction"
    )


class FormatConversionRequest(BaseModel):
    """Request schema for data format conversion with smart defaults."""

    exchange: Optional[str] = Field(
        default=Exchange.BINANCE.value,
        description="Exchange name (defaults to binance)",
    )
    symbol: Optional[str] = Field(
        default=CryptoSymbol.BTCUSDT.value,
        description="Trading symbol (defaults to BTCUSDT)",
    )
    timeframe: Optional[str] = Field(
        default=TimeFrame.HOUR_1.value, description="Source timeframe (defaults to 1h)"
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date in ISO format (defaults to dataset's first date)",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date in ISO format (defaults to dataset's last date)",
    )
    source_format: Optional[str] = Field(
        default=RepositoryType.CSV.value,
        description="Source repository format (csv or parquet, defaults to csv)",
    )
    target_format: Optional[str] = Field(
        default=RepositoryType.PARQUET.value,
        description="Target repository format (csv or parquet, defaults to parquet)",
    )
    target_timeframes: Optional[List[str]] = Field(
        default=[TimeFrame.HOUR_4.value, TimeFrame.DAY_1.value],
        description="Target timeframes for conversion (optional)",
    )
    overwrite_existing: Optional[bool] = Field(
        default=False, description="Whether to overwrite existing data"
    )


class BulkOperationRequest(BaseModel):
    """Request schema for bulk storage operations."""

    operations: List[Dict[str, Any]] = Field(
        description="List of operations to perform"
    )
    max_concurrent: int = Field(
        default=5, ge=1, le=20, description="Maximum concurrent operations"
    )
    continue_on_error: bool = Field(
        default=True, description="Whether to continue on individual operation errors"
    )


class StorageStatsResponse(BaseModel):
    """Response schema for storage statistics."""

    total_objects: int
    total_size_bytes: int
    datasets_by_format: Dict[str, int]
    datasets_by_exchange: Dict[str, int]
    storage_health: Dict[str, Any]
    last_updated: str
