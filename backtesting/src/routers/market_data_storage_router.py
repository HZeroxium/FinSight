# routers/market_data_storage_router.py

"""
Market Data Storage Router

RESTful endpoints for managing market data storage operations including:
- Object storage management (upload, download, list, delete)
- Dataset format conversion (CSV ↔ Parquet)
- Bulk data operations and archiving
- Storage statistics and monitoring

Based on the storage service layer and cross-repository pipeline logic.
"""

from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import tempfile
from pathlib import Path
import asyncio
import json
from datetime import datetime

from ..services.market_data_storage_service import MarketDataStorageService
from ..misc.timeframe_load_convert_save import CrossRepositoryTimeFramePipeline
from ..schemas.ohlcv_schemas import OHLCVQuerySchema
from ..schemas.enums import Exchange, TimeFrame, RepositoryType
from ..utils.dependencies import (
    get_market_data_service,
    get_dependency_manager,
    require_admin_access,
)
from common.logger import LoggerFactory, LoggerType, LogLevel

# Initialize logger
logger = LoggerFactory.get_logger(
    name="storage-router",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    file_level=LogLevel.DEBUG,
    log_file="logs/storage_router.log",
)

# Create router
router = APIRouter(prefix="/storage", tags=["market-data-storage"])


# Request/Response schemas
class DatasetUploadRequest(BaseModel):
    """Request schema for dataset upload operations."""

    exchange: str = Field(default=Exchange.BINANCE.value, description="Exchange name")
    symbol: str = Field(description="Trading symbol (e.g., BTCUSDT)")
    timeframe: str = Field(default=TimeFrame.HOUR_1.value, description="Time interval")
    start_date: str = Field(description="Start date in ISO format")
    end_date: str = Field(description="End date in ISO format")
    source_format: str = Field(
        default="csv", description="Source data format (csv or parquet)"
    )
    target_format: str = Field(
        default="parquet", description="Target format for upload (csv or parquet)"
    )
    compress: bool = Field(default=True, description="Whether to compress the dataset")
    include_metadata: bool = Field(
        default=True, description="Whether to include metadata files"
    )


class DatasetDownloadRequest(BaseModel):
    """Request schema for dataset download operations."""

    object_key: str = Field(description="Object storage key for the dataset")
    extract_archive: bool = Field(
        default=False, description="Whether to extract archive files"
    )
    target_directory: Optional[str] = Field(
        default=None, description="Target directory for extraction"
    )


class FormatConversionRequest(BaseModel):
    """Request schema for data format conversion."""

    exchange: str = Field(default=Exchange.BINANCE.value, description="Exchange name")
    symbol: str = Field(description="Trading symbol")
    timeframe: str = Field(description="Source timeframe")
    start_date: str = Field(description="Start date in ISO format")
    end_date: str = Field(description="End date in ISO format")
    source_format: str = Field(description="Source repository format (csv or parquet)")
    target_format: str = Field(description="Target repository format (csv or parquet)")
    target_timeframes: Optional[List[str]] = Field(
        default=None, description="Target timeframes for conversion (optional)"
    )
    overwrite_existing: bool = Field(
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


# Dependency functions
def get_storage_service() -> MarketDataStorageService:
    """Get market data storage service with dependencies."""
    try:
        # This is a simplified version - in production you'd want proper DI
        from ..utils.storage_client import StorageClient
        from ..adapters.csv_market_data_repository import CSVMarketDataRepository
        from ..adapters.parquet_market_data_repository import (
            ParquetMarketDataRepository,
        )

        storage_client = StorageClient()
        csv_repo = CSVMarketDataRepository()
        parquet_repo = ParquetMarketDataRepository()

        return MarketDataStorageService(
            storage_client=storage_client,
            csv_repository=csv_repo,
            parquet_repository=parquet_repo,
        )
    except Exception as e:
        logger.error(f"Failed to create storage service: {e}")
        raise HTTPException(
            status_code=500, detail=f"Storage service initialization failed: {str(e)}"
        )


def get_conversion_pipeline() -> CrossRepositoryTimeFramePipeline:
    """Get timeframe conversion pipeline."""
    try:
        from ..adapters.csv_market_data_repository import CSVMarketDataRepository
        from ..adapters.parquet_market_data_repository import (
            ParquetMarketDataRepository,
        )

        csv_repo = CSVMarketDataRepository()
        parquet_repo = ParquetMarketDataRepository()

        return CrossRepositoryTimeFramePipeline(
            source_repository=csv_repo, target_repository=parquet_repo
        )
    except Exception as e:
        logger.error(f"Failed to create conversion pipeline: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Conversion pipeline initialization failed: {str(e)}",
        )


# Storage management endpoints


@router.post("/upload")
async def upload_dataset(
    request: DatasetUploadRequest,
    storage_service: MarketDataStorageService = Depends(get_storage_service),
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Upload a dataset to object storage.

    Supports format conversion and compression during upload.
    """
    try:
        logger.info(
            f"Starting dataset upload: {request.exchange}/{request.symbol}/{request.timeframe}"
        )

        result = await storage_service.upload_dataset(
            exchange=request.exchange,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            source_format=request.source_format,
            target_format=request.target_format,
            compress=request.compress,
            include_metadata=request.include_metadata,
        )

        if result["success"]:
            logger.info(f"Dataset upload completed: {result['message']}")
        else:
            logger.warning(f"Dataset upload failed: {result['message']}")

        return result

    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset upload failed: {str(e)}")


@router.post("/download")
async def download_dataset(
    request: DatasetDownloadRequest,
    storage_service: MarketDataStorageService = Depends(get_storage_service),
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Download a dataset from object storage.

    Supports automatic archive extraction.
    """
    try:
        logger.info(f"Starting dataset download: {request.object_key}")

        result = await storage_service.download_dataset(
            object_key=request.object_key,
            extract_archive=request.extract_archive,
            target_directory=request.target_directory,
        )

        if result["success"]:
            logger.info(f"Dataset download completed: {result['message']}")
        else:
            logger.warning(f"Dataset download failed: {result['message']}")

        return result

    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise HTTPException(
            status_code=500, detail=f"Dataset download failed: {str(e)}"
        )


@router.get("/list")
async def list_datasets(
    prefix: Optional[str] = Query(None, description="Object key prefix for filtering"),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of objects to return"
    ),
    storage_service: MarketDataStorageService = Depends(get_storage_service),
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    List available datasets in object storage.
    """
    try:
        logger.info(f"Listing datasets with prefix: {prefix}")

        result = await storage_service.list_datasets(
            prefix=prefix,
            limit=limit,
        )

        logger.info(f"Found {len(result.get('datasets', []))} datasets")
        return result

    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset listing failed: {str(e)}")


@router.delete("/dataset/{object_key:path}")
async def delete_dataset(
    object_key: str,
    storage_service: MarketDataStorageService = Depends(get_storage_service),
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Delete a dataset from object storage.
    """
    try:
        logger.info(f"Deleting dataset: {object_key}")

        result = await storage_service.delete_dataset(object_key)

        if result["success"]:
            logger.info(f"Dataset deleted successfully: {object_key}")
        else:
            logger.warning(f"Dataset deletion failed: {result['message']}")

        return result

    except Exception as e:
        logger.error(f"Error deleting dataset: {e}")
        raise HTTPException(
            status_code=500, detail=f"Dataset deletion failed: {str(e)}"
        )


# Data conversion endpoints


@router.post("/convert")
async def convert_dataset_format(
    request: FormatConversionRequest,
    storage_service: MarketDataStorageService = Depends(get_storage_service),
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Convert dataset between formats (CSV ↔ Parquet).

    Based on timeframe_load_convert_save.py logic.
    """
    try:
        logger.info(
            f"Starting format conversion: {request.source_format} -> {request.target_format} "
            f"for {request.exchange}/{request.symbol}/{request.timeframe}"
        )

        result = await storage_service.convert_dataset_format(
            exchange=request.exchange,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            source_format=request.source_format,
            target_format=request.target_format,
            target_timeframes=request.target_timeframes,
            overwrite_existing=request.overwrite_existing,
        )

        if result["success"]:
            logger.info(f"Format conversion completed: {result['message']}")
        else:
            logger.warning(f"Format conversion failed: {result['message']}")

        return result

    except Exception as e:
        logger.error(f"Error converting dataset format: {e}")
        raise HTTPException(
            status_code=500, detail=f"Format conversion failed: {str(e)}"
        )


@router.post("/convert/timeframes")
async def convert_timeframes(
    exchange: str = Query(Exchange.BINANCE.value, description="Exchange name"),
    symbol: str = Query(..., description="Trading symbol"),
    source_timeframe: str = Query(
        TimeFrame.HOUR_1.value, description="Source timeframe"
    ),
    target_timeframes: List[str] = Query(
        default=[TimeFrame.HOUR_4.value, TimeFrame.DAY_1.value],
        description="Target timeframes for conversion",
    ),
    start_date: str = Query(..., description="Start date in ISO format"),
    end_date: str = Query(..., description="End date in ISO format"),
    source_format: str = Query("csv", description="Source repository format"),
    target_format: str = Query("parquet", description="Target repository format"),
    overwrite_existing: bool = Query(False, description="Overwrite existing data"),
    pipeline: CrossRepositoryTimeFramePipeline = Depends(get_conversion_pipeline),
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Convert timeframes using the cross-repository pipeline.

    References timeframe_load_convert_save.py logic for data conversion.
    """
    try:
        logger.info(
            f"Starting timeframe conversion: {source_timeframe} -> {target_timeframes} "
            f"for {exchange}/{symbol}"
        )

        # Configure pipeline repositories based on formats
        if source_format != target_format:
            # Cross-repository conversion
            if source_format == "csv" and target_format == "parquet":
                from ..adapters.csv_market_data_repository import (
                    CSVMarketDataRepository,
                )
                from ..adapters.parquet_market_data_repository import (
                    ParquetMarketDataRepository,
                )

                pipeline.source_repository = CSVMarketDataRepository()
                pipeline.target_repository = ParquetMarketDataRepository()
            elif source_format == "parquet" and target_format == "csv":
                from ..adapters.parquet_market_data_repository import (
                    ParquetMarketDataRepository,
                )
                from ..adapters.csv_market_data_repository import (
                    CSVMarketDataRepository,
                )

                pipeline.source_repository = ParquetMarketDataRepository()
                pipeline.target_repository = CSVMarketDataRepository()

        # Configure target timeframes
        pipeline.target_timeframes = target_timeframes

        # Run the conversion pipeline
        result = await pipeline.run_cross_repository_pipeline(
            symbols=[symbol],
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
            overwrite_existing=overwrite_existing,
        )

        success = result.get("success", False)
        if success:
            logger.info(f"Timeframe conversion completed for {symbol}")
        else:
            logger.warning(
                f"Timeframe conversion failed for {symbol}: {result.get('message', '')}"
            )

        return {
            "success": success,
            "message": result.get("message", "Timeframe conversion completed"),
            "symbol": symbol,
            "source_timeframe": source_timeframe,
            "target_timeframes": target_timeframes,
            "statistics": result.get("statistics", {}),
            "converted_records": result.get("total_converted", 0),
        }

    except Exception as e:
        logger.error(f"Error converting timeframes: {e}")
        raise HTTPException(
            status_code=500, detail=f"Timeframe conversion failed: {str(e)}"
        )


# Bulk operations


@router.post("/bulk")
async def bulk_operations(
    request: BulkOperationRequest,
    storage_service: MarketDataStorageService = Depends(get_storage_service),
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Perform bulk storage operations.

    Supports concurrent execution with error handling.
    """
    try:
        logger.info(f"Starting bulk operations: {len(request.operations)} operations")

        result = await storage_service.bulk_upload_datasets(
            operations=request.operations,
            max_concurrent=request.max_concurrent,
            continue_on_error=request.continue_on_error,
        )

        logger.info(f"Bulk operations completed: {result.get('message', '')}")
        return result

    except Exception as e:
        logger.error(f"Error in bulk operations: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk operations failed: {str(e)}")


# Storage statistics and monitoring


@router.get("/stats", response_model=StorageStatsResponse)
async def get_storage_statistics(
    storage_service: MarketDataStorageService = Depends(get_storage_service),
    _: bool = Depends(require_admin_access),
) -> StorageStatsResponse:
    """
    Get storage statistics and health information.
    """
    try:
        logger.info("Retrieving storage statistics")

        stats = await storage_service.get_storage_statistics()

        return StorageStatsResponse(
            total_objects=stats.get("total_objects", 0),
            total_size_bytes=stats.get("total_size_bytes", 0),
            datasets_by_format=stats.get("datasets_by_format", {}),
            datasets_by_exchange=stats.get("datasets_by_exchange", {}),
            storage_health=stats.get("storage_health", {}),
            last_updated=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Error retrieving storage statistics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Storage statistics retrieval failed: {str(e)}"
        )


@router.get("/health")
async def storage_health_check(
    storage_service: MarketDataStorageService = Depends(get_storage_service),
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Check storage service health and connectivity.
    """
    try:
        # Test storage client connectivity
        storage_info = await storage_service.storage_client.get_storage_info()

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "storage_client": "connected",
            "storage_info": storage_info,
            "service": "market-data-storage",
        }

    except Exception as e:
        logger.error(f"Storage health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "storage_client": "disconnected",
            "error": str(e),
            "service": "market-data-storage",
        }


# Service information


@router.get("/")
async def storage_service_info() -> Dict[str, Any]:
    """
    Get storage service information and available endpoints.
    """
    return {
        "service": "Market Data Storage Service",
        "version": "1.0.0",
        "description": "Object storage management and data conversion for market data",
        "endpoints": {
            "upload": "POST /storage/upload - Upload dataset to object storage",
            "download": "POST /storage/download - Download dataset from object storage",
            "list": "GET /storage/list - List available datasets",
            "delete": "DELETE /storage/dataset/{object_key} - Delete dataset",
            "convert": "POST /storage/convert - Convert dataset format",
            "convert_timeframes": "POST /storage/convert/timeframes - Convert timeframes",
            "bulk": "POST /storage/bulk - Bulk operations",
            "stats": "GET /storage/stats - Storage statistics",
            "health": "GET /storage/health - Health check",
        },
        "supported_formats": ["csv", "parquet"],
        "supported_operations": [
            "upload",
            "download",
            "format_conversion",
            "timeframe_conversion",
            "bulk_operations",
            "compression",
        ],
        "features": [
            "S3-compatible object storage",
            "Cross-repository format conversion",
            "Timeframe aggregation",
            "Compression and archiving",
            "Bulk operations",
            "Storage monitoring",
        ],
    }
