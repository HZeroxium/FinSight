# routers/dataset_management.py

"""
Dataset Management Router for AI Prediction Module

RESTful endpoints for comprehensive dataset management including:
- Dataset discovery and listing
- Dataset availability checking
- Dataset download and caching
- Cache management and invalidation
- Dataset statistics and health monitoring
"""

import uuid
from typing import Optional

from common.logger.logger_factory import LoggerFactory, LoggerType
from fastapi import APIRouter, Depends, HTTPException, Path, Query

from ..schemas.dataset_schemas import (BulkDatasetOperation,
                                       BulkOperationResponse,
                                       CacheInvalidateRequest,
                                       CacheInvalidateResponse,
                                       CacheListRequest, CacheListResponse,
                                       DatasetAvailabilityRequest,
                                       DatasetAvailabilityResponse,
                                       DatasetDownloadRequest,
                                       DatasetDownloadResponse,
                                       DatasetHealthCheck, DatasetListRequest,
                                       DatasetListResponse, DatasetStatistics)
from ..schemas.enums import CryptoSymbol, DataFormat, Exchange, TimeFrame
from ..services.dataset_management_service import DatasetManagementService
from ..utils.dependencies import get_storage_client_dependency

# Initialize router
router = APIRouter(prefix="/datasets", tags=["Dataset Management"])

# Initialize logger
logger = LoggerFactory.get_logger(
    name="dataset-management-router",
    logger_type=LoggerType.STANDARD,
    log_file="logs/dataset_management_router.log",
)


async def get_dataset_management_service() -> DatasetManagementService:
    """Dependency function to get dataset management service."""
    from ..utils.dependencies import get_dataset_management_service_dependency

    return await get_dataset_management_service_dependency()


# ===== Dataset Discovery and Listing =====


@router.get("/", response_model=DatasetListResponse)
async def list_datasets(
    exchange_filter: Optional[str] = Query(
        default=Exchange.BINANCE.value, description="Filter by exchange"
    ),
    symbol_filter: Optional[str] = Query(
        default=CryptoSymbol.BTCUSDT.value, description="Filter by trading symbol"
    ),
    timeframe_filter: Optional[str] = Query(
        default=TimeFrame.HOUR_1.value, description="Filter by timeframe"
    ),
    format_filter: Optional[str] = Query(
        default=DataFormat.CSV.value, description="Filter by data format"
    ),
    prefix: Optional[str] = Query(None, description="Object key prefix to search"),
    include_cached: bool = Query(True, description="Include cached datasets"),
    include_cloud: bool = Query(True, description="Include cloud datasets"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Result offset for pagination"),
    sort_by: str = Query("last_modified", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    service: DatasetManagementService = Depends(get_dataset_management_service),
) -> DatasetListResponse:
    """
    List available datasets with filtering, pagination, and sorting.

    Returns comprehensive information about datasets available in cloud storage,
    local cache, and local files with advanced filtering capabilities.
    """
    try:
        logger.info(
            f"Listing datasets with filters: {exchange_filter}, {symbol_filter}, {timeframe_filter}"
        )

        # Get datasets
        datasets = await service.list_datasets(
            exchange_filter=exchange_filter,
            symbol_filter=symbol_filter,
            timeframe_filter=timeframe_filter,
            format_filter=format_filter,
            prefix=prefix,
            include_cached=include_cached,
            include_cloud=include_cloud,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        # Calculate summary statistics
        total_count = len(datasets)
        total_size_bytes = sum(ds.size_bytes or 0 for ds in datasets)

        # Extract unique values
        unique_exchanges = list(set(ds.exchange for ds in datasets if ds.exchange))
        unique_symbols = list(set(ds.symbol for ds in datasets if ds.symbol))
        unique_timeframes = list(set(ds.timeframe for ds in datasets if ds.timeframe))

        # Determine if more results are available
        has_more = total_count == limit
        next_offset = offset + limit if has_more else None

        response = DatasetListResponse(
            success=True,
            message=f"Found {total_count} datasets",
            datasets=datasets,
            total_count=total_count,
            filtered_count=total_count,
            total_size_bytes=total_size_bytes,
            unique_exchanges=unique_exchanges,
            unique_symbols=unique_symbols,
            unique_timeframes=unique_timeframes,
            has_more=has_more,
            next_offset=next_offset,
        )

        logger.info(f"Dataset listing completed: {total_count} datasets found")
        return response

    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list datasets: {str(e)}"
        )


@router.post("/list", response_model=DatasetListResponse)
async def list_datasets_post(
    request: DatasetListRequest,
    service: DatasetManagementService = Depends(get_dataset_management_service),
) -> DatasetListResponse:
    """
    List available datasets using POST request with complex filtering.

    This endpoint allows for more complex filtering and search criteria
    that may be too long for GET request query parameters.
    """
    try:
        logger.info(f"POST listing datasets with request: {request}")

        # Get datasets
        datasets = await service.list_datasets(
            exchange_filter=request.exchange_filter,
            symbol_filter=request.symbol_filter,
            timeframe_filter=request.timeframe_filter,
            format_filter=request.format_filter,
            prefix=request.prefix,
            include_cached=request.include_cached,
            include_cloud=request.include_cloud,
            limit=request.limit,
            offset=request.offset,
            sort_by=request.sort_by,
            sort_order=request.sort_order,
        )

        # Calculate summary statistics
        total_count = len(datasets)
        total_size_bytes = sum(ds.size_bytes or 0 for ds in datasets)

        # Extract unique values
        unique_exchanges = list(set(ds.exchange for ds in datasets if ds.exchange))
        unique_symbols = list(set(ds.symbol for ds in datasets if ds.symbol))
        unique_timeframes = list(set(ds.timeframe for ds in datasets if ds.timeframe))

        # Determine if more results are available
        has_more = total_count == request.limit
        next_offset = request.offset + request.limit if has_more else None

        response = DatasetListResponse(
            success=True,
            message=f"Found {total_count} datasets",
            datasets=datasets,
            total_count=total_count,
            filtered_count=total_count,
            total_size_bytes=total_size_bytes,
            unique_exchanges=unique_exchanges,
            unique_symbols=unique_symbols,
            unique_timeframes=unique_timeframes,
            has_more=has_more,
            next_offset=next_offset,
        )

        logger.info(f"POST dataset listing completed: {total_count} datasets found")
        return response

    except Exception as e:
        logger.error(f"Failed to list datasets via POST: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list datasets: {str(e)}"
        )


# ===== Dataset Availability Checking =====


@router.get(
    "/availability/{symbol}/{timeframe}", response_model=DatasetAvailabilityResponse
)
async def check_dataset_availability(
    symbol: str = Path(..., description="Trading symbol (e.g., BTCUSDT)"),
    timeframe: TimeFrame = Path(..., description="Data timeframe"),
    exchange: str = Query(
        default=Exchange.BINANCE.value, description="Exchange to check"
    ),
    check_cloud: bool = Query(True, description="Check cloud storage availability"),
    check_cache: bool = Query(True, description="Check local cache availability"),
    check_local: bool = Query(True, description="Check local file availability"),
    service: DatasetManagementService = Depends(get_dataset_management_service),
) -> DatasetAvailabilityResponse:
    """
    Check dataset availability across all sources.

    Returns comprehensive information about where a dataset is available
    (cloud storage, local cache, local files) with recommendations.
    """
    try:
        logger.info(f"Checking availability for {symbol}_{timeframe.value}")

        response = await service.check_dataset_availability(
            symbol=symbol,
            timeframe=timeframe,
            exchange=exchange,
            check_cloud=check_cloud,
            check_cache=check_cache,
            check_local=check_local,
        )

        logger.info(
            f"Availability check completed for {symbol}_{timeframe.value}: {response.exists}"
        )
        return response

    except Exception as e:
        logger.error(f"Failed to check dataset availability: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to check availability: {str(e)}"
        )


@router.post("/availability", response_model=DatasetAvailabilityResponse)
async def check_dataset_availability_post(
    request: DatasetAvailabilityRequest,
    service: DatasetManagementService = Depends(get_dataset_management_service),
) -> DatasetAvailabilityResponse:
    """
    Check dataset availability using POST request.

    This endpoint allows for more complex availability checking criteria.
    """
    try:
        logger.info(
            f"POST availability check for {request.symbol}_{request.timeframe.value}"
        )

        response = await service.check_dataset_availability(
            symbol=request.symbol,
            timeframe=request.timeframe,
            exchange=request.exchange,
            check_cloud=request.check_cloud,
            check_cache=request.check_cache,
            check_local=request.check_local,
        )

        logger.info(
            f"POST availability check completed for {request.symbol}_{request.timeframe.value}"
        )
        return response

    except Exception as e:
        logger.error(f"Failed to check dataset availability via POST: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to check availability: {str(e)}"
        )


# ===== Dataset Download and Caching =====


@router.post("/download", response_model=DatasetDownloadResponse)
async def download_dataset(
    request: DatasetDownloadRequest,
    service: DatasetManagementService = Depends(get_dataset_management_service),
) -> DatasetDownloadResponse:
    """
    Download dataset with intelligent caching and fallback.

    Downloads a dataset from cloud storage with automatic caching,
    or uses cached data if available and fresh.
    """
    try:
        logger.info(f"Downloading dataset: {request.symbol}_{request.timeframe.value}")

        response = await service.download_dataset(
            symbol=request.symbol,
            timeframe=request.timeframe,
            exchange=request.exchange,
            target_format=request.target_format,
            force_download=request.force_download,
            update_cache=request.update_cache,
            update_local=request.update_local,
            object_key=request.object_key,
            cache_ttl_hours=request.cache_ttl_hours,
        )

        logger.info(
            f"Dataset download completed: {request.symbol}_{request.timeframe.value}, success: {response.success}"
        )
        return response

    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to download dataset: {str(e)}"
        )


# ===== Cache Management =====


@router.get("/cache", response_model=CacheListResponse)
async def list_cached_datasets(
    symbol_filter: Optional[str] = Query(
        default=CryptoSymbol.BTCUSDT.value, description="Filter by symbol"
    ),
    timeframe_filter: Optional[str] = Query(
        default=TimeFrame.HOUR_1.value, description="Filter by timeframe"
    ),
    expired_only: bool = Query(False, description="Show only expired caches"),
    valid_only: bool = Query(False, description="Show only valid caches"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Result offset for pagination"),
    service: DatasetManagementService = Depends(get_dataset_management_service),
) -> CacheListResponse:
    """
    List cached datasets with filtering and pagination.

    Returns information about locally cached datasets including
    cache age, TTL, and expiration status.
    """
    try:
        logger.info("Listing cached datasets")

        cached_datasets = await service.list_cached_datasets(
            symbol_filter=symbol_filter,
            timeframe_filter=timeframe_filter,
            expired_only=expired_only,
            valid_only=valid_only,
            limit=limit,
            offset=offset,
        )

        # Calculate cache statistics
        total_count = len(cached_datasets)
        total_cache_size_bytes = sum(cache.file_size_bytes for cache in cached_datasets)
        expired_count = sum(1 for cache in cached_datasets if cache.is_expired)
        valid_count = total_count - expired_count

        # Get cache directory info
        cache_directory = str(service.cache_dir)

        response = CacheListResponse(
            success=True,
            message=f"Found {total_count} cached datasets",
            cached_datasets=cached_datasets,
            total_count=total_count,
            total_cache_size_bytes=total_cache_size_bytes,
            expired_count=expired_count,
            valid_count=valid_count,
            cache_directory=cache_directory,
        )

        logger.info(f"Cache listing completed: {total_count} cached datasets found")
        return response

    except Exception as e:
        logger.error(f"Failed to list cached datasets: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list cached datasets: {str(e)}"
        )


@router.post("/cache/invalidate", response_model=CacheInvalidateResponse)
async def invalidate_cache(
    request: CacheInvalidateRequest,
    service: DatasetManagementService = Depends(get_dataset_management_service),
) -> CacheInvalidateResponse:
    """
    Invalidate caches based on criteria.

    Allows selective cache invalidation by symbol, timeframe,
    or bulk invalidation of all or expired caches.
    """
    try:
        logger.info(f"Invalidating caches: {request}")

        result = await service.invalidate_cache(
            symbol=request.symbol,
            timeframe=request.timeframe,
            expired_only=request.expired_only,
            all_caches=request.all_caches,
            confirm_invalidation=request.confirm_invalidation,
        )

        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "Cache invalidation failed"),
            )

        response = CacheInvalidateResponse(
            success=True,
            message=result.get("message", "Cache invalidation completed"),
            invalidated_count=result.get("invalidated_count", 0),
            freed_space_bytes=result.get("freed_space_bytes", 0),
            invalidated_caches=result.get("invalidated_caches", []),
            failed_invalidations=result.get("failed_invalidations", []),
            status=result.get("status", "completed"),
        )

        logger.info(
            f"Cache invalidation completed: {response.invalidated_count} caches invalidated"
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to invalidate cache: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to invalidate cache: {str(e)}"
        )


# ===== Dataset Statistics and Health =====


@router.get("/statistics", response_model=DatasetStatistics)
async def get_dataset_statistics(
    service: DatasetManagementService = Depends(get_dataset_management_service),
) -> DatasetStatistics:
    """
    Get comprehensive dataset statistics.

    Returns detailed statistics about datasets including
    counts, sizes, distributions, and cache information.
    """
    try:
        logger.info("Getting dataset statistics")

        statistics = await service.get_dataset_statistics()

        logger.info("Dataset statistics collected successfully")
        return statistics

    except Exception as e:
        logger.error(f"Failed to get dataset statistics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get statistics: {str(e)}"
        )


@router.get("/health", response_model=DatasetHealthCheck)
async def get_dataset_health(
    service: DatasetManagementService = Depends(get_dataset_management_service),
) -> DatasetHealthCheck:
    """
    Get dataset system health information.

    Returns comprehensive health status including component health,
    performance metrics, and resource usage.
    """
    try:
        logger.info("Getting dataset health check")

        health_check = await service.get_health_check()

        logger.info(
            f"Dataset health check completed: score={health_check.health_score:.1f}"
        )
        return health_check

    except Exception as e:
        logger.error(f"Failed to get dataset health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get health: {str(e)}")


# ===== Bulk Operations =====


@router.post("/bulk", response_model=BulkOperationResponse)
async def bulk_dataset_operation(
    request: BulkDatasetOperation,
    service: DatasetManagementService = Depends(get_dataset_management_service),
) -> BulkOperationResponse:
    """
    Execute bulk dataset operations.

    Supports bulk operations like downloading multiple datasets
    or invalidating multiple caches with parallel execution.
    """
    try:
        logger.info(
            f"Starting bulk operation: {request.operation_type} for {len(request.symbols)} symbols"
        )

        # For now, return a placeholder response
        # TODO: Implement bulk operations in the service
        response = BulkOperationResponse(
            success=True,
            message=f"Bulk operation {request.operation_type} initiated",
            operation_id="bulk_" + str(uuid.uuid4()),
            total_operations=len(request.symbols) * len(request.timeframes),
            successful_operations=0,
            failed_operations=0,
            total_duration_seconds=0.0,
            average_duration_per_operation=0.0,
            operation_results=[],
            errors=[],
            status="not_implemented",
            completion_percentage=0.0,
        )

        logger.info(f"Bulk operation initiated: {response.operation_id}")
        return response

    except Exception as e:
        logger.error(f"Failed to execute bulk operation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to execute bulk operation: {str(e)}"
        )


# ===== Utility Endpoints =====


@router.get("/formats")
async def get_supported_formats() -> dict:
    """Get supported data formats."""
    return {
        "success": True,
        "message": "Supported data formats",
        "formats": [fmt.value for fmt in DataFormat],
        "default_format": DataFormat.CSV.value,
        "compression_support": True,
    }


@router.get("/exchanges")
async def get_supported_exchanges() -> dict:
    """Get supported exchanges."""
    return {
        "success": True,
        "message": "Supported exchanges",
        "exchanges": [ex.value for ex in Exchange],
        "default_exchange": Exchange.BINANCE.value,
        "auto_discovery": True,
    }


@router.get("/timeframes")
async def get_supported_timeframes() -> dict:
    """Get supported timeframes."""
    timeframes = [tf.value for tf in TimeFrame]
    return {
        "success": True,
        "message": "Supported timeframes",
        "timeframes": timeframes,
        "default_timeframes": [
            TimeFrame.HOUR_1.value,
            TimeFrame.HOUR_4.value,
            TimeFrame.DAY_1.value,
        ],
        "custom_timeframes": False,
    }


@router.get("/symbols")
async def get_supported_symbols() -> dict:
    """Get supported trading symbols."""
    from ..schemas.enums import CryptoSymbol

    symbols = [sym.value for sym in CryptoSymbol]
    return {
        "success": True,
        "message": "Supported trading symbols",
        "symbols": symbols,
        "default_symbols": [
            CryptoSymbol.BTCUSDT.value,
            CryptoSymbol.ETHUSDT.value,
            CryptoSymbol.BNBUSDT.value,
        ],
        "custom_symbols": True,
    }
