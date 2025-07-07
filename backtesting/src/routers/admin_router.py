# routers/admin_router.py

"""
Admin Router

Provides administrative endpoints for the backtesting system,
including data management, system statistics, and maintenance operations.
Protected by API key authentication.
"""

from typing import Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..services.admin_service import AdminService
from ..schemas.admin_schemas import (
    DataEnsureRequest,
    DataEnsureResponse,
    TimeframeConvertRequest,
    TimeframeConvertResponse,
    AdminStatsResponse,
    SystemHealthResponse,
    CleanupRequest,
    CleanupResponse,
)
from ..core.config import Settings
from ..common.logger import LoggerFactory
from ..factories.admin_factory import get_admin_service
from ..utils.datetime_utils import DateTimeUtils


# Security scheme for API key authentication
security = HTTPBearer()

# Router configuration
router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    responses={
        401: {"description": "Unauthorized - Invalid API key"},
        403: {"description": "Forbidden - Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)

# Logger
logger = LoggerFactory.get_logger(name="admin_router")

# Configuration
settings = Settings()


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> bool:
    """
    Verify API key for admin endpoints.

    Args:
        credentials: HTTP bearer token credentials

    Returns:
        True if API key is valid

    Raises:
        HTTPException: If API key is invalid
    """
    if not credentials.credentials or credentials.credentials != settings.admin_api_key:
        logger.warning(f"Invalid API key attempt: {credentials.credentials[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health(
    admin_service: AdminService = Depends(get_admin_service),
) -> SystemHealthResponse:
    """
    Get system health status.

    This endpoint is not protected to allow monitoring systems to check health.
    """
    try:
        logger.info("Health check requested")
        health = await admin_service.get_system_health()
        return health
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}",
        )


@router.get("/stats", response_model=AdminStatsResponse)
async def get_system_stats(
    _: bool = Depends(verify_api_key),
    admin_service: AdminService = Depends(get_admin_service),
) -> AdminStatsResponse:
    """
    Get comprehensive system statistics.

    Returns detailed information about:
    - Total records in system
    - Available symbols, exchanges, timeframes
    - Storage utilization
    - Server uptime
    """
    try:
        logger.info("System statistics requested")
        stats = await admin_service.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system statistics: {str(e)}",
        )


@router.post("/data/ensure", response_model=DataEnsureResponse)
async def ensure_data_available(
    request: DataEnsureRequest,
    _: bool = Depends(verify_api_key),
    admin_service: AdminService = Depends(get_admin_service),
) -> DataEnsureResponse:
    """
    Ensure OHLCV data is available for specified parameters.

    This endpoint will:
    - Check if data exists for the specified range
    - Fetch missing data from the exchange if needed
    - Return detailed statistics about the operation

    Use `force_refresh=true` to re-fetch data even if it exists.
    """
    try:
        logger.info(
            f"Data ensure requested: {request.symbol} on {request.exchange} "
            f"({request.timeframe}) from {request.start_date} to {request.end_date}"
        )

        response = await admin_service.ensure_data_available(request)

        logger.info(
            f"Data ensure completed: success={response.success}, "
            f"fetched={response.records_fetched}, saved={response.records_saved}"
        )

        return response
    except Exception as e:
        logger.error(f"Data ensure operation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data ensure operation failed: {str(e)}",
        )


@router.post("/data/convert-timeframe", response_model=TimeframeConvertResponse)
async def convert_timeframe_data(
    request: TimeframeConvertRequest,
    _: bool = Depends(verify_api_key),
    admin_service: AdminService = Depends(get_admin_service),
) -> TimeframeConvertResponse:
    """
    Convert OHLCV data to different timeframe.

    This endpoint will:
    - Fetch source data in the specified timeframe
    - Convert it to the target timeframe
    - Optionally save the converted data

    Set `save_converted=false` to get converted data without saving.
    """
    try:
        logger.info(
            f"Timeframe conversion requested: {request.symbol} "
            f"from {request.source_timeframe} to {request.target_timeframe}"
        )

        response = await admin_service.convert_timeframe_data(request)

        logger.info(
            f"Timeframe conversion completed: success={response.success}, "
            f"converted={response.converted_records}, saved={response.saved_records}"
        )

        return response
    except Exception as e:
        logger.error(f"Timeframe conversion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Timeframe conversion failed: {str(e)}",
        )


@router.post("/data/cleanup", response_model=CleanupResponse)
async def cleanup_old_data(
    request: CleanupRequest,
    _: bool = Depends(verify_api_key),
    admin_service: AdminService = Depends(get_admin_service),
) -> CleanupResponse:
    """
    Clean up old data before specified date.

    **WARNING: This is a destructive operation!**

    Set `confirm=true` to actually perform the deletion.
    This endpoint will delete all records before the specified cutoff date.
    """
    try:
        if not request.confirm:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Destructive operation requires confirmation. Set 'confirm=true'.",
            )

        logger.warning(
            f"Data cleanup requested: {request.symbol} on {request.exchange} "
            f"({request.timeframe}) before {request.cutoff_date}"
        )

        result = await admin_service.cleanup_old_data(
            exchange=request.exchange,
            symbol=request.symbol,
            timeframe=request.timeframe,
            cutoff_date=request.cutoff_date,
        )

        response = CleanupResponse(
            success=result["success"],
            records_before=result.get("records_before", 0),
            records_deleted=result.get("records_deleted", 0),
            cutoff_date=request.cutoff_date,
            operation_timestamp=datetime.utcnow(),
            error_message=result.get("error_message"),
        )

        logger.warning(
            f"Data cleanup completed: success={response.success}, "
            f"deleted={response.records_deleted} records"
        )

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data cleanup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data cleanup failed: {str(e)}",
        )


@router.get("/info")
async def get_admin_info(
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get information about available admin operations.

    Returns documentation about all available admin endpoints
    and their capabilities.
    """
    return {
        "admin_endpoints": {
            "GET /health": {
                "description": "Get system health status",
                "authentication": "None (public endpoint)",
                "response": "SystemHealthResponse",
            },
            "GET /stats": {
                "description": "Get comprehensive system statistics",
                "authentication": "API key required",
                "response": "AdminStatsResponse",
            },
            "POST /data/ensure": {
                "description": "Ensure OHLCV data is available",
                "authentication": "API key required",
                "request": "DataEnsureRequest",
                "response": "DataEnsureResponse",
            },
            "POST /data/convert-timeframe": {
                "description": "Convert data to different timeframe",
                "authentication": "API key required",
                "request": "TimeframeConvertRequest",
                "response": "TimeframeConvertResponse",
            },
            "POST /data/cleanup": {
                "description": "Clean up old data (destructive)",
                "authentication": "API key required",
                "request": "CleanupRequest",
                "response": "CleanupResponse",
                "warning": "Destructive operation - requires confirmation",
            },
        },
        "authentication": {
            "type": "Bearer token",
            "header": "Authorization: Bearer <api_key>",
            "environment_variable": "ADMIN_API_KEY",
        },
        "supported_operations": [
            "Data availability checking",
            "Missing data fetching",
            "Timeframe conversion",
            "Data cleanup",
            "System monitoring",
            "Storage statistics",
        ],
        "server_timestamp": DateTimeUtils.now_iso(),
    }
