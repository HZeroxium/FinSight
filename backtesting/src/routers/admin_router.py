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
from common.logger import LoggerFactory
from ..factories.admin_factory import get_admin_service
from ..utils.datetime_utils import DateTimeUtils
from ..utils.dependencies import require_admin_access


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
    _: bool = Depends(require_admin_access),
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
    _: bool = Depends(require_admin_access),
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
    _: bool = Depends(require_admin_access),
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
    _: bool = Depends(require_admin_access),
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
    _: bool = Depends(require_admin_access),
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
            "GET /config": {
                "description": "Get current system configuration",
                "authentication": "API key required",
                "response": "Configuration object with masked sensitive values",
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
            "Configuration viewing",
        ],
        "server_timestamp": DateTimeUtils.now_iso(),
    }


@router.get("/config")
async def get_current_configuration(
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Get current system configuration.

    Returns the current configuration settings, with sensitive values masked
    for security. This helps administrators understand the current system
    configuration and troubleshoot issues.
    """
    try:
        logger.info("Configuration requested")

        def mask_sensitive_value(value: str, show_chars: int = 4) -> str:
            """Mask sensitive values while showing a few characters for identification"""
            if not value:
                return "[NOT_SET]"
            if len(value) <= show_chars:
                return "*" * len(value)
            return value[:show_chars] + "*" * (len(value) - show_chars)

        # Get current settings
        config = {
            "service": {
                "app_name": settings.app_name,
                "environment": settings.environment,
                "debug": settings.debug,
                "host": settings.host,
                "port": settings.port,
            },
            "storage": {
                "base_directory": settings.storage_base_directory,
                "provider": settings.storage_provider,
                "s3_compatible": {
                    "endpoint_url": settings.s3_endpoint_url,
                    "region_name": settings.s3_region_name,
                    "bucket_name": settings.s3_bucket_name,
                    "use_ssl": settings.s3_use_ssl,
                    "verify_ssl": settings.s3_verify_ssl,
                    "signature_version": settings.s3_signature_version,
                    "max_pool_connections": settings.s3_max_pool_connections,
                },
                "digitalocean_spaces": {
                    "endpoint_url": settings.spaces_endpoint_url,
                    "region_name": settings.spaces_region_name,
                    "bucket_name": settings.spaces_bucket_name,
                },
                "aws_s3": {
                    "region_name": settings.aws_region_name,
                    "bucket_name": settings.aws_bucket_name,
                },
            },
            "database": {
                "mongodb_url": mask_sensitive_value(settings.mongodb_url, 10),
                "mongodb_database": settings.mongodb_database,
            },
            "data_collection": {
                "default_symbols": settings.default_symbols,
                "default_timeframes": settings.default_timeframes,
                "max_ohlcv_limit": settings.max_ohlcv_limit,
                "max_trades_limit": settings.max_trades_limit,
                "max_orderbook_limit": settings.max_orderbook_limit,
            },
            "exchange_binance": {
                "api_key": mask_sensitive_value(settings.binance_api_key),
                "secret_key": mask_sensitive_value(settings.binance_secret_key),
                "requests_per_minute": settings.binance_requests_per_minute,
                "orders_per_second": settings.binance_orders_per_second,
                "orders_per_day": settings.binance_orders_per_day,
            },
            "cross_repository": {
                "source_repository_type": settings.source_repository_type,
                "source_timeframe": settings.source_timeframe,
                "target_repository_type": settings.target_repository_type,
                "target_timeframes": settings.target_timeframes,
                "enable_parallel_conversion": settings.enable_parallel_conversion,
                "max_concurrent_conversions": settings.max_concurrent_conversions,
                "conversion_batch_size": settings.conversion_batch_size,
            },
            "logging": {
                "log_level": settings.log_level,
                "log_file_path": settings.log_file_path,
                "enable_structured_logging": settings.enable_structured_logging,
            },
            "caching": {
                "enable_caching": settings.enable_caching,
                "cache_ttl_seconds": settings.cache_ttl_seconds,
                "cache_max_size": settings.cache_max_size,
            },
            "admin": {
                "api_key": mask_sensitive_value(settings.api_key),
            },
            "cron_job": {
                "enabled": settings.cron_job_enabled,
                "schedule": settings.cron_job_schedule,
                "max_symbols_per_run": settings.cron_job_max_symbols_per_run,
                "log_file": settings.cron_job_log_file,
                "pid_file": settings.cron_job_pid_file,
            },
            "demo": {
                "max_symbols": settings.demo_max_symbols,
                "days_back": settings.demo_days_back,
            },
        }

        # Get repository information
        from ..utils.dependencies import get_dependency_manager

        dep_manager = get_dependency_manager()

        try:
            current_repository = dep_manager.get_repository()
            repository_info = {
                "type": type(current_repository).__name__,
                "class": current_repository.__class__.__name__,
                "module": current_repository.__class__.__module__,
                "repository_type": settings.repository_type,
                "source_repository_type": settings.source_repository_type,
                "target_repository_type": settings.target_repository_type,
            }
        except Exception as e:
            repository_info = {"error": str(e)}

        # Get environment variable information
        import os

        env_vars = {
            "SOURCE_REPOSITORY_TYPE": os.getenv("SOURCE_REPOSITORY_TYPE", "[NOT_SET]"),
            "MONGODB_URL": mask_sensitive_value(os.getenv("MONGODB_URL", "")),
            "STORAGE_BASE_DIRECTORY": os.getenv("STORAGE_BASE_DIRECTORY", "[NOT_SET]"),
            "ADMIN_API_KEY": mask_sensitive_value(os.getenv("ADMIN_API_KEY", "")),
            "STORAGE_PROVIDER": os.getenv("STORAGE_PROVIDER", "[NOT_SET]"),
            "S3_ENDPOINT_URL": os.getenv("S3_ENDPOINT_URL", "[NOT_SET]"),
            "S3_ACCESS_KEY": mask_sensitive_value(os.getenv("S3_ACCESS_KEY", "")),
            "S3_SECRET_KEY": mask_sensitive_value(os.getenv("S3_SECRET_KEY", "")),
            "S3_BUCKET_NAME": os.getenv("S3_BUCKET_NAME", "[NOT_SET]"),
            "SPACES_ACCESS_KEY": mask_sensitive_value(
                os.getenv("SPACES_ACCESS_KEY", "")
            ),
            "SPACES_SECRET_KEY": mask_sensitive_value(
                os.getenv("SPACES_SECRET_KEY", "")
            ),
            "AWS_ACCESS_KEY_ID": mask_sensitive_value(
                os.getenv("AWS_ACCESS_KEY_ID", "")
            ),
            "AWS_SECRET_ACCESS_KEY": mask_sensitive_value(
                os.getenv("AWS_SECRET_ACCESS_KEY", "")
            ),
        }

        repository_types = {
            "available_types": [
                "mongodb",
                "csv",
                "parquet",
                "influxdb",
                "object_storage",
            ],
            "current_source": settings.source_repository_type,
            "current_target": settings.target_repository_type,
            "current_repository_instance": repository_info,
            "environment_variables": env_vars,
            "descriptions": {
                "mongodb": "Production-ready document storage with indexing and aggregation",
                "csv": "Simple file-based storage, human-readable format",
                "parquet": "Columnar storage optimized for analytics workloads",
                "influxdb": "Time-series optimized storage with downsampling",
                "object_storage": "S3-compatible object storage (MinIO, DigitalOcean Spaces, AWS S3)",
            },
        }

        return {
            "configuration": config,
            "repository_types": repository_types,
            "timestamp": DateTimeUtils.now_iso(),
            "config_file_location": ".env or environment variables",
            "documentation": "/docs/CONFIGURATION_GUIDE.md",
            "note": "Sensitive values are masked for security. Check environment variables for actual values.",
        }

    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get configuration: {str(e)}",
        )
