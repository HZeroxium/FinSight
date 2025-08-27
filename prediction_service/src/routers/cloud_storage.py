# routers/cloud_storage.py

"""
Cloud Storage Management Router

Provides endpoints for cloud storage operations including health checks,
model synchronization, and storage management.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from common.logger.logger_factory import LoggerFactory
from fastapi import APIRouter, Depends, HTTPException

from ..core.config import get_settings
from ..facades import get_unified_facade
from ..schemas.enums import ModelType, TimeFrame
from ..utils.model_utils import ModelUtils

router = APIRouter(prefix="/cloud-storage", tags=["Cloud Storage Management"])

logger = LoggerFactory.get_logger("CloudStorageRouter")


@router.get("/health", summary="Check cloud storage health")
async def health_check_cloud_storage() -> Dict[str, Any]:
    """
    Perform comprehensive health check on cloud storage.

    Returns:
        Health check result with status and details
    """
    try:
        settings = get_settings()

        if not settings.enable_cloud_storage:
            return {
                "status": "disabled",
                "message": "Cloud storage is not enabled",
                "timestamp": datetime.now().isoformat(),
                "details": {
                    "enable_cloud_storage": False,
                    "storage_provider": settings.storage_provider,
                },
            }

        model_utils = ModelUtils()
        health_result = await model_utils.health_check_cloud_storage()

        return {
            **health_result,
            "timestamp": datetime.now().isoformat(),
            "storage_provider": settings.storage_provider,
            "bucket_name": settings.s3_bucket_name,
        }

    except Exception as e:
        logger.error(f"Cloud storage health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.post("/models/sync-to-cloud", summary="Sync model to cloud storage")
async def sync_model_to_cloud(
    symbol: str,
    timeframe: TimeFrame,
    model_type: ModelType,
    adapter_type: str = "simple",
    force_upload: bool = False,
) -> Dict[str, Any]:
    """
    Synchronize a local model to cloud storage.

    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        model_type: Model type
        adapter_type: Adapter type
        force_upload: Whether to force upload even if cloud version exists

    Returns:
        Sync result with status and details
    """
    try:
        settings = get_settings()

        if not settings.enable_cloud_storage:
            raise HTTPException(status_code=400, detail="Cloud storage is not enabled")

        model_utils = ModelUtils()
        sync_result = await model_utils.sync_model_to_cloud(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            adapter_type=adapter_type,
            force_upload=force_upload,
        )

        if sync_result["success"]:
            logger.info(
                f"Successfully synced model to cloud: {symbol}_{timeframe.value}_{model_type.value}"
            )
        else:
            logger.warning(f"Failed to sync model to cloud: {sync_result.get('error')}")

        return {
            **sync_result,
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "symbol": symbol,
                "timeframe": timeframe.value,
                "model_type": model_type.value,
                "adapter_type": adapter_type,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model sync to cloud failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


@router.post("/models/sync-from-cloud", summary="Sync model from cloud storage")
async def sync_model_from_cloud(
    symbol: str,
    timeframe: TimeFrame,
    model_type: ModelType,
    adapter_type: str = "simple",
    force_download: bool = False,
) -> Dict[str, Any]:
    """
    Synchronize a cloud model to local storage.

    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        model_type: Model type
        adapter_type: Adapter type
        force_download: Whether to force download even if local version exists

    Returns:
        Sync result with status and details
    """
    try:
        settings = get_settings()

        if not settings.enable_cloud_storage:
            raise HTTPException(status_code=400, detail="Cloud storage is not enabled")

        model_utils = ModelUtils()
        sync_result = await model_utils.sync_model_from_cloud(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            adapter_type=adapter_type,
            force_download=force_download,
        )

        if sync_result["success"]:
            logger.info(
                f"Successfully synced model from cloud: {symbol}_{timeframe.value}_{model_type.value}"
            )
        else:
            logger.warning(
                f"Failed to sync model from cloud: {sync_result.get('error')}"
            )

        return {
            **sync_result,
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "symbol": symbol,
                "timeframe": timeframe.value,
                "model_type": model_type.value,
                "adapter_type": adapter_type,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model sync from cloud failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


@router.get("/models/check-exists", summary="Check if model exists in cloud")
async def check_model_exists_in_cloud(
    symbol: str,
    timeframe: TimeFrame,
    model_type: ModelType,
    adapter_type: str = "simple",
) -> Dict[str, Any]:
    """
    Check if a model exists in cloud storage.

    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        model_type: Model type
        adapter_type: Adapter type

    Returns:
        Existence check result
    """
    try:
        settings = get_settings()

        if not settings.enable_cloud_storage:
            return {
                "exists": False,
                "reason": "Cloud storage is not enabled",
                "timestamp": datetime.now().isoformat(),
            }

        model_utils = ModelUtils()
        exists = await model_utils.model_exists_in_cloud(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            adapter_type=adapter_type,
        )

        return {
            "exists": exists,
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "symbol": symbol,
                "timeframe": timeframe.value,
                "model_type": model_type.value,
                "adapter_type": adapter_type,
            },
        }

    except Exception as e:
        logger.error(f"Model existence check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Existence check failed: {str(e)}")


@router.get("/info", summary="Get cloud storage information")
async def get_cloud_storage_info() -> Dict[str, Any]:
    """
    Get comprehensive information about cloud storage configuration and status.

    Returns:
        Cloud storage information
    """
    try:
        settings = get_settings()
        model_utils = ModelUtils()

        # Get storage client info if available
        storage_info = {}
        if model_utils.storage_client:
            try:
                storage_info = await model_utils.storage_client.get_storage_info()
            except Exception as e:
                storage_info = {"error": str(e)}

        return {
            "enabled": settings.enable_cloud_storage,
            "storage_provider": settings.storage_provider,
            "bucket_name": settings.storage_bucket_name,
            "model_storage_prefix": settings.model_storage_prefix,
            "dataset_storage_prefix": settings.dataset_storage_prefix,
            "endpoint_url": settings.storage_endpoint_url,
            "region": settings.storage_region_name,
            "use_ssl": settings.storage_use_ssl,
            "storage_info": storage_info,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get cloud storage info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get storage info: {str(e)}"
        )
