# routers/cleanup.py

"""
Cleanup Router - Provides endpoints for manual cleanup operations

This router provides simple endpoints for:
- Cleaning cloud cache files
- Cleaning dataset files
- Cleaning model files
- Getting cleanup service status
"""

from typing import Any, Dict

from common.logger.logger_factory import LoggerFactory
from fastapi import APIRouter, HTTPException

from ..services.background_cleaner_service import \
    get_background_cleaner_service

router = APIRouter(prefix="/cleanup", tags=["cleanup"])
logger = LoggerFactory.get_logger("CleanupRouter")


@router.post("/cloud-cache")
async def cleanup_cloud_cache() -> Dict[str, Any]:
    """
    Manually clean up cloud cache files.

    This endpoint immediately cleans up old cloud cache files
    based on the configured age threshold.

    Returns:
        Dict containing cleanup results
    """
    try:
        logger.info("Manual cloud cache cleanup requested")

        cleaner_service = get_background_cleaner_service()
        result = await cleaner_service.cleanup_cloud_cache()

        if result["success"]:
            logger.info(
                f"Cloud cache cleanup completed: {result.get('files_removed', 0)} files removed"
            )
        else:
            logger.warning(
                f"Cloud cache cleanup failed: {result.get('error', 'Unknown error')}"
            )

        return {
            "success": result["success"],
            "message": result.get("message", "Cloud cache cleanup completed"),
            "data": result,
        }

    except Exception as e:
        logger.error(f"Cloud cache cleanup endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/datasets")
async def cleanup_datasets() -> Dict[str, Any]:
    """
    Manually clean up old dataset files.

    This endpoint immediately cleans up old dataset files
    based on the configured age threshold.

    Returns:
        Dict containing cleanup results
    """
    try:
        logger.info("Manual datasets cleanup requested")

        cleaner_service = get_background_cleaner_service()
        result = await cleaner_service.cleanup_datasets()

        if result["success"]:
            logger.info(
                f"Datasets cleanup completed: {result.get('files_removed', 0)} files removed"
            )
        else:
            logger.warning(
                f"Datasets cleanup failed: {result.get('error', 'Unknown error')}"
            )

        return {
            "success": result["success"],
            "message": result.get("message", "Datasets cleanup completed"),
            "data": result,
        }

    except Exception as e:
        logger.error(f"Datasets cleanup endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/models")
async def cleanup_models() -> Dict[str, Any]:
    """
    Manually clean up old model files.

    This endpoint immediately cleans up old model files
    based on the configured age threshold.

    Returns:
        Dict containing cleanup results
    """
    try:
        logger.info("Manual models cleanup requested")

        cleaner_service = get_background_cleaner_service()
        result = await cleaner_service.cleanup_models()

        if result["success"]:
            logger.info(
                f"Models cleanup completed: {result.get('models_removed', 0)} models removed"
            )
        else:
            logger.warning(
                f"Models cleanup failed: {result.get('error', 'Unknown error')}"
            )

        return {
            "success": result["success"],
            "message": result.get("message", "Models cleanup completed"),
            "data": result,
        }

    except Exception as e:
        logger.error(f"Models cleanup endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/all")
async def cleanup_all() -> Dict[str, Any]:
    """
    Manually clean up all targets (cloud cache, datasets, models).

    This endpoint immediately cleans up all targets
    based on their configured age thresholds.

    Returns:
        Dict containing cleanup results for all targets
    """
    try:
        logger.info("Manual comprehensive cleanup requested")

        cleaner_service = get_background_cleaner_service()
        result = await cleaner_service.cleanup_all()

        if result["success"]:
            summary = result.get("summary", {})
            logger.info(
                f"Comprehensive cleanup completed: {summary.get('total_items_removed', 0)} items removed"
            )
        else:
            logger.warning(
                f"Comprehensive cleanup failed: {result.get('error', 'Unknown error')}"
            )

        return {
            "success": result["success"],
            "message": result.get("message", "Comprehensive cleanup completed"),
            "data": result,
        }

    except Exception as e:
        logger.error(f"Comprehensive cleanup endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/status")
async def get_cleanup_status() -> Dict[str, Any]:
    """
    Get the current status of the background cleaner service.

    Returns:
        Dict containing service status and configuration
    """
    try:
        logger.debug("Cleanup status requested")

        cleaner_service = get_background_cleaner_service()
        status = await cleaner_service.get_cleanup_status()

        return {
            "success": True,
            "message": "Cleanup service status retrieved",
            "data": status,
        }

    except Exception as e:
        logger.error(f"Cleanup status endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
