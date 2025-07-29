# routers/async_training.py

"""
Enhanced async training router with non-blocking endpoints
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, Optional, List

from ..services.async_training_service import AsyncTrainingService
from ..schemas.model_schemas import TrainingRequest, TrainingResponse
from ..schemas.training_schemas import (
    AsyncTrainingRequest,
    AsyncTrainingResponse,
    TrainingJobStatusResponse,
    TrainingJobListResponse,
    TrainingJobCancelRequest,
    TrainingJobCancelResponse,
    TrainingQueueResponse,
    TrainingJobFilter,
    BackgroundTaskHealthResponse,
    TrainingJobStatus,
)
from ..schemas.enums import ModelType, TimeFrame
from common.logger.logger_factory import LoggerFactory

router = APIRouter(prefix="/training", tags=["training"])
logger = LoggerFactory.get_logger("AsyncTrainingRouter")

# Global service instance
_training_service: Optional[AsyncTrainingService] = None


def get_training_service() -> AsyncTrainingService:
    """Get or create training service instance"""
    global _training_service
    if _training_service is None:
        _training_service = AsyncTrainingService()
    return _training_service


# New asynchronous endpoints


@router.post("/train-async", response_model=AsyncTrainingResponse)
async def train_model_async(
    request: AsyncTrainingRequest,
    training_service: AsyncTrainingService = Depends(get_training_service),
) -> AsyncTrainingResponse:
    """
    Start asynchronous model training (NON-BLOCKING)

    This endpoint submits a training job to the background queue and returns immediately.
    The training will be executed in the background without blocking other requests.

    Returns a job ID that can be used to check training status and progress.
    """
    try:
        logger.info(
            f"Received async training request for {request.symbol} {request.timeframe} {request.model_type}"
        )

        # Start async training
        response = await training_service.start_async_training(request)

        if response.success:
            logger.info(f"Async training submitted successfully: {response.job_id}")
        else:
            logger.warning(f"Async training submission failed: {response.error}")

        return response

    except Exception as e:
        logger.error(f"Async training endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/status/{job_id}", response_model=TrainingJobStatusResponse)
async def get_training_status(
    job_id: str, training_service: AsyncTrainingService = Depends(get_training_service)
) -> TrainingJobStatusResponse:
    """
    Get detailed status of a training job

    Returns comprehensive information about the training job including:
    - Current status and progress
    - Training metrics (if available)
    - Error information (if failed)
    - Estimated completion time
    """
    try:
        logger.debug(f"Getting status for training job {job_id}")

        response = await training_service.get_training_status(job_id)

        if not response.success:
            if "not found" in response.error.lower():
                raise HTTPException(status_code=404, detail=response.error)
            else:
                raise HTTPException(status_code=500, detail=response.error)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training status for {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/jobs", response_model=TrainingJobListResponse)
async def list_training_jobs(
    # Status filters
    statuses: Optional[List[TrainingJobStatus]] = Query(
        None, description="Filter by job statuses"
    ),
    exclude_statuses: Optional[List[TrainingJobStatus]] = Query(
        None, description="Exclude job statuses"
    ),
    # Model filters
    symbols: Optional[List[str]] = Query(None, description="Filter by trading symbols"),
    timeframes: Optional[List[TimeFrame]] = Query(
        None, description="Filter by timeframes"
    ),
    model_types: Optional[List[ModelType]] = Query(
        None, description="Filter by model types"
    ),
    # Pagination
    offset: int = Query(0, ge=0, description="Result offset"),
    limit: int = Query(50, ge=1, le=1000, description="Result limit"),
    # Sorting
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    training_service: AsyncTrainingService = Depends(get_training_service),
) -> TrainingJobListResponse:
    """
    List training jobs with advanced filtering and pagination

    Supports filtering by status, model type, symbol, timeframe, and more.
    Results are paginated and sorted according to the specified criteria.
    """
    try:
        logger.debug(f"Listing training jobs with filters")

        # Create filter object
        filter_criteria = TrainingJobFilter(
            statuses=statuses,
            exclude_statuses=exclude_statuses,
            symbols=symbols,
            timeframes=timeframes,
            model_types=model_types,
            offset=offset,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        response = await training_service.list_training_jobs(filter_criteria)

        return response

    except Exception as e:
        logger.error(f"Error listing training jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/jobs/{job_id}/cancel", response_model=TrainingJobCancelResponse)
async def cancel_training_job(
    job_id: str,
    cancel_request: TrainingJobCancelRequest = TrainingJobCancelRequest(),
    training_service: AsyncTrainingService = Depends(get_training_service),
) -> TrainingJobCancelResponse:
    """
    Cancel a training job

    Attempts to cancel a running or queued training job.
    If the job is in a critical stage (like saving), use force=true to override.
    """
    try:
        logger.info(f"Cancelling training job {job_id}")

        response = await training_service.cancel_training_job(job_id, cancel_request)

        if response.success:
            logger.info(f"Successfully cancelled job {job_id}")
        else:
            logger.warning(f"Failed to cancel job {job_id}: {response.error}")

        return response

    except Exception as e:
        logger.error(f"Error cancelling training job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/queue", response_model=TrainingQueueResponse)
async def get_training_queue_info(
    training_service: AsyncTrainingService = Depends(get_training_service),
) -> TrainingQueueResponse:
    """
    Get information about the training queue

    Returns statistics about the training queue including:
    - Number of pending, running, and completed jobs
    - Queue capacity and health status
    - Average processing times
    - Worker availability
    """
    try:
        logger.debug("Getting training queue information")

        response = await training_service.get_queue_info()

        return response

    except Exception as e:
        logger.error(f"Error getting queue info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/system/health", response_model=BackgroundTaskHealthResponse)
async def get_background_system_health(
    training_service: AsyncTrainingService = Depends(get_training_service),
) -> BackgroundTaskHealthResponse:
    """
    Get health status of the background training system

    Returns detailed health information including:
    - System resource usage (CPU, memory, disk)
    - Worker status and availability
    - Recent error counts
    - Queue performance metrics
    """
    try:
        logger.debug("Getting background system health")

        response = await training_service.get_background_health()

        return response

    except Exception as e:
        logger.error(f"Error getting background health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Legacy synchronous endpoints for backward compatibility


@router.post("/train", response_model=TrainingResponse)
async def train_model_sync(
    request: TrainingRequest,
    training_service: AsyncTrainingService = Depends(get_training_service),
) -> TrainingResponse:
    """
    Train a time series model (BLOCKING - DEPRECATED)

    ⚠️ DEPRECATED: This endpoint blocks other requests during training.
    Use /training/train-async for non-blocking training.

    This endpoint trains a model synchronously and returns only when training is complete.
    For long-running training jobs, this can cause request timeouts.
    """
    try:
        logger.warning(
            f"Using deprecated sync training endpoint for {request.symbol} {request.timeframe} {request.model_type}"
        )

        # Use legacy sync method
        response = training_service.start_training(request)

        if response.success:
            logger.info(f"Sync training completed successfully: {response.training_id}")
        else:
            logger.warning(f"Sync training failed: {response.error}")

        return response

    except Exception as e:
        logger.error(f"Sync training endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/active", response_model=Dict[str, Any])
async def get_active_trainings(
    training_service: AsyncTrainingService = Depends(get_training_service),
) -> Dict[str, Any]:
    """
    Get all active training jobs (LEGACY)

    ⚠️ DEPRECATED: Use /training/jobs with status filters instead.

    Returns a list of all currently active training jobs with their status.
    This endpoint is kept for backward compatibility.
    """
    try:
        logger.warning("Using deprecated /active endpoint")

        # Get active jobs using new method
        filter_criteria = TrainingJobFilter(
            exclude_statuses=[
                TrainingJobStatus.COMPLETED,
                TrainingJobStatus.FAILED,
                TrainingJobStatus.CANCELLED,
            ],
            limit=1000,
        )

        response = await training_service.list_training_jobs(filter_criteria)

        if response.success:
            # Convert to legacy format
            legacy_format = {}
            for job in response.jobs:
                legacy_format[job.job_id] = {
                    "status": job.status.value,
                    "progress": job.progress,
                    "start_time": (
                        job.started_at.timestamp() if job.started_at else None
                    ),
                    "request": {
                        "symbol": job.symbol,
                        "timeframe": job.timeframe,
                        "model_type": job.model_type,
                    },
                }

            return {
                "success": True,
                "message": f"Found {len(legacy_format)} active training jobs",
                "data": {"count": len(legacy_format), "trainings": legacy_format},
            }
        else:
            return {
                "success": False,
                "message": "Failed to get active trainings",
                "error": response.error,
            }

    except Exception as e:
        logger.error(f"Error getting active trainings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Additional utility endpoints


@router.get("/models/supported", response_model=Dict[str, Any])
async def get_supported_models() -> Dict[str, Any]:
    """
    Get list of supported model types and their default configurations

    Returns information about available model types and their default parameters.
    """
    try:
        from ..core.constants import DEFAULT_MODEL_CONFIGS

        return {
            "success": True,
            "message": "Supported models retrieved",
            "data": {
                "model_types": [model_type.value for model_type in ModelType],
                "default_configs": DEFAULT_MODEL_CONFIGS,
                "supported_timeframes": [tf.value for tf in TimeFrame],
            },
        }

    except Exception as e:
        logger.error(f"Error getting supported models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/jobs/{job_id}/retry", response_model=AsyncTrainingResponse)
async def retry_failed_training(
    job_id: str, training_service: AsyncTrainingService = Depends(get_training_service)
) -> AsyncTrainingResponse:
    """
    Retry a failed training job with the same parameters

    Creates a new training job using the same configuration as the failed job.
    The original job ID will remain failed, and a new job ID will be generated.
    """
    try:
        logger.info(f"Retrying failed training job {job_id}")

        # Get the original job
        job_response = await training_service.get_training_status(job_id)

        if not job_response.success:
            raise HTTPException(status_code=404, detail="Original job not found")

        original_job = job_response.job

        if original_job.status != TrainingJobStatus.FAILED:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} is not in failed status (current: {original_job.status})",
            )

        # Create new request from original job config
        retry_request = AsyncTrainingRequest(
            symbol=original_job.symbol,
            timeframe=TimeFrame(original_job.timeframe),
            model_type=ModelType(original_job.model_type),
            config=original_job.config,
            tags={**(original_job.tags or {}), "retry_of": job_id},
        )

        # Submit retry
        response = await training_service.start_async_training(retry_request)

        if response.success:
            logger.info(f"Retry job submitted successfully: {response.job_id}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrying training job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Cleanup endpoint for managing service lifecycle


@router.post("/system/shutdown", response_model=Dict[str, Any])
async def shutdown_training_system(
    force: bool = Query(False, description="Force shutdown even with active jobs"),
    training_service: AsyncTrainingService = Depends(get_training_service),
) -> Dict[str, Any]:
    """
    Shutdown the training system gracefully

    ⚠️ ADMIN ONLY: This endpoint should be protected in production.

    Stops accepting new jobs and waits for active jobs to complete.
    Use force=true to cancel active jobs and shutdown immediately.
    """
    try:
        logger.warning("Training system shutdown requested")

        if not force:
            # Check for active jobs
            active_jobs = await training_service.job_repository.get_active_jobs()
            if active_jobs:
                return {
                    "success": False,
                    "message": f"Cannot shutdown with {len(active_jobs)} active jobs",
                    "data": {"active_jobs": len(active_jobs), "use_force": True},
                }

        # Shutdown the service
        await training_service.shutdown()

        logger.info("Training system shutdown completed")

        return {
            "success": True,
            "message": "Training system shutdown completed",
            "data": {"forced": force},
        }

    except Exception as e:
        logger.error(f"Error during system shutdown: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
