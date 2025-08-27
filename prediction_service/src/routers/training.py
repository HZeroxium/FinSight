# routers/training.py

"""
Consolidated training router with both legacy and async endpoints
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from common.logger.logger_factory import LoggerFactory
from fastapi import APIRouter, Depends, HTTPException, Query

from ..schemas.enums import ModelType, TimeFrame
from ..schemas.model_schemas import TrainingRequest, TrainingResponse
from ..schemas.training_schemas import (AsyncTrainingRequest,
                                        AsyncTrainingResponse,
                                        BackgroundTaskHealthResponse,
                                        TrainingJobCancelRequest,
                                        TrainingJobCancelResponse,
                                        TrainingJobFilter,
                                        TrainingJobListResponse,
                                        TrainingJobPriority, TrainingJobStatus,
                                        TrainingJobStatusResponse,
                                        TrainingQueueResponse)
from ..services.training_service import TrainingService

router = APIRouter(prefix="/training", tags=["training"])
logger = LoggerFactory.get_logger("TrainingRouter")

# Global service instance
_training_service: Optional[TrainingService] = None


def get_training_service() -> TrainingService:
    """Get or create training service instance"""
    global _training_service
    if _training_service is None:
        _training_service = TrainingService()
    return _training_service


async def get_initialized_training_service() -> TrainingService:
    """Get training service and ensure it's initialized"""
    service = get_training_service()
    await service._ensure_initialized()
    return service


# Main training endpoints


@router.post("/train", response_model=AsyncTrainingResponse)
async def train_model(
    request: TrainingRequest,
    training_service: TrainingService = Depends(get_initialized_training_service),
) -> AsyncTrainingResponse:
    """
    Train a time series model (NON-BLOCKING ASYNC)

    This endpoint converts legacy TrainingRequest to AsyncTrainingRequest and submits the job
    to the background queue for processing. The training is executed asynchronously to prevent
    blocking other requests.

    Returns:
        AsyncTrainingResponse: Contains job_id for tracking progress via /status/{job_id}

    Migration Notes:
        - Previously this was a blocking synchronous endpoint
        - Now returns immediately with job_id for status tracking
        - Use /status/{job_id} to monitor training progress
        - Backward compatible response structure
    """
    try:
        logger.info(
            f"Received training request for {request.symbol} {request.timeframe} {request.model_type}"
        )

        # Convert legacy TrainingRequest to AsyncTrainingRequest
        async_request = AsyncTrainingRequest(
            symbol=request.symbol,
            timeframe=request.timeframe,
            model_type=request.model_type,
            config=request.config,
            priority=TrainingJobPriority.NORMAL,  # Default priority for legacy requests
            tags={"source": "legacy_endpoint", "migrated": "true"},
            force_retrain=False,  # Default to prevent duplicate training
        )

        # Start async training (non-blocking)
        response = await training_service.start_async_training(async_request)

        if response.success:
            logger.info(
                f"Async training submitted successfully: {response.job_id} "
                f"(estimated duration: {response.estimated_duration_seconds}s)"
            )
        else:
            logger.warning(f"Async training submission failed: {response.error}")

        return response

    except Exception as e:
        logger.error(f"Training endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/train-async", response_model=AsyncTrainingResponse)
async def train_model_async(
    request: AsyncTrainingRequest,
    training_service: TrainingService = Depends(get_initialized_training_service),
) -> AsyncTrainingResponse:
    """
    Start asynchronous model training with full configuration (NON-BLOCKING)

    This endpoint accepts the full AsyncTrainingRequest with priority, tags, and other
    advanced options. The training is executed in the background without blocking other requests.

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


# Status and monitoring endpoints


@router.get("/status/{job_id}", response_model=TrainingJobStatusResponse)
async def get_training_status(
    job_id: str,
    training_service: TrainingService = Depends(get_initialized_training_service),
) -> TrainingJobStatusResponse:
    """
    Get detailed status of a training job

    Returns comprehensive information about the training job including:
    - Current status and progress
    - Training metrics (if available)
    - Error information (if failed)
    - Estimated completion time

    Args:
        job_id: Training job identifier

    Returns:
        TrainingJobStatusResponse: Job status information
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
    training_service: TrainingService = Depends(get_initialized_training_service),
) -> TrainingJobListResponse:
    """
    List training jobs with advanced filtering and pagination

    Supports filtering by status, model type, symbol, timeframe, and more.
    Results are paginated and sorted according to the specified criteria.
    """
    try:
        # Build filter criteria
        filter_criteria = TrainingJobFilter(
            statuses=statuses,
            exclude_statuses=exclude_statuses,
            symbols=symbols,
            timeframes=[tf.value for tf in timeframes] if timeframes else None,
            model_types=[mt.value for mt in model_types] if model_types else None,
            offset=offset,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        response = await training_service.list_training_jobs(filter_criteria)

        if not response.success:
            raise HTTPException(status_code=500, detail=response.error)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing training jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Job management endpoints


@router.delete("/jobs/{job_id}/cancel", response_model=TrainingJobCancelResponse)
async def cancel_training_job(
    job_id: str,
    cancel_request: TrainingJobCancelRequest = TrainingJobCancelRequest(),
    training_service: TrainingService = Depends(get_initialized_training_service),
) -> TrainingJobCancelResponse:
    """
    Cancel a training job

    Cancels a pending or running training job. Use force=true to cancel running jobs.

    Args:
        job_id: Job identifier
        cancel_request: Cancellation request parameters

    Returns:
        TrainingJobCancelResponse: Cancellation result
    """
    try:
        logger.info(f"Cancelling training job {job_id} (force: {cancel_request.force})")

        response = await training_service.cancel_training_job(job_id, cancel_request)

        if not response.success:
            if "not found" in response.error.lower():
                raise HTTPException(status_code=404, detail=response.error)
            else:
                raise HTTPException(status_code=400, detail=response.error)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling training job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Queue and system status endpoints


@router.get("/queue", response_model=TrainingQueueResponse)
async def get_training_queue_info(
    training_service: TrainingService = Depends(get_initialized_training_service),
) -> TrainingQueueResponse:
    """
    Get information about the training queue

    Returns current queue status including pending, running, and completed jobs.
    """
    try:
        response = await training_service.get_queue_info()

        if not response.success:
            raise HTTPException(status_code=500, detail=response.error)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting queue info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/system/health", response_model=BackgroundTaskHealthResponse)
async def get_background_system_health(
    training_service: TrainingService = Depends(get_initialized_training_service),
) -> BackgroundTaskHealthResponse:
    """
    Get health status of the training system

    Returns comprehensive health information including resource usage and system status.
    """
    try:
        response = await training_service.get_background_health()

        if not response.success:
            raise HTTPException(status_code=500, detail=response.error)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Additional utility endpoints


@router.get("/active", response_model=Dict[str, Any])
async def get_active_trainings(
    training_service: TrainingService = Depends(get_initialized_training_service),
) -> Dict[str, Any]:
    """
    Get all active training jobs

    Returns a list of all currently active training jobs with their status.
    """
    try:
        # Get active jobs from the async training system
        filter_criteria = TrainingJobFilter(
            statuses=TrainingJobStatus.get_active_statuses(),
            sort_by="created_at",
            sort_order="desc",
            limit=100,
        )

        response = await training_service.list_training_jobs(filter_criteria)

        if not response.success:
            raise HTTPException(status_code=500, detail=response.error)

        return {
            "success": True,
            "message": "Active trainings retrieved",
            "data": {
                "active_jobs": response.jobs,
                "total_active": response.active_count,
                "queue_info": {
                    "running": len(
                        [
                            j
                            for j in response.jobs
                            if j.status == TrainingJobStatus.TRAINING
                        ]
                    ),
                    "queued": len(
                        [
                            j
                            for j in response.jobs
                            if j.status == TrainingJobStatus.QUEUED
                        ]
                    ),
                    "initializing": len(
                        [
                            j
                            for j in response.jobs
                            if j.status
                            in [
                                TrainingJobStatus.INITIALIZING,
                                TrainingJobStatus.LOADING_DATA,
                                TrainingJobStatus.PREPARING_FEATURES,
                            ]
                        ]
                    ),
                },
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting active trainings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/models/supported", response_model=Dict[str, Any])
async def get_supported_models() -> Dict[str, Any]:
    """
    Get list of supported model types and their configurations

    Returns information about available model types and their default configurations.
    """
    try:
        supported_models = {
            ModelType.PATCHTST.value: {
                "name": "PatchTST",
                "description": "Patch Time Series Transformer for forecasting",
                "default_config": {
                    "context_length": 64,
                    "prediction_length": 1,
                    "num_epochs": 10,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                },
            },
            ModelType.PATCHTSMIXER.value: {
                "name": "PatchTSMixer",
                "description": "Patch Time Series Mixer for forecasting",
                "default_config": {
                    "context_length": 64,
                    "prediction_length": 1,
                    "num_epochs": 10,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                },
            },
        }

        supported_timeframes = [tf.value for tf in TimeFrame]

        return {
            "success": True,
            "message": "Supported models retrieved",
            "data": {
                "models": supported_models,
                "timeframes": supported_timeframes,
                "max_concurrent_trainings": 3,  # From TrainingConstants
                "max_queue_size": 50,  # From TrainingConstants
            },
        }

    except Exception as e:
        logger.error(f"Error getting supported models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Legacy compatibility endpoints


@router.post("/jobs/{job_id}/retry", response_model=AsyncTrainingResponse)
async def retry_failed_training(
    job_id: str,
    training_service: TrainingService = Depends(get_initialized_training_service),
) -> AsyncTrainingResponse:
    """
    Retry a failed training job

    Creates a new training job based on the configuration of a failed job.

    Args:
        job_id: Original job identifier

    Returns:
        AsyncTrainingResponse: New training job information
    """
    try:
        logger.info(f"Retrying failed training job {job_id}")

        # Get the original job info
        status_response = await training_service.get_training_status(job_id)

        if not status_response.success:
            raise HTTPException(status_code=404, detail="Original job not found")

        original_job = status_response.job

        if original_job.status != TrainingJobStatus.FAILED:
            raise HTTPException(status_code=400, detail="Job is not in failed state")

        # Create new request based on original job
        retry_request = AsyncTrainingRequest(
            symbol=original_job.symbol,
            timeframe=TimeFrame(original_job.timeframe),
            model_type=ModelType(original_job.model_type),
            config=original_job.config,
            priority=TrainingJobPriority.HIGH,  # Higher priority for retries
            force_retrain=True,  # Allow retry even if similar job exists
            tags={
                "source": "retry",
                "original_job_id": job_id,
                "retry_timestamp": str(int(time.time())),
            },
        )

        # Start the retry
        response = await training_service.start_async_training(retry_request)

        if response.success:
            logger.info(
                f"Retry job {response.job_id} created for original job {job_id}"
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrying training job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# System management endpoints


@router.post("/system/shutdown", response_model=Dict[str, Any])
async def shutdown_training_system(
    force: bool = Query(False, description="Force shutdown even with active jobs"),
    training_service: TrainingService = Depends(get_initialized_training_service),
) -> Dict[str, Any]:
    """
    Shutdown the training system

    Gracefully shuts down the background training system. If force=false, will only
    shutdown if no jobs are active.

    Args:
        force: Force shutdown even with active jobs

    Returns:
        Dict[str, Any]: Shutdown status
    """
    try:
        logger.info(f"Shutdown request received (force: {force})")

        if not force:
            # Check for active jobs
            active_response = await get_active_trainings(training_service)
            if active_response["data"]["total_active"] > 0:
                return {
                    "success": False,
                    "message": "Cannot shutdown with active jobs",
                    "error": f"Found {active_response['data']['total_active']} active jobs. Use force=true to shutdown anyway.",
                    "active_jobs": active_response["data"]["total_active"],
                }

        # Perform shutdown
        await training_service.shutdown()

        logger.info("Training system shutdown completed")

        return {
            "success": True,
            "message": "Training system shutdown completed",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error during system shutdown: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
