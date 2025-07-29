# routers/training.py

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, Optional

from ..services.training_service import AsyncTrainingService
from ..schemas.model_schemas import TrainingRequest
from ..schemas.training_schemas import (
    AsyncTrainingRequest,
    AsyncTrainingResponse,
    TrainingJobPriority,
)
from common.logger.logger_factory import LoggerFactory

router = APIRouter(prefix="/training", tags=["training"])
logger = LoggerFactory.get_logger("TrainingRouter")

# Global service instance
_training_service: Optional[AsyncTrainingService] = None


def get_training_service() -> AsyncTrainingService:
    """Get or create training service instance"""
    global _training_service
    if _training_service is None:
        _training_service = AsyncTrainingService()
    return _training_service


async def get_initialized_training_service() -> AsyncTrainingService:
    """Get training service and ensure it's initialized"""
    service = get_training_service()
    await service._ensure_initialized()
    return service


@router.post("/train", response_model=AsyncTrainingResponse)
async def train_model(
    request: TrainingRequest,
    training_service: AsyncTrainingService = Depends(get_initialized_training_service),
) -> AsyncTrainingResponse:
    """
    Train a time series model (NON-BLOCKING ASYNC)

    This endpoint now uses the async training system to prevent blocking other requests.
    It converts legacy TrainingRequest to AsyncTrainingRequest and submits the job
    to the background queue for processing.

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
            priority=TrainingJobPriority.MEDIUM,  # Default priority for legacy requests
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


@router.get("/status/{training_id}", response_model=Dict[str, Any])
async def get_training_status(
    training_id: str,
    training_service: AsyncTrainingService = Depends(get_initialized_training_service),
) -> Dict[str, Any]:
    """
    Get status of a training job

    Returns the current status, progress, and metrics of a training job.
    """
    try:
        # Try async training status first (new system)
        async_response = await training_service.get_training_status(training_id)

        if async_response.success:
            return {
                "success": True,
                "message": "Training status retrieved",
                "data": async_response.job.model_dump() if async_response.job else None,
            }

        # Fallback to legacy status for backward compatibility
        status = training_service.get_training_status(training_id)

        if status is None:
            raise HTTPException(
                status_code=404, detail=f"Training job {training_id} not found"
            )

        return {"success": True, "message": "Training status retrieved", "data": status}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/active", response_model=Dict[str, Any])
async def get_active_trainings(
    training_service: AsyncTrainingService = Depends(get_initialized_training_service),
) -> Dict[str, Any]:
    """
    Get all active training jobs

    Returns a list of all currently active training jobs with their status.
    """
    try:
        # Get active jobs from the async training system
        from ..schemas.training_schemas import TrainingJobFilter, TrainingJobStatus

        filter_criteria = TrainingJobFilter(
            statuses=[
                TrainingJobStatus.PENDING,
                TrainingJobStatus.RUNNING,
                TrainingJobStatus.QUEUED,
            ]
        )

        async_jobs_response = await training_service.list_training_jobs(filter_criteria)

        if async_jobs_response.success:
            active_jobs = [job.model_dump() for job in async_jobs_response.jobs]

            return {
                "success": True,
                "message": f"Found {len(active_jobs)} active training jobs",
                "data": {"count": len(active_jobs), "trainings": active_jobs},
            }

        # Fallback to legacy method
        active_trainings = training_service.active_trainings

        return {
            "success": True,
            "message": f"Found {len(active_trainings)} active training jobs",
            "data": {"count": len(active_trainings), "trainings": active_trainings},
        }

    except Exception as e:
        logger.error(f"Error getting active trainings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/queue", response_model=Dict[str, Any])
async def get_training_queue_info(
    training_service: AsyncTrainingService = Depends(get_initialized_training_service),
) -> Dict[str, Any]:
    """
    Get information about the training queue

    Returns current queue status including pending, running, and completed jobs.
    """
    try:
        queue_response = await training_service.get_queue_info()

        if queue_response.success:
            return {
                "success": True,
                "message": "Training queue information retrieved",
                "data": queue_response.queue_info.model_dump(),
            }
        else:
            raise HTTPException(status_code=500, detail=queue_response.error)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting queue info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/cancel/{training_id}", response_model=Dict[str, Any])
async def cancel_training_job(
    training_id: str,
    force: bool = Query(False, description="Force cancel even if running"),
    training_service: AsyncTrainingService = Depends(get_initialized_training_service),
) -> Dict[str, Any]:
    """
    Cancel a training job

    Cancels a pending or running training job. Use force=true to cancel running jobs.
    """
    try:
        from ..schemas.training_schemas import TrainingJobCancelRequest

        cancel_request = TrainingJobCancelRequest(
            force=force, reason="User cancellation"
        )

        cancel_response = await training_service.cancel_training_job(
            training_id, cancel_request
        )

        if cancel_response.success:
            logger.info(f"Training job {training_id} cancelled successfully")
            return {
                "success": True,
                "message": f"Training job {training_id} cancelled successfully",
                "data": {"training_id": training_id, "cancelled": True},
            }
        else:
            raise HTTPException(status_code=400, detail=cancel_response.error)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling training job {training_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def get_training_system_health(
    training_service: AsyncTrainingService = Depends(get_initialized_training_service),
) -> Dict[str, Any]:
    """
    Get health status of the training system

    Returns comprehensive health information including resource usage and system status.
    """
    try:
        health_response = await training_service.get_background_health()

        if health_response.success:
            return {
                "success": True,
                "message": "Training system health retrieved",
                "data": health_response.health.model_dump(),
            }
        else:
            raise HTTPException(status_code=500, detail=health_response.error)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training system health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
