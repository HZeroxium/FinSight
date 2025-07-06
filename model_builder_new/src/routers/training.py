# routers/training.py

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any

from ..services.training_service import TrainingService
from ..schemas.model_schemas import TrainingRequest, TrainingResponse
from ..schemas.base_schemas import BaseResponse
from ..logger.logger_factory import LoggerFactory

router = APIRouter(prefix="/training", tags=["training"])
training_service = TrainingService()
logger = LoggerFactory.get_logger("TrainingRouter")


@router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest) -> TrainingResponse:
    """
    Train a time series model

    This endpoint trains a model based on the provided parameters.
    It validates that the required dataset exists and the model type is supported.
    """
    try:
        logger.info(
            f"Received training request for {request.symbol} {request.timeframe} {request.model_type}"
        )

        # Start training (synchronous for now, could be made async with background tasks)
        response = training_service.start_training(request)

        if response.success:
            logger.info(f"Training completed successfully: {response.training_id}")
        else:
            logger.warning(f"Training failed: {response.error}")

        return response

    except Exception as e:
        logger.error(f"Training endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/status/{training_id}", response_model=Dict[str, Any])
async def get_training_status(training_id: str) -> Dict[str, Any]:
    """
    Get status of a training job

    Returns the current status, progress, and metrics of a training job.
    """
    try:
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
async def get_active_trainings() -> Dict[str, Any]:
    """
    Get all active training jobs

    Returns a list of all currently active training jobs with their status.
    """
    try:
        active_trainings = training_service.active_trainings

        return {
            "success": True,
            "message": f"Found {len(active_trainings)} active training jobs",
            "data": {"count": len(active_trainings), "trainings": active_trainings},
        }

    except Exception as e:
        logger.error(f"Error getting active trainings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
