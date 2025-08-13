# routers/prediction.py

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from ..services.prediction_service import PredictionService
from ..schemas.model_schemas import PredictionRequest, PredictionResponse
from common.logger.logger_factory import LoggerFactory

router = APIRouter(prefix="/prediction", tags=["prediction"])
prediction_service = PredictionService()
logger = LoggerFactory.get_logger("PredictionRouter")


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Make predictions using a trained model

    This endpoint uses a trained model to make predictions for the specified
    symbol and timeframe. It automatically selects the best available model
    if no specific model type is provided.
    """
    try:
        logger.info(
            f"Received prediction request for {request.symbol} {request.timeframe} ({request.n_steps} steps)"
        )

        response = await prediction_service.predict(request)

        if response.success:
            logger.info(f"Prediction completed successfully")
        else:
            logger.warning(f"Prediction failed: {response.error}")

        return response

    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/models", response_model=Dict[str, Any])
async def get_available_models() -> Dict[str, Any]:
    """
    Get information about available trained models

    Returns a comprehensive list of all trained models with their metadata,
    organized by symbol and timeframe.
    """
    try:
        models_info = prediction_service.get_available_models()

        return {
            "success": True,
            "message": f"Found {len(models_info)} trained models",
            "data": {
                "models": models_info,
                "total_count": len(models_info),
            },
        }

    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
