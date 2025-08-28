# routers/prediction.py

from typing import Any, Dict

from common.logger.logger_factory import LoggerFactory
from fastapi import APIRouter, HTTPException

from ..schemas.model_schemas import PredictionRequest, PredictionResponse
from ..services.prediction_service import PredictionService
from ..schemas.enums import CryptoSymbol, ModelType, TimeFrame

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


@router.get("/predict/info", response_model=Dict[str, Any])
async def get_predict_info() -> Dict[str, Any]:
    """
    Get available options for POST /prediction/predict
    """
    try:
        data = {
            "symbols": [s.value for s in CryptoSymbol],
            # "timeframes": [t.value for t in TimeFrame],
            "timeframes": ["1h", "1d"],
            "model_types": [m.value for m in ModelType],
            "request_defaults": {
                "symbol": CryptoSymbol.BTCUSDT.value,
                "timeframe": TimeFrame.DAY_1.value,
                "model_type": None,
                "n_steps": 1,
                "enable_fallback": True,
            },
            "constraints": {
                "n_steps": {"min": 1, "max": 100},
            },
        }
        return {
            "success": True,
            "message": "Predict request options",
            "data": data,
        }
    except Exception as e:
        logger.error(f"Error getting predict info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
