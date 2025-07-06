# routers/models.py

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional

from ..services.model_service import ModelService
from ..schemas.base_schemas import ModelInfoResponse, BaseResponse
from ..schemas.enums import ModelType, TimeFrame
from ..logger.logger_factory import LoggerFactory

router = APIRouter(prefix="/models", tags=["models"])
model_service = ModelService()
logger = LoggerFactory.get_logger("ModelsRouter")


@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info() -> ModelInfoResponse:
    """
    Get comprehensive information about available and trained models

    Returns information about:
    - Available model types that can be trained
    - Currently trained models with their metadata
    - Supported timeframes and symbols
    """
    try:
        logger.info("Received request for model information")

        model_info = model_service.get_model_info()

        logger.info(
            f"Returning model info with {len(model_info.trained_models)} trained models"
        )
        return model_info

    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/available", response_model=Dict[str, Any])
async def get_available_models() -> Dict[str, Any]:
    """
    Get list of available model types that can be trained
    """
    try:
        from ..models.model_factory import ModelFactory

        available_models = [
            model_type.value for model_type in ModelFactory.get_supported_models()
        ]

        return {
            "success": True,
            "message": f"Found {len(available_models)} available model types",
            "data": {
                "available_models": available_models,
                "count": len(available_models),
            },
        }

    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/trained", response_model=Dict[str, Any])
async def get_trained_models() -> Dict[str, Any]:
    """
    Get information about all trained models
    """
    try:
        trained_models = model_service._scan_trained_models()

        return {
            "success": True,
            "message": f"Found {len(trained_models)} trained models",
            "data": {"trained_models": trained_models, "count": len(trained_models)},
        }

    except Exception as e:
        logger.error(f"Error getting trained models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/check", response_model=Dict[str, Any])
async def check_model_exists(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: TimeFrame = Query(..., description="Data timeframe"),
    model_type: Optional[ModelType] = Query(None, description="Specific model type"),
) -> Dict[str, Any]:
    """
    Check if a trained model exists

    Checks whether a trained model exists for the specified symbol, timeframe,
    and optionally model type.
    """
    try:
        logger.info(f"Checking model existence for {symbol} {timeframe} {model_type}")

        exists = model_service.check_model_exists(symbol, timeframe, model_type)

        return {
            "success": True,
            "message": f"Model {'exists' if exists else 'does not exist'}",
            "data": {
                "exists": exists,
                "symbol": symbol,
                "timeframe": timeframe,
                "model_type": model_type,
            },
        }

    except Exception as e:
        logger.error(f"Error checking model existence: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/details", response_model=Dict[str, Any])
async def get_model_details(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: TimeFrame = Query(..., description="Data timeframe"),
    model_type: ModelType = Query(..., description="Model type"),
) -> Dict[str, Any]:
    """
    Get detailed information about a specific model

    Returns comprehensive details about a trained model including
    configuration, training metrics, and file information.
    """
    try:
        logger.info(f"Getting model details for {symbol} {timeframe} {model_type}")

        details = model_service.get_model_details(symbol, timeframe, model_type)

        if details is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found for {symbol} {timeframe} {model_type}",
            )

        return {"success": True, "message": "Model details retrieved", "data": details}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/delete", response_model=BaseResponse)
async def delete_model(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: TimeFrame = Query(..., description="Data timeframe"),
    model_type: ModelType = Query(..., description="Model type"),
) -> BaseResponse:
    """
    Delete a trained model

    Permanently removes a trained model and all associated files.
    This action cannot be undone.
    """
    try:
        logger.info(f"Deleting model {symbol} {timeframe} {model_type}")

        success = model_service.delete_model(symbol, timeframe, model_type)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found for {symbol} {timeframe} {model_type}",
            )

        return BaseResponse(
            success=True,
            message=f"Model {symbol} {timeframe} {model_type} deleted successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/performance", response_model=Dict[str, Any])
async def get_model_performance(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: TimeFrame = Query(..., description="Data timeframe"),
    model_type: ModelType = Query(..., description="Model type"),
) -> Dict[str, Any]:
    """
    Get performance metrics for a specific model

    Returns training and validation performance metrics for the specified model.
    """
    try:
        logger.info(
            f"Getting performance metrics for {symbol} {timeframe} {model_type}"
        )

        performance = model_service.get_model_performance(symbol, timeframe, model_type)

        if performance is None:
            raise HTTPException(
                status_code=404,
                detail=f"Performance data not found for {symbol} {timeframe} {model_type}",
            )

        return {
            "success": True,
            "message": "Performance metrics retrieved",
            "data": performance,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
