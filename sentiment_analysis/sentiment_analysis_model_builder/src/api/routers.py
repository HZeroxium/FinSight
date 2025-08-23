# api/routers.py

"""FastAPI routers for sentiment analysis API."""

import uuid
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from loguru import logger

from ..services.inference_service import SentimentInferenceService, InferenceError
from ..schemas.api_schemas import (
    SentimentRequest,
    SentimentResponse,
    BatchSentimentRequest,
    BatchSentimentResponse,
    ModelInfo,
    HealthStatus,
    ErrorResponse,
    MetricsResponse,
    ResponseStatus,
)
from ..core.enums import APIEndpoint
from ..utils.dependencies import get_inference_service, get_metrics_tracker


# Create router
router = APIRouter(
    prefix="/api/v1",
    tags=["Sentiment Analysis"],
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
    },
)


@router.get(
    APIEndpoint.HEALTH.value,
    response_model=HealthStatus,
    summary="Health Check",
    description="Check the health status of the sentiment analysis service",
)
async def health_check(
    inference_service: SentimentInferenceService = Depends(get_inference_service),
    metrics_tracker=Depends(get_metrics_tracker),
) -> HealthStatus:
    """Health check endpoint."""
    try:
        uptime = metrics_tracker.get_uptime()
        memory_usage = metrics_tracker.get_memory_usage()

        return HealthStatus(
            status="healthy" if inference_service.is_ready() else "not_ready",
            model_loaded=inference_service.is_ready(),
            device=inference_service.device,
            uptime_seconds=uptime,
            memory_usage_mb=memory_usage,
            version="1.0.0",
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}",
        )


@router.post(
    APIEndpoint.PREDICT.value,
    response_model=SentimentResponse,
    summary="Analyze Sentiment",
    description="Analyze sentiment of a single text using the fine-tuned FinBERT model",
    response_description="Sentiment analysis result with confidence scores",
)
async def predict_sentiment(
    request: SentimentRequest,
    inference_service: SentimentInferenceService = Depends(get_inference_service),
    metrics_tracker=Depends(get_metrics_tracker),
) -> SentimentResponse:
    """Predict sentiment for a single text."""
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Processing sentiment request {request_id}")
        metrics_tracker.increment_requests()

        # Perform inference
        result = await inference_service.predict_sentiment(request.text)

        metrics_tracker.increment_successful_requests()
        metrics_tracker.add_processing_time(result.processing_time_ms or 0)

        response = SentimentResponse(
            status=ResponseStatus.SUCCESS,
            result=result,
            request_id=request_id,
        )

        logger.info(
            f"Request {request_id} completed successfully: "
            f"{result.label.value} (confidence: {result.confidence})"
        )

        return response

    except InferenceError as e:
        logger.error(f"Inference error for request {request_id}: {e}")
        metrics_tracker.increment_failed_requests()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error for request {request_id}: {e}")
        metrics_tracker.increment_failed_requests()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post(
    APIEndpoint.BATCH_PREDICT.value,
    response_model=BatchSentimentResponse,
    summary="Batch Analyze Sentiment",
    description="Analyze sentiment of multiple texts in a single request",
    response_description="Batch sentiment analysis results",
)
async def predict_batch_sentiment(
    request: BatchSentimentRequest,
    inference_service: SentimentInferenceService = Depends(get_inference_service),
    metrics_tracker=Depends(get_metrics_tracker),
) -> BatchSentimentResponse:
    """Predict sentiment for multiple texts."""
    request_id = str(uuid.uuid4())

    try:
        logger.info(
            f"Processing batch sentiment request {request_id} with {len(request.texts)} texts"
        )
        metrics_tracker.increment_requests()

        start_time = time.time()

        # Perform batch inference
        logger.info(f"Batch request {request_id} texts: {request.texts}")
        results = await inference_service.predict_batch(request.texts)

        total_processing_time = (time.time() - start_time) * 1000

        metrics_tracker.increment_successful_requests()
        metrics_tracker.add_processing_time(total_processing_time)

        response = BatchSentimentResponse(
            status=ResponseStatus.SUCCESS,
            results=results,
            total_processed=len(results),
            total_processing_time_ms=round(total_processing_time, 2),
            request_id=request_id,
        )

        logger.info(
            f"Batch request {request_id} completed successfully: "
            f"{len(results)} texts processed in {total_processing_time:.2f}ms"
        )

        return response

    except InferenceError as e:
        logger.error(f"Batch inference error for request {request_id}: {e}")
        metrics_tracker.increment_failed_requests()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch sentiment analysis failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error for batch request {request_id}: {e}")
        metrics_tracker.increment_failed_requests()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    APIEndpoint.MODEL_INFO.value,
    response_model=ModelInfo,
    summary="Get Model Information",
    description="Get detailed information about the loaded sentiment analysis model",
)
async def get_model_info(
    inference_service: SentimentInferenceService = Depends(get_inference_service),
) -> ModelInfo:
    """Get model information."""
    try:
        if not inference_service.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )

        model_info = inference_service.get_model_info()

        return ModelInfo(
            model_name=model_info.get("model_name", "FinBERT-Sentiment"),
            model_version="1.0.0",
            backbone="ProsusAI/finbert",
            num_labels=model_info.get("num_labels", 3),
            max_sequence_length=model_info.get("max_sequence_length", 512),
            labels=model_info.get("labels", ["NEGATIVE", "NEUTRAL", "POSITIVE"]),
            device=model_info.get("device", "cpu"),
            model_size_mb=model_info.get("model_size_mb"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information",
        )


@router.get(
    APIEndpoint.METRICS.value,
    response_model=MetricsResponse,
    summary="Get API Metrics",
    description="Get performance and usage metrics for the API",
)
async def get_metrics(
    metrics_tracker=Depends(get_metrics_tracker),
) -> MetricsResponse:
    """Get API metrics."""
    try:
        stats = metrics_tracker.get_stats()

        return MetricsResponse(
            total_requests=stats["total_requests"],
            successful_requests=stats["successful_requests"],
            failed_requests=stats["failed_requests"],
            average_processing_time_ms=stats["average_processing_time_ms"],
            uptime_seconds=stats["uptime_seconds"],
            memory_usage_mb=stats.get("memory_usage_mb"),
        )

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics",
        )
