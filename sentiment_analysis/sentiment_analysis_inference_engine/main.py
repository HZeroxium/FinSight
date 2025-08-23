# main.py
"""
Sentiment Analysis Inference Engine

Automated Triton Server deployment and REST API for FinBERT sentiment analysis.
"""

import time
from datetime import datetime
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import InferenceConfig
from src.core.enums import ServerStatus, ModelStatus
from src.services.triton_manager import TritonServerManager, TritonServerError
from src.services.sentiment_service import (
    SentimentAnalysisService,
    SentimentAnalysisError,
)
from src.models.schemas import (
    SentimentRequest,
    BatchSentimentRequest,
    SentimentResult,
    BatchSentimentResult,
    HealthStatus,
    ModelInfo,
    MetricsResponse,
    ErrorResponse,
    ServerInfo,
)
from common.logger.logger_factory import LoggerFactory, LoggerType
from common.logger.logger_interface import LogLevel

# Global variables
config: InferenceConfig
triton_manager: TritonServerManager
sentiment_service: SentimentAnalysisService
app_startup_time: float
logger = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global config, triton_manager, sentiment_service, app_startup_time, logger

    # Startup
    try:
        # Load configuration
        config = InferenceConfig()

        # Setup logging
        logger = LoggerFactory.get_logger(
            name="sentiment_inference_engine",
            logger_type=LoggerType.STANDARD,
            console_level=LogLevel(config.api.log_level.value),
            log_file=str(config.log_dir / "inference_engine.log"),
        )

        logger.info("Starting FinSight Sentiment Analysis Inference Engine...")
        app_startup_time = time.time()

        # Initialize Triton server manager
        triton_manager = TritonServerManager(config.triton)

        # Start Triton server
        await triton_manager.start_server()

        # Initialize sentiment analysis service
        sentiment_service = SentimentAnalysisService(
            config.sentiment,
            triton_host=config.triton.host,
            triton_port=config.triton.http_port,
        )
        await sentiment_service.initialize()

        logger.info("All services started successfully")

        yield

    except Exception as e:
        if logger:
            logger.error(f"Failed to start services: {e}")
        raise

    # Shutdown
    finally:
        if logger:
            logger.info("Shutting down services...")

        try:
            if "sentiment_service" in globals() and sentiment_service:
                await sentiment_service.cleanup()
        except Exception as e:
            if logger:
                logger.warning(f"Error stopping sentiment service: {e}")

        try:
            if "triton_manager" in globals() and triton_manager:
                await triton_manager.stop_server()
        except Exception as e:
            if logger:
                logger.warning(f"Error stopping Triton server: {e}")

        if logger:
            logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="FinSight Sentiment Analysis Engine",
    description="Automated sentiment analysis using fine-tuned FinBERT model with Triton Inference Server",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    if logger:
        logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(TritonServerError)
async def triton_exception_handler(request, exc):
    """Handle Triton server errors."""
    if logger:
        logger.error(f"Triton server error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error="TritonServerError", message="Triton server error", detail=str(exc)
        ).model_dump(),
    )


@app.exception_handler(SentimentAnalysisError)
async def sentiment_exception_handler(request, exc):
    """Handle sentiment analysis errors."""
    if logger:
        logger.error(f"Sentiment analysis error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="SentimentAnalysisError",
            message="Sentiment analysis failed",
            detail=str(exc),
        ).model_dump(),
    )


# API Routes


@app.get("/", response_model=ServerInfo)
async def root():
    """Get server information."""
    # Ensure startup_time is a datetime (not float)
    return ServerInfo(
        model_name=config.sentiment.model_name,
        startup_time=datetime.fromtimestamp(app_startup_time),
    )


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint."""
    try:
        # Check Triton server health
        triton_health = await triton_manager.get_server_health()
        triton_status = (
            ServerStatus.RUNNING
            if triton_health["status"] == "healthy"
            else ServerStatus.ERROR
        )

        # Check model status
        model_status = await triton_manager.get_model_status(
            config.sentiment.model_name
        )

        # Overall status
        overall_status = (
            "healthy"
            if triton_status == ServerStatus.RUNNING
            and model_status == ModelStatus.READY
            else "unhealthy"
        )

        return HealthStatus(
            status=overall_status,
            triton_status=triton_status,
            model_status=model_status,
            uptime_seconds=triton_manager.uptime_seconds,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthStatus(
            status="unhealthy",
            triton_status=ServerStatus.ERROR,
            model_status=ModelStatus.UNAVAILABLE,
            uptime_seconds=0,
        )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    try:
        model_info_data = await triton_manager.get_model_info(
            config.sentiment.model_name
        )

        if not model_info_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model information not available",
            )

        return ModelInfo(
            name=model_info_data.get("name", config.sentiment.model_name),
            version=model_info_data.get("versions", ["1"])[0],
            platform=model_info_data.get("platform", "onnxruntime_onnx"),
            status=await triton_manager.get_model_status(config.sentiment.model_name),
            inputs=model_info_data.get("inputs", []),
            outputs=model_info_data.get("outputs", []),
            max_batch_size=model_info_data.get(
                "max_batch_size", config.sentiment.max_batch_size
            ),
        )

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to retrieve model information",
        )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get service metrics."""
    try:
        stats = sentiment_service.get_statistics()

        return MetricsResponse(
            total_requests=stats["total_requests"],
            successful_requests=stats["successful_requests"],
            failed_requests=stats["failed_requests"],
            average_processing_time_ms=stats["average_processing_time_ms"],
            uptime_seconds=triton_manager.uptime_seconds,
        )

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics",
        )


@app.post("/predict", response_model=SentimentResult)
async def predict_sentiment(request: SentimentRequest):
    """Analyze sentiment for a single text."""
    try:
        logger.info(
            f"Processing sentiment analysis request for text: {request.text[:100]}..."
        )

        result = await sentiment_service.analyze_sentiment(request.text)

        logger.info(
            f"Sentiment analysis completed: {result.label} (confidence: {result.confidence:.3f})"
        )
        return result

    except SentimentAnalysisError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in sentiment prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction",
        )


@app.post("/predict/batch", response_model=BatchSentimentResult)
async def predict_batch_sentiment(request: BatchSentimentRequest):
    """Analyze sentiment for multiple texts."""
    try:
        start_time = time.time()

        logger.info(
            f"Processing batch sentiment analysis for {len(request.texts)} texts..."
        )

        results = await sentiment_service.analyze_batch(request.texts)

        total_processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Batch sentiment analysis completed: {len(results)} results in {total_processing_time:.2f}ms"
        )

        return BatchSentimentResult(
            results=results,
            total_processing_time_ms=total_processing_time,
            batch_size=len(results),
        )

    except SentimentAnalysisError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in batch sentiment prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during batch prediction",
        )


# Admin endpoints


@app.post("/admin/restart")
async def restart_triton_server():
    """Restart Triton server (admin endpoint)."""
    try:
        logger.info("Restarting Triton server...")
        await triton_manager.restart_server()

        # Reinitialize sentiment service
        await sentiment_service.cleanup()
        await sentiment_service.initialize()

        logger.info("Triton server restarted successfully")
        return {"message": "Triton server restarted successfully"}

    except Exception as e:
        logger.error(f"Failed to restart Triton server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restart server: {e}",
        )


@app.post("/admin/reset-metrics")
async def reset_metrics():
    """Reset service metrics (admin endpoint)."""
    try:
        sentiment_service.reset_statistics()
        logger.info("Service metrics reset")
        return {"message": "Metrics reset successfully"}

    except Exception as e:
        logger.error(f"Failed to reset metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset metrics: {e}",
        )


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False, log_level="info")
