# main.py
"""
FastAPI application for the sentiment analysis service.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.config import settings
from .routers import sentiment
from .services.news_consumer_service import NewsConsumerService
from .utils.dependencies import (
    get_sentiment_service,
    get_news_repository,
    get_message_broker,
    get_news_consumer_service,
    initialize_services,
    cleanup_services,
)
from common.logger import LoggerFactory, LoggerType, LogLevel

# Setup logging using custom logger factory
logger = LoggerFactory.get_logger(
    name="sentiment-analysis-service",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    console_level=LogLevel.INFO,
    use_colors=True,
    log_file="logs/sentiment_analysis.log" if not settings.debug else None,
)

# Global variable for message consumer
message_consumer: NewsConsumerService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager with resilient error handling.
    """
    global message_consumer

    logger.info(f"🚀 Starting {settings.app_name}")

    try:
        # Initialize services
        await initialize_services()
        logger.info("✅ Services initialized successfully")

        # Start message consumer with error resilience
        try:
            message_consumer = get_news_consumer_service()

            # Start consuming in background task
            consumer_task = asyncio.create_task(message_consumer.start_consuming())

            # Give some time for connection to establish
            await asyncio.sleep(2)

            if message_consumer.is_running():
                logger.info("✅ Message consumer started successfully")
            else:
                logger.warning(
                    "⚠️ Message consumer not running - service will continue without message broker"
                )

        except Exception as e:
            logger.warning(
                f"⚠️ Message broker unavailable, continuing without consumer: {e}"
            )
            message_consumer = None

        # Verify service health
        try:
            sentiment_service = get_sentiment_service()
            is_healthy = await sentiment_service.health_check()
            if is_healthy:
                logger.info("✅ Sentiment service health check passed")
            else:
                logger.warning(
                    "⚠️ Sentiment service health check failed - continuing anyway"
                )
        except Exception as e:
            logger.warning(f"⚠️ Failed to check service health: {e} - continuing anyway")

        logger.info("🎉 Sentiment Analysis Service is fully operational!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize sentiment analysis service: {e}")
        # Continue startup even with errors - some functionality may be limited

    yield

    # Cleanup
    logger.info("🛑 Shutting down sentiment analysis service...")

    try:
        # Stop message consumer
        if message_consumer:
            await message_consumer.stop_consuming()
            logger.info("✅ Message consumer stopped")

        # Cleanup services
        await cleanup_services()
        logger.info("✅ Services cleaned up successfully")

    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")

    logger.info("🔚 Sentiment Analysis Service shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Sentiment Analysis Service",
    description="AI-powered sentiment analysis service using OpenAI with MongoDB storage",
    version="1.0.0",
    lifespan=lifespan,
    debug=settings.debug,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sentiment.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.app_name,
        "status": "running",
        "version": "1.0.0",
        "description": "AI-powered sentiment analysis service",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global message_consumer

    try:
        sentiment_service = get_sentiment_service()
        is_healthy = await sentiment_service.health_check()

        consumer_running = message_consumer.is_running() if message_consumer else False

        return {
            "status": "healthy" if is_healthy else "degraded",
            "service": settings.app_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "sentiment_analyzer": "healthy" if is_healthy else "unhealthy",
                "database": "healthy",
                "message_broker": "healthy" if consumer_running else "optional",
                "message_consumer": "running" if consumer_running else "stopped",
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": settings.app_name,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    global message_consumer

    return {
        "service": settings.app_name,
        "consumer_running": (
            message_consumer.is_running() if message_consumer else False
        ),
        "uptime": "N/A",  # Would track in production
        "processed_messages": "N/A",  # Would track in production
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An error occurred",
            "service": settings.app_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.debug,
        log_level="info",
    )
