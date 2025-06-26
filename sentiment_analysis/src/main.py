# sentiment_analysis/src/main.py
"""
FastAPI application for the sentiment analysis service.
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.config import settings
from .routers import sentiment
from .services.message_consumer import MessageConsumerService
from .utils.dependencies import (
    get_sentiment_service,
    get_sentiment_repository,
    get_message_broker,
)
from .common.logger import LoggerFactory, LoggerType, LogLevel

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
message_consumer: MessageConsumerService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    """
    global message_consumer

    logger.info(f"Starting {settings.app_name}")

    # Initialize services
    sentiment_service = get_sentiment_service()
    sentiment_repository = get_sentiment_repository()
    message_broker = get_message_broker()

    # Initialize database indexes
    try:
        await sentiment_repository.initialize()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")

    # Initialize message consumer
    try:
        message_consumer = MessageConsumerService(
            message_broker=message_broker,
            sentiment_service=sentiment_service,
        )

        # Start consuming in background task
        asyncio.create_task(message_consumer.start_consuming())
        logger.info("Message consumer started")

    except Exception as e:
        logger.error(f"Failed to start message consumer: {str(e)}")

    # Verify service health
    try:
        is_healthy = await sentiment_service.health_check()
        if not is_healthy:
            logger.warning("Sentiment service health check failed during startup")
        else:
            logger.info("Sentiment service health check passed")
    except Exception as e:
        logger.error(f"Failed to check service health: {str(e)}")

    yield

    # Cleanup
    try:
        if message_consumer:
            await message_consumer.stop_consuming()
        await sentiment_repository.close()
        logger.info("All services cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

    logger.info(f"Shutting down {settings.app_name}")


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
            "status": "healthy" if is_healthy and consumer_running else "degraded",
            "service": settings.app_name,
            "components": {
                "sentiment_analyzer": "healthy" if is_healthy else "unhealthy",
                "database": "healthy",  # Would check in production
                "message_broker": "healthy" if consumer_running else "unhealthy",
                "message_consumer": "running" if consumer_running else "stopped",
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": settings.app_name,
                "error": str(e),
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
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again later.",
        },
    )
