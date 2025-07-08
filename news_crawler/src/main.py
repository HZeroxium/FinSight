# main.py

"""
FastAPI application for the news crawler service.
Entry point for the News Crawler Service REST API.
"""

import sys
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.config import settings
from .routers import search, news
from .services.sentiment_consumer import SentimentConsumerService
from .utils.dependencies import (
    get_search_service,
    get_message_broker,
    get_news_service,
    initialize_services,
)
from .common.logger import LoggerFactory, LoggerType, LogLevel

# Setup application logger
logger = LoggerFactory.get_logger(
    name="news-crawler-main",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    console_level=LogLevel.INFO,
    use_colors=True,
    log_file=f"{settings.log_file_path}news_crawler_main.log",
)

# Global variable for sentiment consumer
sentiment_consumer: SentimentConsumerService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager with improved error handling and startup sequence.
    """
    global sentiment_consumer

    logger.info(f"🚀 Starting {settings.app_name} v1.0.0")
    logger.info(f"📊 Environment: {settings.environment}")
    logger.info(f"🌐 Host: {settings.host}:{settings.port}")

    startup_errors = []

    try:
        # Step 1: Initialize core services (MongoDB, basic dependencies)
        logger.info("📋 Step 1: Initializing core services...")
        await initialize_services()
        logger.info("✅ Core services initialized successfully")

        # Step 2: Wait a moment for services to stabilize
        await asyncio.sleep(0.5)

        # Step 3: Initialize sentiment consumer with improved error handling
        logger.info("📋 Step 2: Initializing sentiment consumer...")
        try:
            message_broker = get_message_broker()
            sentiment_consumer = SentimentConsumerService(message_broker=message_broker)

            # Start consuming in background task with error handling
            consumer_task = asyncio.create_task(sentiment_consumer.start_consuming())

            # Give consumer time to start properly
            await asyncio.sleep(1.0)

            if sentiment_consumer.is_running():
                logger.info("✅ Sentiment consumer started successfully")
            else:
                logger.warning("⚠️ Sentiment consumer failed to start properly")
                startup_errors.append("Sentiment consumer startup issues")

        except Exception as consumer_error:
            logger.error(f"⚠️ Failed to start sentiment consumer: {consumer_error}")
            startup_errors.append(f"Sentiment consumer error: {str(consumer_error)}")
            sentiment_consumer = None

        # Step 4: Health check for search service (with timeout and error handling)
        logger.info("📋 Step 3: Performing health checks...")
        try:
            search_service = get_search_service()

            # Add timeout to health check to prevent hanging
            is_healthy = await asyncio.wait_for(
                search_service.health_check(), timeout=10.0
            )

            if is_healthy:
                logger.info("✅ Search service health check passed")
            else:
                logger.warning("⚠️ Search service health check failed")
                startup_errors.append("Search service health issues")

        except asyncio.TimeoutError:
            logger.warning("⚠️ Search service health check timed out")
            startup_errors.append("Search service health check timeout")
        except Exception as health_error:
            logger.warning(f"⚠️ Search service health check error: {health_error}")
            startup_errors.append(f"Search service health error: {str(health_error)}")

        # Final startup validation
        if startup_errors:
            logger.warning(f"⚠️ Service started with warnings: {startup_errors}")
            logger.info("🔧 Service is operational but some components may be degraded")
        else:
            logger.info("🎉 News Crawler Service is fully operational!")

    except Exception as e:
        logger.error(f"❌ Critical startup error: {str(e)}")
        # Don't raise the error - let the service start in degraded mode
        logger.warning("🔧 Starting service in degraded mode due to startup errors")
        startup_errors.append(f"Critical error: {str(e)}")

    yield

    # Cleanup phase
    logger.info("🛑 Shutting down services...")
    try:
        if sentiment_consumer:
            await sentiment_consumer.stop_consuming()
            logger.info("✅ Sentiment consumer stopped")

        # Additional cleanup can be added here
        logger.info("✅ All services cleaned up successfully")

    except Exception as e:
        logger.error(f"❌ Error during cleanup: {str(e)}")

    logger.info(f"👋 {settings.app_name} shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="News Crawler Service",
    description="AI-powered news and content search service using Tavily with MongoDB storage",
    version="1.0.0",
    lifespan=lifespan,
    debug=settings.debug,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search.router)
app.include_router(news.router)


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": settings.app_name,
        "status": "running",
        "version": "1.0.0",
        "environment": settings.environment,
        "description": "AI-powered news crawler and sentiment analysis service",
        "docs_url": "/docs" if settings.debug else "disabled",
        "features": {
            "tavily_search": True,
            "sentiment_analysis": True,
            "mongodb_storage": True,
            "rabbitmq_messaging": True,
            "caching": settings.enable_caching,
        },
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    global sentiment_consumer

    health_status = {
        "status": "healthy",
        "service": settings.app_name,
        "version": "1.0.0",
        "timestamp": None,
        "components": {},
        "metrics": {},
    }

    try:
        # Check search service
        search_service = get_search_service()
        search_healthy = await search_service.health_check()
        health_status["components"]["search_service"] = (
            "healthy" if search_healthy else "unhealthy"
        )

        # Check sentiment consumer
        consumer_running = sentiment_consumer._running if sentiment_consumer else False
        health_status["components"]["sentiment_consumer"] = (
            "running" if consumer_running else "stopped"
        )

        # Check news service
        news_service = get_news_service()
        news_stats = await news_service.get_repository_stats()
        health_status["components"]["database"] = "healthy"
        health_status["metrics"]["total_articles"] = news_stats.get("total_articles", 0)

        # Overall health determination
        unhealthy_components = [
            k
            for k, v in health_status["components"].items()
            if v in ["unhealthy", "stopped"]
        ]

        if unhealthy_components:
            health_status["status"] = "degraded"
            health_status["issues"] = unhealthy_components

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": settings.app_name,
                "error": str(e),
                "timestamp": None,
            },
        )


@app.get("/metrics")
async def get_metrics():
    """Get service metrics and statistics."""
    global sentiment_consumer

    try:
        news_service = get_news_service()
        stats = await news_service.get_repository_stats()

        return {
            "service": settings.app_name,
            "version": "1.0.0",
            "environment": settings.environment,
            "consumer_running": (
                sentiment_consumer._running if sentiment_consumer else False
            ),
            "database_stats": stats,
            "configuration": {
                "max_concurrent_crawls": settings.max_concurrent_crawls,
                "cache_enabled": settings.enable_caching,
                "cache_ttl": settings.cache_ttl_seconds,
                "rate_limit": settings.rate_limit_requests_per_minute,
            },
        }

    except Exception as e:
        logger.error(f"Metrics retrieval failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve metrics", "detail": str(e)},
        )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler with proper logging."""
    logger.error(f"Unhandled exception on {request.method} {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again later.",
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
    )


def main():
    """Main function for running the FastAPI server."""
    try:
        logger.info(f"🚀 Starting FastAPI server on {settings.host}:{settings.port}")

        uvicorn.run(
            "src.main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_level=settings.log_level.lower(),
            access_log=settings.debug,
        )

    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start server: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
