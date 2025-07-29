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
from .routers import news_router, search
from .services.sentiment_consumer import SentimentConsumerService
from .grpc_services import create_grpc_server, GrpcServer
from .utils.dependencies import (
    get_search_service,
    get_message_broker,
    get_news_service,
    initialize_services,
)
from common.logger import LoggerFactory, LoggerType, LogLevel

# Setup application logger
logger = LoggerFactory.get_logger(
    name="news-crawler-main",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    console_level=LogLevel.INFO,
    use_colors=True,
    log_file=f"{settings.log_file_path}news_crawler_main.log",
)

# Global variables for services
sentiment_consumer: SentimentConsumerService = None
grpc_server: GrpcServer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager with improved error handling and startup sequence.
    Includes both REST API and gRPC server initialization.
    """
    global sentiment_consumer, grpc_server

    logger.info(f"🚀 Starting {settings.app_name} v1.0.0")
    logger.info(f"📊 Environment: {settings.environment}")
    logger.info(f"🌐 FastAPI Host: {settings.host}:{settings.port}")

    # Check if gRPC is enabled
    grpc_enabled = getattr(settings, "enable_grpc", True)
    grpc_port = getattr(settings, "grpc_port", 50051)

    if grpc_enabled:
        logger.info(f"🔌 gRPC Server: {settings.host}:{grpc_port}")

    startup_errors = []

    try:
        # Step 1: Initialize core services (MongoDB, basic dependencies)
        logger.info("📋 Step 1: Initializing core services...")
        await initialize_services()
        logger.info("✅ Core services initialized successfully")

        # Step 2: Wait a moment for services to stabilize
        await asyncio.sleep(0.5)

        # Step 3: Initialize gRPC server if enabled
        if grpc_enabled:
            logger.info("📋 Step 2: Initializing gRPC server...")
            try:
                news_service = get_news_service()
                grpc_server = await create_grpc_server(
                    news_service=news_service, host=settings.host, port=grpc_port
                )
                await grpc_server.start()
                logger.info("✅ gRPC server started successfully")
            except Exception as grpc_error:
                logger.error(f"⚠️ Failed to start gRPC server: {grpc_error}")
                startup_errors.append(f"gRPC server error: {str(grpc_error)}")
                grpc_server = None

        # Step 4: Initialize sentiment consumer with improved error handling
        logger.info("📋 Step 3: Initializing sentiment consumer...")
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

        # Step 5: Health check for search service (with timeout and error handling)
        logger.info("📋 Step 4: Performing health checks...")
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
            if grpc_enabled:
                logger.info(
                    "🎉 News Crawler Service is fully operational (REST + gRPC)!"
                )
            else:
                logger.info("🎉 News Crawler Service is fully operational (REST only)!")

    except Exception as e:
        logger.error(f"❌ Critical startup error: {str(e)}")
        # Don't raise the error - let the service start in degraded mode
        logger.warning("🔧 Starting service in degraded mode due to startup errors")
        startup_errors.append(f"Critical error: {str(e)}")

    yield

    # Cleanup phase
    logger.info("🛑 Shutting down services...")
    try:
        # Stop gRPC server first
        if grpc_server:
            await grpc_server.stop()
            logger.info("✅ gRPC server stopped")

        # Stop sentiment consumer
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
# app.include_router(search.router)
app.include_router(news_router.router)


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
    """Comprehensive health check endpoint including gRPC status."""
    global sentiment_consumer, grpc_server

    health_status = {
        "status": "healthy",
        "service": settings.app_name,
        "version": "1.0.0",
        "timestamp": None,
        "components": {},
        "metrics": {},
        "protocols": {
            "rest_api": {"enabled": True, "port": settings.port, "status": "healthy"}
        },
    }

    # Add gRPC status if enabled
    grpc_enabled = getattr(settings, "enable_grpc", True)
    if grpc_enabled:
        grpc_status = (
            "healthy" if grpc_server and grpc_server.is_running() else "stopped"
        )
        health_status["protocols"]["grpc"] = {
            "enabled": grpc_enabled,
            "port": getattr(settings, "grpc_port", 50051),
            "status": grpc_status,
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

        # Check if gRPC is supposed to be running but isn't
        if grpc_enabled and health_status["protocols"]["grpc"]["status"] == "stopped":
            unhealthy_components.append("grpc_server")

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
