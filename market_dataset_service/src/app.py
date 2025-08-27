# app.py

"""
FastAPI application for the backtesting system.

Provides RESTful endpoints for:
- Administrative operations (data management, system stats)
- Backtesting operations (run strategies, get results)
- Market data operations (fetch, query, manage OHLCV data)

Uses centralized configuration, logging, and error handling.
"""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from common.logger import LoggerFactory
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import Response

from .core.config import settings
from .interfaces.errors import (BacktestingServiceError, CollectionError,
                                RepositoryError, ValidationError)
from .routers import (admin_router, backtesting_router, market_data_job_router,
                      market_data_router, market_data_storage_router)
from .routers.eureka_router import router as eureka_router
from .utils.dependencies import get_eureka_client_service

# Initialize configuration and logging
logger = LoggerFactory.get_logger(name="fastapi_app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager with improved error handling and startup sequence.
    Includes both REST API and Eureka client initialization.
    """
    # Startup
    logger.info("🚀 Starting FinSight Market Dataset Service server")
    logger.info(f"📊 Environment: {settings.environment}")
    logger.info(f"🌐 FastAPI Host: {settings.host}:{settings.port}")
    logger.info(f"🔧 Debug mode: {settings.debug}")

    # Check if Eureka client is enabled
    eureka_enabled = getattr(settings, "enable_eureka_client", True)
    if eureka_enabled:
        logger.info(f"🔗 Eureka Client: {settings.eureka_server_url}")

    startup_errors = []

    try:
        # Step 1: Initialize core services
        logger.info("📋 Step 1: Initializing core services...")

        # Initialize dependency manager
        from .utils.dependencies import get_dependency_manager

        dependency_manager = get_dependency_manager()
        logger.info("✅ Dependency manager initialized successfully")

        # Step 2: Initialize Eureka client service
        if eureka_enabled:
            logger.info("📋 Step 2: Initializing Eureka client service...")
            try:
                eureka_service = get_eureka_client_service()
                success = await eureka_service.start()
                if success:
                    logger.info("✅ Eureka client service initialized successfully")
                else:
                    logger.warning("⚠️ Eureka client service initialization failed")
                    startup_errors.append("Eureka client service initialization failed")
            except Exception as eureka_error:
                logger.error(f"⚠️ Failed to start Eureka client service: {eureka_error}")
                startup_errors.append(
                    f"Eureka client service error: {str(eureka_error)}"
                )
        else:
            logger.info("🔄 Eureka client service is disabled")

        # Final startup validation
        if startup_errors:
            logger.warning(f"⚠️ Service started with warnings: {startup_errors}")
            logger.info("🔧 Service is operational but some components may be degraded")
        else:
            if eureka_enabled:
                logger.info(
                    "🎉 FinSight Market Dataset Service is fully operational (REST + Eureka)!"
                )
            else:
                logger.info(
                    "🎉 FinSight Market Dataset Service is fully operational (REST only)!"
                )

    except Exception as e:
        logger.error(f"❌ Critical startup error: {str(e)}")
        # Don't raise the error - let the service start in degraded mode
        logger.warning("🔧 Starting service in degraded mode due to startup errors")
        startup_errors.append(f"Critical error: {str(e)}")

    yield

    # Cleanup phase
    logger.info("🛑 Shutting down services...")
    try:
        # Stop Eureka client service first
        if eureka_enabled:
            eureka_service = get_eureka_client_service()
            if eureka_service.is_registered():
                await eureka_service.stop()
                logger.info("✅ Eureka client service stopped")

        # Cleanup dependency manager
        dependency_manager.shutdown()
        logger.info("✅ Dependency manager cleaned up successfully")

    except Exception as e:
        logger.error(f"❌ Error during cleanup: {str(e)}")

    logger.info("👋 FinSight Market Dataset Service shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="FinSight Market Dataset Service",
    description="""
    Advanced market dataset service for cryptocurrency trading strategies.

    ## Features

    * **Administrative Operations**: Data management, system statistics, maintenance
    * **Market Data Management**: Fetch, store, and convert OHLCV data across timeframes
    * **Strategy Backtesting**: Run and analyze trading strategies with multiple engines
    * **Cross-Repository Support**: MongoDB, CSV, InfluxDB storage adapters

    ## Authentication

    Admin endpoints require API key authentication via Bearer token.
    Set the `ADMIN_API_KEY` environment variable or use the default key.

    ## Architecture

    Built with Ports & Adapters (Hexagonal Architecture) for maximum flexibility:
    - **Service Layer**: Business logic and orchestration
    - **Repository Layer**: Pluggable storage adapters  
    - **Strategy Layer**: Extensible trading strategies
    - **Adapter Layer**: Exchange connectors and backtesting engines
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware log request và response."""
    start_time = time.time()

    # Đọc request body (phải clone lại vì body chỉ đọc 1 lần)
    request_body = await request.body()

    logger.debug(f"=== Incoming Request ===")
    logger.debug(f"Client: {request.client.host if request.client else 'unknown'}")
    logger.debug(f"{request.method} {request.url}")
    logger.debug(f"Query Params: {request.query_params}")
    logger.debug(f"Body: {request_body.decode('utf-8') if request_body else None}")

    # Gọi tiếp middleware/route
    response = await call_next(request)

    # Đọc response body (lưu lại vì response gốc là streaming)
    resp_body = b""
    async for chunk in response.body_iterator:
        resp_body += chunk

    process_time = (time.time() - start_time) * 1000

    logger.debug(f"=== Outgoing Response ===")
    logger.debug(f"Status code: {response.status_code}")
    logger.debug(f"Process time: {process_time:.2f} ms")
    logger.debug(f"Response Body: {resp_body.decode('utf-8') if resp_body else None}")

    # Tạo lại response mới với body cũ
    return Response(
        content=resp_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )


# Global exception handlers
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "type": "validation_error",
        },
    )


@app.exception_handler(RepositoryError)
async def repository_error_handler(request: Request, exc: RepositoryError):
    """Handle repository errors."""
    logger.error(f"Repository error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Repository Error",
            "detail": str(exc),
            "type": "repository_error",
        },
    )


@app.exception_handler(BacktestingServiceError)
async def backtesting_error_handler(request: Request, exc: BacktestingServiceError):
    """Handle backtesting service errors."""
    logger.error(f"Backtesting error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Backtesting Error",
            "detail": str(exc),
            "type": "backtesting_error",
        },
    )


@app.exception_handler(CollectionError)
async def collection_error_handler(request: Request, exc: CollectionError):
    """Handle data collection errors."""
    logger.error(f"Collection error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Collection Error",
            "detail": str(exc),
            "type": "collection_error",
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "type": "internal_error",
        },
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware log request và response."""
    start_time = time.time()

    # Đọc request body (phải clone lại vì body chỉ đọc 1 lần)
    request_body = await request.body()

    logger.info(f"=== Incoming Request ===")
    logger.info(f"Client: {request.client.host if request.client else 'unknown'}")
    logger.info(f"{request.method} {request.url}")
    logger.info(f"Query Params: {request.query_params}")
    logger.info(f"Body: {request_body.decode('utf-8') if request_body else None}")

    # Gọi tiếp middleware/route
    response = await call_next(request)

    # Đọc response body (lưu lại vì response gốc là streaming)
    resp_body = b""
    async for chunk in response.body_iterator:
        resp_body += chunk

    process_time = (time.time() - start_time) * 1000

    logger.info(f"=== Outgoing Response ===")
    logger.info(f"Status code: {response.status_code}")
    logger.info(f"Process time: {process_time:.2f} ms")
    logger.info(f"Response Body: {resp_body.decode('utf-8') if resp_body else None}")

    # Tạo lại response mới với body cũ
    return Response(
        content=resp_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )


# Root endpoint
@app.get("/", response_model=Dict[str, Any])
async def root():
    """
    API root endpoint with service information.
    """
    # Check if Eureka client is enabled
    eureka_enabled = getattr(settings, "enable_eureka_client", True)

    return {
        "service": "FinSight Backtesting API",
        "version": "1.0.0",
        "description": "Advanced backtesting system for cryptocurrency trading strategies",
        "status": "operational",
        "endpoints": {
            "documentation": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "health": "/admin/health",
            "admin": "/admin",
            "eureka": "/api/v1/eureka" if eureka_enabled else None,
        },
        "features": [
            "Multi-exchange data collection",
            "Cross-timeframe data conversion",
            "Multiple backtesting engines",
            "Extensible strategy framework",
            "Cross-repository data management",
            "Administrative operations",
            "Eureka service discovery" if eureka_enabled else None,
        ],
        "supported_exchanges": ["binance"],
        "supported_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
        "supported_strategies": [
            "moving_average_crossover",
            "rsi_strategy",
            "bollinger_bands",
            "macd_strategy",
            "simple_buy_hold",
        ],
        "eureka_client": {
            "enabled": eureka_enabled,
            "server_url": settings.eureka_server_url if eureka_enabled else None,
            "app_name": settings.eureka_app_name if eureka_enabled else None,
        },
    }


# Health check endpoint (public)
@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.

    This is a lightweight health check that doesn't require authentication.
    For detailed system health, use /admin/health.
    """
    # Add Eureka client status if enabled
    eureka_enabled = getattr(settings, "enable_eureka_client", True)
    dependencies = {}

    if eureka_enabled:
        try:
            eureka_service = get_eureka_client_service()
            eureka_status = (
                "healthy"
                if eureka_service and eureka_service.is_registered()
                else "stopped"
            )
            dependencies["eureka_client"] = eureka_status
        except Exception as e:
            dependencies["eureka_client"] = "error"
            logger.error(f"Error checking Eureka status: {e}")

    # Determine overall health status
    overall_status = "healthy"
    if any(status in ["stopped", "error"] for status in dependencies.values()):
        overall_status = "degraded"

    return {
        "status": overall_status,
        "timestamp": time.time(),
        "service": "backtesting-api",
        "dependencies": dependencies if dependencies else None,
    }


# Include routers
app.include_router(
    admin_router.router,
    prefix="/api/v1",
    tags=["admin"],
)

app.include_router(
    market_data_router.router,
    prefix="/api/v1",
    tags=["market-data"],
)

app.include_router(
    backtesting_router.router,
    prefix="/api/v1",
    tags=["backtesting"],
)

app.include_router(
    market_data_storage_router.router,
    prefix="/api/v1",
    tags=["market-data-storage"],
)

app.include_router(
    market_data_job_router.router,
    prefix="/api/v1",
    tags=["market-data-job-management"],
)

app.include_router(
    eureka_router,
    prefix="/api/v1",
    tags=["eureka-client-management"],
)


def create_app() -> FastAPI:
    """
    Application factory function.

    Returns:
        Configured FastAPI application instance
    """
    return app


if __name__ == "__main__":
    """
    Run the application with uvicorn.

    For production deployment, use a proper ASGI server like uvicorn
    with appropriate configuration for workers, SSL, etc.
    """
    logger.info("Starting development server")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info",
        access_log=True,
    )
