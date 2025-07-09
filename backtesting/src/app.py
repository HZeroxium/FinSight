# app.py

"""
FastAPI application for the backtesting system.

Provides RESTful endpoints for:
- Administrative operations (data management, system stats)
- Backtesting operations (run strategies, get results)
- Market data operations (fetch, query, manage OHLCV data)

Uses centralized configuration, logging, and error handling.
"""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, Any
import time

from .routers import admin_router
from .routers import market_data_router, backtesting_router
from .core.config import settings
from common.logger import LoggerFactory
from .interfaces.errors import (
    ValidationError,
    RepositoryError,
    CollectionError,
    BacktestingServiceError,
)


# Initialize configuration and logging
logger = LoggerFactory.get_logger(name="fastapi_app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("Starting FinSight Backtesting API server")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")

    yield

    # Shutdown
    logger.info("Shutting down FinSight Backtesting API server")


# Create FastAPI application
app = FastAPI(
    title="FinSight Backtesting API",
    description="""
    Advanced backtesting system for cryptocurrency trading strategies.
    
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
    """Log all HTTP requests with timing information."""
    start_time = time.time()

    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )

    # Process request
    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"Response: {response.status_code} "
        f"({process_time:.3f}s) for {request.method} {request.url.path}"
    )

    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)

    return response


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


# Root endpoint
@app.get("/", response_model=Dict[str, Any])
async def root():
    """
    API root endpoint with service information.
    """
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
        },
        "features": [
            "Multi-exchange data collection",
            "Cross-timeframe data conversion",
            "Multiple backtesting engines",
            "Extensible strategy framework",
            "Cross-repository data management",
            "Administrative operations",
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
    }


# Health check endpoint (public)
@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.

    This is a lightweight health check that doesn't require authentication.
    For detailed system health, use /admin/health.
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "backtesting-api",
    }


# Include routers
app.include_router(
    admin_router.router,
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
