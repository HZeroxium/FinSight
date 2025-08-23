# api/app.py

"""FastAPI application for sentiment analysis API."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn
from loguru import logger

from ..core.config import Config
from ..schemas.api_schemas import ErrorResponse, ResponseStatus
from ..utils.dependencies import get_config, cleanup_dependencies
from .routers import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting FinBERT Sentiment Analysis API...")

    try:
        # Initialize services during startup
        config = get_config()
        logger.info(f"Configuration loaded: {config.api.title}")

        # Pre-load the inference service
        from ..utils.dependencies import get_inference_service

        inference_service = await get_inference_service()
        logger.info("Inference service initialized successfully")

        yield

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    finally:
        logger.info("Shutting down FinBERT Sentiment Analysis API...")
        try:
            await cleanup_dependencies()
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config()
    api_config = config.api

    app = FastAPI(
        title=api_config.title,
        description=api_config.description,
        version=api_config.version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_config.allow_origins,
        allow_credentials=api_config.allow_credentials,
        allow_methods=api_config.allow_methods,
        allow_headers=api_config.allow_headers,
    )

    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all incoming requests."""
        start_time = datetime.utcnow()

        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )

        # Process request
        response = await call_next(request)

        # Log response
        process_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(f"Response: {response.status_code} " f"({process_time:.2f}ms)")

        return response

    # Include routers
    app.include_router(router)

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with API information."""
        return {
            "service": api_config.title,
            "version": api_config.version,
            "description": api_config.description,
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "docs_url": "/docs",
            "health_url": "/api/v1/health",
        }

    # Global exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """Handle validation errors."""
        logger.error(f"Validation error: {exc}")

        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                status=ResponseStatus.ERROR,
                error="Validation error",
                error_code="VALIDATION_ERROR",
                details={"errors": exc.errors()},
            ).model_dump(),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions."""
        logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")

        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                status=ResponseStatus.ERROR,
                error=exc.detail,
                error_code=f"HTTP_{exc.status_code}",
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error(f"Unexpected error: {exc}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                status=ResponseStatus.ERROR,
                error="Internal server error",
                error_code="INTERNAL_ERROR",
            ).model_dump(),
        )

    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    # Run the application
    config = get_config()
    api_config = config.api

    uvicorn.run(
        "src.api.app:app",
        host=api_config.host,
        port=api_config.port,
        reload=api_config.reload,
        debug=api_config.debug,
        workers=1,  # Always use 1 worker for model loading
        log_level="info",
        access_log=True,
    )
