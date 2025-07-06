# app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import uvicorn

from .core.config import get_settings
from .routers import training_router, prediction_router, models_router
from .schemas.base_schemas import HealthResponse
from .logger.logger_factory import LoggerFactory


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger = LoggerFactory.get_logger("FinSightApp")
    settings = get_settings()

    logger.info("Starting FinSight Model Builder API")
    logger.info(f"Data directory: {settings.data_dir}")
    logger.info(f"Models directory: {settings.models_dir}")
    logger.info(f"Logs directory: {settings.logs_dir}")

    yield

    # Shutdown
    logger.info("Shutting down FinSight Model Builder API")


# Create FastAPI app
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Time Series Model Training and Prediction API for Financial Data",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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
app.include_router(training_router)
app.include_router(prediction_router)
app.include_router(models_router)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "training": "/training",
            "prediction": "/prediction",
            "models": "/models",
            "health": "/health",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    settings = get_settings()

    dependencies = {
        "data_dir": "available" if settings.data_dir.exists() else "missing",
        "models_dir": "available" if settings.models_dir.exists() else "missing",
        "logs_dir": "available" if settings.logs_dir.exists() else "missing",
    }

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version=settings.app_version,
        dependencies=dependencies,
    )


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app:app", host=settings.api_host, port=settings.api_port, reload=settings.debug
    )
