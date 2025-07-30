# app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import uvicorn

from .core.config import get_settings
from .routers import prediction_router, training_router
from .routers.models import router as models_router
from .routers.serving import router as serving_router
from .schemas.base_schemas import HealthResponse
from common.logger import LoggerFactory


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger = LoggerFactory.get_logger("FinSightApp")
    settings = get_settings()

    logger.info("Starting FinSight Model Builder API")
    logger.info(f"Data directory: {settings.data_dir}")
    logger.info(f"Models directory: {settings.models_dir}")
    logger.info(f"Logs directory: {settings.logs_dir}")

    # Initialize async services
    try:
        logger.info("Initializing async training services...")
        from .routers.training import get_training_service

        training_service = get_training_service()
        await training_service.initialize()

        logger.info("✅ Async training services initialized successfully")

        # Initialize serving adapter
        logger.info("Initializing serving adapter...")
        from .models.model_facade import ModelFacade

        facade = ModelFacade()
        await facade.initialize_serving()
        logger.info("✅ Serving adapter initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize async services: {e}")
        # Don't raise - let the service start in degraded mode
        logger.warning("🔧 Starting API in degraded mode due to initialization errors")

    yield

    # Shutdown
    logger.info("Shutting down FinSight Model Builder API")

    # Cleanup async services
    try:
        logger.info("Cleaning up async services...")
        from .routers.training import get_training_service

        training_service = get_training_service()
        await training_service.shutdown()

        logger.info("✅ Async services cleaned up successfully")

    except Exception as e:
        logger.error(f"❌ Error during async services cleanup: {e}")


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
app.include_router(
    training_router
)  # Consolidated training endpoints (both legacy and async)
app.include_router(prediction_router)
app.include_router(models_router)
app.include_router(serving_router)  # Model serving management endpoints


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "training": "/training",  # Consolidated training endpoints
            "prediction": "/prediction",
            "models": "/models",
            "serving": "/serving",  # Model serving management
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
