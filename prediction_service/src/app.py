# app.py

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import time
from contextlib import asynccontextmanager
from datetime import datetime
import uvicorn

from .core.config import get_settings
from .routers import prediction_router, training_router
from .routers.models import router as models_router
from .routers.serving import router as serving_router
from .routers.dataset_management import router as dataset_management_router
from .routers.cloud_storage import router as cloud_storage_router
from .routers.eureka_router import router as eureka_router
from .schemas.base_schemas import HealthResponse
from .services.eureka_client_service import eureka_client_service
from common.logger import LoggerFactory, LogLevel

logger = LoggerFactory.get_logger("FinSightApp", console_level=LogLevel.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager with improved error handling and startup sequence.
    Includes both REST API and Eureka client initialization.
    """
    global eureka_client_service

    settings = get_settings()

    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"📊 Environment: {getattr(settings, 'environment', 'development')}")
    logger.info(f"🌐 FastAPI Host: {settings.api_host}:{settings.api_port}")
    logger.info(f"📁 Data directory: {settings.data_dir}")
    logger.info(f"📁 Models directory: {settings.models_dir}")
    logger.info(f"📁 Logs directory: {settings.logs_dir}")

    # Check if Eureka client is enabled
    eureka_enabled = getattr(settings, "enable_eureka_client", True)
    if eureka_enabled:
        logger.info(
            f"🔗 Eureka Client: {getattr(settings, 'eureka_server_url', 'http://localhost:8761')}"
        )

    startup_errors = []

    try:
        # Step 1: Initialize core services
        logger.info("📋 Step 1: Initializing core services...")

        # Initialize async training services
        logger.info("Initializing async training services...")
        from .routers.training import get_training_service

        training_service = get_training_service()
        await training_service.initialize()

        logger.info("✅ Async training services initialized successfully")

        # Initialize serving adapter
        logger.info("Initializing serving adapter...")
        from .facades import get_serving_facade, get_unified_facade

        # Initialize the serving facade singleton
        serving_facade = get_serving_facade()
        await serving_facade.initialize()

        # Also initialize the unified facade singleton
        unified_facade = get_unified_facade()
        await unified_facade.initialize()

        logger.info("✅ Serving adapter initialized successfully")

        # Step 2: Initialize Eureka client service
        if eureka_enabled:
            logger.info("📋 Step 2: Initializing Eureka client service...")
            try:
                success = await eureka_client_service.start()
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
                    "🎉 FinSight Model Builder API is fully operational (REST + Eureka)!"
                )
            else:
                logger.info(
                    "🎉 FinSight Model Builder API is fully operational (REST only)!"
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
        if eureka_enabled and eureka_client_service.is_registered():
            await eureka_client_service.stop()
            logger.info("✅ Eureka client service stopped")

        # Cleanup async services
        logger.info("Cleaning up async services...")
        from .routers.training import get_training_service

        training_service = get_training_service()
        await training_service.shutdown()

        logger.info("✅ Async services cleaned up successfully")

    except Exception as e:
        logger.error(f"❌ Error during cleanup: {str(e)}")

    logger.info(f"👋 {settings.app_name} shutdown complete")


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


# Include routers
app.include_router(
    training_router
)  # Consolidated training endpoints (both legacy and async)
app.include_router(prediction_router)
app.include_router(models_router)
app.include_router(serving_router)  # Model serving management endpoints
app.include_router(dataset_management_router)  # Dataset management endpoints
app.include_router(cloud_storage_router)  # Cloud storage management endpoints
app.include_router(eureka_router)  # Eureka client management endpoints


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    settings = get_settings()
    eureka_enabled = getattr(settings, "enable_eureka_client", True)

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
            "datasets": "/datasets",  # Dataset management
            "cloud_storage": "/cloud-storage",  # Cloud storage management
            "eureka": "/eureka" if eureka_enabled else None,  # Eureka client management
            "health": "/health",
        },
        "features": {
            "model_training": True,
            "model_serving": True,
            "dataset_management": True,
            "cloud_storage": True,
            "eureka_client": eureka_enabled,
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    settings = get_settings()

    # Check directory availability (handle case where settings might not have these directories)
    dependencies = {}

    if hasattr(settings, "data_dir") and settings.data_dir:
        dependencies["data_dir"] = (
            "available" if settings.data_dir.exists() else "missing"
        )
    else:
        dependencies["data_dir"] = "not_configured"

    if hasattr(settings, "models_dir") and settings.models_dir:
        dependencies["models_dir"] = (
            "available" if settings.models_dir.exists() else "missing"
        )
    else:
        dependencies["models_dir"] = "not_configured"

    if hasattr(settings, "logs_dir") and settings.logs_dir:
        dependencies["logs_dir"] = (
            "available" if settings.logs_dir.exists() else "missing"
        )
    else:
        dependencies["logs_dir"] = "not_configured"

    # Add Eureka client status if enabled
    eureka_enabled = getattr(settings, "enable_eureka_client", True)
    if eureka_enabled:
        eureka_status = (
            "healthy"
            if eureka_client_service and eureka_client_service.is_registered()
            else "stopped"
        )
        dependencies["eureka_client"] = eureka_status

    # Determine overall health status
    overall_status = "healthy"
    if any(status in ["missing", "stopped"] for status in dependencies.values()):
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        version=settings.app_version,
        dependencies=dependencies,
    )


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app:app", host=settings.api_host, port=settings.api_port, reload=settings.debug
    )
