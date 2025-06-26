# news_crawler/src/main.py
"""
FastAPI application for the news crawler service.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.config import settings
from .routers import search, crawler
from .utils.dependencies import get_search_service, get_article_repository
from .common.logger import LoggerFactory, LoggerType, LogLevel


# Setup logging using custom logger factory
logger = LoggerFactory.get_logger(
    name="news-crawler-service",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    console_level=LogLevel.INFO,
    use_colors=True,
    log_file="logs/news_crawler.log" if not settings.debug else None,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    """
    logger.info(f"Starting {settings.app_name}")

    # Initialize services
    search_service = get_search_service()
    article_repository = get_article_repository()

    # Initialize database indexes
    try:
        await article_repository.initialize()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")

    # Verify search engine connectivity
    try:
        is_healthy = await search_service.health_check()
        if not is_healthy:
            logger.warning("Search engine health check failed during startup")
        else:
            logger.info("Search engine health check passed")
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {str(e)}")

    yield

    # Cleanup
    try:
        await article_repository.close()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

    logger.info(f"Shutting down {settings.app_name}")


# Create FastAPI application
app = FastAPI(
    title="News Crawler Service",
    description="AI-powered news and content search service using Tavily with MongoDB storage",
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
app.include_router(search.router)
app.include_router(crawler.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.app_name,
        "status": "running",
        "version": "1.0.0",
        "description": "AI-powered news crawler and sentiment analysis service",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        search_service = get_search_service()
        is_healthy = await search_service.health_check()

        return {
            "status": "healthy" if is_healthy else "degraded",
            "service": settings.app_name,
            "search_engine": "tavily",
            "components": {
                "search_service": "healthy" if is_healthy else "unhealthy",
                "database": "healthy",  # Would check in production
                "message_broker": "healthy",  # Would check in production
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
