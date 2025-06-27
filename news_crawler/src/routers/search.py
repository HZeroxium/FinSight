# routers/search.py

"""
REST API routes for search operations.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse

from ..schemas.search_schemas import SearchRequestSchema, SearchResponseSchema
from ..schemas.common_schemas import HealthCheckSchema, ErrorResponseSchema
from ..services.search_service import SearchService
from ..interfaces.search_engine import SearchEngineError
from ..utils.dependencies import get_search_service
from ..common.logger import LoggerFactory, LoggerType, LogLevel

logger = LoggerFactory.get_logger(
    name="search-router", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)

router = APIRouter(prefix="/api/v1/search", tags=["search"])


@router.post("/", response_model=SearchResponseSchema)
async def search_content(
    request: SearchRequestSchema,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponseSchema:
    """
    Perform a general content search.

    Args:
        request: Search parameters
        search_service: Injected search service

    Returns:
        SearchResponseSchema: Search results
    """
    try:
        logger.info(
            f"Search request: {request.query} (crawler: {request.enable_crawler})"
        )

        result = await search_service.search_news(request)
        return result

    except SearchEngineError as e:
        logger.error(f"Search engine error: {e.message}")
        raise HTTPException(status_code=502, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/financial-sentiment/{symbol}", response_model=SearchResponseSchema)
async def get_financial_sentiment(
    symbol: str,
    days: int = Query(7, ge=1, le=30, description="Number of days to look back"),
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponseSchema:
    """
    Get financial sentiment for a specific symbol.

    Args:
        symbol: Financial symbol (e.g., BTC, AAPL)
        days: Number of days to analyze
        search_service: Injected search service

    Returns:
        SearchResponseSchema: Financial sentiment results
    """
    try:
        logger.info(f"Financial sentiment search for {symbol} ({days} days)")
        result = await search_service.search_financial_sentiment(symbol, days)
        return result

    except SearchEngineError as e:
        logger.error(f"Financial sentiment search failed: {e.message}")
        raise HTTPException(status_code=502, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/trending", response_model=SearchResponseSchema)
async def get_trending_topics(
    topic: str = Query("finance", description="Topic category"),
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponseSchema:
    """
    Get trending topics in a specific category.

    Args:
        topic: Topic category
        search_service: Injected search service

    Returns:
        SearchResponseSchema: Trending topics
    """
    try:
        logger.info(f"Trending topics search for {topic}")
        result = await search_service.get_trending_topics(topic)
        return result

    except SearchEngineError as e:
        logger.error(f"Trending topics search failed: {e.message}")
        raise HTTPException(status_code=502, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health", response_model=HealthCheckSchema)
async def health_check(
    search_service: SearchService = Depends(get_search_service),
) -> HealthCheckSchema:
    """
    Health check endpoint.

    Returns:
        HealthCheckSchema: Health status
    """
    try:
        is_healthy = await search_service.health_check()

        dependencies = {
            "search_engine": "healthy" if is_healthy else "unhealthy",
            "message_broker": "healthy" if is_healthy else "unhealthy",
            "crawler_service": "healthy" if is_healthy else "unhealthy",
        }

        return HealthCheckSchema(
            status="healthy" if is_healthy else "unhealthy",
            service="news-crawler-service",
            dependencies=dependencies,
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content=ErrorResponseSchema(
                error="ServiceUnavailable", message=f"Health check failed: {str(e)}"
            ).dict(),
        )
        dependencies = {
            "search_engine": "healthy" if is_healthy else "unhealthy",
            "message_broker": "healthy" if is_healthy else "unhealthy",
            "crawler_service": "healthy" if is_healthy else "unhealthy",
        }

        return HealthCheckSchema(
            status="healthy" if is_healthy else "unhealthy",
            service="news-crawler-service",
            dependencies=dependencies,
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content=ErrorResponseSchema(
                error="ServiceUnavailable", message=f"Health check failed: {str(e)}"
            ).dict(),
        )
