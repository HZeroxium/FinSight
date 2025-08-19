# routers/news.py

from typing import Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.responses import JSONResponse

from ..services.news_service import NewsService, NewsSearchRequest
from ..schemas.news_schemas import (
    NewsSource,
    NewsResponse,
    TimeRangeSearchParams,
)
from ..utils.dependencies import get_news_service, require_admin_access
from ..utils.response_converters import build_news_response, build_filters_summary
from ..utils.cache_utils import (
    get_cache_statistics as get_cache_statistics_util,
    check_cache_health as check_cache_health_util,
    invalidate_all_news_cache as invalidate_all_news_cache_util,
)
from common.logger import LoggerFactory, LoggerType, LogLevel

# Initialize router
router = APIRouter(prefix="/news", tags=["news"])

# Setup logging
logger = LoggerFactory.get_logger(
    name="news-router",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    file_level=LogLevel.DEBUG,
    log_file="logs/news_router.log",
)


@router.get("/", response_model=NewsResponse)
async def search_news(
    source: Optional[NewsSource] = Query(None, description="Filter by news source"),
    keywords: Optional[str] = Query(None, description="Comma-separated keywords"),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    start_date: Optional[datetime] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[datetime] = Query(None, description="End date (ISO format)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    news_service: NewsService = Depends(get_news_service),
) -> NewsResponse:
    """
    Search news articles with flexible filtering options

    - **source**: Filter by specific news source
    - **keywords**: Search in title and description (comma-separated)
    - **tags**: Filter by tags (comma-separated)
    - **start_date**: Filter articles from this date onwards
    - **end_date**: Filter articles up to this date
    - **limit**: Maximum number of articles to return (1-1000)
    - **offset**: Number of articles to skip for pagination
    """
    try:
        logger.info(
            f"Searching news with filters: source={source}, keywords={keywords}, tags={tags}"
        )

        # Parse keywords and tags
        keyword_list = None
        if keywords:
            keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]

        tag_list = None
        if tags:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        # Create search request
        search_request = NewsSearchRequest(
            source=source,
            keywords=keyword_list,
            tags=tag_list,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )

        # Execute search
        items = await news_service.search_news(search_request)

        # Get total count for pagination
        total_count = await news_service.count_news(
            source=source,
            keywords=keyword_list,
            tags=tag_list,
            start_date=start_date,
            end_date=end_date,
        )

        # Build filters summary
        filters_applied = build_filters_summary(
            source=source.value if source else None,
            keywords=keyword_list,
            tags=tag_list,
            start_date=start_date,
            end_date=end_date,
        )

        # Build response using converter
        response = build_news_response(
            items=items,
            total_count=total_count,
            limit=limit,
            offset=offset,
            filters_applied=filters_applied,
        )

        logger.info(
            f"Search completed: {len(items)} items returned, {total_count} total"
        )
        return response

    except Exception as e:
        logger.error(f"Error searching news: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/recent", response_model=NewsResponse)
async def get_recent_news(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back (max 1 week)"),
    source: Optional[NewsSource] = Query(None, description="Filter by news source"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum items to return"),
    news_service: NewsService = Depends(get_news_service),
) -> NewsResponse:
    """
    Get recent news articles from the last N hours

    - **hours**: Number of hours to look back (1-168, default 24)
    - **source**: Optional source filter
    - **limit**: Maximum number of articles to return
    """
    try:
        logger.info(f"Getting recent news: {hours}h, source={source}")

        items = await news_service.get_recent_news(
            source=source, hours=hours, limit=limit
        )

        # Calculate time range for count
        start_date = datetime.now(timezone.utc) - timedelta(hours=hours)
        total_count = await news_service.count_news(
            source=source, start_date=start_date
        )

        # Build filters summary
        filters_applied = build_filters_summary(
            source=source.value if source else None,
            hours=hours,
            start_date=start_date,
            end_date=datetime.now(timezone.utc),
        )

        # Build response using converter
        response = build_news_response(
            items=items,
            total_count=total_count,
            limit=limit,
            offset=0,
            filters_applied=filters_applied,
        )

        logger.info(f"Recent news retrieved: {len(items)} items")
        return response

    except Exception as e:
        logger.error(f"Error getting recent news: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get recent news: {str(e)}"
        )


@router.post("/search/time-range", response_model=NewsResponse)
async def search_by_time_range(
    params: TimeRangeSearchParams, news_service: NewsService = Depends(get_news_service)
) -> NewsResponse:
    """
    Advanced time-based search with optimized performance

    Provide ONE of the following time parameters:
    - **hours**: Look back N hours from now
    - **days**: Look back N days from now
    - **start_date**: Search from specific date (with optional end_date)
    """
    try:
        logger.info(f"Time-range search: {params.model_dump()}")

        # Calculate date range based on provided parameters
        start_date = None
        end_date = params.end_date

        if params.hours:
            start_date = datetime.now(timezone.utc) - timedelta(hours=params.hours)
        elif params.days:
            start_date = datetime.now(timezone.utc) - timedelta(days=params.days)
        elif params.start_date:
            start_date = params.start_date
        else:
            # Default to last 24 hours if no time parameter provided
            start_date = datetime.now(timezone.utc) - timedelta(hours=24)

        # Create search request
        search_request = NewsSearchRequest(
            source=params.source,
            keywords=params.keywords,
            start_date=start_date,
            end_date=end_date,
            limit=params.limit,
            offset=params.offset,
        )

        # Execute search
        items = await news_service.search_news(search_request)

        # Get total count
        total_count = await news_service.count_news(
            source=params.source, start_date=start_date, end_date=end_date
        )

        filters_applied = {
            "source": params.source.value if params.source else None,
            "keywords": params.keywords,
            "start_date": start_date,
            "end_date": end_date,
            "hours_back": params.hours,
            "days_back": params.days,
            "optimization": "time-range-optimized",
        }

        response = build_news_response(
            items=items,
            total_count=total_count,
            limit=params.limit,
            offset=params.offset,
            filters_applied=filters_applied,
        )

        logger.info(f"Time-range search completed: {len(items)} items")
        return response

    except ValueError as e:
        logger.warning(f"Invalid time-range parameters: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in time-range search: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Time-range search failed: {str(e)}"
        )


@router.get("/keywords/{keywords}", response_model=NewsResponse)
async def search_by_keywords(
    keywords: str = Path(..., description="Comma-separated keywords"),
    source: Optional[NewsSource] = Query(None, description="Optional source filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    hours: Optional[int] = Query(None, ge=1, le=8760, description="Hours to look back"),
    news_service: NewsService = Depends(get_news_service),
) -> NewsResponse:
    """
    Search news by keywords with optional filters

    - **keywords**: Comma-separated keywords to search for
    - **source**: Optional source filter
    - **limit**: Maximum number of articles to return
    - **hours**: Optional time filter (hours to look back)
    """
    try:
        logger.info(f"Searching by keywords: {keywords}")

        # Parse keywords
        keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
        if not keyword_list:
            raise HTTPException(
                status_code=400, detail="At least one keyword is required"
            )

        # Calculate date range if hours specified
        start_date = None
        if hours:
            start_date = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Execute search
        items = await news_service.get_news_by_keywords(
            keywords=keyword_list, limit=limit
        )

        # Filter by source and date if needed (note: this is not optimal for large datasets)
        if source or start_date:
            search_request = NewsSearchRequest(
                source=source,
                keywords=keyword_list,
                start_date=start_date,
                limit=limit,
                offset=offset,
            )
            items = await news_service.search_news(search_request)

        total_count = await news_service.count_news(
            source=source,
            keywords=keyword_list,
        )

        filters_applied = {
            "keywords": keyword_list,
            "source": source.value if source else None,
            "hours_back": hours,
            "start_date": start_date,
        }

        response = build_news_response(
            items=items,
            total_count=total_count,
            limit=limit,
            offset=offset,
            filters_applied=filters_applied,
        )

        logger.info(f"Keyword search completed: {len(items)} items")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching by keywords: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Keyword search failed: {str(e)}")


@router.get("/by-tag/{tags}", response_model=NewsResponse)
async def get_news_by_tags(
    tags: str = Path(..., description="Comma-separated tags"),
    source: Optional[NewsSource] = Query(None, description="Optional source filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    hours: Optional[int] = Query(None, ge=1, le=8760, description="Hours to look back"),
    news_service: NewsService = Depends(get_news_service),
) -> NewsResponse:
    """
    Get news articles by tags with optional filters

    - **tags**: Comma-separated tags to filter by
    - **source**: Optional source filter
    - **limit**: Maximum number of articles to return
    - **offset**: Number of articles to skip for pagination
    - **hours**: Optional time filter (hours to look back)
    """
    try:
        logger.info(f"Getting news by tags: {tags}")

        # Parse tags
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        if not tag_list:
            raise HTTPException(status_code=400, detail="At least one tag is required")

        # Calculate date range if hours specified
        start_date = None
        if hours:
            start_date = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Create search request
        search_request = NewsSearchRequest(
            source=source,
            tags=tag_list,
            start_date=start_date,
            limit=limit,
            offset=offset,
        )

        # Execute search
        items = await news_service.search_news(search_request)

        # Get total count
        total_count = await news_service.count_news(
            source=source,
            tags=tag_list,
            start_date=start_date,
        )

        # Build filters summary
        filters_applied = build_filters_summary(
            source=source.value if source else None,
            tags=tag_list,
            hours=hours,
            start_date=start_date,
        )

        # Build response using converter
        response = build_news_response(
            items=items,
            total_count=total_count,
            limit=limit,
            offset=offset,
            filters_applied=filters_applied,
        )

        logger.info(f"Tag search completed: {len(items)} items")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching by tags: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tag search failed: {str(e)}")


@router.get("/tags/available", response_model=Dict[str, Any])
async def get_available_tags(
    source: Optional[NewsSource] = Query(None, description="Optional source filter"),
    limit: int = Query(100, ge=1, le=500, description="Maximum tags to return"),
    news_service: NewsService = Depends(get_news_service),
) -> Dict[str, Any]:
    """
    Get available tags from news articles

    - **source**: Optional source filter
    - **limit**: Maximum number of tags to return (sorted by frequency)
    """
    try:
        logger.info(f"Getting available tags for source: {source}")

        tags = await news_service.get_unique_tags(source=source, limit=limit)

        response = {
            "tags": tags,
            "total_count": len(tags),
            "source_filter": source.value if source else None,
            "limit": limit,
        }

        logger.info(f"Available tags retrieved: {len(tags)} tags")
        return response

    except Exception as e:
        logger.error(f"Error getting available tags: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get tags: {str(e)}")


# Health check endpoint for the news router
@router.get("/health/check")
async def news_health_check(
    news_service: NewsService = Depends(get_news_service),
) -> Dict[str, Any]:
    """
    Health check for news service functionality
    """
    try:
        # Try to get a simple count to verify database connectivity
        total_count = await news_service.count_news()

        return {
            "status": "healthy",
            "service": "news-service",
            "database_accessible": True,
            "total_articles": total_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"News service health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "news-service",
                "database_accessible": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


# Cache management endpoints (admin only)
@router.get("/cache/stats", response_model=Dict[str, Any])
async def cache_stats(
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Get cache statistics and information (Admin only)

    Returns:
        Dict containing cache statistics, hit rates, and key information
    """
    try:
        logger.info("Getting cache statistics")
        stats = await get_cache_statistics_util()

        return {
            "cache_statistics": stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting cache statistics: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get cache statistics: {str(e)}"
        )


@router.get("/cache/health")
async def cache_health(
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Check cache health status (Admin only)

    Returns:
        Dict containing cache health status
    """
    try:
        logger.info("Checking cache health")
        is_healthy = await check_cache_health_util()

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "cache_accessible": is_healthy,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Cache health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "cache_accessible": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


@router.post("/cache/invalidate")
async def cache_invalidate(
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Invalidate all news cache entries (Admin only)

    This endpoint clears all cached data for news endpoints.
    Use this after data updates to ensure fresh data is served.

    Returns:
        Dict containing invalidation result
    """
    try:
        logger.info("Invalidating all news cache")
        success = await invalidate_all_news_cache_util()

        if success:
            return {
                "status": "success",
                "message": "All news cache entries invalidated successfully",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to invalidate cache")

    except Exception as e:
        logger.error(f"Error invalidating cache: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to invalidate cache: {str(e)}"
        )
