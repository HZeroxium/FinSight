# routers/news.py

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.responses import JSONResponse

from ..services.news_service import NewsService, NewsSearchRequest
from ..schemas.news_schemas import (
    NewsItem,
    NewsItemResponse,
    NewsSource,
    NewsResponse,
    NewsStatsResponse,
    TimeRangeSearchParams,
)
from ..utils.dependencies import get_news_service
from ..utils.response_converters import build_news_response, build_filters_summary
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


@router.get("/by-source/{source}", response_model=NewsResponse)
async def get_news_by_source(
    source: NewsSource = Path(..., description="News source"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    news_service: NewsService = Depends(get_news_service),
) -> NewsResponse:
    """
    Get news articles from a specific source with optional date filtering

    - **source**: News source (coindesk, cointelegraph, etc.)
    - **limit**: Maximum number of articles to return
    - **offset**: Number of articles to skip for pagination
    - **start_date**: Optional start date filter
    - **end_date**: Optional end date filter
    """
    try:
        logger.info(f"Getting news by source: {source}")

        if start_date or end_date:
            # Use search with date filters
            search_request = NewsSearchRequest(
                source=source,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                offset=offset,
            )
            items = await news_service.search_news(search_request)
        else:
            # Use optimized source-only query
            items = await news_service.get_news_by_source(
                source=source, limit=limit, offset=offset
            )

        # Get total count
        total_count = await news_service.count_news(
            source=source, start_date=start_date, end_date=end_date
        )

        filters_applied = {
            "source": source.value,
            "start_date": start_date,
            "end_date": end_date,
            "has_date_filter": start_date is not None or end_date is not None,
        }

        response = NewsResponse(
            items=items,
            total_count=total_count,
            limit=limit,
            offset=offset,
            has_more=(offset + len(items)) < total_count,
            filters_applied=filters_applied,
        )

        logger.info(f"Source query completed: {len(items)} items from {source}")
        return response

    except Exception as e:
        logger.error(f"Error getting news by source {source}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get news by source: {str(e)}"
        )


@router.get("/keywords/{keywords}", response_model=NewsResponse)
async def search_by_keywords(
    keywords: str = Path(..., description="Comma-separated keywords"),
    source: Optional[NewsSource] = Query(None, description="Optional source filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum items to return"),
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
                offset=0,
            )
            items = await news_service.search_news(search_request)

        filters_applied = {
            "keywords": keyword_list,
            "source": source.value if source else None,
            "hours_back": hours,
            "start_date": start_date,
        }

        response = NewsResponse(
            items=items,
            total_count=len(items),  # Simplified for keyword search
            limit=limit,
            offset=0,
            has_more=len(items) >= limit,
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


@router.get("/{item_id}", response_model=NewsItemResponse)
async def get_news_item(
    item_id: str = Path(..., description="News item ID"),
    news_service: NewsService = Depends(get_news_service),
) -> NewsItemResponse:
    """
    Get a specific news item by ID

    - **item_id**: Unique identifier of the news item
    """
    try:
        logger.info(f"Getting news item: {item_id}")

        item = await news_service.get_news_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="News item not found")

        # Convert to response format
        from ..utils.response_converters import convert_news_item_to_response

        response = convert_news_item_to_response(item)

        logger.info(f"News item retrieved: {item_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting news item {item_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get news item: {str(e)}"
        )


@router.get("/stats/summary", response_model=NewsStatsResponse)
async def get_news_statistics(
    news_service: NewsService = Depends(get_news_service),
) -> NewsStatsResponse:
    """
    Get comprehensive news database statistics

    Returns statistics about the news database including:
    - Total article count
    - Articles by source
    - Recent activity
    - Date range information
    """
    try:
        logger.info("Getting news statistics")

        stats = await news_service.get_repository_stats()

        response = NewsStatsResponse(
            total_articles=stats.get("total_articles", 0),
            articles_by_source=stats.get("articles_by_source", {}),
            recent_articles_24h=stats.get("recent_articles_24h", 0),
            oldest_article=stats.get("oldest_article"),
            newest_article=stats.get("newest_article"),
            database_info={
                "database_name": stats.get("database_name"),
                "collection_name": stats.get("collection_name"),
            },
        )

        logger.info(f"Statistics retrieved: {response.total_articles} total articles")
        return response

    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get statistics: {str(e)}"
        )


@router.delete("/{item_id}")
async def delete_news_item(
    item_id: str = Path(..., description="News item ID to delete"),
    news_service: NewsService = Depends(get_news_service),
) -> Dict[str, Any]:
    """
    Delete a specific news article

    - **item_id**: Unique identifier of the news article to delete
    """
    try:
        logger.info(f"Deleting news item: {item_id}")

        # Check if item exists
        item = await news_service.get_news_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="News item not found")

        # Delete item
        success = await news_service.delete_news_item(item_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete news item")

        logger.info(f"News item deleted: {item_id}")
        return {
            "success": True,
            "message": "News item deleted successfully",
            "item_id": item_id,
            "deleted_title": (
                item.title[:50] + "..." if len(item.title) > 50 else item.title
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting news item {item_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete news item: {str(e)}"
        )


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
