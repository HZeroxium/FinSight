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
