# utils/response_converters.py

"""
Response converter utilities for transforming internal models to API responses.

This module provides clean conversion functions between internal data models
and external API response schemas, ensuring separation of concerns and
maintaining API stability.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from common.logger import LoggerFactory, LoggerType, LogLevel

from ..schemas.news_schemas import NewsItem, NewsItemResponse, NewsResponse

# Initialize logger
logger = LoggerFactory.get_logger(
    name="response-converters",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    file_level=LogLevel.DEBUG,
    log_file="logs/response_converters.log",
)


def convert_news_item_to_response(news_item: NewsItem) -> NewsItemResponse:
    """
    Convert NewsItem to NewsItemResponse for frontend consumption.

    Args:
        news_item: Internal NewsItem model

    Returns:
        NewsItemResponse: Streamlined response model
    """
    try:
        return NewsItemResponse.from_news_item(news_item)
    except Exception as e:
        logger.error(f"Failed to convert news item: {str(e)}")
        # Return a minimal response in case of error
        return NewsItemResponse(
            source=news_item.source,
            title=news_item.title or "Title unavailable",
            url=news_item.url,
            description=news_item.description,
            published_at=news_item.published_at,
            author=news_item.author,
            tags=news_item.tags or [],
        )


def convert_news_items_to_response(
    news_items: List[NewsItem],
) -> List[NewsItemResponse]:
    """
    Convert list of NewsItem to list of NewsItemResponse.

    Args:
        news_items: List of internal NewsItem models

    Returns:
        List[NewsItemResponse]: List of streamlined response models
    """
    if not news_items:
        return []

    try:
        responses = []
        for item in news_items:
            response_item = convert_news_item_to_response(item)
            responses.append(response_item)

        logger.debug(f"Converted {len(responses)} news items to response format")
        return responses

    except Exception as e:
        logger.error(f"Failed to convert news items list: {str(e)}")
        return []


def build_news_response(
    items: List[NewsItem],
    total_count: int,
    limit: int,
    offset: int,
    filters_applied: Dict[str, Any],
) -> NewsResponse:
    """
    Build complete NewsResponse with converted items and metadata.

    Args:
        items: List of internal NewsItem models
        total_count: Total number of matching items
        limit: Applied limit
        offset: Applied offset
        filters_applied: Dictionary of applied filters

    Returns:
        NewsResponse: Complete response with converted items
    """
    try:
        # Convert items to response format
        response_items = convert_news_items_to_response(items)

        # Calculate pagination info
        has_more = (offset + len(items)) < total_count

        response = NewsResponse(
            items=response_items,
            total_count=total_count,
            limit=limit,
            offset=offset,
            has_more=has_more,
            filters_applied=filters_applied,
        )

        logger.debug(
            f"Built news response: {len(response_items)} items, "
            f"total: {total_count}, has_more: {has_more}"
        )

        return response

    except Exception as e:
        logger.error(f"Failed to build news response: {str(e)}")
        # Return empty response in case of error
        return NewsResponse(
            items=[],
            total_count=0,
            limit=limit,
            offset=offset,
            has_more=False,
            filters_applied=filters_applied,
        )


def build_filters_summary(
    source: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    hours: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Build standardized filters summary for API responses.

    Args:
        source: Source filter
        keywords: Keywords filter
        tags: Tags filter
        start_date: Start date filter
        end_date: End date filter
        hours: Hours filter (for recent searches)
        **kwargs: Additional filter parameters

    Returns:
        Dict[str, Any]: Standardized filters summary
    """
    filters = {
        "source": source,
        "keywords": keywords,
        "tags": tags,
        "start_date": start_date,
        "end_date": end_date,
        "has_date_filter": start_date is not None or end_date is not None,
        "has_keywords_filter": keywords is not None and len(keywords) > 0,
        "has_tags_filter": tags is not None and len(tags) > 0,
    }

    # Add time range info for recent searches
    if hours is not None:
        filters["time_range_hours"] = hours
        filters["is_recent_search"] = True
    else:
        filters["is_recent_search"] = False

    # Add any additional filters
    filters.update(kwargs)

    return filters
