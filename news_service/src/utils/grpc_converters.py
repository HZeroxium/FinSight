# utils/grpc_converters.py

"""
gRPC converter utilities for transforming between internal models and gRPC messages.

This module provides conversion functions between internal Pydantic models
and gRPC protocol buffer messages, ensuring clean separation between
transport layer and business logic.
"""

from datetime import datetime, timezone
from typing import Optional

from common.logger import LoggerFactory, LoggerType, LogLevel
from google.protobuf.timestamp_pb2 import Timestamp

from ..schemas.news_schemas import NewsItemResponse as PydanticNewsItemResponse
from ..schemas.news_schemas import NewsResponse as PydanticNewsResponse
from ..schemas.news_schemas import NewsSource as PydanticNewsSource
from ..schemas.news_schemas import \
    NewsStatsResponse as PydanticNewsStatsResponse

# Initialize logger
logger = LoggerFactory.get_logger(
    name="grpc-converters",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    file_level=LogLevel.DEBUG,
    log_file="logs/grpc_converters.log",
)


def datetime_to_timestamp(dt: Optional[datetime]) -> Optional[Timestamp]:
    """Convert datetime to protobuf Timestamp."""
    if dt is None:
        return None

    timestamp = Timestamp()
    timestamp.FromDatetime(dt)
    return timestamp


def timestamp_to_datetime(timestamp: Optional[Timestamp]) -> Optional[datetime]:
    """Convert protobuf Timestamp to datetime."""
    if timestamp is None or not timestamp.HasField("seconds"):
        return None

    return timestamp.ToDatetime().replace(tzinfo=timezone.utc)


def pydantic_source_to_grpc(source: Optional[PydanticNewsSource]) -> int:
    """Convert Pydantic NewsSource to gRPC enum value."""
    if source is None:
        return 0  # NEWS_SOURCE_UNSPECIFIED

    source_mapping = {
        PydanticNewsSource.COINDESK: 1,  # NEWS_SOURCE_COINDESK
        PydanticNewsSource.COINTELEGRAPH: 2,  # NEWS_SOURCE_COINTELEGRAPH
    }

    return source_mapping.get(source, 0)


def grpc_source_to_pydantic(source: int) -> Optional[PydanticNewsSource]:
    """Convert gRPC enum value to Pydantic NewsSource."""
    if source == 0:
        return None

    source_mapping = {
        1: PydanticNewsSource.COINDESK,  # NEWS_SOURCE_COINDESK
        2: PydanticNewsSource.COINTELEGRAPH,  # NEWS_SOURCE_COINTELEGRAPH
    }

    return source_mapping.get(source)


def convert_pydantic_news_item_to_grpc(item: PydanticNewsItemResponse):
    """Convert Pydantic NewsItemResponse to gRPC NewsItemResponse."""
    try:
        # Import here to avoid circular imports
        from ..grpc_generated import news_service_pb2

        grpc_item = news_service_pb2.NewsItemResponse()
        grpc_item.source = pydantic_source_to_grpc(item.source)
        grpc_item.title = item.title or ""
        grpc_item.url = str(item.url)
        grpc_item.description = item.description or ""

        if item.published_at:
            timestamp = datetime_to_timestamp(item.published_at)
            if timestamp:
                grpc_item.published_at.CopyFrom(timestamp)

        grpc_item.author = item.author or ""
        grpc_item.tags.extend(item.tags or [])

        return grpc_item

    except Exception as e:
        logger.error(f"Failed to convert news item to gRPC: {e}")
        raise


def convert_pydantic_news_response_to_grpc(response: PydanticNewsResponse):
    """Convert Pydantic NewsResponse to gRPC NewsResponse."""
    try:
        # Import here to avoid circular imports
        from ..grpc_generated import news_service_pb2

        grpc_response = news_service_pb2.NewsResponse()

        # Convert items
        for item in response.items:
            grpc_item = convert_pydantic_news_item_to_grpc(item)
            grpc_response.items.append(grpc_item)

        # Set metadata
        grpc_response.total_count = response.total_count
        grpc_response.limit = response.limit
        grpc_response.offset = response.offset
        grpc_response.has_more = response.has_more

        # Convert filters
        if response.filters_applied:
            filters = news_service_pb2.FiltersApplied()

            filters.source = response.filters_applied.get("source", "") or ""

            keywords = response.filters_applied.get("keywords", [])
            if keywords:
                filters.keywords.extend(keywords)

            tags = response.filters_applied.get("tags", [])
            if tags:
                filters.tags.extend(tags)

            start_date = response.filters_applied.get("start_date")
            if start_date:
                timestamp = datetime_to_timestamp(start_date)
                if timestamp:
                    filters.start_date.CopyFrom(timestamp)

            end_date = response.filters_applied.get("end_date")
            if end_date:
                timestamp = datetime_to_timestamp(end_date)
                if timestamp:
                    filters.end_date.CopyFrom(timestamp)

            filters.has_date_filter = response.filters_applied.get(
                "has_date_filter", False
            )
            filters.has_keywords_filter = response.filters_applied.get(
                "has_keywords_filter", False
            )
            filters.has_tags_filter = response.filters_applied.get(
                "has_tags_filter", False
            )
            filters.time_range_hours = response.filters_applied.get(
                "time_range_hours", 0
            )
            filters.is_recent_search = response.filters_applied.get(
                "is_recent_search", False
            )

            grpc_response.filters_applied.CopyFrom(filters)

        return grpc_response

    except Exception as e:
        logger.error(f"Failed to convert news response to gRPC: {e}")
        raise


def convert_pydantic_stats_to_grpc(stats: PydanticNewsStatsResponse):
    """Convert Pydantic NewsStatsResponse to gRPC NewsStatsResponse."""
    try:
        # Import here to avoid circular imports
        from ..grpc_generated import news_service_pb2

        grpc_stats = news_service_pb2.NewsStatsResponse()
        grpc_stats.total_articles = stats.total_articles
        grpc_stats.recent_articles_24h = stats.recent_articles_24h

        # Convert articles by source
        for source, count in stats.articles_by_source.items():
            grpc_stats.articles_by_source[source] = count

        # Convert timestamps
        if stats.oldest_article:
            timestamp = datetime_to_timestamp(stats.oldest_article)
            if timestamp:
                grpc_stats.oldest_article.CopyFrom(timestamp)

        if stats.newest_article:
            timestamp = datetime_to_timestamp(stats.newest_article)
            if timestamp:
                grpc_stats.newest_article.CopyFrom(timestamp)

        # Convert database info
        for key, value in stats.database_info.items():
            grpc_stats.database_info[key] = str(value) if value else ""

        return grpc_stats

    except Exception as e:
        logger.error(f"Failed to convert stats to gRPC: {e}")
        raise


def convert_grpc_search_request_to_pydantic(request):
    """Convert gRPC SearchNewsRequest to internal search parameters."""
    try:
        params = {}

        if request.HasField("source") and request.source != 0:
            params["source"] = grpc_source_to_pydantic(request.source)

        if request.keywords:
            params["keywords"] = list(request.keywords)

        if request.tags:
            params["tags"] = list(request.tags)

        if request.HasField("start_date"):
            params["start_date"] = timestamp_to_datetime(request.start_date)

        if request.HasField("end_date"):
            params["end_date"] = timestamp_to_datetime(request.end_date)

        if request.limit > 0:
            params["limit"] = min(request.limit, 1000)  # Cap at 1000
        else:
            params["limit"] = 100  # Default

        if request.offset >= 0:
            params["offset"] = request.offset
        else:
            params["offset"] = 0  # Default

        return params

    except Exception as e:
        logger.error(f"Failed to convert gRPC search request: {e}")
        raise


def build_grpc_error_response(error_message: str, response_type):
    """Build empty gRPC response for error cases."""
    try:
        # Import here to avoid circular imports
        from ..grpc_generated import news_service_pb2

        if response_type == "NewsResponse":
            response = news_service_pb2.NewsResponse()
            response.total_count = 0
            response.limit = 0
            response.offset = 0
            response.has_more = False
            return response

        elif response_type == "AvailableTagsResponse":
            response = news_service_pb2.AvailableTagsResponse()
            response.total_count = 0
            response.limit = 0
            return response

        elif response_type == "DeleteResponse":
            response = news_service_pb2.DeleteResponse()
            response.success = False
            response.message = error_message
            return response

        else:
            logger.error(f"Unknown response type for error: {response_type}")
            return None

    except Exception as e:
        logger.error(f"Failed to build error response: {e}")
        return None
