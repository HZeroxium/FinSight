# grpc_services/news_grpc_service.py

"""
gRPC service implementation for news operations.

This module implements the gRPC service interface defined in the protocol buffer
definitions, providing high-performance RPC access to news data.
"""

from datetime import datetime, timezone, timedelta

import grpc

from ..services.news_service import NewsService, NewsSearchRequest
from ..utils.grpc_converters import (
    convert_pydantic_news_response_to_grpc,
    convert_pydantic_news_item_to_grpc,
    convert_pydantic_stats_to_grpc,
    convert_grpc_search_request_to_pydantic,
    grpc_source_to_pydantic,
    timestamp_to_datetime,
)
from ..utils.response_converters import (
    build_news_response,
    build_filters_summary,
    convert_news_item_to_response,
)
from common.logger import LoggerFactory, LoggerType, LogLevel

# Initialize logger
logger = LoggerFactory.get_logger(
    name="news-grpc-service",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    file_level=LogLevel.DEBUG,
    log_file="logs/news_grpc_service.log",
)


class NewsGrpcService:
    """gRPC service implementation for news operations."""

    def __init__(self, news_service: NewsService):
        """
        Initialize gRPC service with news service dependency.

        Args:
            news_service: Core news service for business logic
        """
        self.news_service = news_service
        logger.info("NewsGrpcService initialized")

    async def SearchNews(self, request, context):
        """Search news with flexible filters."""
        try:
            logger.info(
                f"gRPC SearchNews called with {len(request.keywords)} keywords, {len(request.tags)} tags"
            )

            # Convert gRPC request to internal parameters
            params = convert_grpc_search_request_to_pydantic(request)

            # Create search request
            search_request = NewsSearchRequest(**params)

            # Execute search
            items = await self.news_service.search_news(search_request)

            # Get total count
            total_count = await self.news_service.count_news(
                source=params.get("source"),
                keywords=params.get("keywords"),
                tags=params.get("tags"),
                start_date=params.get("start_date"),
                end_date=params.get("end_date"),
            )

            # Build filters summary
            filters_applied = build_filters_summary(
                source=params.get("source").value if params.get("source") else None,
                keywords=params.get("keywords"),
                tags=params.get("tags"),
                start_date=params.get("start_date"),
                end_date=params.get("end_date"),
            )

            # Build response
            pydantic_response = build_news_response(
                items=items,
                total_count=total_count,
                limit=params["limit"],
                offset=params["offset"],
                filters_applied=filters_applied,
            )

            # Convert to gRPC response
            grpc_response = convert_pydantic_news_response_to_grpc(pydantic_response)

            logger.info(f"gRPC SearchNews completed: {len(items)} items")
            return grpc_response

        except Exception as e:
            logger.error(f"gRPC SearchNews error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Search failed: {str(e)}")

    async def GetRecentNews(self, request, context):
        """Get recent news articles."""
        try:
            logger.info(
                f"gRPC GetRecentNews called: {request.hours}h, source={request.source}"
            )

            # Validate parameters
            hours = request.hours if request.hours > 0 else 24
            hours = min(hours, 168)  # Cap at 1 week

            source = (
                grpc_source_to_pydantic(request.source) if request.source != 0 else None
            )
            limit = min(request.limit, 1000) if request.limit > 0 else 100

            # Get recent news
            items = await self.news_service.get_recent_news(
                source=source, hours=hours, limit=limit
            )

            # Calculate time range for count
            start_date = datetime.now(timezone.utc) - timedelta(hours=hours)
            total_count = await self.news_service.count_news(
                source=source, start_date=start_date
            )

            # Build filters summary
            filters_applied = build_filters_summary(
                source=source.value if source else None,
                hours=hours,
                start_date=start_date,
                end_date=datetime.now(timezone.utc),
            )

            # Build response
            pydantic_response = build_news_response(
                items=items,
                total_count=total_count,
                limit=limit,
                offset=0,
                filters_applied=filters_applied,
            )

            # Convert to gRPC response
            grpc_response = convert_pydantic_news_response_to_grpc(pydantic_response)

            logger.info(f"gRPC GetRecentNews completed: {len(items)} items")
            return grpc_response

        except Exception as e:
            logger.error(f"gRPC GetRecentNews error: {e}")
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Get recent news failed: {str(e)}"
            )

    async def GetNewsBySource(self, request, context):
        """Get news articles from a specific source."""
        try:
            logger.info(f"gRPC GetNewsBySource called: source={request.source}")

            # Validate parameters
            source = grpc_source_to_pydantic(request.source)
            if not source:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "Valid source is required"
                )
                return

            limit = min(request.limit, 1000) if request.limit > 0 else 100
            offset = max(request.offset, 0)

            start_date = (
                timestamp_to_datetime(request.start_date)
                if request.HasField("start_date")
                else None
            )
            end_date = (
                timestamp_to_datetime(request.end_date)
                if request.HasField("end_date")
                else None
            )

            # Get news by source
            if start_date or end_date:
                search_request = NewsSearchRequest(
                    source=source,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit,
                    offset=offset,
                )
                items = await self.news_service.search_news(search_request)
            else:
                items = await self.news_service.get_news_by_source(
                    source=source, limit=limit, offset=offset
                )

            # Get total count
            total_count = await self.news_service.count_news(
                source=source, start_date=start_date, end_date=end_date
            )

            # Build filters summary
            filters_applied = build_filters_summary(
                source=source.value,
                start_date=start_date,
                end_date=end_date,
            )

            # Build response
            pydantic_response = build_news_response(
                items=items,
                total_count=total_count,
                limit=limit,
                offset=offset,
                filters_applied=filters_applied,
            )

            # Convert to gRPC response
            grpc_response = convert_pydantic_news_response_to_grpc(pydantic_response)

            logger.info(f"gRPC GetNewsBySource completed: {len(items)} items")
            return grpc_response

        except Exception as e:
            logger.error(f"gRPC GetNewsBySource error: {e}")
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Get news by source failed: {str(e)}"
            )

    async def SearchByKeywords(self, request, context):
        """Search news by keywords."""
        try:
            logger.info(
                f"gRPC SearchByKeywords called: {len(request.keywords)} keywords"
            )

            # Validate parameters
            if not request.keywords:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "At least one keyword is required"
                )
                return

            keywords = list(request.keywords)
            source = (
                grpc_source_to_pydantic(request.source) if request.source != 0 else None
            )
            limit = min(request.limit, 1000) if request.limit > 0 else 100

            # Calculate date range if hours specified
            start_date = None
            if request.hours > 0:
                start_date = datetime.now(timezone.utc) - timedelta(hours=request.hours)

            # Create search request
            search_request = NewsSearchRequest(
                source=source,
                keywords=keywords,
                start_date=start_date,
                limit=limit,
                offset=0,
            )

            # Execute search
            items = await self.news_service.search_news(search_request)

            # Get total count
            total_count = await self.news_service.count_news(
                source=source,
                keywords=keywords,
                start_date=start_date,
            )

            # Build filters summary
            filters_applied = build_filters_summary(
                source=source.value if source else None,
                keywords=keywords,
                hours=request.hours if request.hours > 0 else None,
                start_date=start_date,
            )

            # Build response
            pydantic_response = build_news_response(
                items=items,
                total_count=total_count,
                limit=limit,
                offset=0,
                filters_applied=filters_applied,
            )

            # Convert to gRPC response
            grpc_response = convert_pydantic_news_response_to_grpc(pydantic_response)

            logger.info(f"gRPC SearchByKeywords completed: {len(items)} items")
            return grpc_response

        except Exception as e:
            logger.error(f"gRPC SearchByKeywords error: {e}")
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Search by keywords failed: {str(e)}"
            )

    async def GetNewsByTags(self, request, context):
        """Get news articles by tags."""
        try:
            logger.info(f"gRPC GetNewsByTags called: {len(request.tags)} tags")

            # Validate parameters
            if not request.tags:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "At least one tag is required"
                )
                return

            tags = list(request.tags)
            source = (
                grpc_source_to_pydantic(request.source) if request.source != 0 else None
            )
            limit = min(request.limit, 1000) if request.limit > 0 else 100
            offset = max(request.offset, 0)

            # Calculate date range if hours specified
            start_date = None
            if request.hours > 0:
                start_date = datetime.now(timezone.utc) - timedelta(hours=request.hours)

            # Create search request
            search_request = NewsSearchRequest(
                source=source,
                tags=tags,
                start_date=start_date,
                limit=limit,
                offset=offset,
            )

            # Execute search
            items = await self.news_service.search_news(search_request)

            # Get total count
            total_count = await self.news_service.count_news(
                source=source,
                tags=tags,
                start_date=start_date,
            )

            # Build filters summary
            filters_applied = build_filters_summary(
                source=source.value if source else None,
                tags=tags,
                hours=request.hours if request.hours > 0 else None,
                start_date=start_date,
            )

            # Build response
            pydantic_response = build_news_response(
                items=items,
                total_count=total_count,
                limit=limit,
                offset=offset,
                filters_applied=filters_applied,
            )

            # Convert to gRPC response
            grpc_response = convert_pydantic_news_response_to_grpc(pydantic_response)

            logger.info(f"gRPC GetNewsByTags completed: {len(items)} items")
            return grpc_response

        except Exception as e:
            logger.error(f"gRPC GetNewsByTags error: {e}")
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Get news by tags failed: {str(e)}"
            )

    async def GetNewsItem(self, request, context):
        """Get a specific news item by ID."""
        try:
            logger.info(f"gRPC GetNewsItem called: {request.item_id}")

            # Validate parameters
            if not request.item_id:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "Item ID is required"
                )
                return

            # Get news item
            item = await self.news_service.get_news_item(request.item_id)
            if not item:
                await context.abort(grpc.StatusCode.NOT_FOUND, "News item not found")
                return

            # Convert to response format
            response_item = convert_news_item_to_response(item)

            # Convert to gRPC response
            grpc_response = convert_pydantic_news_item_to_grpc(response_item)

            logger.info(f"gRPC GetNewsItem completed: {request.item_id}")
            return grpc_response

        except Exception as e:
            logger.error(f"gRPC GetNewsItem error: {e}")
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Get news item failed: {str(e)}"
            )

    async def GetAvailableTags(self, request, context):
        """Get available tags from news articles."""
        try:
            logger.info(f"gRPC GetAvailableTags called: source={request.source}")

            # Validate parameters
            source = (
                grpc_source_to_pydantic(request.source) if request.source != 0 else None
            )
            limit = min(request.limit, 500) if request.limit > 0 else 100

            # Get available tags
            tags = await self.news_service.get_unique_tags(source=source, limit=limit)

            # Import here to avoid circular imports
            from ..grpc_generated import news_service_pb2

            # Build gRPC response
            grpc_response = news_service_pb2.AvailableTagsResponse()
            grpc_response.tags.extend(tags)
            grpc_response.total_count = len(tags)
            grpc_response.source_filter = source.value if source else ""
            grpc_response.limit = limit

            logger.info(f"gRPC GetAvailableTags completed: {len(tags)} tags")
            return grpc_response

        except Exception as e:
            logger.error(f"gRPC GetAvailableTags error: {e}")
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Get available tags failed: {str(e)}"
            )

    async def GetNewsStatistics(self, request, context):
        """Get comprehensive news statistics."""
        try:
            logger.info("gRPC GetNewsStatistics called")

            # Get statistics
            stats = await self.news_service.get_repository_stats()

            # Convert to Pydantic model first
            from ..schemas.news_schemas import NewsStatsResponse

            pydantic_stats = NewsStatsResponse(
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

            # Convert to gRPC response
            grpc_response = convert_pydantic_stats_to_grpc(pydantic_stats)

            logger.info(
                f"gRPC GetNewsStatistics completed: {pydantic_stats.total_articles} articles"
            )
            return grpc_response

        except Exception as e:
            logger.error(f"gRPC GetNewsStatistics error: {e}")
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Get statistics failed: {str(e)}"
            )

    async def DeleteNewsItem(self, request, context):
        """Delete a specific news item."""
        try:
            logger.info(f"gRPC DeleteNewsItem called: {request.item_id}")

            # Validate parameters
            if not request.item_id:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "Item ID is required"
                )
                return

            # Check if item exists
            item = await self.news_service.get_news_item(request.item_id)
            if not item:
                await context.abort(grpc.StatusCode.NOT_FOUND, "News item not found")
                return

            # Delete item
            success = await self.news_service.delete_news_item(request.item_id)

            # Import here to avoid circular imports
            from ..grpc_generated import news_service_pb2

            # Build response
            grpc_response = news_service_pb2.DeleteResponse()
            grpc_response.success = success
            grpc_response.message = (
                "News item deleted successfully"
                if success
                else "Failed to delete news item"
            )

            logger.info(
                f"gRPC DeleteNewsItem completed: {request.item_id}, success={success}"
            )
            return grpc_response

        except Exception as e:
            logger.error(f"gRPC DeleteNewsItem error: {e}")
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Delete news item failed: {str(e)}"
            )


# Create the servicer class that implements the proto service
def create_news_servicer(news_service: NewsService):
    """
    Create gRPC servicer with news service dependency.

    Args:
        news_service: Core news service instance

    Returns:
        gRPC servicer instance
    """
    try:
        # Import the generated servicer class
        from ..grpc_generated.news_service_pb2_grpc import NewsServiceServicer

        class NewsServiceServicer(NewsServiceServicer):
            def __init__(self, news_service: NewsService):
                self.grpc_service = NewsGrpcService(news_service)

            async def SearchNews(self, request, context):
                return await self.grpc_service.SearchNews(request, context)

            async def GetRecentNews(self, request, context):
                return await self.grpc_service.GetRecentNews(request, context)

            async def GetNewsBySource(self, request, context):
                return await self.grpc_service.GetNewsBySource(request, context)

            async def SearchByKeywords(self, request, context):
                return await self.grpc_service.SearchByKeywords(request, context)

            async def GetNewsByTags(self, request, context):
                return await self.grpc_service.GetNewsByTags(request, context)

            async def GetNewsItem(self, request, context):
                return await self.grpc_service.GetNewsItem(request, context)

            async def GetAvailableTags(self, request, context):
                return await self.grpc_service.GetAvailableTags(request, context)

            async def GetNewsStatistics(self, request, context):
                return await self.grpc_service.GetNewsStatistics(request, context)

            async def DeleteNewsItem(self, request, context):
                return await self.grpc_service.DeleteNewsItem(request, context)

        return NewsServiceServicer(news_service)

    except ImportError:
        logger.error(
            "Failed to import generated gRPC code. Please run generate_grpc_code.py first."
        )
        raise
