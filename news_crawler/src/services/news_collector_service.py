from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from ..core.news_collector_facade import NewsCollectorFacade
from ..schemas.news_schemas import NewsSource, NewsItem
from ..common.logger import LoggerFactory, LoggerType, LogLevel
from .news_service import NewsService


class CollectionRequest(BaseModel):
    """Request model for news collection operations"""

    source: NewsSource = Field(..., description="News source to collect from")
    max_items: Optional[int] = Field(None, description="Maximum items to collect")
    config_overrides: Optional[Dict[str, Any]] = Field(
        None, description="Config overrides"
    )


class MultiSourceCollectionRequest(BaseModel):
    """Request model for multi-source news collection"""

    sources: List[NewsSource] = Field(..., description="News sources to collect from")
    max_items_per_source: Optional[int] = Field(
        None, description="Max items per source"
    )
    config_overrides: Optional[Dict[NewsSource, Dict[str, Any]]] = Field(
        None, description="Config overrides per source"
    )


class CollectionResult(BaseModel):
    """Result model for news collection operations"""

    source: str = Field(..., description="Source name")
    collection_success: bool = Field(..., description="Collection success status")
    items_collected: int = Field(..., description="Number of items collected")
    items_stored: int = Field(..., description="Number of items stored")
    items_duplicated: int = Field(..., description="Number of duplicate items")
    items_failed: int = Field(..., description="Number of failed items")
    collection_error: Optional[str] = Field(
        None, description="Collection error message"
    )
    storage_errors: List[str] = Field(
        default_factory=list, description="Storage errors"
    )


class NewsCollectorService:
    """
    Orchestrates news collection and storage operations.
    Aggregates NewsCollectorFacade and NewsService to provide complete
    news collection workflow with persistence.
    """

    def __init__(self, news_service: NewsService, use_cache: bool = True):
        """
        Initialize news collector service

        Args:
            news_service: News service for repository operations
            use_cache: Whether to cache collector instances
        """
        self.news_service = news_service
        self.collector_facade = NewsCollectorFacade(use_cache=use_cache)

        self.logger = LoggerFactory.get_logger(
            name="news-collector-service",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file="logs/news_collector_service.log",
        )

        self.logger.info("NewsCollectorService initialized")

    async def collect_and_store_from_source(
        self, request: CollectionRequest
    ) -> CollectionResult:
        """
        Collect news from a specific source and store via news service

        Args:
            request: Collection request parameters

        Returns:
            CollectionResult: Collection and storage results
        """
        self.logger.info(f"Starting collection from {request.source.value}")

        # Collect news using facade
        collection_result = await self.collector_facade.collect_from_source(
            source=request.source,
            max_items=request.max_items,
            config_overrides=request.config_overrides,
        )

        # Store results using news service
        if collection_result.success and collection_result.items:
            storage_result = await self.news_service.store_news_items_bulk(
                collection_result.items
            )
        else:
            storage_result = {
                "stored_count": 0,
                "duplicate_count": 0,
                "failed_count": 0,
                "errors": [],
            }

        result = CollectionResult(
            source=request.source.value,
            collection_success=collection_result.success,
            items_collected=collection_result.total_items,
            items_stored=storage_result["stored_count"],
            items_duplicated=storage_result["duplicate_count"],
            items_failed=storage_result["failed_count"],
            collection_error=collection_result.error_message,
            storage_errors=storage_result["errors"],
        )

        self.logger.info(
            f"Collection from {request.source.value} completed - "
            f"Collected: {result.items_collected}, "
            f"Stored: {result.items_stored}, "
            f"Duplicates: {result.items_duplicated}"
        )

        return result

    async def collect_and_store_from_multiple_sources(
        self, request: MultiSourceCollectionRequest
    ) -> Dict[str, Any]:
        """
        Collect news from multiple sources and store via news service

        Args:
            request: Multi-source collection request

        Returns:
            Dict[str, Any]: Aggregated collection and storage results
        """
        self.logger.info(f"Starting collection from {len(request.sources)} sources")

        # Collect from multiple sources
        collection_results = await self.collector_facade.collect_from_multiple_sources(
            sources=request.sources,
            max_items_per_source=request.max_items_per_source,
            config_overrides=request.config_overrides,
        )

        # Process each source result
        source_results = {}
        all_items = []

        for source, collection_result in collection_results.items():
            all_items.extend(collection_result.items)
            source_results[source.value] = {
                "collection_success": collection_result.success,
                "items_collected": collection_result.total_items,
                "collection_error": collection_result.error_message,
            }

        # Store all items in bulk
        if all_items:
            storage_result = await self.news_service.store_news_items_bulk(all_items)
        else:
            storage_result = {
                "stored_count": 0,
                "duplicate_count": 0,
                "failed_count": 0,
                "errors": [],
            }

        # Create aggregated result
        result = {
            "sources": list(source_results.keys()),
            "total_items_collected": len(all_items),
            "total_items_stored": storage_result["stored_count"],
            "total_items_duplicated": storage_result["duplicate_count"],
            "total_items_failed": storage_result["failed_count"],
            "source_results": source_results,
            "storage_errors": storage_result["errors"],
        }

        self.logger.info(
            f"Multi-source collection completed - "
            f"Sources: {len(request.sources)}, "
            f"Collected: {result['total_items_collected']}, "
            f"Stored: {result['total_items_stored']}, "
            f"Duplicates: {result['total_items_duplicated']}"
        )

        return result

    async def collect_and_store_all_supported(
        self, max_items_per_source: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Collect news from all supported sources

        Args:
            max_items_per_source: Maximum items per source

        Returns:
            Dict[str, Any]: Collection and storage results
        """
        self.logger.info("Starting collection from all supported sources")

        # Get all supported sources
        from ..core.news_collector_factory import NewsCollectorFactory

        supported_sources = NewsCollectorFactory.get_supported_sources()

        request = MultiSourceCollectionRequest(
            sources=supported_sources, max_items_per_source=max_items_per_source
        )

        return await self.collect_and_store_from_multiple_sources(request)

    async def get_recent_news(
        self,
        source: Optional[NewsSource] = None,
        hours: int = 24,
        limit: int = 100,
    ) -> List[NewsItem]:
        """
        Get recent news from storage

        Args:
            source: Filter by news source
            hours: Number of hours to look back
            limit: Maximum number of items to return

        Returns:
            List[NewsItem]: List of recent news items
        """
        return await self.news_service.get_recent_news(
            source=source, hours=hours, limit=limit
        )

    async def search_stored_news(
        self,
        source: Optional[NewsSource] = None,
        keywords: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[NewsItem]:
        """
        Search stored news with filters

        Args:
            source: Filter by news source
            keywords: Keywords to search in title/description
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of items to return
            offset: Number of items to skip

        Returns:
            List[NewsItem]: List of matching news items
        """
        from .news_service import NewsSearchRequest

        search_request = NewsSearchRequest(
            source=source,
            keywords=keywords,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )

        return await self.news_service.search_news(search_request)

    async def get_repository_stats(self) -> Dict[str, Any]:
        """
        Get repository statistics

        Returns:
            Dict[str, Any]: Repository statistics
        """
        return await self.news_service.get_repository_stats()

    async def cleanup_old_news(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """
        Clean up old news items

        Args:
            days_to_keep: Number of days of news to keep

        Returns:
            Dict[str, Any]: Cleanup results
        """
        return await self.news_service.cleanup_old_news(days_to_keep)

    async def get_news_by_source(
        self, source: NewsSource, limit: int = 100, offset: int = 0
    ) -> List[NewsItem]:
        """
        Get news items from specific source

        Args:
            source: News source to filter by
            limit: Maximum number of items
            offset: Number of items to skip

        Returns:
            List[NewsItem]: News items from source
        """
        return await self.news_service.get_news_by_source(
            source=source, limit=limit, offset=offset
        )

    async def close(self) -> None:
        """Close service and cleanup resources"""
        try:
            self.collector_facade.clear_cache()
            self.logger.info("NewsCollectorService closed")
        except Exception as e:
            self.logger.error(f"Error closing NewsCollectorService: {e}")
