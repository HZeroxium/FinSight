# services/news_collector_service.py

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from ..core.news_collector_facade import NewsCollectorFacade
from ..core.news_collector_factory import CollectorType
from ..schemas.news_schemas import NewsSource, NewsItem
from ..common.logger import LoggerFactory, LoggerType, LogLevel
from .news_service import NewsService


class CollectionRequest(BaseModel):
    """Enhanced request model for news collection operations"""

    source: NewsSource = Field(..., description="News source to collect from")
    collector_type: Optional[CollectorType] = Field(
        None, description="Preferred collector type"
    )
    max_items: Optional[int] = Field(None, description="Maximum items to collect")
    config_overrides: Optional[Dict[str, Any]] = Field(
        None, description="Config overrides"
    )
    enable_fallback: bool = Field(
        True, description="Enable fallback collectors on failure"
    )


class BatchCollectionRequest(BaseModel):
    """Request model for batch news collection from multiple sources"""

    sources: List[NewsSource] = Field(..., description="News sources to collect from")
    collector_preferences: Optional[Dict[str, str]] = Field(
        None, description="Collector preferences per source"
    )
    max_items_per_source: Optional[int] = Field(
        None, description="Max items per source"
    )
    config_overrides: Optional[Dict[str, Dict[str, Any]]] = Field(
        None, description="Config overrides per source"
    )
    enable_fallback: bool = Field(True, description="Enable fallback collectors")


class CollectionResult(BaseModel):
    """Result model for individual source collection operations"""

    source: str = Field(..., description="Source name")
    collection_success: bool = Field(..., description="Whether collection succeeded")
    items_collected: int = Field(..., description="Number of items collected")
    items_stored: int = Field(..., description="Number of items stored")
    items_duplicated: int = Field(..., description="Number of duplicate items")
    items_failed: int = Field(..., description="Number of items that failed to store")
    collection_error: Optional[str] = Field(
        None, description="Collection error message"
    )
    storage_errors: List[str] = Field(
        default_factory=list, description="Storage errors"
    )


class NewsCollectorService:
    """
    Enhanced news collector service with flexible adapter support and fallback mechanisms
    """

    def __init__(
        self,
        news_service: NewsService,
        use_cache: bool = True,
        enable_fallback: bool = True,
    ):
        """
        Initialize enhanced news collector service

        Args:
            news_service: News service for repository operations
            use_cache: Whether to cache collector instances
            enable_fallback: Whether to enable fallback mechanisms
        """
        self.news_service = news_service
        self.collector_facade = NewsCollectorFacade(
            use_cache=use_cache, enable_fallback=enable_fallback
        )

        self.logger = LoggerFactory.get_logger(
            name="news-collector-service",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file="logs/news_collector_service.log",
        )

        self.logger.info("Enhanced NewsCollectorService initialized")

    async def collect_and_store(self, request: CollectionRequest) -> Dict[str, Any]:
        """
        Enhanced collection with flexible collector selection and fallback

        Args:
            request: Enhanced collection request parameters

        Returns:
            Detailed collection and storage results
        """
        self.logger.info(f"Starting enhanced collection from {request.source.value}")

        # Collect news using enhanced facade
        collection_result = await self.collector_facade.collect_from_source(
            source=request.source,
            collector_type=request.collector_type,
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

        result = {
            "source": request.source.value,
            "collector_type": (
                request.collector_type.value if request.collector_type else "auto"
            ),
            "collection_success": collection_result.success,
            "items_collected": collection_result.total_items,
            "items_stored": storage_result["stored_count"],
            "items_duplicated": storage_result["duplicate_count"],
            "items_failed": storage_result["failed_count"],
            "collection_error": collection_result.error_message,
            "storage_errors": storage_result["errors"],
        }

        self.logger.info(
            f"Enhanced collection from {request.source.value} completed - "
            f"Collected: {result['items_collected']}, "
            f"Stored: {result['items_stored']}"
        )

        return result

    async def collect_and_store_batch(
        self, request: BatchCollectionRequest
    ) -> Dict[str, Any]:
        """
        Batch collection from multiple sources with flexible configuration

        Args:
            request: Batch collection request

        Returns:
            Aggregated collection and storage results
        """
        self.logger.info(
            f"Starting batch collection from {len(request.sources)} sources"
        )

        # Convert string collector preferences to enum
        collector_preferences = None
        if request.collector_preferences:
            collector_preferences = {
                NewsSource(source): CollectorType(collector_type)
                for source, collector_type in request.collector_preferences.items()
            }

        # Convert string source keys to NewsSource enum for config overrides
        config_overrides = None
        if request.config_overrides:
            config_overrides = {
                NewsSource(source): config
                for source, config in request.config_overrides.items()
            }

        # Collect from multiple sources
        collection_results = await self.collector_facade.collect_from_multiple_sources(
            sources=request.sources,
            collector_preferences=collector_preferences,
            max_items_per_source=request.max_items_per_source,
            config_overrides=config_overrides,
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
            f"Batch collection completed - "
            f"Sources: {len(request.sources)}, "
            f"Collected: {result['total_items_collected']}, "
            f"Stored: {result['total_items_stored']}"
        )

        return result

    async def collect_all_with_best_adapters(
        self, max_items_per_source: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Collect from all sources using the best available adapters

        Args:
            max_items_per_source: Maximum items per source

        Returns:
            Collection and storage results
        """
        self.logger.info("Starting collection from all sources with best adapters")

        # Use all supported sources with best collectors
        collection_results = await self.collector_facade.collect_all_supported_sources(
            max_items_per_source=max_items_per_source, use_best_collectors=True
        )

        # Process and store results
        all_items = []
        source_results = {}

        for source, collection_result in collection_results.items():
            all_items.extend(collection_result.items)
            source_results[source.value] = {
                "collection_success": collection_result.success,
                "items_collected": collection_result.total_items,
                "collection_error": collection_result.error_message,
            }

        # Store all items
        if all_items:
            storage_result = await self.news_service.store_news_items_bulk(all_items)
        else:
            storage_result = {
                "stored_count": 0,
                "duplicate_count": 0,
                "failed_count": 0,
                "errors": [],
            }

        return {
            "sources": list(source_results.keys()),
            "total_items_collected": len(all_items),
            "total_items_stored": storage_result["stored_count"],
            "total_items_duplicated": storage_result["duplicate_count"],
            "source_results": source_results,
        }

    def get_available_adapters(self) -> Dict[str, List[str]]:
        """Get available adapters for each source"""
        adapters = {}
        for source in NewsSource:
            adapters[source.value] = self.collector_facade.get_available_collectors(
                source
            )
        return adapters

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
