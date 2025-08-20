# services/news_service.py

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta

from ..interfaces.news_repository_interface import NewsRepositoryInterface
from ..schemas.news_schemas import (
    NewsItem,
    NewsSource,
    NewsStorageResult,
    NewsSearchRequest,
)
from ..utils.cache_utils import CacheEndpoint, get_cache_manager, CacheManager
from .news_message_producer_service import NewsMessageProducerService
from common.logger import LoggerFactory, LoggerType, LogLevel


class NewsService:
    """
    News service providing abstraction layer for repository operations.
    Handles business logic, validation, and provides clean API for news operations.
    """

    def __init__(
        self,
        repository: NewsRepositoryInterface,
        cache_manager: Optional[CacheManager] = None,
        message_producer: Optional[NewsMessageProducerService] = None,
    ):
        """
        Initialize news service

        Args:
            repository: News repository implementation
            cache_manager: Optional cache manager for performance
            message_producer: Optional message producer for sentiment analysis
        """
        self.repository = repository
        self.message_producer = message_producer

        self.logger = LoggerFactory.get_logger(
            name="news-service",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.DEBUG,
            file_level=LogLevel.DEBUG,
            console_level=LogLevel.DEBUG,
            log_file="logs/news_service.log",
        )
        self.logger.info("NewsService initialized")

        # Cache manager (injected if available, otherwise lazily initialized once)
        self._cache_manager: Optional[CacheManager] = cache_manager
        self._cache_initialized: bool = cache_manager is not None

    async def _ensure_cache_manager(self) -> CacheManager:
        """Ensure a single CacheManager instance is available and reused.

        Returns:
            CacheManager: Ready-to-use cache manager instance
        """
        if not self._cache_initialized or self._cache_manager is None:
            self._cache_manager = await get_cache_manager()
            self._cache_initialized = True
            self.logger.debug("[CACHE] CacheManager initialized for NewsService")
        return self._cache_manager

    async def store_news_item(self, news_item: NewsItem) -> NewsStorageResult:
        """
        Store a single news item with duplicate detection and optional message publishing

        Args:
            news_item: News item to store

        Returns:
            NewsStorageResult: Storage operation result
        """
        try:
            # Check for duplicates
            is_duplicate = await self._check_duplicate(news_item)

            if is_duplicate:
                self.logger.debug(
                    f"Duplicate news item detected: {news_item.title[:50]}..."
                )
                return NewsStorageResult(
                    is_duplicate=True, success=True, error_message="Item already exists"
                )

            # Store the item
            item_id = await self.repository.save_news_item(news_item)

            self.logger.debug(f"Published news item with ID: {item_id}")

            if not self.message_producer:
                self.logger.warning(
                    "Message producer not available, skipping publishing"
                )

            # Publish to sentiment analysis queue if message producer is available
            if self.message_producer and item_id:
                try:
                    await self.message_producer.publish_news_for_sentiment(
                        news_item=news_item, article_id=item_id
                    )
                    self.logger.debug(
                        f"Published news item to sentiment queue: {item_id}"
                    )
                except Exception as e:
                    # Don't fail the storage operation if publishing fails
                    self.logger.warning(
                        f"Failed to publish news item to sentiment queue: {e}"
                    )

            # Invalidate relevant caches after storing new item
            await self._invalidate_related_caches(news_item)

            self.logger.debug(f"Stored news item: {item_id}")
            return NewsStorageResult(item_id=item_id, is_duplicate=False, success=True)

        except Exception as e:
            error_msg = f"Failed to store news item: {str(e)}"
            self.logger.error(error_msg)
            return NewsStorageResult(success=False, error_message=error_msg)

    async def store_news_items_bulk(self, news_items: List[NewsItem]) -> Dict[str, Any]:
        """
        Store multiple news items in bulk with detailed results

        Args:
            news_items: List of news items to store

        Returns:
            Dict[str, Any]: Bulk storage results
        """
        stored_count = 0
        duplicate_count = 0
        failed_count = 0
        errors = []
        stored_ids = []

        self.logger.info(f"Starting bulk storage of {len(news_items)} news items")

        for item in news_items:
            result = await self.store_news_item(item)

            if result.success:
                if result.is_duplicate:
                    duplicate_count += 1
                else:
                    stored_count += 1
                    if result.item_id:
                        stored_ids.append(result.item_id)
            else:
                failed_count += 1
                if result.error_message:
                    errors.append(result.error_message)

        # Invalidate all cache after bulk storage
        if stored_count > 0:
            await self._invalidate_all_cache()

        result_summary = {
            "total_items": len(news_items),
            "stored_count": stored_count,
            "duplicate_count": duplicate_count,
            "failed_count": failed_count,
            "stored_ids": stored_ids,
            "errors": errors,
        }

        self.logger.info(
            f"Bulk storage completed: {stored_count} stored, "
            f"{duplicate_count} duplicates, {failed_count} failed"
        )

        return result_summary

    async def get_news_item(self, item_id: str) -> Optional[NewsItem]:
        """
        Retrieve a news item by ID

        Args:
            item_id: ID of the news item

        Returns:
            Optional[NewsItem]: News item if found
        """
        try:
            cache_manager = await self._ensure_cache_manager()

            # Try to get from cache first
            cached_item = await cache_manager.get_cached_data(
                CacheEndpoint.NEWS_ITEM, item_id
            )

            if cached_item is not None:
                self.logger.debug(f"[CACHE] HIT news_item id={item_id}")
                return NewsItem(**cached_item)

            self.logger.debug(f"[CACHE] MISS news_item id={item_id}")

            # Get from repository
            item = await self.repository.get_news_item(item_id)

            if item:
                # Cache the result
                await cache_manager.set_cached_data(
                    CacheEndpoint.NEWS_ITEM, item.model_dump(), item_id
                )
                self.logger.debug(f"[CACHE] SET news_item id={item_id}")
            else:
                self.logger.debug(f"[CACHE] NO_DATA news_item id={item_id}")

            return item

        except Exception as e:
            self.logger.error(f"Failed to get news item {item_id}: {str(e)}")
            return None

    async def search_news(self, search_request: NewsSearchRequest) -> List[NewsItem]:
        """
        Search news items with filters

        Args:
            search_request: Search parameters

        Returns:
            List[NewsItem]: Matching news items
        """
        try:
            self.logger.debug(
                f"Searching news with filters: {search_request.model_dump()}"
            )

            cache_manager = await self._ensure_cache_manager()
            cache_key = self._generate_search_cache_key(search_request)

            # Try to get from cache first
            cached_result = await cache_manager.get_cached_data(
                CacheEndpoint.SEARCH_NEWS, cache_key
            )

            if cached_result is not None:
                self.logger.debug(f"[CACHE] HIT search key={cache_key}")
                return [NewsItem(**item) for item in cached_result]

            self.logger.debug(f"[CACHE] MISS search key={cache_key}")

            # Execute search
            items = await self.repository.search_news(
                source=search_request.source,
                keywords=search_request.keywords,
                tags=search_request.tags,
                start_date=search_request.start_date,
                end_date=search_request.end_date,
                limit=search_request.limit,
                offset=search_request.offset,
            )

            # Cache the result
            items_data = [item.model_dump() for item in items]
            await cache_manager.set_cached_data(
                CacheEndpoint.SEARCH_NEWS, items_data, cache_key
            )
            self.logger.debug(f"[CACHE] SET search key={cache_key}")

            return items

        except Exception as e:
            self.logger.error(f"Failed to search news: {str(e)}")
            return []

    async def get_recent_news(
        self, source: Optional[NewsSource] = None, hours: int = 24, limit: int = 100
    ) -> List[NewsItem]:
        """
        Get recent news items

        Args:
            source: Filter by news source
            hours: Number of hours to look back
            limit: Maximum number of items

        Returns:
            List[NewsItem]: Recent news items
        """
        try:
            cache_manager = await self._ensure_cache_manager()
            # Try to get from cache first
            cached_result = await cache_manager.get_cached_data(
                CacheEndpoint.RECENT_NEWS, source, hours, limit
            )

            if cached_result is not None:
                self.logger.debug(
                    f"[CACHE] HIT recent source={getattr(source,'value',None)} hours={hours} limit={limit}"
                )
                return [NewsItem(**item) for item in cached_result]

            self.logger.debug(
                f"[CACHE] MISS recent source={getattr(source,'value',None)} hours={hours} limit={limit}"
            )

            # Get from repository
            items = await self.repository.get_recent_news(
                source=source, hours=hours, limit=limit
            )

            # Cache the result
            items_data = [item.model_dump() for item in items]
            await cache_manager.set_cached_data(
                CacheEndpoint.RECENT_NEWS, items_data, source, hours, limit
            )
            self.logger.debug(
                f"[CACHE] SET recent source={getattr(source,'value',None)} hours={hours} limit={limit}"
            )

            return items

        except Exception as e:
            self.logger.error(f"Failed to get recent news: {str(e)}")
            return []

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
        search_request = NewsSearchRequest(source=source, limit=limit, offset=offset)
        return await self.search_news(search_request)

    async def get_news_by_keywords(
        self, keywords: List[str], limit: int = 100
    ) -> List[NewsItem]:
        """
        Get news items matching keywords

        Args:
            keywords: Keywords to search for
            limit: Maximum number of items

        Returns:
            List[NewsItem]: Matching news items
        """
        search_request = NewsSearchRequest(keywords=keywords, limit=limit)
        return await self.search_news(search_request)

    async def get_news_in_date_range(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        source: Optional[NewsSource] = None,
        limit: int = 100,
    ) -> List[NewsItem]:
        """
        Get news items within date range

        Args:
            start_date: Start date for search
            end_date: End date for search (defaults to now)
            source: Optional source filter
            limit: Maximum number of items

        Returns:
            List[NewsItem]: News items in date range
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        search_request = NewsSearchRequest(
            source=source, start_date=start_date, end_date=end_date, limit=limit
        )
        return await self.search_news(search_request)

    async def count_news(
        self,
        source: Optional[NewsSource] = None,
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """
        Count news items with filters

        Args:
            source: Filter by news source
            keywords: Keywords to search for
            tags: Tags to filter by
            start_date: Start date filter
            end_date: End date filter

        Returns:
            int: Number of matching items
        """
        try:
            return await self.repository.count_news(
                source=source,
                keywords=keywords,
                tags=tags,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as e:
            self.logger.error(f"Failed to count news: {str(e)}")
            return 0

    async def get_news_by_tags(
        self, tags: List[str], limit: int = 100, offset: int = 0
    ) -> List[NewsItem]:
        """
        Get news items matching specific tags

        Args:
            tags: Tags to filter by
            limit: Maximum number of items
            offset: Number of items to skip

        Returns:
            List[NewsItem]: Matching news items
        """
        search_request = NewsSearchRequest(tags=tags, limit=limit, offset=offset)
        return await self.search_news(search_request)

    async def get_unique_tags(
        self, source: Optional[NewsSource] = None, limit: int = 100
    ) -> List[str]:
        """
        Get unique tags from news items

        Args:
            source: Optional source filter
            limit: Maximum number of tags to return

        Returns:
            List[str]: Unique tags sorted by frequency
        """
        try:
            cache_manager = await self._ensure_cache_manager()
            # Try to get from cache first
            cached_result = await cache_manager.get_cached_data(
                CacheEndpoint.AVAILABLE_TAGS, source, limit
            )

            if cached_result is not None:
                self.logger.debug(
                    f"[CACHE] HIT available_tags source={getattr(source,'value',None)} limit={limit}"
                )
                return cached_result

            self.logger.debug(
                f"[CACHE] MISS available_tags source={getattr(source,'value',None)} limit={limit}"
            )

            # Get from repository
            tags = await self.repository.get_unique_tags(source=source, limit=limit)

            # Cache the result
            await cache_manager.set_cached_data(
                CacheEndpoint.AVAILABLE_TAGS, tags, source, limit
            )
            self.logger.debug(
                f"[CACHE] SET available_tags source={getattr(source,'value',None)} limit={limit} count={len(tags)}"
            )

            return tags

        except Exception as e:
            self.logger.error(f"Failed to get unique tags: {str(e)}")
            return []

    async def delete_news_item(self, item_id: str) -> bool:
        """
        Delete a news item

        Args:
            item_id: ID of item to delete

        Returns:
            bool: True if deleted successfully
        """
        try:
            result = await self.repository.delete_news_item(item_id)
            if result:
                # Invalidate related caches
                await self._invalidate_related_caches_by_id(item_id)
                self.logger.info(f"Deleted news item: {item_id}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to delete news item {item_id}: {str(e)}")
            return False

    async def get_repository_stats(self) -> Dict[str, Any]:
        """
        Get repository statistics

        Returns:
            Dict[str, Any]: Repository statistics
        """
        try:
            cache_manager = await self._ensure_cache_manager()
            # Try to get from cache first
            cached_result = await cache_manager.get_cached_data(
                CacheEndpoint.REPOSITORY_STATS
            )

            if cached_result is not None:
                self.logger.debug("[CACHE] HIT repository_stats")
                return cached_result

            self.logger.debug("[CACHE] MISS repository_stats")

            # Get from repository
            stats = await self.repository.get_repository_stats()

            # Cache the result
            await cache_manager.set_cached_data(CacheEndpoint.REPOSITORY_STATS, stats)
            self.logger.debug("[CACHE] SET repository_stats")

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get repository stats: {str(e)}")
            return {}

    async def cleanup_old_news(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """
        Clean up old news items

        Args:
            days_to_keep: Number of days of news to keep

        Returns:
            Dict[str, Any]: Cleanup results
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

            # Count items to be deleted
            count_to_delete = await self.count_news(end_date=cutoff_date)

            if count_to_delete == 0:
                self.logger.info("No old news items found for cleanup")
                return {"deleted_count": 0, "cutoff_date": cutoff_date}

            # Get old items and delete them
            old_items = await self.search_news(
                NewsSearchRequest(end_date=cutoff_date, limit=count_to_delete)
            )

            deleted_count = 0
            for item in old_items:
                if item.metadata.get("_id"):
                    if await self.delete_news_item(str(item.metadata["_id"])):
                        deleted_count += 1

            # Invalidate all cache after cleanup
            if deleted_count > 0:
                await self._invalidate_all_cache()

            self.logger.info(f"Cleaned up {deleted_count} old news items")

            return {
                "deleted_count": deleted_count,
                "cutoff_date": cutoff_date,
                "days_kept": days_to_keep,
            }

        except Exception as e:
            error_msg = f"Failed to cleanup old news: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}

    async def _check_duplicate(self, news_item: NewsItem) -> bool:
        """
        Check if news item is a duplicate

        Args:
            news_item: News item to check

        Returns:
            bool: True if duplicate exists
        """
        try:
            # Check by URL first
            existing_by_url = await self.repository.get_news_by_url(str(news_item.url))
            if existing_by_url:
                return True

            # Check by GUID if available
            if news_item.guid:
                existing_by_guid = await self.repository.get_news_by_guid(
                    news_item.source, news_item.guid
                )
                if existing_by_guid:
                    return True

            return False

        except Exception as e:
            self.logger.warning(f"Error checking duplicates: {str(e)}")
            # If we can't check, assume it's not a duplicate to avoid losing data
            return False

    def _generate_search_cache_key(self, search_request: NewsSearchRequest) -> str:
        """Generate cache key for search request"""
        key_parts = [
            str(search_request.source.value) if search_request.source else "all",
            str(search_request.limit),
            str(search_request.offset),
        ]

        if search_request.keywords:
            key_parts.append("keywords_" + "_".join(search_request.keywords))

        if search_request.tags:
            key_parts.append("tags_" + "_".join(search_request.tags))

        if search_request.start_date:
            key_parts.append(f"start_{search_request.start_date.isoformat()}")

        if search_request.end_date:
            key_parts.append(f"end_{search_request.end_date.isoformat()}")

        return "_".join(key_parts)

    async def _invalidate_related_caches(self, news_item: NewsItem) -> None:
        """Invalidate caches related to a news item"""
        try:
            cache_manager = await self._ensure_cache_manager()

            # Invalidate source-specific caches
            await cache_manager.invalidate_endpoint_cache(
                CacheEndpoint.NEWS_BY_SOURCE, news_item.source
            )

            # Invalidate tag-related caches
            if news_item.tags:
                for tag in news_item.tags:
                    await cache_manager.invalidate_endpoint_cache(
                        CacheEndpoint.NEWS_BY_TAGS, [tag]
                    )

            # Invalidate recent news cache
            await cache_manager.invalidate_endpoint_cache(CacheEndpoint.RECENT_NEWS)

            # Invalidate available tags cache
            await cache_manager.invalidate_endpoint_cache(CacheEndpoint.AVAILABLE_TAGS)

            # Invalidate repository stats cache
            await cache_manager.invalidate_endpoint_cache(
                CacheEndpoint.REPOSITORY_STATS
            )

            self.logger.debug(
                f"Invalidated related caches for news item: {news_item.title[:50]}"
            )

        except Exception as e:
            self.logger.error(f"Error invalidating related caches: {e}")

    async def _invalidate_related_caches_by_id(self, item_id: str) -> None:
        """Invalidate caches related to a news item by ID"""
        try:
            cache_manager = await get_cache_manager()

            # Invalidate the specific news item cache
            await cache_manager.invalidate_endpoint_cache(
                CacheEndpoint.NEWS_ITEM, item_id
            )

            # Invalidate other related caches
            await cache_manager.invalidate_endpoint_cache(CacheEndpoint.RECENT_NEWS)
            await cache_manager.invalidate_endpoint_cache(
                CacheEndpoint.REPOSITORY_STATS
            )

            self.logger.debug(f"Invalidated related caches for news item ID: {item_id}")

        except Exception as e:
            self.logger.error(f"Error invalidating related caches by ID: {e}")

    async def _invalidate_all_cache(self) -> None:
        """Invalidate all cache entries"""
        try:
            cache_manager = await get_cache_manager()
            success = await cache_manager.invalidate_all_cache()
            if success:
                self.logger.info("[CACHE] INVALIDATE_ALL completed")
            else:
                self.logger.warning("[CACHE] INVALIDATE_ALL no entries found")
        except Exception as e:
            self.logger.error(f"[CACHE] INVALIDATE_ALL error: {e}")
