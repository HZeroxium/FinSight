from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field

from ..interfaces.news_repository_interface import NewsRepositoryInterface
from ..schemas.news_schemas import NewsItem, NewsSource
from ..common.logger import LoggerFactory, LoggerType, LogLevel


class NewsSearchRequest(BaseModel):
    """Request model for news search operations"""

    source: Optional[NewsSource] = Field(None, description="Filter by news source")
    keywords: Optional[List[str]] = Field(None, description="Keywords to search")
    start_date: Optional[datetime] = Field(None, description="Start date filter")
    end_date: Optional[datetime] = Field(None, description="End date filter")
    limit: int = Field(100, ge=1, le=1000, description="Maximum items to return")
    offset: int = Field(0, ge=0, description="Number of items to skip")


class NewsStorageResult(BaseModel):
    """Result model for news storage operations"""

    item_id: Optional[str] = Field(None, description="ID of stored item")
    is_duplicate: bool = Field(False, description="Whether item was a duplicate")
    success: bool = Field(True, description="Whether operation succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class NewsService:
    """
    News service providing abstraction layer for repository operations.
    Handles business logic, validation, and provides clean API for news operations.
    """

    def __init__(self, repository: NewsRepositoryInterface):
        """
        Initialize news service

        Args:
            repository: News repository implementation
        """
        self.repository = repository
        self.logger = LoggerFactory.get_logger(
            name="news-service",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file="logs/news_service.log",
        )
        self.logger.info("NewsService initialized")

    async def store_news_item(self, news_item: NewsItem) -> NewsStorageResult:
        """
        Store a single news item with duplicate detection

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
            return await self.repository.get_news_item(item_id)
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

            return await self.repository.search_news(
                source=search_request.source,
                keywords=search_request.keywords,
                start_date=search_request.start_date,
                end_date=search_request.end_date,
                limit=search_request.limit,
                offset=search_request.offset,
            )
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
            return await self.repository.get_recent_news(
                source=source, hours=hours, limit=limit
            )
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
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """
        Count news items with filters

        Args:
            source: Filter by news source
            start_date: Start date filter
            end_date: End date filter

        Returns:
            int: Number of matching items
        """
        try:
            return await self.repository.count_news(
                source=source, start_date=start_date, end_date=end_date
            )
        except Exception as e:
            self.logger.error(f"Failed to count news: {str(e)}")
            return 0

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
            return await self.repository.get_repository_stats()
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
