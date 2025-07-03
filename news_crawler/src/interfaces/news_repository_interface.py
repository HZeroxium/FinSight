from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..schemas.news_schemas import NewsItem, NewsSource


class NewsRepositoryInterface(ABC):
    """Abstract interface for news repository operations"""

    @abstractmethod
    async def save_news_item(self, news_item: NewsItem) -> str:
        """
        Save a news item to the repository

        Args:
            news_item: NewsItem to save

        Returns:
            str: ID of the saved news item
        """
        pass

    @abstractmethod
    async def get_news_item(self, item_id: str) -> Optional[NewsItem]:
        """
        Retrieve a news item by ID

        Args:
            item_id: ID of the news item

        Returns:
            Optional[NewsItem]: News item if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_news_by_url(self, url: str) -> Optional[NewsItem]:
        """
        Retrieve a news item by URL

        Args:
            url: URL of the news item

        Returns:
            Optional[NewsItem]: News item if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_news_by_guid(
        self, source: NewsSource, guid: str
    ) -> Optional[NewsItem]:
        """
        Retrieve a news item by source and GUID

        Args:
            source: News source
            guid: GUID of the news item

        Returns:
            Optional[NewsItem]: News item if found, None otherwise
        """
        pass

    @abstractmethod
    async def search_news(
        self,
        source: Optional[NewsSource] = None,
        keywords: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[NewsItem]:
        """
        Search news items with filters

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
        pass

    @abstractmethod
    async def get_recent_news(
        self,
        source: Optional[NewsSource] = None,
        hours: int = 24,
        limit: int = 100,
    ) -> List[NewsItem]:
        """
        Get recent news items

        Args:
            source: Filter by news source
            hours: Number of hours to look back
            limit: Maximum number of items to return

        Returns:
            List[NewsItem]: List of recent news items
        """
        pass

    @abstractmethod
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
            int: Number of matching news items
        """
        pass

    @abstractmethod
    async def delete_news_item(self, item_id: str) -> bool:
        """
        Delete a news item

        Args:
            item_id: ID of the news item to delete

        Returns:
            bool: True if deleted successfully, False otherwise
        """
        pass

    @abstractmethod
    async def get_repository_stats(self) -> Dict[str, Any]:
        """
        Get repository statistics

        Returns:
            Dict[str, Any]: Repository statistics
        """
        pass
