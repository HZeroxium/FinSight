# interfaces/news_repository_interface.py

"""
News repository interface for sentiment analysis service - synchronized with news_service
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..schemas.news_schemas import NewsSource, NewsItem


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
        tags: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        has_sentiment: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[NewsItem]:
        """
        Search for news items with filters

        Args:
            source: Filter by news source
            keywords: Filter by keywords in title/description
            tags: Filter by tags/categories
            start_date: Filter by published date (start)
            end_date: Filter by published date (end)
            has_sentiment: Filter by sentiment analysis presence
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
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        has_sentiment: Optional[bool] = None,
    ) -> int:
        """
        Count news items with filters

        Args:
            source: Filter by news source
            keywords: Filter by keywords
            tags: Filter by tags
            start_date: Start date filter
            end_date: End date filter
            has_sentiment: Filter by sentiment analysis presence

        Returns:
            int: Number of matching news items
        """
        pass

    @abstractmethod
    async def get_unique_tags(
        self, source: Optional[NewsSource] = None, limit: int = 100
    ) -> List[str]:
        """
        Get unique tags from news items

        Args:
            source: Filter by news source
            limit: Maximum number of tags to return

        Returns:
            List[str]: List of unique tags
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

    @abstractmethod
    async def update_news_sentiment(
        self,
        item_id: str,
        sentiment_label: str,
        sentiment_scores: Dict[str, float],
        sentiment_confidence: float,
        sentiment_reasoning: Optional[str] = None,
        analyzer_version: Optional[str] = None,
    ) -> bool:
        """
        Update sentiment analysis results for a news item

        Args:
            item_id: ID of the news item
            sentiment_label: Sentiment classification
            sentiment_scores: Sentiment scores
            sentiment_confidence: Analysis confidence
            sentiment_reasoning: Optional analysis reasoning
            analyzer_version: Optional analyzer version

        Returns:
            bool: True if update was successful, False otherwise
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close database connection"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check repository health

        Returns:
            bool: True if healthy, False otherwise
        """
        pass
