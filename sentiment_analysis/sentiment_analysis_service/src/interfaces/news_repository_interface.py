# interfaces/news_repository_interface.py

"""
News repository interface for sentiment analysis service.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from datetime import datetime

from ..schemas.news_schemas import NewsSource
from ..models.news_model import NewsModel


class NewsRepositoryInterface(ABC):
    """Abstract interface for news repository operations"""

    @abstractmethod
    async def get_news_item(self, item_id: str) -> Optional[NewsModel]:
        """
        Retrieve a news item by ID

        Args:
            item_id: ID of the news item

        Returns:
            Optional[NewsModel]: News item if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_news_by_url(self, url: str) -> Optional[NewsModel]:
        """
        Retrieve a news item by URL

        Args:
            url: URL of the news item

        Returns:
            Optional[NewsModel]: News item if found, None otherwise
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
    async def search_news(
        self,
        source: Optional[NewsSource] = None,
        keywords: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        has_sentiment: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[NewsModel]:
        """
        Search for news items with filters

        Args:
            source: Filter by news source
            keywords: Filter by keywords in title/description
            start_date: Filter by published date (start)
            end_date: Filter by published date (end)
            has_sentiment: Filter by sentiment analysis presence
            limit: Maximum number of items to return
            offset: Number of items to skip

        Returns:
            List[NewsModel]: List of matching news items
        """
        pass

    @abstractmethod
    async def count_news(
        self,
        source: Optional[NewsSource] = None,
        keywords: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        has_sentiment: Optional[bool] = None,
    ) -> int:
        """
        Count news items with filters

        Args:
            source: Filter by news source
            keywords: Filter by keywords in title/description
            start_date: Filter by published date (start)
            end_date: Filter by published date (end)
            has_sentiment: Filter by sentiment analysis presence

        Returns:
            int: Number of matching news items
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the repository (e.g., create indexes)"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check repository health

        Returns:
            bool: True if repository is healthy, False otherwise
        """
        pass
