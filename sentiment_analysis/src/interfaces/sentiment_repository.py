"""
Repository interface for sentiment data storage.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..models.sentiment import (
    ProcessedSentiment,
    SentimentQueryFilter,
    SentimentAggregation,
)


class SentimentRepository(ABC):
    """Abstract base class for sentiment data repositories."""

    @abstractmethod
    async def save_sentiment(self, sentiment: ProcessedSentiment) -> str:
        """
        Save processed sentiment to storage.

        Args:
            sentiment: ProcessedSentiment to save

        Returns:
            str: Saved sentiment ID
        """
        pass

    @abstractmethod
    async def get_sentiment(self, sentiment_id: str) -> Optional[ProcessedSentiment]:
        """
        Retrieve sentiment by ID.

        Args:
            sentiment_id: Sentiment ID

        Returns:
            Optional[ProcessedSentiment]: Sentiment or None if not found
        """
        pass

    @abstractmethod
    async def get_sentiment_by_article_id(
        self, article_id: str
    ) -> Optional[ProcessedSentiment]:
        """
        Retrieve sentiment by article ID.

        Args:
            article_id: Article ID

        Returns:
            Optional[ProcessedSentiment]: Sentiment or None if not found
        """
        pass

    @abstractmethod
    async def search_sentiments(
        self, filter_params: SentimentQueryFilter
    ) -> List[ProcessedSentiment]:
        """
        Search sentiments based on filter parameters.

        Args:
            filter_params: Filter parameters

        Returns:
            List[ProcessedSentiment]: Matching sentiments
        """
        pass

    @abstractmethod
    async def get_sentiment_aggregation(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        source_domain: Optional[str] = None,
    ) -> SentimentAggregation:
        """
        Get aggregated sentiment statistics.

        Args:
            date_from: Start date filter
            date_to: End date filter
            source_domain: Source domain filter

        Returns:
            SentimentAggregation: Aggregated statistics
        """
        pass

    @abstractmethod
    async def delete_sentiment(self, sentiment_id: str) -> bool:
        """
        Delete sentiment by ID.

        Args:
            sentiment_id: Sentiment ID

        Returns:
            bool: True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def sentiment_exists(self, article_id: str) -> bool:
        """
        Check if sentiment exists for article.

        Args:
            article_id: Article ID

        Returns:
            bool: True if exists, False otherwise
        """
        pass


class SentimentRepositoryError(Exception):
    """Base exception for sentiment repository operations."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
