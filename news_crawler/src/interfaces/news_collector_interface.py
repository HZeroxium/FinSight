from abc import ABC, abstractmethod
from typing import List, Optional
from ..schemas.news_schemas import NewsItem, NewsCollectionResult, NewsCollectorConfig


class NewsCollectorInterface(ABC):
    """Abstract interface for news collectors"""

    def __init__(self, config: NewsCollectorConfig):
        """
        Initialize news collector with configuration

        Args:
            config: Configuration for the news collector
        """
        self.config = config

    @abstractmethod
    async def collect_news(
        self, max_items: Optional[int] = None
    ) -> NewsCollectionResult:
        """
        Collect news items from the source

        Args:
            max_items: Maximum number of items to collect (overrides config)

        Returns:
            NewsCollectionResult containing collected news items
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the news source is available

        Returns:
            True if source is accessible, False otherwise
        """
        pass

    @abstractmethod
    def get_source_info(self) -> dict:
        """
        Get information about the news source

        Returns:
            Dictionary with source information
        """
        pass
