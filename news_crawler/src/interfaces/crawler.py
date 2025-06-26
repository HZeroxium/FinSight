# interfaces/crawler.py
"""
Defines the abstract interface for news crawler services.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models.article import CrawledArticle


class NewsCrawler(ABC):
    """
    Abstract base class for news crawler implementations.
    """

    @abstractmethod
    async def fetch_listings(self) -> List[str]:
        """
        Retrieve a list of article URLs to crawl.

        Returns:
            List[str]: URLs of articles to be fetched.
        """
        pass

    @abstractmethod
    async def fetch_article(self, url: str) -> Optional[CrawledArticle]:
        """
        Fetch and parse a single article by its URL.

        Args:
            url (str): The URL of the article to fetch.

        Returns:
            Optional[CrawledArticle]: Parsed article data or None if failed.
        """
        pass
