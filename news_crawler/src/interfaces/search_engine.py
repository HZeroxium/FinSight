"""
Search engine interface for news and content discovery.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from ..models.search import SearchRequest, SearchResponse


class SearchEngine(ABC):
    """Abstract base class for search engines."""

    @abstractmethod
    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Perform a search query.

        Args:
            request: Search parameters and configuration

        Returns:
            SearchResponse: Search results and metadata

        Raises:
            SearchEngineError: When search operation fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the search engine is healthy and accessible.

        Returns:
            bool: True if healthy, False otherwise
        """
        pass


class SearchEngineError(Exception):
    """Base exception for search engine operations."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
