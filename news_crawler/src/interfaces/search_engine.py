# interfaces/search_engine.py

"""
Search engine interface for news and content discovery.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class SearchEngine(ABC):
    """Abstract base class for search engines."""

    @abstractmethod
    async def search(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a search query.

        Args:
            request: Search parameters as dictionary

        Returns:
            Dict[str, Any]: Search results and metadata as dictionary

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
        self.details = details or {}
