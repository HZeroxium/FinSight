"""
Sentiment analyzer interface for text analysis.
"""

from abc import ABC, abstractmethod
from typing import List
from ..models.sentiment import SentimentAnalysisResult, SentimentRequest


class SentimentAnalyzer(ABC):
    """Abstract base class for sentiment analyzers."""

    @abstractmethod
    async def analyze(self, text: str, title: str = None) -> SentimentAnalysisResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text content to analyze
            title: Optional title for context

        Returns:
            SentimentAnalysisResult: Analysis results

        Raises:
            SentimentAnalysisError: When analysis fails
        """
        pass

    @abstractmethod
    async def analyze_batch(
        self, requests: List[SentimentRequest]
    ) -> List[SentimentAnalysisResult]:
        """
        Analyze sentiment of multiple texts in batch.

        Args:
            requests: List of sentiment requests

        Returns:
            List[SentimentAnalysisResult]: Analysis results

        Raises:
            SentimentAnalysisError: When batch analysis fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the sentiment analyzer is healthy and accessible.

        Returns:
            bool: True if healthy, False otherwise
        """
        pass


class SentimentAnalysisError(Exception):
    """Base exception for sentiment analysis operations."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
