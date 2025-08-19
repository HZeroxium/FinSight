# interfaces/__init__.py

"""
Interface definitions for the sentiment analysis service.
"""

from .sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisError
from .message_broker import MessageBroker, MessageBrokerError
from .news_repository_interface import NewsRepositoryInterface

__all__ = [
    "SentimentAnalyzer",
    "SentimentAnalysisError",
    "NewsRepositoryInterface",
    "MessageBroker",
    "MessageBrokerError",
]
