# utils/__init__.py

"""
Utility functions and dependency injection.
"""

from .dependencies import (
    get_sentiment_analyzer,
    get_news_repository,
    get_message_broker,
    get_sentiment_service,
    get_news_consumer_service,
)

__all__ = [
    "get_sentiment_analyzer",
    "get_news_repository",
    "get_message_broker",
    "get_sentiment_service",
    "get_news_consumer_service",
]
