# utils/__init__.py

"""
Utility functions and dependency injection.
"""

from .dependencies import (
    get_sentiment_analyzer,
    get_news_repository,
    get_message_broker,
    get_sentiment_service,
    get_sentiment_message_producer,
    get_news_message_consumer,
)

__all__ = [
    "get_sentiment_analyzer",
    "get_news_repository",
    "get_message_broker",
    "get_sentiment_service",
    "get_sentiment_message_producer",
    "get_news_message_consumer",
]
