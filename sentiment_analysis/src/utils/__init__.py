"""
Utility functions and dependency injection.
"""

from .dependencies import (
    get_sentiment_analyzer,
    get_sentiment_repository,
    get_message_broker,
    get_sentiment_service,
)

__all__ = [
    "get_sentiment_analyzer",
    "get_sentiment_repository",
    "get_message_broker",
    "get_sentiment_service",
]
