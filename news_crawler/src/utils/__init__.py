"""
Utility functions and dependency injection.
"""

from .dependencies import (
    get_search_engine,
    get_message_broker,
    get_article_repository,
    get_crawler_service,
    get_search_service,
)

__all__ = [
    "get_search_engine",
    "get_message_broker",
    "get_article_repository",
    "get_crawler_service",
    "get_search_service",
]
