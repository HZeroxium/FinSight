"""
Interface definitions for the news crawler service.
"""

from .search_engine import SearchEngine, SearchEngineError
from .message_broker import MessageBroker, MessageBrokerError
from .crawler import NewsCrawler

__all__ = [
    "SearchEngine",
    "SearchEngineError",
    "MessageBroker",
    "MessageBrokerError",
    "NewsCrawler",
]
