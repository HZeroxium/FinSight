# interfaces/__init__.py

"""
Interface definitions for the news crawler service.
"""

from .message_broker import MessageBroker, MessageBrokerError
from .search_engine import SearchEngine, SearchEngineError

__all__ = [
    "SearchEngine",
    "SearchEngineError",
    "MessageBroker",
    "MessageBrokerError",
]
