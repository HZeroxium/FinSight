# interfaces/__init__.py

"""
Interface definitions for the news crawler service.
"""

from .search_engine import SearchEngine, SearchEngineError
from .message_broker import MessageBroker, MessageBrokerError

__all__ = [
    "SearchEngine",
    "SearchEngineError",
    "MessageBroker",
    "MessageBrokerError",
]
