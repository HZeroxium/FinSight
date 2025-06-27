# adapters/__init__.py

"""
External service adapters and implementations.
"""

from .tavily_search_engine import TavilySearchEngine
from .rabbitmq_broker import RabbitMQBroker
from .default_crawler import DefaultNewsCrawler

__all__ = [
    "TavilySearchEngine",
    "RabbitMQBroker",
    "DefaultNewsCrawler",
]
