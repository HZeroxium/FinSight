# repositories/__init__.py

"""
Data access layer repositories.
"""

from .mongodb_sentiment_repository import MongoDBSentimentRepository

__all__ = [
    "MongoDBSentimentRepository",
]
