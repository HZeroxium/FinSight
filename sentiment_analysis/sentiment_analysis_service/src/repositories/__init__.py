# repositories/__init__.py

"""
Data access layer repositories.
"""

from .mongo_news_repository import MongoNewsRepository

__all__ = [
    "MongoNewsRepository",
]
