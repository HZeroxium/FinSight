# routers/__init__.py

"""
Routers package for the news crawler service.
Contains all FastAPI route definitions.
"""

from . import search, news

__all__ = ["search", "news"]
