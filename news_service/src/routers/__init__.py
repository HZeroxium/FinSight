# routers/__init__.py

"""
Routers package for the news crawler service.
Contains all FastAPI route definitions.
"""

from . import news_router, search, job_router

__all__ = ["search", "news_router", "job_router"]
