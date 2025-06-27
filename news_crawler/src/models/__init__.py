# models/__init__.py

"""
Data models for persistence and domain logic.
"""

from .article import (
    ArticleSource,
    ArticleMetadata,
    CrawledArticle,
    ProcessedArticle,
    ArticleSearchQuery,
    PyObjectId,
)

# Search models have been moved to schemas/search_schemas.py
# since they are API DTOs, not data persistence models

__all__ = [
    # Article models
    "ArticleSource",
    "ArticleMetadata",
    "CrawledArticle",
    "ProcessedArticle",
    "ArticleSearchQuery",
    "PyObjectId",
    "ArticleSearchQuery",
    "PyObjectId",
    # Search models
    "SearchResult",
    "SearchResponse",
    "SearchRequest",
    "SearchError",
]
