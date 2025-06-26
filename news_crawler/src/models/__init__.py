"""
Data models for persistence and domain logic.
"""

from .article import (
    ArticleSource,
    ArticleMetadata,
    CrawledArticle,
    ProcessedArticle,
    ArticleSearchQuery,
    ArticleSearchResponse,
    PyObjectId,
)
from .search import (
    SearchResult,
    SearchResponse,
    SearchRequest,
    SearchError,
)

__all__ = [
    # Article models
    "ArticleSource",
    "ArticleMetadata",
    "CrawledArticle",
    "ProcessedArticle",
    "ArticleSearchQuery",
    "ArticleSearchResponse",
    "PyObjectId",
    # Search models
    "SearchResult",
    "SearchResponse",
    "SearchRequest",
    "SearchError",
]
