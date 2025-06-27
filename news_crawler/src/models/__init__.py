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
from .crawler import (
    CrawlerConfig,
    CrawlerStats,
    CrawlJob,
    CrawlResult,
    CrawlerStatus,
)

__all__ = [
    # Article models
    "ArticleSource",
    "ArticleMetadata",
    "CrawledArticle",
    "ProcessedArticle",
    "ArticleSearchQuery",
    "PyObjectId",
    # Crawler models
    "CrawlerConfig",
    "CrawlerStats",
    "CrawlJob",
    "CrawlResult",
    "CrawlerStatus",
]
