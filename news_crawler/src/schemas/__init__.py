# schemas/__init__.py

from .search_schemas import *
from .crawler_schemas import *
from .common_schemas import *

__all__ = [
    # Search schemas (primary API DTOs)
    "SearchRequestSchema",
    "SearchResponseSchema",
    "SearchResultSchema",
    "SearchErrorSchema",
    # Crawler schemas
    "CrawlerConfigSchema",
    "CrawlJobSchema",
    "CrawlResultSchema",
    "CrawlStatsSchema",
    # Common schemas
    "HealthCheckSchema",
    "ErrorResponseSchema",
    "PaginationSchema",
    "MetricsSchema",
]
