# schemas/__init__.py

from .common_schemas import *
from .job_schemas import *
from .search_schemas import *

__all__ = [
    # Search schemas (primary API DTOs)
    "SearchRequestSchema",
    "SearchResponseSchema",
    "SearchResultSchema",
    "SearchErrorSchema",
    # Common schemas
    "HealthCheckSchema",
    "ErrorResponseSchema",
    "PaginationSchema",
    "MetricsSchema",
    # Job schemas
    "JobStatus",
    "JobConfigModel",
    "JobStatsModel",
    "JobStatusResponse",
    "JobStartRequest",
    "JobStopRequest",
    "ManualJobRequest",
    "ManualJobResponse",
    "JobConfigUpdateRequest",
    "JobOperationResponse",
]
