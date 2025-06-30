# schemas/__init__.py

"""
API schemas for request/response DTOs.
"""

from .sentiment_schemas import *
from .common_schemas import *

__all__ = [
    # Sentiment schemas
    "SentimentAnalysisRequestSchema",
    "SentimentBatchRequestSchema",
    "SentimentAnalysisResponseSchema",
    "SentimentBatchResponseSchema",
    "ProcessedSentimentSchema",
    "SentimentSearchRequestSchema",
    "SentimentSearchResponseSchema",
    "SentimentAggregationSchema",
    "SentimentErrorSchema",
    "SentimentScore"
    # Common schemas
    "HealthCheckSchema",
    "ErrorResponseSchema",
    "PaginationSchema",
]
