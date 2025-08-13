# schemas/sentiment_schemas.py

"""
Sentiment analysis API schemas for request/response DTOs.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl

from ..models.sentiment import SentimentLabel, SentimentScore


class SentimentAnalysisRequestSchema(BaseModel):
    """Sentiment analysis request DTO."""

    text: str = Field(
        ..., min_length=1, max_length=10000, description="Text to analyze"
    )
    title: Optional[str] = Field(
        None, max_length=500, description="Optional title for context"
    )
    article_id: Optional[str] = Field(None, description="Article ID for storage")
    source_url: Optional[HttpUrl] = Field(None, description="Source URL")
    save_result: bool = Field(True, description="Whether to save the analysis result")


class SentimentBatchItemSchema(BaseModel):
    """Single item in batch sentiment analysis request."""

    text: str = Field(..., min_length=1, max_length=10000)
    title: Optional[str] = Field(None, max_length=500)
    source_url: Optional[HttpUrl] = Field(None)


class SentimentBatchRequestSchema(BaseModel):
    """Batch sentiment analysis request DTO."""

    items: List[SentimentBatchItemSchema] = Field(..., min_items=1, max_items=50)
    save_results: bool = Field(True, description="Whether to save analysis results")


class SentimentAnalysisResponseSchema(BaseModel):
    """Sentiment analysis response DTO."""

    sentiment_label: SentimentLabel
    scores: SentimentScore
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: Optional[str] = Field(None, description="Analysis reasoning")
    processing_time_ms: Optional[float] = Field(
        None, description="Processing time in milliseconds"
    )


class SentimentBatchResponseSchema(BaseModel):
    """Batch sentiment analysis response DTO."""

    results: List[SentimentAnalysisResponseSchema]
    total_processed: int
    success_count: int
    error_count: int


class ProcessedSentimentSchema(BaseModel):
    """Processed sentiment DTO for API responses."""

    id: str
    article_id: str
    url: Optional[str] = None  # Changed from HttpUrl to str for consistency
    title: Optional[str] = None
    content_preview: Optional[str] = None
    sentiment_label: SentimentLabel
    scores: SentimentScore
    confidence: float
    reasoning: Optional[str] = None
    processed_at: datetime
    processing_time_ms: Optional[float] = None
    model_version: Optional[str] = None
    source_domain: Optional[str] = None
    source_category: Optional[str] = None
    published_at: Optional[datetime] = None


class SentimentSearchRequestSchema(BaseModel):
    """Sentiment search request DTO."""

    sentiment_label: Optional[SentimentLabel] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    source_domain: Optional[str] = None
    source_category: Optional[str] = None
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)


class SentimentSearchResponseSchema(BaseModel):
    """Sentiment search response DTO."""

    sentiments: List[ProcessedSentimentSchema]
    total_count: int
    limit: int
    offset: int


class SentimentAggregationSchema(BaseModel):
    """Sentiment aggregation DTO."""

    total_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    average_confidence: float
    sentiment_distribution: Dict[str, float]
    time_period: Optional[str] = None


class SentimentErrorSchema(BaseModel):
    """Sentiment error response DTO."""

    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthCheckSchema(BaseModel):
    """Health check response DTO."""

    status: str
    service: str
    dependencies: Optional[Dict[str, str]] = None


class ErrorResponseSchema(BaseModel):
    """Generic error response DTO."""

    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
