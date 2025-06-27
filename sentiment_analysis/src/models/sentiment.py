# models/sentiment.py

"""
Data models for sentiment analysis persistence and domain logic.
"""

from datetime import datetime, timezone
from typing import Optional, Dict
from enum import Enum

from bson import ObjectId
from pydantic import BaseModel, Field, HttpUrl, ConfigDict


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic v2."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str) and ObjectId.is_valid(v):
            return ObjectId(v)
        raise ValueError("Invalid ObjectId")

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, model_field) -> dict:
        return {"type": "string"}


class SentimentLabel(str, Enum):
    """Sentiment label enumeration."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentScore(BaseModel):
    """Individual sentiment scores."""

    positive: float = Field(..., ge=0.0, le=1.0, description="Positive sentiment score")
    negative: float = Field(..., ge=0.0, le=1.0, description="Negative sentiment score")
    neutral: float = Field(..., ge=0.0, le=1.0, description="Neutral sentiment score")

    model_config = ConfigDict()


class SentimentRequest(BaseModel):
    """Request for sentiment analysis - domain model."""

    text: str = Field(..., min_length=1, description="Text to analyze")
    title: Optional[str] = Field(None, description="Optional title for context")
    source_url: Optional[HttpUrl] = Field(None, description="Source URL")

    model_config = ConfigDict()


class SentimentAnalysisResult(BaseModel):
    """Result of sentiment analysis for a piece of text."""

    label: SentimentLabel
    scores: SentimentScore
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence score"
    )
    reasoning: Optional[str] = Field(None, description="AI reasoning for the sentiment")

    model_config = ConfigDict()


class ProcessedSentiment(BaseModel):
    """Processed sentiment data stored in database."""

    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    article_id: str = Field(..., description="Reference to original article")
    url: Optional[HttpUrl] = Field(None, description="Article URL")
    title: Optional[str] = Field(None, description="Article title")
    content_preview: Optional[str] = Field(
        None, description="First 200 chars of content"
    )

    # Sentiment analysis results
    sentiment_label: SentimentLabel
    sentiment_scores: SentimentScore
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: Optional[str] = None

    # Metadata
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time_ms: Optional[float] = Field(
        None, description="Processing time in milliseconds"
    )
    model_version: Optional[str] = Field(None, description="Model version used")

    # Source information
    source_domain: Optional[str] = None
    source_category: Optional[str] = None
    published_at: Optional[datetime] = None

    model_config = ConfigDict(
        validate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )


class SentimentQueryFilter(BaseModel):
    """Filter parameters for sentiment queries."""

    sentiment_label: Optional[SentimentLabel] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    source_domain: Optional[str] = None
    source_category: Optional[str] = None
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)

    model_config = ConfigDict()


class SentimentAggregation(BaseModel):
    """Aggregated sentiment statistics."""

    total_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    average_confidence: float
    sentiment_distribution: Dict[str, float]
    time_period: Optional[str] = None

    model_config = ConfigDict()
    source_domain: Optional[str] = None
    source_category: Optional[str] = None
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)

    model_config = ConfigDict()


class SentimentAggregation(BaseModel):
    """Aggregated sentiment statistics."""

    total_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    average_confidence: float
    sentiment_distribution: Dict[str, float]
    time_period: Optional[str] = None

    model_config = ConfigDict()
