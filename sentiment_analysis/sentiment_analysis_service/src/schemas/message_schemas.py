# schemas/message_schemas.py

"""
Message schemas for inter-service communication in sentiment analysis service.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class NewsMessageSchema(BaseModel):
    """Schema for news messages received from news service."""

    # News identification
    id: str = Field(..., description="Unique news item identifier")
    url: str = Field(..., description="Article URL")
    title: str = Field(..., description="Article title")
    description: Optional[str] = Field(None, description="Article description/summary")

    # News metadata
    source: str = Field(..., description="News source identifier")
    published_at: datetime = Field(..., description="Publication timestamp")
    author: Optional[str] = Field(None, description="Article author")
    tags: List[str] = Field(default_factory=list, description="Article tags")

    # Processing metadata
    fetched_at: datetime = Field(..., description="When item was fetched")
    message_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(), description="When message was created"
    )

    # Additional context
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("tags", mode="before")
    @classmethod
    def ensure_tags_list(cls, v):
        """Ensure tags is always a list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v) if v else []


class SentimentResultMessageSchema(BaseModel):
    """Schema for sentiment analysis results sent back to news service."""

    # Reference to original news
    news_id: str = Field(..., description="Reference to original news item")
    url: Optional[str] = Field(None, description="Article URL")
    title: Optional[str] = Field(None, description="Article title")

    # Sentiment results
    sentiment_label: str = Field(..., description="Sentiment classification")
    sentiment_scores: Dict[str, float] = Field(..., description="Sentiment scores")
    confidence: float = Field(..., description="Analysis confidence")
    reasoning: Optional[str] = Field(None, description="Analysis reasoning")

    # Processing metadata
    processed_at: datetime = Field(..., description="Processing timestamp")
    processing_time_ms: Optional[float] = Field(
        None, description="Processing time in milliseconds"
    )
    analyzer_version: Optional[str] = Field(None, description="Analyzer version")

    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional processing metadata"
    )