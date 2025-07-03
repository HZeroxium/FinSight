# schemas/message_schemas.py

"""
Shared message schemas for communication between services.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ArticleMessageSchema(BaseModel):
    """Schema for article messages sent from news crawler to sentiment analysis."""

    id: str = Field(..., description="Unique article identifier")
    url: str = Field(..., description="Article URL")
    title: str = Field(..., description="Article title")
    content: str = Field(..., description="Article content")
    source: Optional[str] = Field(None, description="Article source")
    published_at: Optional[str] = Field(
        None, description="Publication timestamp ISO format"
    )
    score: Optional[float] = Field(None, description="Search relevance score")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )
    search_query: Optional[str] = Field(None, description="Original search query")
    search_timestamp: str = Field(..., description="Search timestamp ISO format")


class SentimentResultMessageSchema(BaseModel):
    """Schema for sentiment analysis results sent back to news crawler."""

    article_id: str = Field(..., description="Reference to original article")
    url: Optional[str] = Field(None, description="Article URL")
    title: Optional[str] = Field(None, description="Article title")
    content_preview: Optional[str] = Field(None, description="Content preview")
    search_query: Optional[str] = Field(None, description="Original search query")

    # Sentiment results
    sentiment_label: str = Field(..., description="Sentiment classification")
    scores: Dict[str, float] = Field(..., description="Sentiment scores")
    confidence: float = Field(..., description="Analysis confidence")
    reasoning: Optional[str] = Field(None, description="Analysis reasoning")

    # Processing metadata
    processed_at: str = Field(..., description="Processing timestamp ISO format")
    processing_time_ms: Optional[float] = Field(None, description="Processing time")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )
