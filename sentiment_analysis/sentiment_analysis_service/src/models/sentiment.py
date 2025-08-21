# models/sentiment.py

"""
Data models for sentiment analysis persistence and domain logic.
"""

from datetime import datetime, timezone
from typing import Optional, Dict
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl, ConfigDict


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

    @classmethod
    def from_dict(cls, data: dict) -> "SentimentAnalysisResult":
        """Create instance from dictionary (for cache deserialization)."""
        if isinstance(data.get("scores"), dict):
            data["scores"] = SentimentScore(**data["scores"])
        return cls(**data)
