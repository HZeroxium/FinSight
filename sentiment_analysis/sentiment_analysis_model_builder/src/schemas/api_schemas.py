# schemas/api_schemas.py

"""API schemas for sentiment analysis REST endpoints."""

from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator

from ..core.enums import SentimentLabel, ResponseStatus


class SentimentRequest(BaseModel):
    """Request schema for sentiment analysis."""

    text: str = Field(
        ...,
        description="Text to analyze for sentiment",
        min_length=1,
        max_length=2048,
        examples=["Bitcoin prices are soaring to new heights!"],
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate and clean input text."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class BatchSentimentRequest(BaseModel):
    """Request schema for batch sentiment analysis."""

    texts: List[str] = Field(
        ...,
        description="List of texts to analyze for sentiment",
        min_length=1,
        max_length=32,
        examples=[["Bitcoin is rising", "Market is volatile"]],
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        """Validate and clean input texts."""
        if not v:
            raise ValueError("texts list cannot be empty")

        cleaned_texts = []
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(
                    f"Text at index {i} cannot be empty or whitespace only"
                )
            if len(text) > 2048:
                raise ValueError(
                    f"Text at index {i} exceeds maximum length of 2048 characters"
                )
            cleaned_texts.append(text.strip())

        return cleaned_texts


class SentimentScore(BaseModel):
    """Sentiment score breakdown."""

    positive: float = Field(
        ..., description="Positive sentiment probability", ge=0.0, le=1.0
    )
    negative: float = Field(
        ..., description="Negative sentiment probability", ge=0.0, le=1.0
    )
    neutral: float = Field(
        ..., description="Neutral sentiment probability", ge=0.0, le=1.0
    )

    @field_validator("positive", "negative", "neutral")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Validate probability values."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Probability must be between 0.0 and 1.0")
        return round(v, 4)


class SentimentResult(BaseModel):
    """Result schema for sentiment analysis."""

    label: SentimentLabel = Field(..., description="Predicted sentiment label")
    confidence: float = Field(
        ..., description="Confidence score for the prediction", ge=0.0, le=1.0
    )
    scores: SentimentScore = Field(..., description="Detailed sentiment scores")
    processing_time_ms: Optional[float] = Field(
        default=None, description="Processing time in milliseconds", ge=0.0
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence score."""
        return round(v, 4)


class SentimentResponse(BaseModel):
    """Response schema for single sentiment analysis."""

    status: ResponseStatus = Field(
        default=ResponseStatus.SUCCESS, description="Response status"
    )
    result: SentimentResult = Field(..., description="Sentiment analysis result")
    request_id: Optional[str] = Field(
        default=None, description="Unique request identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )


class BatchSentimentResponse(BaseModel):
    """Response schema for batch sentiment analysis."""

    status: ResponseStatus = Field(
        default=ResponseStatus.SUCCESS, description="Response status"
    )
    results: List[SentimentResult] = Field(
        ..., description="List of sentiment analysis results"
    )
    total_processed: int = Field(
        ..., description="Total number of texts processed", ge=0
    )
    total_processing_time_ms: Optional[float] = Field(
        default=None, description="Total processing time in milliseconds", ge=0.0
    )
    request_id: Optional[str] = Field(
        default=None, description="Unique request identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )


class ModelInfo(BaseModel):
    """Model information schema."""

    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    backbone: str = Field(..., description="Model backbone")
    num_labels: int = Field(..., description="Number of sentiment labels")
    max_sequence_length: int = Field(..., description="Maximum sequence length")
    labels: List[str] = Field(..., description="Available sentiment labels")
    device: str = Field(..., description="Device used for inference")
    model_size_mb: Optional[float] = Field(default=None, description="Model size in MB")


class HealthStatus(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device information")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    memory_usage_mb: Optional[float] = Field(
        default=None, description="Memory usage in MB"
    )
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Health check timestamp"
    )


class ErrorResponse(BaseModel):
    """Error response schema."""

    status: ResponseStatus = Field(
        default=ResponseStatus.ERROR, description="Response status"
    )
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    request_id: Optional[str] = Field(default=None, description="Request identifier")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )


class MetricsResponse(BaseModel):
    """API metrics response schema."""

    total_requests: int = Field(..., description="Total number of requests processed")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    average_processing_time_ms: float = Field(
        ..., description="Average processing time in milliseconds"
    )
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    memory_usage_mb: Optional[float] = Field(
        default=None, description="Current memory usage in MB"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Metrics timestamp"
    )
