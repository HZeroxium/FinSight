# models/schemas.py

"""Pydantic schemas for the sentiment analysis API."""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict

from ..core.enums import SentimentLabel, ServerStatus, ModelStatus


class BaseSchema(BaseModel):
    """Base schema with Pydantic v2-compatible config."""

    # Avoid protected namespace warnings (fields like model_name, model_status)
    # and serialize Enum values as their underlying .value
    model_config = ConfigDict(protected_namespaces=(), use_enum_values=True)


class SentimentRequest(BaseSchema):
    """Request schema for single text sentiment analysis."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to analyze for sentiment",
        example="The market is showing strong positive signals today.",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate and clean input text."""
        # Remove excessive whitespace
        cleaned = " ".join(v.split())
        if not cleaned:
            raise ValueError("Text cannot be empty after cleaning")
        return cleaned


class BatchSentimentRequest(BaseSchema):
    """Request schema for batch sentiment analysis."""

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts to analyze for sentiment",
        example=[
            "The market is bullish today.",
            "Economic indicators show decline.",
            "Neutral market conditions observed.",
        ],
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        """Validate and clean input texts."""
        cleaned_texts = []
        for i, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} must be a string")

            cleaned = " ".join(text.split())
            if not cleaned:
                raise ValueError(f"Text at index {i} cannot be empty after cleaning")

            if len(cleaned) > 10000:
                raise ValueError(
                    f"Text at index {i} exceeds maximum length of 10000 characters"
                )

            cleaned_texts.append(cleaned)

        return cleaned_texts


class SentimentScore(BaseSchema):
    """Sentiment score breakdown."""

    negative: float = Field(
        ..., ge=0.0, le=1.0, description="Negative sentiment probability"
    )
    neutral: float = Field(
        ..., ge=0.0, le=1.0, description="Neutral sentiment probability"
    )
    positive: float = Field(
        ..., ge=0.0, le=1.0, description="Positive sentiment probability"
    )

    @field_validator("negative", "neutral", "positive")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Validate probability values."""
        return round(v, 4)


class SentimentResult(BaseSchema):
    """Result schema for sentiment analysis."""

    text: str = Field(..., description="Original input text")
    label: SentimentLabel = Field(..., description="Predicted sentiment label")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the prediction"
    )
    scores: SentimentScore = Field(..., description="Detailed sentiment scores")
    processing_time_ms: float = Field(
        ..., ge=0.0, description="Processing time in milliseconds"
    )


class BatchSentimentResult(BaseSchema):
    """Result schema for batch sentiment analysis."""

    results: List[SentimentResult] = Field(
        ..., description="List of sentiment analysis results"
    )
    total_processing_time_ms: float = Field(
        ..., ge=0.0, description="Total processing time in milliseconds"
    )
    batch_size: int = Field(..., ge=1, description="Number of texts processed")

    @field_validator("results")
    @classmethod
    def validate_results(cls, v: List[SentimentResult]) -> List[SentimentResult]:
        """Validate results list."""
        if not v:
            raise ValueError("Results list cannot be empty")
        return v


class HealthStatus(BaseSchema):
    """Health check response schema."""

    status: str = Field(..., description="Overall service status")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Health check timestamp",
    )
    triton_status: ServerStatus = Field(..., description="Triton server status")
    model_status: ModelStatus = Field(..., description="Model status in Triton")
    uptime_seconds: float = Field(..., ge=0.0, description="Service uptime in seconds")
    version: str = Field(default="1.0.0", description="Service version")


class ModelInfo(BaseSchema):
    """Model information schema."""

    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    platform: str = Field(..., description="Model platform (e.g., onnxruntime_onnx)")
    status: ModelStatus = Field(..., description="Current model status")
    inputs: List[Dict[str, Any]] = Field(..., description="Model input specifications")
    outputs: List[Dict[str, Any]] = Field(
        ..., description="Model output specifications"
    )
    max_batch_size: int = Field(..., ge=1, description="Maximum batch size")


class MetricsResponse(BaseSchema):
    """Metrics response schema."""

    total_requests: int = Field(
        ..., ge=0, description="Total number of requests processed"
    )
    successful_requests: int = Field(
        ..., ge=0, description="Number of successful requests"
    )
    failed_requests: int = Field(..., ge=0, description="Number of failed requests")
    average_processing_time_ms: float = Field(
        ..., ge=0.0, description="Average processing time"
    )
    uptime_seconds: float = Field(..., ge=0.0, description="Service uptime")

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return round(self.successful_requests / self.total_requests, 4)


class ErrorResponse(BaseSchema):
    """Error response schema."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp",
    )
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class ServerInfo(BaseSchema):
    """Server information schema."""

    name: str = Field(
        default="FinSight Sentiment Analysis Engine", description="Service name"
    )
    version: str = Field(default="1.0.0", description="Service version")
    description: str = Field(
        default="Automated sentiment analysis using fine-tuned FinBERT model",
        description="Service description",
    )
    triton_version: Optional[str] = Field(None, description="Triton server version")
    model_name: str = Field(..., description="Loaded model name")
    startup_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Service startup time",
    )
