"""Data schemas for sentiment analysis."""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from ..core.enums import DataSplit, SentimentLabel


class NewsArticle(BaseModel):
    """Represents a single news article with extracted fields."""

    id: Optional[str] = Field(default=None, description="Article identifier")
    text: str = Field(..., description="Main text content for sentiment analysis")
    label: SentimentLabel = Field(..., description="Sentiment label")
    title: Optional[str] = Field(default=None, description="Article title")
    published_at: Optional[datetime] = Field(
        default=None, description="Publication date"
    )
    tickers: Optional[List[str]] = Field(
        default=None, description="Related ticker symbols"
    )
    split: Optional[DataSplit] = Field(
        default=None, description="Data split assignment"
    )
    source: Optional[str] = Field(default=None, description="News source")
    url: Optional[str] = Field(default=None, description="Article URL")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate text content."""
        if not v or not v.strip():
            raise ValueError("Text content cannot be empty")
        if len(v.strip()) < 10:
            raise ValueError("Text content must be at least 10 characters")
        return v.strip()

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate ticker symbols."""
        if v is not None:
            # Remove empty strings and normalize
            v = [ticker.strip().upper() for ticker in v if ticker and ticker.strip()]
            # Remove duplicates while preserving order
            seen = set()
            v = [x for x in v if not (x in seen or seen.add(x))]
        return v


class TrainingExample(BaseModel):
    """Training example for model training."""

    text: str = Field(..., description="Input text")
    label: int = Field(..., description="Label ID")
    label_text: str = Field(..., description="Label text")
    id: str = Field(default="", description="Example ID")
    title: str = Field(default="", description="Article title")
    source: str = Field(default="", description="News source")
    published_at: str = Field(default="", description="Publication date")
    tickers: List[str] = Field(default_factory=list, description="Related tickers")

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: int) -> int:
        """Validate label ID."""
        if v < 0:
            raise ValueError("Label ID must be non-negative")
        return v


class DatasetStats(BaseModel):
    """Statistics about a dataset."""

    split_name: str = Field(..., description="Dataset split name")
    total_samples: int = Field(..., description="Total number of samples")
    class_distribution: dict[str, int] = Field(..., description="Samples per class")
    avg_text_length: float = Field(..., description="Average text length")
    std_text_length: float = Field(..., description="Standard deviation of text length")
    min_text_length: int = Field(..., description="Minimum text length")
    max_text_length: int = Field(..., description="Maximum text length")

    @field_validator("total_samples")
    @classmethod
    def validate_total_samples(cls, v: int) -> int:
        """Validate total samples."""
        if v <= 0:
            raise ValueError("Total samples must be positive")
        return v

    @field_validator("avg_text_length", "std_text_length")
    @classmethod
    def validate_length_stats(cls, v: float) -> float:
        """Validate length statistics."""
        if v < 0:
            raise ValueError("Length statistics must be non-negative")
        return round(v, 2)


class TokenizerConfig(BaseModel):
    """Configuration for tokenizer setup."""

    max_length: int = Field(..., description="Maximum sequence length")
    truncation: bool = Field(default=True, description="Enable truncation")
    padding: str = Field(default="max_length", description="Padding strategy")
    return_tensors: Optional[str] = Field(
        default=None, description="Return tensor format"
    )

    @field_validator("max_length")
    @classmethod
    def validate_max_length(cls, v: int) -> int:
        """Validate maximum length."""
        if v < 64:
            raise ValueError("max_length must be at least 64")
        if v > 2048:
            raise ValueError("max_length must not exceed 2048")
        return v

    @field_validator("padding")
    @classmethod
    def validate_padding(cls, v: str) -> str:
        """Validate padding strategy."""
        valid_padding = ["longest", "max_length", "do_not_pad"]
        if v not in valid_padding:
            raise ValueError(f"padding must be one of {valid_padding}")
        return v
