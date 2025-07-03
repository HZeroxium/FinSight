from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, HttpUrl, Field, field_validator
from enum import Enum


class NewsSource(str, Enum):
    """Supported news sources"""

    COINDESK = "coindesk"
    COINTELEGRAPH = "cointelegraph"


class NewsItem(BaseModel):
    """
    Core news item schema with essential fields found in most RSS feeds.
    Additional fields are stored in metadata.
    """

    source: NewsSource = Field(..., description="News source identifier")
    title: str = Field(..., description="Article title")
    url: HttpUrl = Field(..., description="Article URL")
    description: Optional[str] = Field(None, description="Article description/summary")
    published_at: datetime = Field(..., description="Publication timestamp")
    author: Optional[str] = Field(None, description="Article author")
    guid: Optional[str] = Field(None, description="Unique identifier from RSS feed")
    tags: List[str] = Field(default_factory=list, description="Article tags/categories")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional fields"
    )

    @field_validator("tags", pre=True)
    def clean_tags(cls, v):
        """Clean and normalize tags"""
        if isinstance(v, str):
            return [v.strip()]
        if isinstance(v, list):
            return [str(tag).strip() for tag in v if tag]
        return []

    @field_validator("description", pre=True)
    def clean_description(cls, v):
        """Clean description by removing HTML tags and extra whitespace"""
        if not v:
            return None

        import re

        # Remove HTML tags
        v = re.sub(r"<[^>]+>", "", str(v))
        # Clean up whitespace
        v = " ".join(v.split())
        return v.strip() if v else None


class NewsCollectionResult(BaseModel):
    """Result of news collection operation"""

    source: NewsSource
    items: List[NewsItem]
    collected_at: datetime = Field(default_factory=datetime.now)
    total_items: int = Field(..., description="Total number of items collected")
    success: bool = Field(True, description="Whether collection was successful")
    error_message: Optional[str] = Field(
        None, description="Error message if collection failed"
    )

    @field_validator("total_items", always=True)
    def set_total_items(cls, v, values):
        """Set total_items based on items list length"""
        return len(values.get("items", []))


class NewsCollectorConfig(BaseModel):
    """Configuration for news collectors"""

    source: NewsSource
    url: HttpUrl
    timeout: int = Field(30, description="Request timeout in seconds")
    max_items: Optional[int] = Field(None, description="Maximum items to collect")
    user_agent: str = Field(
        "FinSight-NewsCollector/1.0", description="User agent for HTTP requests"
    )
    retry_attempts: int = Field(3, description="Number of retry attempts on failure")
    retry_delay: float = Field(1.0, description="Delay between retries in seconds")
