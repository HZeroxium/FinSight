# schemas/news_schemas.py

from typing import Optional, Dict, Any, List, Annotated
from datetime import datetime
from pydantic import BaseModel, HttpUrl, Field, field_validator, model_validator
from enum import Enum
import re


class NewsSource(str, Enum):
    COINDESK = "coindesk"
    COINTELEGRAPH = "cointelegraph"


class NewsItem(BaseModel):
    source: NewsSource = Field(..., description="News source identifier")
    title: str = Field(..., description="Article title")
    url: HttpUrl = Field(..., description="Article URL")
    description: Optional[str] = Field(None, description="Article description/summary")
    published_at: datetime = Field(..., description="Publication timestamp")
    author: Optional[str] = Field(None, description="Article author")
    guid: Optional[str] = Field(None, description="Unique identifier from RSS feed")
    tags: Annotated[
        Optional[List[str]], Field(default_factory=list, validate_default=True)
    ] = None
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional fields"
    )

    @field_validator("tags", mode="before")
    @classmethod
    def clean_tags(cls, v):
        if isinstance(v, str):
            return [v.strip()]
        if isinstance(v, list):
            return [str(tag).strip() for tag in v if tag]
        return []

    @field_validator("description", mode="before")
    @classmethod
    def clean_description(cls, v):
        if not v:
            return None
        v = re.sub(r"<[^>]+>", "", str(v))
        v = " ".join(v.split())
        return v.strip() or None


class NewsCollectionResult(BaseModel):
    source: Optional[NewsSource]
    items: List[NewsItem]
    collected_at: datetime = Field(default_factory=datetime.now)
    total_items: int = Field(
        ..., description="Total number of items collected", default_factory=lambda: 0
    )
    success: bool = Field(True, description="Whether collection was successful")
    error_message: Optional[str] = Field(
        None, description="Error message if collection failed"
    )

    @field_validator("total_items")
    @classmethod
    def set_total_items(cls, v, info):
        return len(info.data.get("items") or [])

    @model_validator(mode="after")
    def compute_total(self):
        self.total_items = len(self.items)
        return self


class NewsCollectorConfig(BaseModel):
    source: NewsSource
    url: HttpUrl
    timeout: int = Field(30, description="Request timeout in seconds")
    max_items: Optional[int] = Field(None, description="Maximum items to collect")
    user_agent: str = Field(
        "FinSight-NewsCollector/1.0", description="User agent for HTTP requests"
    )
    retry_attempts: int = Field(3, description="Number of retry attempts on failure")
    retry_delay: float = Field(1.0, description="Delay between retries in seconds")


# Pydantic models for request/response
class NewsSearchParams(BaseModel):
    """News search parameters with validation"""

    source: Optional[NewsSource] = Field(None, description="News source filter")
    keywords: Optional[List[str]] = Field(None, description="Keywords to search for")
    start_date: Optional[datetime] = Field(None, description="Start date (ISO format)")
    end_date: Optional[datetime] = Field(None, description="End date (ISO format)")
    limit: int = Field(100, ge=1, le=1000, description="Maximum items to return")
    offset: int = Field(0, ge=0, description="Number of items to skip")

    @field_validator("end_date")
    def validate_date_range(cls, v, values):
        """Validate that end_date is after start_date"""
        if v and values.get("start_date") and v <= values["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v


class TimeRangeSearchParams(BaseModel):
    """Optimized time-based search parameters"""

    hours: Optional[int] = Field(
        None, ge=1, le=8760, description="Hours to look back from now"
    )
    days: Optional[int] = Field(
        None, ge=1, le=365, description="Days to look back from now"
    )
    start_date: Optional[datetime] = Field(None, description="Specific start date")
    end_date: Optional[datetime] = Field(None, description="Specific end date")
    source: Optional[NewsSource] = Field(None, description="News source filter")
    keywords: Optional[List[str]] = Field(None, description="Keywords to search for")
    limit: int = Field(100, ge=1, le=1000, description="Maximum items to return")
    offset: int = Field(0, ge=0, description="Number of items to skip")

    @model_validator(mode="before")
    def validate_time_params(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that only one time parameter is provided:
        hours, days, or start_date (mutually exclusive)
        """
        provided = [
            values.get("hours"),
            values.get("days"),
            values.get("start_date"),
        ]
        if sum(1 for v in provided if v is not None) > 1:
            raise ValueError(
                "Only one of hours, days, or start_date should be provided"
            )
        return values


class NewsItemResponse(BaseModel):
    """Streamlined news item response for frontend consumption"""

    source: str = Field(..., description="News source identifier")
    title: str = Field(..., description="Article title")
    url: HttpUrl = Field(..., description="Article URL")
    description: Optional[str] = Field(None, description="Article description/summary")
    published_at: datetime = Field(..., description="Publication timestamp")
    author: Optional[str] = Field(None, description="Article author")
    tags: List[str] = Field(default_factory=list, description="Article tags")
    image_url: Optional[HttpUrl] = Field(None, description="Image URL if available")
    sentiment: Optional[str] = Field(None, description="Article sentiment")

    @classmethod
    def from_news_item(cls, news_item: NewsItem) -> "NewsItemResponse":
        """Convert NewsItem to NewsItemResponse"""
        source_override = None
        if news_item.metadata:
            si = news_item.metadata.get("source_info")
            if isinstance(si, dict):
                source_override = si.get("source_key")
        return cls(
            source=source_override or news_item.source,
            title=news_item.title,
            url=news_item.url,
            description=news_item.description,
            published_at=news_item.published_at,
            author=news_item.author,
            tags=news_item.tags or [],
            image_url=(
                news_item.metadata.get("image_url") if news_item.metadata else None
            ),
            sentiment=(
                news_item.metadata.get("sentiment") if news_item.metadata else None
            ),
        )


class NewsResponse(BaseModel):
    """Standardized news response"""

    items: List[NewsItemResponse] = Field(..., description="List of news items")
    total_count: int = Field(..., description="Total number of matching items")
    limit: int = Field(..., description="Applied limit")
    offset: int = Field(..., description="Applied offset")
    has_more: bool = Field(..., description="Whether more items are available")
    filters_applied: Dict[str, Any] = Field(..., description="Applied filters summary")


class NewsStatsResponse(BaseModel):
    """News statistics response"""

    total_articles: int = Field(..., description="Total articles in database")
    articles_by_source: Dict[str, int] = Field(
        ..., description="Article count by source"
    )
    recent_articles_24h: int = Field(..., description="Articles from last 24 hours")
    oldest_article: Optional[datetime] = Field(None, description="Oldest article date")
    newest_article: Optional[datetime] = Field(None, description="Newest article date")
    database_info: Dict[str, Any] = Field(..., description="Database information")
