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
