# schemas/news_schemas.py

"""
News-related schemas for sentiment analysis service.
"""

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
