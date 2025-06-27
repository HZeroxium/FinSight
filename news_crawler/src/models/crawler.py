# models/crawler.py

"""
Data models for crawler configuration and management.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class CrawlerStatus(str, Enum):
    """Crawler status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DISABLED = "disabled"


class CrawlerConfig(BaseModel):
    """Configuration for website crawler."""

    name: str = Field(..., description="Unique crawler name")
    base_url: str = Field(..., description="Base URL for the website")
    listing_url: str = Field(..., description="URL for article listings")
    listing_selector: str = Field(..., description="CSS selector for article links")
    title_selector: str = Field(..., description="CSS selector for article title")
    content_selector: str = Field(..., description="CSS selector for article content")
    date_selector: str = Field(..., description="CSS selector for publication date")
    author_selector: Optional[str] = Field(None, description="CSS selector for author")
    date_format: str = Field("%Y-%m-%d", description="Date format for parsing")
    category: Optional[str] = Field(None, description="Article category")
    credibility_score: float = Field(
        0.8, ge=0.0, le=1.0, description="Source credibility"
    )
    enabled: bool = Field(True, description="Whether crawler is enabled")

    model_config = ConfigDict()


class CrawlerStats(BaseModel):
    """Crawler statistics and metrics."""

    name: str
    status: CrawlerStatus
    total_articles_found: int = 0
    total_articles_crawled: int = 0
    total_articles_saved: int = 0
    success_rate: float = 0.0
    average_crawl_time: float = 0.0
    last_crawl_time: Optional[datetime] = None
    errors: List[str] = Field(default_factory=list)

    model_config = ConfigDict()


class CrawlJob(BaseModel):
    """Crawl job specification."""

    job_id: str
    crawler_name: str
    source_name: str
    max_articles: Optional[int] = Field(None, ge=1, le=1000)
    priority: int = Field(1, ge=1, le=10, description="Job priority (1=highest)")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")

    model_config = ConfigDict()


class CrawlResult(BaseModel):
    """Result of a crawl operation."""

    job_id: str
    source_name: str
    articles_found: int
    articles_crawled: int
    articles_saved: int
    duration: float
    status: str
    errors: List[str] = Field(default_factory=list)
    started_at: datetime
    completed_at: Optional[datetime] = None

    model_config = ConfigDict()
