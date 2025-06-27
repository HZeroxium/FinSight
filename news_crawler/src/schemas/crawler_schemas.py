# schemas/crawler_schemas.py

"""
Crawler API schemas for request/response DTOs.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl


class CrawlerConfigSchema(BaseModel):
    """Crawler configuration DTO."""

    name: str = Field(..., description="Crawler name")
    base_url: str = Field(..., description="Base URL for the crawler")
    listing_url: str = Field(..., description="URL for article listings")
    listing_selector: str = Field(..., description="CSS selector for article links")
    title_selector: str = Field(..., description="CSS selector for article title")
    content_selector: str = Field(..., description="CSS selector for article content")
    date_selector: str = Field(..., description="CSS selector for publication date")
    author_selector: Optional[str] = Field(None, description="CSS selector for author")
    date_format: str = Field("%Y-%m-%d", description="Date format for parsing")
    category: Optional[str] = Field(None, description="Article category")
    credibility_score: float = Field(
        0.8, ge=0.0, le=1.0, description="Source credibility score"
    )
    enabled: bool = Field(True, description="Whether crawler is enabled")


class CrawlJobSchema(BaseModel):
    """Crawl job DTO."""

    source_name: str = Field(..., description="Source name to crawl")
    max_articles: Optional[int] = Field(
        None, ge=1, le=1000, description="Maximum articles to crawl"
    )
    priority: int = Field(1, ge=1, le=10, description="Job priority (1=highest)")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")


class CrawlResultSchema(BaseModel):
    """Crawl result DTO."""

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


class CrawlStatsSchema(BaseModel):
    """Crawler statistics DTO."""

    total_crawlers: int
    active_crawlers: int
    total_articles_crawled: int
    total_articles_saved: int
    success_rate: float
    average_crawl_time: float
    last_crawl_time: Optional[datetime] = None
    sources: List[str]
