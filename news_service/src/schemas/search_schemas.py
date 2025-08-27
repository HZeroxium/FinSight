# schemas/search_schemas.py

"""
Search API schemas for request/response DTOs.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl


class SearchRequestSchema(BaseModel):
    """Search request DTO."""

    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    topic: Optional[str] = Field(
        None, description="Search topic (e.g., 'finance', 'news')"
    )
    search_depth: str = Field(
        "basic", description="Search depth: 'basic' or 'advanced'"
    )
    time_range: Optional[str] = Field(
        None, description="Time range: 'day', 'week', 'month', 'year'"
    )
    include_answer: bool = Field(False, description="Include AI-generated answer")
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of results")
    chunks_per_source: int = Field(
        3, ge=1, le=10, description="Content chunks per source"
    )


class SearchResultSchema(BaseModel):
    """Individual search result DTO."""

    url: HttpUrl
    title: str
    content: str
    score: float = Field(ge=0.0, le=1.0)
    published_at: Optional[datetime] = None
    source: Optional[str] = None
    is_crawled: bool = Field(False, description="Whether result was deep crawled")
    metadata: Optional[Dict[str, Any]] = None


class SearchResponseSchema(BaseModel):
    """Search response DTO."""

    query: str
    total_results: int
    results: List[SearchResultSchema]
    answer: Optional[str] = None
    follow_up_questions: Optional[List[str]] = None
    response_time: float
    search_depth: str
    topic: Optional[str] = None
    time_range: Optional[str] = None
    crawler_used: bool = Field(False, description="Whether crawler was used")


class SearchErrorSchema(BaseModel):
    """Search error response DTO."""

    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
