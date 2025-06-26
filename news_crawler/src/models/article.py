# models/article.py

"""
Article data models for news crawler service.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from bson import ObjectId
from pydantic import BaseModel, Field, HttpUrl, ConfigDict


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic v2."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str) and ObjectId.is_valid(v):
            return ObjectId(v)
        raise ValueError("Invalid ObjectId")

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, model_field) -> dict:
        # Replace `__modify_schema__`
        return {"type": "string"}


class ArticleSource(BaseModel):
    """Article source information."""

    name: str
    domain: str
    url: HttpUrl
    credibility_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    category: Optional[str] = None

    model_config = ConfigDict()


class ArticleMetadata(BaseModel):
    """Article metadata."""

    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    content_length: int = 0
    language: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    entities: Dict[str, Any] = Field(default_factory=dict)
    summary: Optional[str] = None

    model_config = ConfigDict()


class CrawledArticle(BaseModel):
    """Raw article data from crawling."""

    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    url: HttpUrl
    title: str
    content: str
    published_at: Optional[datetime] = None
    author: Optional[str] = None
    source: ArticleSource
    metadata: ArticleMetadata = Field(default_factory=ArticleMetadata)
    raw_html: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        validate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )


class ProcessedArticle(BaseModel):
    """Processed article with sentiment and analysis."""

    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    original_article_id: PyObjectId
    url: HttpUrl
    title: str
    content: str
    published_at: Optional[datetime] = None
    author: Optional[str] = None
    source: ArticleSource
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    sentiment_label: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    entities: Dict[str, Any] = Field(default_factory=dict)
    categories: List[str] = Field(default_factory=list)
    processed_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        validate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )


class ArticleSearchQuery(BaseModel):
    """Article search query parameters."""

    query: Optional[str] = None
    source: Optional[str] = None
    category: Optional[str] = None
    author: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    sentiment_min: Optional[float] = Field(None, ge=-1.0, le=1.0)
    sentiment_max: Optional[float] = Field(None, ge=-1.0, le=1.0)
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)

    model_config = ConfigDict()


class ArticleSearchResponse(BaseModel):
    """Article search response."""

    articles: List[ProcessedArticle]
    total_count: int
    query: ArticleSearchQuery
    response_time: float

    model_config = ConfigDict()
