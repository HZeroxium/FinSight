# models/news_model.py

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pydantic import BaseModel, Field, field_serializer, field_validator

from ..schemas.news_schemas import NewsItem, NewsSource


class NewsModel(BaseModel):
    """MongoDB model for news items"""

    id: Optional[str] = Field(default=None, alias="_id")
    source: NewsSource = Field(..., description="News source identifier")
    title: str = Field(..., description="Article title")
    url: str = Field(..., description="Article URL")
    description: Optional[str] = Field(None, description="Article description/summary")
    published_at: datetime = Field(..., description="Publication timestamp")
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When item was fetched",
    )
    author: Optional[str] = Field(None, description="Article author")
    guid: Optional[str] = Field(None, description="Unique identifier from RSS feed")
    tags: List[str] = Field(default_factory=list, description="Article tags/categories")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional fields"
    )

    # Indexing fields for fast lookups
    url_hash: str = Field(..., description="Hash of URL for duplicate detection")
    guid_source_hash: Optional[str] = Field(
        None, description="Hash of GUID+source for duplicate detection"
    )

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

    @field_validator("id", mode="before")
    @classmethod
    def validate_object_id(cls, v):
        """Convert ObjectId to string"""
        if isinstance(v, ObjectId):
            return str(v)
        return v

    @field_serializer("id")
    def serialize_id(self, value: Optional[str]) -> Optional[str]:
        """Serialize id field"""
        return value

    @classmethod
    def from_news_item(cls, news_item: "NewsItem") -> "NewsModel":
        """
        Create NewsModel from NewsItem schema

        Args:
            news_item: NewsItem to convert

        Returns:
            NewsModel: Converted news model
        """
        import hashlib

        # Create URL hash for duplicate detection
        url_hash = hashlib.md5(str(news_item.url).encode()).hexdigest()

        # Create GUID+source hash if GUID exists
        guid_source_hash = None
        if news_item.guid:
            guid_source_key = f"{news_item.source.value}:{news_item.guid}"
            guid_source_hash = hashlib.md5(guid_source_key.encode()).hexdigest()

        return cls(
            source=news_item.source,
            title=news_item.title,
            url=str(news_item.url),
            description=news_item.description,
            published_at=news_item.published_at,
            author=news_item.author,
            guid=news_item.guid,
            tags=news_item.tags,
            metadata=news_item.metadata,
            url_hash=url_hash,
            guid_source_hash=guid_source_hash,
        )

    def to_news_item(self) -> "NewsItem":
        """
        Convert NewsModel to NewsItem schema

        Returns:
            NewsItem: Converted news item
        """
        from ..schemas.news_schemas import NewsItem

        # Add _id to metadata if available
        metadata = self.metadata.copy()
        if self.id:
            metadata["_id"] = self.id

        return NewsItem(
            source=self.source,
            title=self.title,
            url=self.url,
            description=self.description,
            published_at=self.published_at,
            author=self.author,
            guid=self.guid,
            tags=self.tags,
            metadata=metadata,
        )
