# models/news_model.py

"""
News model for MongoDB storage in sentiment analysis service.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_serializer, field_validator
from bson import ObjectId

from ..schemas.news_schemas import NewsSource, NewsItem


class NewsModel(BaseModel):
    """MongoDB model for news items with sentiment analysis fields"""

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

    # Sentiment analysis fields
    sentiment_label: Optional[str] = Field(None, description="Sentiment classification")
    sentiment_scores: Optional[Dict[str, float]] = Field(
        None, description="Sentiment scores"
    )
    sentiment_confidence: Optional[float] = Field(
        None, description="Analysis confidence"
    )
    sentiment_reasoning: Optional[str] = Field(None, description="Analysis reasoning")
    sentiment_analyzed_at: Optional[datetime] = Field(
        None, description="When sentiment was analyzed"
    )
    sentiment_analyzer_version: Optional[str] = Field(
        None, description="Analyzer version used"
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
        """Serialize ID field"""
        return value

    @classmethod
    def from_news_item(
        cls, news_item: NewsItem, url_hash: str, guid_source_hash: Optional[str] = None
    ) -> "NewsModel":
        """
        Create NewsModel from NewsItem

        Args:
            news_item: NewsItem schema
            url_hash: Hash of URL for duplicate detection
            guid_source_hash: Hash of GUID+source for duplicate detection

        Returns:
            NewsModel: MongoDB model instance
        """
        return cls(
            source=news_item.source,
            title=news_item.title,
            url=str(news_item.url),
            description=news_item.description,
            published_at=news_item.published_at,
            author=news_item.author,
            guid=news_item.guid,
            tags=news_item.tags or [],
            metadata=news_item.metadata or {},
            url_hash=url_hash,
            guid_source_hash=guid_source_hash,
        )

    def to_news_item(self) -> NewsItem:
        """
        Convert NewsModel to NewsItem

        Returns:
            NewsItem: Pydantic schema
        """
        return NewsItem(
            source=self.source,
            title=self.title,
            url=self.url,
            description=self.description,
            published_at=self.published_at,
            author=self.author,
            guid=self.guid,
            tags=self.tags,
            metadata=self.metadata,
        )

    def update_sentiment(
        self,
        label: str,
        scores: Dict[str, float],
        confidence: float,
        reasoning: Optional[str] = None,
        analyzer_version: Optional[str] = None,
    ) -> None:
        """
        Update sentiment analysis fields

        Args:
            label: Sentiment classification
            scores: Sentiment scores
            confidence: Analysis confidence
            reasoning: Optional analysis reasoning
            analyzer_version: Optional analyzer version
        """
        self.sentiment_label = label
        self.sentiment_scores = scores
        self.sentiment_confidence = confidence
        self.sentiment_reasoning = reasoning
        self.sentiment_analyzer_version = analyzer_version
        self.sentiment_analyzed_at = datetime.now(timezone.utc)

    def has_sentiment(self) -> bool:
        """
        Check if this news item has sentiment analysis

        Returns:
            bool: True if sentiment analysis exists
        """
        return self.sentiment_label is not None
