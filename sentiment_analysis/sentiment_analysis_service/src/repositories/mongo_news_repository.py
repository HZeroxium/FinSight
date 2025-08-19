# repositories/mongo_news_repository.py

"""
MongoDB implementation of news repository for sentiment analysis service.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import hashlib

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorDatabase,
    AsyncIOMotorCollection,
)
from pymongo import ASCENDING, DESCENDING, TEXT
from pymongo.errors import DuplicateKeyError
from bson import ObjectId

from ..interfaces.news_repository_interface import NewsRepositoryInterface
from ..schemas.news_schemas import NewsItem, NewsSource
from ..models.news_model import NewsModel
from common.logger import LoggerFactory

logger = LoggerFactory.get_logger(
    name="mongo-news-repository", log_file="logs/mongo_news_repository.log"
)


class MongoNewsRepository(NewsRepositoryInterface):
    """MongoDB implementation of NewsRepositoryInterface for sentiment analysis"""

    def __init__(self, mongo_url: str, database_name: str = "finsight_news"):
        """
        Initialize MongoDB news repository

        Args:
            mongo_url: MongoDB connection URL
            database_name: Database name
        """
        self.mongo_url = mongo_url
        self.database_name = database_name
        self.client: AsyncIOMotorClient = AsyncIOMotorClient(mongo_url)
        self.db: AsyncIOMotorDatabase = self.client[database_name]
        self.collection: AsyncIOMotorCollection = self.db.news_items

        logger.info(
            f"MongoDB news repository initialized with database: {database_name}"
        )

    async def initialize(self) -> None:
        """Initialize database indexes for optimal performance"""
        try:
            # Create indexes for efficient querying
            await self.collection.create_index([("url_hash", ASCENDING)], unique=True)
            await self.collection.create_index(
                [("guid_source_hash", ASCENDING)], unique=True, sparse=True
            )
            await self.collection.create_index([("source", ASCENDING)])
            await self.collection.create_index([("published_at", DESCENDING)])
            await self.collection.create_index([("fetched_at", DESCENDING)])
            await self.collection.create_index(
                [("sentiment_label", ASCENDING)], sparse=True
            )
            await self.collection.create_index(
                [("sentiment_analyzed_at", DESCENDING)], sparse=True
            )

            # Text index for searching
            await self.collection.create_index(
                [("title", TEXT), ("description", TEXT), ("tags", TEXT)]
            )

            # Compound indexes for common queries
            await self.collection.create_index(
                [("source", ASCENDING), ("published_at", DESCENDING)]
            )

            await self.collection.create_index(
                [("sentiment_label", ASCENDING), ("sentiment_confidence", DESCENDING)],
                sparse=True,
            )

            logger.info("MongoDB indexes created successfully")

        except Exception as e:
            logger.error(f"Failed to create MongoDB indexes: {e}")
            raise

    async def get_news_item(self, item_id: str) -> Optional[NewsModel]:
        """
        Retrieve a news item by ID

        Args:
            item_id: ID of the news item

        Returns:
            Optional[NewsModel]: News item if found, None otherwise
        """
        try:
            document = await self.collection.find_one({"_id": ObjectId(item_id)})
            if document:
                return NewsModel(**document)
            return None
        except Exception as e:
            logger.error(f"Failed to get news item by ID {item_id}: {e}")
            return None

    async def get_news_by_url(self, url: str) -> Optional[NewsModel]:
        """
        Retrieve a news item by URL

        Args:
            url: URL of the news item

        Returns:
            Optional[NewsModel]: News item if found, None otherwise
        """
        try:
            url_hash = hashlib.sha256(url.encode()).hexdigest()
            document = await self.collection.find_one({"url_hash": url_hash})
            if document:
                return NewsModel(**document)
            return None
        except Exception as e:
            logger.error(f"Failed to get news item by URL {url}: {e}")
            return None

    async def update_news_sentiment(
        self,
        item_id: str,
        sentiment_label: str,
        sentiment_scores: Dict[str, float],
        sentiment_confidence: float,
        sentiment_reasoning: Optional[str] = None,
        analyzer_version: Optional[str] = None,
    ) -> bool:
        """
        Update sentiment analysis results for a news item

        Args:
            item_id: ID of the news item
            sentiment_label: Sentiment classification
            sentiment_scores: Sentiment scores
            sentiment_confidence: Analysis confidence
            sentiment_reasoning: Optional analysis reasoning
            analyzer_version: Optional analyzer version

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            update_data = {
                "sentiment_label": sentiment_label,
                "sentiment_scores": sentiment_scores,
                "sentiment_confidence": sentiment_confidence,
                "sentiment_analyzed_at": datetime.now(timezone.utc),
            }

            if sentiment_reasoning:
                update_data["sentiment_reasoning"] = sentiment_reasoning
            if analyzer_version:
                update_data["sentiment_analyzer_version"] = analyzer_version

            result = await self.collection.update_one(
                {"_id": ObjectId(item_id)}, {"$set": update_data}
            )

            success = result.modified_count > 0
            if success:
                logger.debug(
                    f"Updated sentiment for news item {item_id}: {sentiment_label}"
                )
            else:
                logger.warning(
                    f"No news item found with ID {item_id} for sentiment update"
                )

            return success

        except Exception as e:
            logger.error(f"Failed to update sentiment for news item {item_id}: {e}")
            return False

    async def search_news(
        self,
        source: Optional[NewsSource] = None,
        keywords: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        has_sentiment: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[NewsModel]:
        """
        Search for news items with filters

        Args:
            source: Filter by news source
            keywords: Filter by keywords in title/description
            start_date: Filter by published date (start)
            end_date: Filter by published date (end)
            has_sentiment: Filter by sentiment analysis presence
            limit: Maximum number of items to return
            offset: Number of items to skip

        Returns:
            List[NewsModel]: List of matching news items
        """
        try:
            query = {}

            # Source filter
            if source:
                query["source"] = source.value

            # Date range filter
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = start_date
                if end_date:
                    date_filter["$lte"] = end_date
                query["published_at"] = date_filter

            # Sentiment filter
            if has_sentiment is not None:
                if has_sentiment:
                    query["sentiment_label"] = {"$exists": True, "$ne": None}
                else:
                    query["sentiment_label"] = {"$exists": False}

            # Keywords filter (text search)
            if keywords:
                keyword_query = " ".join(keywords)
                query["$text"] = {"$search": keyword_query}

            # Execute query
            cursor = self.collection.find(query)
            cursor = cursor.sort("published_at", DESCENDING)
            cursor = cursor.skip(offset).limit(limit)

            documents = await cursor.to_list(length=limit)
            return [NewsModel(**doc) for doc in documents]

        except Exception as e:
            logger.error(f"Failed to search news items: {e}")
            return []

    async def count_news(
        self,
        source: Optional[NewsSource] = None,
        keywords: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        has_sentiment: Optional[bool] = None,
    ) -> int:
        """
        Count news items with filters

        Args:
            source: Filter by news source
            keywords: Filter by keywords in title/description
            start_date: Filter by published date (start)
            end_date: Filter by published date (end)
            has_sentiment: Filter by sentiment analysis presence

        Returns:
            int: Number of matching news items
        """
        try:
            query = {}

            # Source filter
            if source:
                query["source"] = source.value

            # Date range filter
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = start_date
                if end_date:
                    date_filter["$lte"] = end_date
                query["published_at"] = date_filter

            # Sentiment filter
            if has_sentiment is not None:
                if has_sentiment:
                    query["sentiment_label"] = {"$exists": True, "$ne": None}
                else:
                    query["sentiment_label"] = {"$exists": False}

            # Keywords filter (text search)
            if keywords:
                keyword_query = " ".join(keywords)
                query["$text"] = {"$search": keyword_query}

            return await self.collection.count_documents(query)

        except Exception as e:
            logger.error(f"Failed to count news items: {e}")
            return 0

    async def health_check(self) -> bool:
        """
        Check repository health

        Returns:
            bool: True if repository is healthy, False otherwise
        """
        try:
            # Simple ping to check connection
            await self.client.admin.command("ping")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
