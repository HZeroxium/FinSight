# repositories/mongo_news_repository.py

"""
MongoDB implementation of news repository - synchronized with news_service
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
import hashlib

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorDatabase,
    AsyncIOMotorCollection,
)
from pymongo import ASCENDING, DESCENDING, TEXT
from pymongo.errors import DuplicateKeyError

from ..interfaces.news_repository_interface import NewsRepositoryInterface
from ..schemas.news_schemas import NewsItem, NewsSource
from ..models.news_model import NewsModel
from common.logger import LoggerFactory, LoggerType, LogLevel


class MongoNewsRepository(NewsRepositoryInterface):
    """MongoDB implementation of NewsRepositoryInterface - matches news_service exactly"""

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

        self.logger = LoggerFactory.get_logger(
            name="mongo-news-repository",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file="logs/mongo_news_repository.log",
        )

        self.logger.info(
            f"MongoDB news repository initialized with database: {database_name}"
        )

    async def initialize(self) -> None:
        """Create all necessary indexes for the collection"""
        try:
            # Define index specifications
            indexes = [
                # Unique indexes for duplicate prevention
                (
                    [("url_hash", ASCENDING)],
                    {"unique": True, "name": "url_hash_unique"},
                ),
                (
                    [("guid_source_hash", ASCENDING)],
                    {"unique": True, "sparse": True, "name": "guid_source_unique"},
                ),
                # Query optimization indexes
                ([("source", ASCENDING)], {"name": "source_idx"}),
                ([("published_at", DESCENDING)], {"name": "published_at_idx"}),
                ([("fetched_at", DESCENDING)], {"name": "fetched_at_idx"}),
                ([("tags", ASCENDING)], {"name": "tags_idx"}),
                # Text search index
                ([("title", TEXT), ("description", TEXT)], {"name": "text_search_idx"}),
                # Compound indexes for common queries
                (
                    [("source", ASCENDING), ("published_at", DESCENDING)],
                    {"name": "source_published_idx"},
                ),
                (
                    [("published_at", DESCENDING), ("source", ASCENDING)],
                    {"name": "published_source_idx"},
                ),
            ]

            # Create indexes
            for keys, kwargs in indexes:
                await self.collection.create_index(keys, **kwargs)

            self.logger.info("MongoDB news repository indexes created successfully")

        except Exception as e:
            self.logger.error(
                f"Failed to initialize MongoDB news repository indexes: {e}"
            )
            raise

    async def save_news_item(self, news_item: NewsItem) -> str:
        """
        Save a news item to MongoDB with duplicate prevention

        Args:
            news_item: NewsItem to save

        Returns:
            str: ID of the saved news item
        """
        try:
            # Convert to MongoDB model
            news_model = NewsModel.from_news_item(news_item)

            # Try to insert
            doc = news_model.model_dump(by_alias=True, exclude={"id"})
            result = await self.collection.insert_one(doc)

            item_id = str(result.inserted_id)
            self.logger.debug(
                f"Saved news item: {news_item.title[:50]}... (ID: {item_id})"
            )

            return item_id

        except DuplicateKeyError as e:
            # Check which constraint was violated
            if "url_hash_unique" in str(e):
                self.logger.debug(
                    f"News item already exists with same URL: {news_item.url}"
                )
                # Find existing item and return its ID
                existing = await self.collection.find_one(
                    {"url_hash": news_model.url_hash}
                )
                return str(existing["_id"]) if existing else ""

            elif "guid_source_unique" in str(e):
                self.logger.debug(
                    f"News item already exists with same GUID: {news_item.guid} from {news_item.source}"
                )
                # Find existing item and return its ID
                existing = await self.collection.find_one(
                    {"guid_source_hash": news_model.guid_source_hash}
                )
                return str(existing["_id"]) if existing else ""

            else:
                self.logger.warning(f"Unexpected duplicate key error: {e}")
                raise

        except Exception as e:
            self.logger.error(f"Failed to save news item: {e}")
            raise

    async def get_news_item(self, item_id: str) -> Optional[NewsItem]:
        """Retrieve a news item by ID"""
        try:
            from bson import ObjectId

            doc = await self.collection.find_one({"_id": ObjectId(item_id)})
            if doc:
                news_model = NewsModel(**doc)
                return news_model.to_news_item()
            return None

        except Exception as e:
            self.logger.error(f"Failed to get news item by ID {item_id}: {e}")
            return None

    async def get_news_by_url(self, url: str) -> Optional[NewsItem]:
        """Retrieve a news item by URL"""
        try:
            url_hash = hashlib.md5(str(url).encode()).hexdigest()
            doc = await self.collection.find_one({"url_hash": url_hash})

            if doc:
                news_model = NewsModel(**doc)
                return news_model.to_news_item()
            return None

        except Exception as e:
            self.logger.error(f"Failed to get news item by URL: {e}")
            return None

    async def get_news_by_guid(
        self, source: NewsSource, guid: str
    ) -> Optional[NewsItem]:
        """Retrieve a news item by source and GUID"""
        try:
            guid_source_key = f"{source.value}:{guid}"
            guid_source_hash = hashlib.md5(guid_source_key.encode()).hexdigest()

            doc = await self.collection.find_one({"guid_source_hash": guid_source_hash})

            if doc:
                news_model = NewsModel(**doc)
                return news_model.to_news_item()
            return None

        except Exception as e:
            self.logger.error(f"Failed to get news item by GUID: {e}")
            return None

    async def search_news(
        self,
        source: Optional[NewsSource] = None,
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        has_sentiment: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[NewsItem]:
        """Search news items with filters including tags and sentiment"""
        try:
            # Build query filter
            query_filter = {}

            if source:
                query_filter["source"] = source.value

            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = start_date
                if end_date:
                    date_filter["$lte"] = end_date
                query_filter["published_at"] = date_filter

            if keywords:
                # Use text search for keywords
                search_text = " ".join(keywords)
                query_filter["$text"] = {"$search": search_text}

            if tags:
                # Filter by tags - documents must contain all specified tags
                query_filter["tags"] = {"$all": tags}

            if has_sentiment is not None:
                # Filter by sentiment analysis presence
                if has_sentiment:
                    query_filter["metadata.sentiment.sentiment_label"] = {
                        "$exists": True
                    }
                else:
                    query_filter["metadata.sentiment.sentiment_label"] = {
                        "$exists": False
                    }

            # Execute query
            cursor = (
                self.collection.find(query_filter)
                .sort("published_at", DESCENDING)
                .skip(offset)
                .limit(limit)
            )

            docs = await cursor.to_list(length=limit)
            news_items = []

            for doc in docs:
                news_model = NewsModel(**doc)
                news_items.append(news_model.to_news_item())

            self.logger.debug(f"Found {len(news_items)} news items for search query")
            return news_items

        except Exception as e:
            self.logger.error(f"Failed to search news items: {e}")
            return []

    async def get_recent_news(
        self,
        source: Optional[NewsSource] = None,
        hours: int = 24,
        limit: int = 100,
    ) -> List[NewsItem]:
        """Get recent news items"""
        start_date = datetime.now(timezone.utc) - timedelta(hours=hours)
        return await self.search_news(
            source=source,
            start_date=start_date,
            limit=limit,
        )

    async def count_news(
        self,
        source: Optional[NewsSource] = None,
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        has_sentiment: Optional[bool] = None,
    ) -> int:
        """Count news items with filters including tags and sentiment"""
        try:
            query_filter = {}

            if source:
                query_filter["source"] = source.value

            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = start_date
                if end_date:
                    date_filter["$lte"] = end_date
                query_filter["published_at"] = date_filter

            if keywords:
                search_text = " ".join(keywords)
                query_filter["$text"] = {"$search": search_text}

            if tags:
                query_filter["tags"] = {"$all": tags}

            if has_sentiment is not None:
                if has_sentiment:
                    query_filter["metadata.sentiment.sentiment_label"] = {
                        "$exists": True
                    }
                else:
                    query_filter["metadata.sentiment.sentiment_label"] = {
                        "$exists": False
                    }

            count = await self.collection.count_documents(query_filter)

            self.logger.debug(f"Found {count} total news items for count query")
            return count

        except Exception as e:
            self.logger.error(f"Failed to count news items: {e}")
            return 0

    async def get_unique_tags(
        self, source: Optional[NewsSource] = None, limit: int = 100
    ) -> List[str]:
        """Get unique tags from news items"""
        try:
            pipeline = []

            # Add source filter if specified
            if source:
                pipeline.append({"$match": {"source": source.value}})

            # Unwind tags array and group to get unique tags
            pipeline.extend(
                [
                    {"$unwind": "$tags"},
                    {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": limit},
                    {"$project": {"_id": 1}},
                ]
            )

            cursor = self.collection.aggregate(pipeline)
            tags = [doc["_id"] for doc in await cursor.to_list(length=limit)]

            self.logger.debug(f"Found {len(tags)} unique tags")
            return tags

        except Exception as e:
            self.logger.error(f"Failed to get unique tags: {e}")
            return []

    async def delete_news_item(self, item_id: str) -> bool:
        """Delete a news item"""
        try:
            from bson import ObjectId

            result = await self.collection.delete_one({"_id": ObjectId(item_id)})
            success = result.deleted_count > 0

            if success:
                self.logger.debug(f"Deleted news item: {item_id}")
            else:
                self.logger.warning(f"News item not found for deletion: {item_id}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to delete news item {item_id}: {e}")
            return False

    async def get_repository_stats(self) -> Dict[str, Any]:
        """Get repository statistics"""
        try:
            # Get total count
            total_count = await self.collection.count_documents({})

            # Get count by source
            source_pipeline = [
                {"$group": {"_id": "$source", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
            ]
            source_stats = await self.collection.aggregate(source_pipeline).to_list(
                length=None
            )

            # Get recent activity (last 24 hours)
            recent_threshold = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_count = await self.collection.count_documents(
                {"fetched_at": {"$gte": recent_threshold}}
            )

            # Get sentiment analysis stats
            sentiment_count = await self.collection.count_documents(
                {"metadata.sentiment.sentiment_label": {"$exists": True}}
            )

            # Get date range
            date_pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "oldest": {"$min": "$published_at"},
                        "newest": {"$max": "$published_at"},
                    }
                }
            ]
            date_result = await self.collection.aggregate(date_pipeline).to_list(
                length=1
            )
            date_range = date_result[0] if date_result else {}

            return {
                "total_articles": total_count,
                "articles_by_source": {
                    item["_id"]: item["count"] for item in source_stats
                },
                "recent_articles_24h": recent_count,
                "articles_with_sentiment": sentiment_count,
                "sentiment_analysis_coverage": (
                    sentiment_count / total_count * 100 if total_count > 0 else 0
                ),
                "oldest_article": date_range.get("oldest"),
                "newest_article": date_range.get("newest"),
                "database_name": self.database_name,
            }

        except Exception as e:
            self.logger.error(f"Failed to get repository stats: {e}")
            return {"error": str(e)}

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
            from bson import ObjectId

            # Prepare sentiment data
            sentiment_data = {
                "sentiment_label": sentiment_label,
                "sentiment_scores": sentiment_scores,
                "sentiment_confidence": sentiment_confidence,
                "sentiment_analyzed_at": datetime.now(timezone.utc).isoformat(),
            }

            if sentiment_reasoning:
                sentiment_data["sentiment_reasoning"] = sentiment_reasoning
            if analyzer_version:
                sentiment_data["sentiment_analyzer_version"] = analyzer_version

            # Update the document
            update_data = {"metadata.sentiment": sentiment_data}

            result = await self.collection.update_one(
                {"_id": ObjectId(item_id)}, {"$set": update_data}
            )

            success = result.modified_count > 0

            if success:
                self.logger.debug(
                    f"Updated sentiment for news item {item_id}: {sentiment_label}"
                )
            else:
                self.logger.warning(
                    f"No news item found for sentiment update: {item_id}"
                )

            return success

        except Exception as e:
            self.logger.error(
                f"Failed to update sentiment for news item {item_id}: {e}"
            )
            return False

    async def close(self) -> None:
        """Close database connection"""
        if self.client:
            self.client.close()
            self.logger.info("MongoDB connection closed")

    async def health_check(self) -> bool:
        """
        Check repository health

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Simple ping to check database connectivity
            await self.client.admin.command("ping")
            return True
        except Exception as e:
            self.logger.error(f"Repository health check failed: {e}")
            return False
