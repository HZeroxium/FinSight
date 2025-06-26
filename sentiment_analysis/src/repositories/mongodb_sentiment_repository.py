"""
MongoDB repository for sentiment data operations.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorDatabase,
    AsyncIOMotorCollection,
)
from bson import ObjectId

from ..interfaces.sentiment_repository import (
    SentimentRepository,
    SentimentRepositoryError,
)
from ..models.sentiment import (
    ProcessedSentiment,
    SentimentQueryFilter,
    SentimentAggregation,
    SentimentLabel,
)
from ..common.logger import LoggerFactory, LoggerType, LogLevel

logger = LoggerFactory.get_logger(
    name="mongodb-sentiment-repository",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
)

# Constants for index direction
ASCENDING = 1
DESCENDING = -1
TEXT = "text"


class MongoDBSentimentRepository(SentimentRepository):
    """MongoDB repository for sentiment data operations."""

    def __init__(self, mongo_url: str, database_name: str):
        """
        Initialize repository.

        Args:
            mongo_url: MongoDB connection URL
            database_name: Database name
        """
        self.client: AsyncIOMotorClient = AsyncIOMotorClient(mongo_url)
        self.db: AsyncIOMotorDatabase = self.client[database_name]
        self.collection: AsyncIOMotorCollection = self.db.sentiments

        logger.info(
            f"MongoDB sentiment repository initialized with database: {database_name}"
        )

    async def initialize(self) -> None:
        """Create all necessary indexes for the collection."""
        try:
            # Define index specifications
            indexes = [
                ([("article_id", ASCENDING)], {"unique": True}),
                ([("sentiment_label", ASCENDING)], {}),
                ([("confidence", DESCENDING)], {}),
                ([("processed_at", DESCENDING)], {}),
                ([("published_at", DESCENDING)], {}),
                ([("source_domain", ASCENDING)], {}),
                ([("source_category", ASCENDING)], {}),
                ([("title", TEXT), ("content_preview", TEXT)], {"name": "text_idx"}),
                # Compound indexes for common queries
                ([("sentiment_label", ASCENDING), ("processed_at", DESCENDING)], {}),
                ([("source_domain", ASCENDING), ("sentiment_label", ASCENDING)], {}),
                ([("published_at", DESCENDING), ("confidence", DESCENDING)], {}),
            ]

            # Create indexes
            for keys, kwargs in indexes:
                await self.collection.create_index(keys, **kwargs)

            logger.info("MongoDB sentiment repository indexes created successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize MongoDB sentiment repository indexes: {e}"
            )
            raise SentimentRepositoryError(f"Failed to initialize indexes: {e}")

    async def save_sentiment(self, sentiment: ProcessedSentiment) -> str:
        """
        Save processed sentiment to MongoDB.

        Args:
            sentiment: ProcessedSentiment to save

        Returns:
            str: Saved sentiment ID
        """
        try:
            doc = sentiment.dict(by_alias=True, exclude_unset=True)

            # Upsert based on article_id
            result = await self.collection.replace_one(
                {"article_id": sentiment.article_id}, doc, upsert=True
            )

            sentiment_id = (
                result.upserted_id
                or await self._get_sentiment_id_by_article(sentiment.article_id)
            )

            logger.debug(f"Saved sentiment for article: {sentiment.article_id}")
            return str(sentiment_id)

        except Exception as e:
            logger.error(f"Failed to save sentiment: {e}")
            raise SentimentRepositoryError(f"Failed to save sentiment: {e}")

    async def get_sentiment(self, sentiment_id: str) -> Optional[ProcessedSentiment]:
        """
        Retrieve sentiment by ID.

        Args:
            sentiment_id: Sentiment ID

        Returns:
            Optional[ProcessedSentiment]: Sentiment or None if not found
        """
        try:
            doc = await self.collection.find_one({"_id": ObjectId(sentiment_id)})
            return ProcessedSentiment(**doc) if doc else None

        except Exception as e:
            logger.error(f"Failed to get sentiment: {e}")
            return None

    async def get_sentiment_by_article_id(
        self, article_id: str
    ) -> Optional[ProcessedSentiment]:
        """
        Retrieve sentiment by article ID.

        Args:
            article_id: Article ID

        Returns:
            Optional[ProcessedSentiment]: Sentiment or None if not found
        """
        try:
            doc = await self.collection.find_one({"article_id": article_id})
            return ProcessedSentiment(**doc) if doc else None

        except Exception as e:
            logger.error(f"Failed to get sentiment by article ID: {e}")
            return None

    async def search_sentiments(
        self, filter_params: SentimentQueryFilter
    ) -> List[ProcessedSentiment]:
        """
        Search sentiments based on filter parameters.

        Args:
            filter_params: Filter parameters

        Returns:
            List[ProcessedSentiment]: Matching sentiments
        """
        try:
            mongo_filter = self._build_filter(filter_params)

            cursor = (
                self.collection.find(mongo_filter)
                .sort("processed_at", DESCENDING)
                .skip(filter_params.offset)
                .limit(filter_params.limit)
            )

            docs = await cursor.to_list(length=filter_params.limit)
            sentiments = [ProcessedSentiment(**doc) for doc in docs]

            logger.debug(f"Found {len(sentiments)} sentiments for search query")
            return sentiments

        except Exception as e:
            logger.error(f"Failed to search sentiments: {e}")
            return []

    async def get_sentiment_aggregation(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        source_domain: Optional[str] = None,
    ) -> SentimentAggregation:
        """
        Get aggregated sentiment statistics.

        Args:
            date_from: Start date filter
            date_to: End date filter
            source_domain: Source domain filter

        Returns:
            SentimentAggregation: Aggregated statistics
        """
        try:
            # Build match filter
            match_filter: Dict[str, Any] = {}

            if date_from or date_to:
                date_filter: Dict[str, datetime] = {}
                if date_from:
                    date_filter["$gte"] = date_from
                if date_to:
                    date_filter["$lte"] = date_to
                match_filter["processed_at"] = date_filter

            if source_domain:
                match_filter["source_domain"] = source_domain

            # Aggregation pipeline
            pipeline = []

            if match_filter:
                pipeline.append({"$match": match_filter})

            # Group by sentiment label and calculate stats
            pipeline.extend(
                [
                    {
                        "$group": {
                            "_id": "$sentiment_label",
                            "count": {"$sum": 1},
                            "avg_confidence": {"$avg": "$confidence"},
                            "total_confidence": {"$sum": "$confidence"},
                        }
                    },
                    {
                        "$group": {
                            "_id": None,
                            "labels": {
                                "$push": {
                                    "label": "$_id",
                                    "count": "$count",
                                    "avg_confidence": "$avg_confidence",
                                }
                            },
                            "total_count": {"$sum": "$count"},
                            "total_confidence": {"$sum": "$total_confidence"},
                        }
                    },
                ]
            )

            # Execute aggregation
            cursor = self.collection.aggregate(pipeline)
            result = await cursor.to_list(length=1)

            if not result:
                return SentimentAggregation(
                    total_count=0,
                    positive_count=0,
                    negative_count=0,
                    neutral_count=0,
                    average_confidence=0.0,
                    sentiment_distribution={},
                )

            # Process results
            data = result[0]
            total_count = data["total_count"]
            average_confidence = (
                data["total_confidence"] / total_count if total_count > 0 else 0.0
            )

            # Initialize counts
            positive_count = negative_count = neutral_count = 0
            sentiment_distribution = {}

            for label_data in data["labels"]:
                label = label_data["label"]
                count = label_data["count"]

                if label == SentimentLabel.POSITIVE:
                    positive_count = count
                elif label == SentimentLabel.NEGATIVE:
                    negative_count = count
                elif label == SentimentLabel.NEUTRAL:
                    neutral_count = count

                sentiment_distribution[label] = (
                    count / total_count if total_count > 0 else 0.0
                )

            # Determine time period
            time_period = None
            if date_from and date_to:
                time_period = f"{date_from.strftime('%Y-%m-%d')} to {date_to.strftime('%Y-%m-%d')}"

            return SentimentAggregation(
                total_count=total_count,
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=neutral_count,
                average_confidence=average_confidence,
                sentiment_distribution=sentiment_distribution,
                time_period=time_period,
            )

        except Exception as e:
            logger.error(f"Failed to get sentiment aggregation: {e}")
            return SentimentAggregation(
                total_count=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                average_confidence=0.0,
                sentiment_distribution={},
            )

    async def delete_sentiment(self, sentiment_id: str) -> bool:
        """
        Delete sentiment by ID.

        Args:
            sentiment_id: Sentiment ID

        Returns:
            bool: True if deleted, False if not found
        """
        try:
            result = await self.collection.delete_one({"_id": ObjectId(sentiment_id)})
            success = result.deleted_count > 0

            if success:
                logger.debug(f"Deleted sentiment: {sentiment_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to delete sentiment: {e}")
            return False

    async def sentiment_exists(self, article_id: str) -> bool:
        """
        Check if sentiment exists for article.

        Args:
            article_id: Article ID

        Returns:
            bool: True if exists, False otherwise
        """
        try:
            count = await self.collection.count_documents(
                {"article_id": article_id}, limit=1
            )
            return count > 0

        except Exception as e:
            logger.error(f"Failed to check sentiment existence: {e}")
            return False

    def _build_filter(self, filter_params: SentimentQueryFilter) -> Dict[str, Any]:
        """Build MongoDB filter from query parameters."""
        mongo_filter: Dict[str, Any] = {}

        if filter_params.sentiment_label:
            mongo_filter["sentiment_label"] = filter_params.sentiment_label

        if (
            filter_params.min_confidence is not None
            or filter_params.max_confidence is not None
        ):
            confidence_filter: Dict[str, float] = {}
            if filter_params.min_confidence is not None:
                confidence_filter["$gte"] = filter_params.min_confidence
            if filter_params.max_confidence is not None:
                confidence_filter["$lte"] = filter_params.max_confidence
            mongo_filter["confidence"] = confidence_filter

        if filter_params.date_from or filter_params.date_to:
            date_filter: Dict[str, datetime] = {}
            if filter_params.date_from:
                date_filter["$gte"] = filter_params.date_from
            if filter_params.date_to:
                date_filter["$lte"] = filter_params.date_to
            mongo_filter["processed_at"] = date_filter

        if filter_params.source_domain:
            mongo_filter["source_domain"] = {
                "$regex": filter_params.source_domain,
                "$options": "i",
            }

        if filter_params.source_category:
            mongo_filter["source_category"] = filter_params.source_category

        return mongo_filter

    async def _get_sentiment_id_by_article(self, article_id: str) -> Optional[ObjectId]:
        """Get sentiment ID by article ID."""
        doc = await self.collection.find_one({"article_id": article_id}, {"_id": 1})
        return doc["_id"] if doc else None

    async def close(self) -> None:
        """Close the MongoDB connection."""
        self.client.close()
        logger.info("MongoDB sentiment repository connection closed")
