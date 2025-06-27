# repositores/article_repository.py

"""
MongoDB repository for article operations using Motor only.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta, timezone

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorDatabase,
    AsyncIOMotorCollection,
)
from bson import ObjectId

from ..models.article import (
    CrawledArticle,
    ProcessedArticle,
    ArticleSearchQuery,
)
from ..common.logger import LoggerFactory, LoggerType, LogLevel

logger = LoggerFactory.get_logger(
    name="article-repository",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
)

# Constants for index direction and type
ASCENDING = 1
DESCENDING = -1
TEXT = "text"


class ArticleRepository:
    """MongoDB repository for article operations with Motor."""

    def __init__(self, mongo_url: str, database_name: str):
        """
        Initialize repository.

        Args:
            mongo_url: MongoDB connection URL
            database_name: Database name
        """
        self.client: AsyncIOMotorClient = AsyncIOMotorClient(mongo_url)
        self.db: AsyncIOMotorDatabase = self.client[database_name]
        self.crawled_collection: AsyncIOMotorCollection = self.db.crawled_articles
        self.processed_collection: AsyncIOMotorCollection = self.db.processed_articles
        self.sources_collection: AsyncIOMotorCollection = self.db.sources

        logger.info(f"Article repository initialized with database: {database_name}")

    async def initialize(self) -> None:
        """Create all necessary indexes for each collection."""
        try:
            # Define index specifications per collection
            index_map: Dict[
                AsyncIOMotorCollection,
                List[Tuple[List[Tuple[str, Any]], Dict[str, Any]]],
            ] = {
                self.crawled_collection: [
                    ([("url", ASCENDING)], {"unique": True}),
                    ([("source.domain", ASCENDING)], {}),
                    ([("published_at", DESCENDING)], {}),
                    ([("created_at", DESCENDING)], {}),
                    ([("title", TEXT), ("content", TEXT)], {"name": "text_idx"}),
                    ([("metadata.tags", ASCENDING)], {}),
                ],
                self.processed_collection: [
                    ([("original_article_id", ASCENDING)], {}),
                    ([("url", ASCENDING)], {"unique": True}),
                    ([("source.domain", ASCENDING)], {}),
                    ([("published_at", DESCENDING)], {}),
                    ([("processed_at", DESCENDING)], {}),
                    ([("sentiment_score", ASCENDING)], {}),
                    ([("sentiment_label", ASCENDING)], {}),
                    ([("categories", ASCENDING)], {}),
                    ([("title", TEXT), ("content", TEXT)], {"name": "text_idx"}),
                ],
                self.sources_collection: [
                    ([("domain", ASCENDING)], {"unique": True}),
                    ([("name", ASCENDING)], {}),
                    ([("category", ASCENDING)], {}),
                ],
            }

            # Loop through each collection and create its indexes
            for collection, specs in index_map.items():
                for keys, kwargs in specs:
                    await collection.create_index(keys, **kwargs)

            logger.info("Database indexes created successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database indexes: {e}")
            raise

    async def save_crawled_article(self, article: CrawledArticle) -> str:
        """
        Upsert a crawled article into the database.

        Args:
            article: CrawledArticle instance to save

        Returns:
            str: The MongoDB document _id as a string.
        """
        article.updated_at = datetime.now(timezone.utc)
        doc = article.model_dump(by_alias=True, exclude_unset=True)

        try:
            result = await self.crawled_collection.replace_one(
                {"url": str(article.url)}, doc, upsert=True
            )
            article_id = result.upserted_id or await self._get_article_id_by_url(
                str(article.url)
            )
            logger.debug(f"Saved crawled article: {article.title}")
            return str(article_id)
        except Exception as e:
            logger.error(f"Failed to save crawled article: {e}")
            raise

    async def save_processed_article(self, article: ProcessedArticle) -> str:
        """
        Upsert a processed article into the database.

        Args:
            article: ProcessedArticle instance to save

        Returns:
            str: The MongoDB document _id as a string.
        """
        doc = article.model_dump(by_alias=True, exclude_unset=True)

        try:
            result = await self.processed_collection.replace_one(
                {"original_article_id": article.original_article_id},
                doc,
                upsert=True,
            )
            article_id = result.upserted_id or await self._get_processed_article_id(
                article.original_article_id
            )
            logger.debug(f"Saved processed article: {article.title}")
            return str(article_id)
        except Exception as e:
            logger.error(f"Failed to save processed article: {e}")
            raise

    async def get_crawled_article(self, article_id: str) -> Optional[CrawledArticle]:
        """
        Retrieve a crawled article by its ID.

        Args:
            article_id: The string representation of the ObjectId

        Returns:
            CrawledArticle or None if not found
        """
        try:
            doc = await self.crawled_collection.find_one({"_id": ObjectId(article_id)})
            return CrawledArticle(**doc) if doc else None
        except Exception as e:
            logger.error(f"Failed to get crawled article: {e}")
            return None

    async def get_processed_article(
        self, article_id: str
    ) -> Optional[ProcessedArticle]:
        """
        Retrieve a processed article by its ID.

        Args:
            article_id: The string representation of the ObjectId

        Returns:
            ProcessedArticle or None if not found
        """
        try:
            doc = await self.processed_collection.find_one(
                {"_id": ObjectId(article_id)}
            )
            return ProcessedArticle(**doc) if doc else None
        except Exception as e:
            logger.error(f"Failed to get processed article: {e}")
            return None

    async def search_articles(
        self, query: ArticleSearchQuery
    ) -> List[ProcessedArticle]:
        """
        Search processed articles based on query parameters.

        Args:
            query: ArticleSearchQuery with filters

        Returns:
            List[ProcessedArticle]: matching articles
        """
        try:
            mongo_filter: Dict[str, Any] = {}

            if query.query:
                mongo_filter["$text"] = {"$search": query.query}
            if query.source:
                mongo_filter["source.domain"] = {
                    "$regex": query.source,
                    "$options": "i",
                }
            if query.category:
                mongo_filter["categories"] = query.category
            if query.author:
                mongo_filter["author"] = {"$regex": query.author, "$options": "i"}
            if query.date_from or query.date_to:
                date_filter: Dict[str, datetime] = {}
                if query.date_from:
                    date_filter["$gte"] = query.date_from
                if query.date_to:
                    date_filter["$lte"] = query.date_to
                mongo_filter["published_at"] = date_filter
            if query.sentiment_min is not None or query.sentiment_max is not None:
                sentiment_filter: Dict[str, float] = {}
                if query.sentiment_min is not None:
                    sentiment_filter["$gte"] = query.sentiment_min
                if query.sentiment_max is not None:
                    sentiment_filter["$lte"] = query.sentiment_max
                mongo_filter["sentiment_score"] = sentiment_filter

            cursor = (
                self.processed_collection.find(mongo_filter)
                .sort("published_at", DESCENDING)
                .skip(query.offset)
                .limit(query.limit)
            )
            docs = await cursor.to_list(length=query.limit)
            articles = [ProcessedArticle(**doc) for doc in docs]
            logger.debug(f"Found {len(articles)} articles for search query")
            return articles

        except Exception as e:
            logger.error(f"Failed to search articles: {e}")
            return []

    async def count_articles(self, query: ArticleSearchQuery) -> int:
        """
        Count the number of processed articles that match the query.

        Args:
            query: ArticleSearchQuery with filters

        Returns:
            int: count of matching documents
        """
        try:
            # Reuse the same filter logic
            mongo_filter = await self._build_filter(query)
            return await self.processed_collection.count_documents(mongo_filter)
        except Exception as e:
            logger.error(f"Failed to count articles: {e}")
            return 0

    async def get_recent_articles(
        self, hours: int = 24, limit: int = 100
    ) -> List[ProcessedArticle]:
        """
        Get recently published processed articles.

        Args:
            hours: look-back window
            limit: maximum number of documents

        Returns:
            List[ProcessedArticle]
        """
        try:
            since = datetime.now(timezone.utc) - timedelta(hours=hours)
            cursor = (
                self.processed_collection.find({"published_at": {"$gte": since}})
                .sort("published_at", DESCENDING)
                .limit(limit)
            )
            docs = await cursor.to_list(length=limit)
            articles = [ProcessedArticle(**doc) for doc in docs]
            logger.debug(f"Retrieved {len(articles)} recent articles")
            return articles
        except Exception as e:
            logger.error(f"Failed to get recent articles: {e}")
            return []

    async def article_exists(self, url: str) -> bool:
        """
        Check if a crawled article already exists by URL.

        Args:
            url: article URL

        Returns:
            bool
        """
        try:
            count = await self.crawled_collection.count_documents({"url": url}, limit=1)
            return count > 0
        except Exception as e:
            logger.error(f"Failed to check article existence: {e}")
            return False

    async def _get_article_id_by_url(self, url: str) -> Optional[ObjectId]:
        """Helper to retrieve a crawled article's _id by URL."""
        doc = await self.crawled_collection.find_one({"url": url}, {"_id": 1})
        return doc["_id"] if doc else None

    async def _get_processed_article_id(
        self, original_article_id: ObjectId
    ) -> Optional[ObjectId]:
        """Helper to retrieve a processed article's _id by original_article_id."""
        doc = await self.processed_collection.find_one(
            {"original_article_id": original_article_id}, {"_id": 1}
        )
        return doc["_id"] if doc else None

    async def _build_filter(self, query: ArticleSearchQuery) -> Dict[str, Any]:
        """Internal helper to build MongoDB filter from ArticleSearchQuery."""
        mongo_filter: Dict[str, Any] = {}
        if query.query:
            mongo_filter["$text"] = {"$search": query.query}
        if query.source:
            mongo_filter["source.domain"] = {"$regex": query.source, "$options": "i"}
        if query.category:
            mongo_filter["categories"] = query.category
        if query.author:
            mongo_filter["author"] = {"$regex": query.author, "$options": "i"}
        if query.date_from or query.date_to:
            date_filter: Dict[str, datetime] = {}
            if query.date_from:
                date_filter["$gte"] = query.date_from
            if query.date_to:
                date_filter["$lte"] = query.date_to
            mongo_filter["published_at"] = date_filter
        if query.sentiment_min is not None or query.sentiment_max is not None:
            sentiment_filter: Dict[str, float] = {}
            if query.sentiment_min is not None:
                sentiment_filter["$gte"] = query.sentiment_min
            if query.sentiment_max is not None:
                sentiment_filter["$lte"] = query.sentiment_max
            mongo_filter["sentiment_score"] = sentiment_filter
        return mongo_filter

    async def close(self) -> None:
        """Close the MongoDB connection."""
        self.client.close()
        logger.info("Database connection closed")
