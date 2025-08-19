# services/news_message_producer_service.py

"""
News message producer service for publishing news to sentiment analysis queue.
"""

from typing import Optional, Dict, Any
from datetime import datetime

from ..interfaces.message_broker import MessageBroker, MessageBrokerError
from ..schemas.message_schemas import NewsMessageSchema
from ..schemas.news_schemas import NewsItem
from ..models.news_model import NewsModel
from ..core.config import settings
from common.logger import LoggerFactory

logger = LoggerFactory.get_logger(
    name="news-message-producer", log_file="logs/news_message_producer.log"
)


class NewsMessageProducerService:
    """Service for publishing news messages to sentiment analysis queue."""

    def __init__(self, message_broker: Optional[MessageBroker] = None):
        """
        Initialize news message producer service.

        Args:
            message_broker: Message broker instance for publishing
        """
        self.message_broker = message_broker
        self._exchange_name = getattr(
            settings, "rabbitmq_news_exchange", "news_exchange"
        )
        self._routing_key = getattr(
            settings, "rabbitmq_routing_key_news_sentiment", "sentiment.analysis.news"
        )
        logger.info("NewsMessageProducerService initialized")

    async def publish_news_for_sentiment(
        self,
        news_item: NewsItem,
        article_id: str,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Publish news item to sentiment analysis queue.

        Args:
            news_item: News item to publish
            article_id: ID of the stored news item
            additional_metadata: Additional metadata to include

        Returns:
            bool: True if published successfully, False otherwise
        """
        if not self.message_broker:
            logger.warning("Message broker not available, skipping message publishing")
            return False

        try:
            # Create message schema
            message_data = self._create_news_message(
                news_item, article_id, additional_metadata
            )

            # Publish message
            await self.message_broker.publish_message(
                exchange_name=self._exchange_name,
                routing_key=self._routing_key,
                message=message_data.model_dump(mode="json"),
                message_type="news_for_sentiment",
            )

            logger.info(
                f"Successfully published news message for sentiment analysis: "
                f"id={article_id}, title='{news_item.title[:50]}...'"
            )
            return True

        except MessageBrokerError as e:
            logger.error(f"Message broker error when publishing news: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error when publishing news message: {e}")
            return False

    async def publish_news_model_for_sentiment(
        self,
        news_model: NewsModel,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Publish news model to sentiment analysis queue.

        Args:
            news_model: News model to publish
            additional_metadata: Additional metadata to include

        Returns:
            bool: True if published successfully, False otherwise
        """
        if not self.message_broker:
            logger.warning("Message broker not available, skipping message publishing")
            return False

        try:
            # Create message schema from model
            message_data = self._create_news_message_from_model(
                news_model, additional_metadata
            )

            # Publish message
            await self.message_broker.publish_message(
                exchange_name=self._exchange_name,
                routing_key=self._routing_key,
                message=message_data.model_dump(mode="json"),
                message_type="news_for_sentiment",
            )

            logger.info(
                f"Successfully published news model message for sentiment analysis: "
                f"id={news_model.id}, title='{news_model.title[:50]}...'"
            )
            return True

        except MessageBrokerError as e:
            logger.error(f"Message broker error when publishing news model: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error when publishing news model message: {e}")
            return False

    def _create_news_message(
        self,
        news_item: NewsItem,
        article_id: str,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> NewsMessageSchema:
        """
        Create NewsMessageSchema from NewsItem.

        Args:
            news_item: News item to convert
            article_id: ID of the stored news item
            additional_metadata: Additional metadata to include

        Returns:
            NewsMessageSchema: Formatted message schema
        """
        # Prepare metadata
        metadata = dict(news_item.metadata) if news_item.metadata else {}
        if additional_metadata:
            metadata.update(additional_metadata)

        # Create message schema
        return NewsMessageSchema(
            id=article_id,
            url=str(news_item.url),
            title=news_item.title,
            description=news_item.description,
            source=news_item.source.value,
            published_at=news_item.published_at,
            author=news_item.author,
            tags=news_item.tags or [],
            fetched_at=datetime.now(),
            metadata=metadata,
        )

    def _create_news_message_from_model(
        self,
        news_model: NewsModel,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> NewsMessageSchema:
        """
        Create NewsMessageSchema from NewsModel.

        Args:
            news_model: News model to convert
            additional_metadata: Additional metadata to include

        Returns:
            NewsMessageSchema: Formatted message schema
        """
        # Prepare metadata
        metadata = dict(news_model.metadata) if news_model.metadata else {}
        if additional_metadata:
            metadata.update(additional_metadata)

        # Create message schema
        return NewsMessageSchema(
            id=news_model.id or "",
            url=news_model.url,
            title=news_model.title,
            description=news_model.description,
            source=news_model.source.value,
            published_at=news_model.published_at,
            author=news_model.author,
            tags=news_model.tags or [],
            fetched_at=news_model.fetched_at,
            metadata=metadata,
        )

    async def publish_batch_news_for_sentiment(
        self,
        news_items: list[tuple[NewsItem, str]],
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Publish multiple news items to sentiment analysis queue.

        Args:
            news_items: List of tuples (NewsItem, article_id) to publish
            additional_metadata: Additional metadata to include for all items

        Returns:
            int: Number of successfully published messages
        """
        if not self.message_broker:
            logger.warning(
                "Message broker not available, skipping batch message publishing"
            )
            return 0

        successful_count = 0
        for news_item, article_id in news_items:
            if await self.publish_news_for_sentiment(
                news_item, article_id, additional_metadata
            ):
                successful_count += 1

        logger.info(
            f"Batch publishing completed: {successful_count}/{len(news_items)} messages published"
        )
        return successful_count

    def set_message_broker(self, message_broker: MessageBroker) -> None:
        """
        Set or update the message broker instance.

        Args:
            message_broker: New message broker instance
        """
        self.message_broker = message_broker
        logger.info("Message broker updated for NewsMessageProducerService")

    def is_available(self) -> bool:
        """
        Check if message producer is available.

        Returns:
            bool: True if message broker is available, False otherwise
        """
        return self.message_broker is not None


# Legacy class for backward compatibility
class MessagePublisherService(NewsMessageProducerService):
    """Legacy class - use NewsMessageProducerService instead."""

    def __init__(self, message_broker: MessageBroker) -> None:
        super().__init__(message_broker)
        logger.warning(
            "MessagePublisherService is deprecated, use NewsMessageProducerService instead"
        )

    async def publish_new_article(self, item: NewsItem, article_id: str) -> bool:
        """Legacy method - use publish_news_for_sentiment instead."""
        logger.warning(
            "publish_new_article is deprecated, use publish_news_for_sentiment instead"
        )
        return await self.publish_news_for_sentiment(item, article_id)
