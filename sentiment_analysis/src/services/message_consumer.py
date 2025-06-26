"""
Message consumer service for processing messages from RabbitMQ.
"""

import asyncio
import json
from typing import Dict, Any

from ..interfaces.message_broker import MessageBroker
from ..services.sentiment_service import SentimentService
from ..models.sentiment import SentimentRequest
from ..common.logger import LoggerFactory, LoggerType, LogLevel

logger = LoggerFactory.get_logger(
    name="message-consumer", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


class MessageConsumerService:
    """
    Service for consuming and processing messages from RabbitMQ.
    """

    def __init__(
        self,
        message_broker: MessageBroker,
        sentiment_service: SentimentService,
    ):
        """
        Initialize message consumer service.

        Args:
            message_broker: Message broker instance
            sentiment_service: Sentiment service instance
        """
        self.message_broker = message_broker
        self.sentiment_service = sentiment_service
        self._running = False

    async def start_consuming(self) -> None:
        """Start consuming messages from RabbitMQ."""
        try:
            logger.info("Starting message consumer service")

            # Connect to message broker
            await self.message_broker.connect()

            # Declare necessary exchanges and queues
            await self._setup_message_infrastructure()

            # Start consuming from raw articles queue
            await self.message_broker.consume(
                queue="raw_articles_queue",
                callback=self._process_raw_article_message,
                auto_ack=False,
            )

            self._running = True
            logger.info("Message consumer service started successfully")

        except Exception as e:
            logger.error(f"Failed to start message consumer: {str(e)}")
            raise

    async def stop_consuming(self) -> None:
        """Stop consuming messages."""
        try:
            self._running = False
            await self.message_broker.disconnect()
            logger.info("Message consumer service stopped")
        except Exception as e:
            logger.error(f"Error stopping message consumer: {str(e)}")

    async def _setup_message_infrastructure(self) -> None:
        """Setup RabbitMQ exchanges and queues."""
        try:
            # Declare exchanges
            await self.message_broker.declare_exchange(
                "news_crawler_exchange", "topic", durable=True
            )
            await self.message_broker.declare_exchange(
                "sentiment_analysis_exchange", "topic", durable=True
            )

            # Declare queues
            await self.message_broker.declare_queue("raw_articles_queue", durable=True)
            await self.message_broker.declare_queue(
                "processed_sentiments_queue", durable=True
            )

            # Bind queues to exchanges
            await self.message_broker.bind_queue(
                "raw_articles_queue", "news_crawler_exchange", "article.crawled"
            )
            await self.message_broker.bind_queue(
                "processed_sentiments_queue",
                "sentiment_analysis_exchange",
                "sentiment.processed",
            )

            logger.info("Message infrastructure setup completed")

        except Exception as e:
            logger.error(f"Failed to setup message infrastructure: {str(e)}")
            raise

    async def _process_raw_article_message(self, message: Dict[str, Any]) -> None:
        """
        Process a raw article message from news crawler.

        Args:
            message: Raw article message data
        """
        try:
            logger.info(
                f"Processing raw article message: {message.get('id', 'unknown')}"
            )

            # Extract article data
            article_id = message.get("id")
            title = message.get("title", "")
            content = message.get("content", "")
            url = message.get("url")

            if not article_id or not content:
                logger.warning("Skipping message with missing required fields")
                return

            # Create sentiment request
            request = SentimentRequest(
                text=content,
                title=title,
                source_url=url,
            )

            # Analyze sentiment
            result = await self.sentiment_service.analyze_text(
                text=content,
                title=title,
                article_id=article_id,
                source_url=url,
                save_result=True,
            )

            # Publish processed sentiment
            sentiment_message = {
                "article_id": article_id,
                "url": url,
                "title": title,
                "sentiment_label": result.label.value,
                "scores": {
                    "positive": result.scores.positive,
                    "negative": result.scores.negative,
                    "neutral": result.scores.neutral,
                },
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "processed_at": str(message.get("created_at", "")),
            }

            await self.message_broker.publish(
                exchange="sentiment_analysis_exchange",
                routing_key="sentiment.processed",
                message=sentiment_message,
            )

            logger.info(f"Successfully processed article {article_id}: {result.label}")

        except Exception as e:
            logger.error(f"Failed to process raw article message: {str(e)}")
            # Don't raise exception to avoid requeue loops

    def is_running(self) -> bool:
        """Check if consumer is running."""
        return self._running
