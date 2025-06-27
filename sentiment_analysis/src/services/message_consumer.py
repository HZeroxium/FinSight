# services/message_consumer.py

"""
Message consumer service for processing messages from RabbitMQ.
"""

from typing import Dict, Any

from ..interfaces.message_broker import MessageBroker
from ..services.sentiment_service import SentimentService
from ..core.config import settings
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

            # Start consuming from article sentiment analysis queue
            await self.message_broker.consume(
                queue=settings.rabbitmq_queue_raw_articles,
                callback=self._process_article_sentiment_message,
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
                settings.rabbitmq_exchange, "topic", durable=True
            )
            await self.message_broker.declare_exchange(
                "sentiment_analysis_exchange", "topic", durable=True
            )

            # Declare queues
            await self.message_broker.declare_queue(
                settings.rabbitmq_queue_raw_articles, durable=True
            )
            await self.message_broker.declare_queue(
                settings.rabbitmq_queue_processed, durable=True
            )

            # Bind queues to exchanges
            await self.message_broker.bind_queue(
                settings.rabbitmq_queue_raw_articles,
                settings.rabbitmq_exchange,
                "article.sentiment_analysis",
            )
            await self.message_broker.bind_queue(
                settings.rabbitmq_queue_processed,
                "sentiment_analysis_exchange",
                "sentiment.processed",
            )

            logger.info("Message infrastructure setup completed")

        except Exception as e:
            logger.error(f"Failed to setup message infrastructure: {str(e)}")
            raise

    async def _process_article_sentiment_message(self, message: Dict[str, Any]) -> None:
        """
        Process an article message for sentiment analysis.

        Args:
            message: Article message data from news crawler
        """
        try:
            article_id = message.get("id")
            title = message.get("title", "")
            content = message.get("content", "")
            url = message.get("url")
            search_query = message.get("search_query", "")

            logger.info(f"Processing sentiment analysis for article: {article_id}")

            if not article_id or not content:
                logger.warning("Skipping message with missing required fields")
                return

            # Combine title and content for better context
            analysis_text = f"{title}\n\n{content}" if title else content

            # Truncate if too long (API limits)
            if len(analysis_text) > 8000:
                analysis_text = analysis_text[:8000] + "..."
                logger.debug(f"Truncated analysis text for article {article_id}")

            # Analyze sentiment
            result = await self.sentiment_service.analyze_text(
                text=analysis_text,
                title=title,
                article_id=article_id,
                source_url=url,
                save_result=True,
            )

            # Publish processed sentiment result
            sentiment_message = {
                "article_id": article_id,
                "url": url,
                "title": title,
                "content_preview": content[:200] if content else "",
                "search_query": search_query,
                "sentiment_label": result.label.value,
                "scores": {
                    "positive": result.scores.positive,
                    "negative": result.scores.negative,
                    "neutral": result.scores.neutral,
                },
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "processed_at": message.get("search_timestamp", ""),
                "metadata": message.get("metadata", {}),
            }

            # Publish to processed sentiments queue
            await self.message_broker.publish(
                exchange="sentiment_analysis_exchange",
                routing_key="sentiment.processed",
                message=sentiment_message,
            )

            logger.info(
                f"Successfully processed sentiment for article {article_id}: {result.label} (confidence: {result.confidence:.2f})"
            )

        except Exception as e:
            logger.error(f"Failed to process article sentiment message: {str(e)}")
            # Don't raise exception to avoid requeue loops
            # In production, might want to send to dead letter queue

    def is_running(self) -> bool:
        """Check if consumer is running."""
        return self._running
