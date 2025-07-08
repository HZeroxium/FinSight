# service/sentiment_consumer.py

"""
Sentiment consumer service for processing sentiment analysis results from RabbitMQ.
"""

import asyncio
from typing import Dict, Any

from ..schemas.message_schemas import SentimentResultMessageSchema
from ..interfaces.message_broker import MessageBroker
from ..core.config import settings
from ..common.logger import LoggerFactory, LoggerType, LogLevel

logger = LoggerFactory.get_logger(
    name="sentiment-consumer", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


class SentimentConsumerService:
    """
    Service for consuming and processing sentiment analysis results from RabbitMQ.
    """

    def __init__(self, message_broker: MessageBroker):
        """
        Initialize sentiment consumer service.

        Args:
            message_broker: Message broker instance
        """
        self.message_broker = message_broker
        self._running = False

    async def start_consuming(self) -> None:
        """Start consuming sentiment analysis results from RabbitMQ with improved error handling."""
        try:
            logger.info("Starting sentiment consumer service")

            # Connect to message broker with timeout
            connection_timeout = 10  # seconds
            try:
                await asyncio.wait_for(
                    self.message_broker.connect(), timeout=connection_timeout
                )
                logger.info("Sentiment consumer connected to RabbitMQ")
            except asyncio.TimeoutError:
                logger.error(
                    f"Sentiment consumer: RabbitMQ connection timeout after {connection_timeout}s"
                )
                raise Exception("RabbitMQ connection timeout")
            except Exception as conn_error:
                logger.error(
                    f"Sentiment consumer: Failed to connect to RabbitMQ: {conn_error}"
                )
                raise

            # Declare necessary exchanges and queues
            await self._setup_message_infrastructure()

            # Start consuming from processed sentiment queue
            await self.message_broker.consume(
                queue=settings.rabbitmq_queue_processed_sentiments,
                callback=self._process_sentiment_result_message,
                auto_ack=False,
            )

            self._running = True
            logger.info("Sentiment consumer service started successfully")

        except Exception as e:
            logger.error(f"Failed to start sentiment consumer: {str(e)}")
            self._running = False
            # Don't re-raise to prevent blocking startup

    async def stop_consuming(self) -> None:
        """Stop consuming messages."""
        try:
            self._running = False
            await self.message_broker.disconnect()
            logger.info("Sentiment consumer service stopped")
        except Exception as e:
            logger.error(f"Error stopping sentiment consumer: {str(e)}")

    async def _setup_message_infrastructure(self) -> None:
        """Setup RabbitMQ exchanges and queues using config."""
        try:
            # Declare exchanges using config
            await self.message_broker.declare_exchange(
                settings.rabbitmq_article_exchange, "topic", durable=True
            )

            # Declare sentiment analysis exchange
            await self.message_broker.declare_exchange(
                "sentiment_analysis_exchange", "topic", durable=True
            )

            # Declare queues using config
            await self.message_broker.declare_queue(
                settings.rabbitmq_queue_raw_articles, durable=True
            )
            await self.message_broker.declare_queue(
                settings.rabbitmq_queue_processed_sentiments, durable=True
            )

            # Bind processed sentiments queue to sentiment analysis exchange using config
            await self.message_broker.bind_queue(
                settings.rabbitmq_queue_processed_sentiments,
                "sentiment_analysis_exchange",
                settings.rabbitmq_routing_key_sentiment_processed,
            )

            logger.info("Sentiment consumer infrastructure setup completed")

        except Exception as e:
            logger.error(f"Failed to setup sentiment consumer infrastructure: {str(e)}")
            raise

    async def _process_sentiment_result_message(self, message: Dict[str, Any]) -> None:
        """
        Process a sentiment analysis result message using schemas.

        Args:
            message: Sentiment analysis result message from sentiment service
        """
        try:
            # Validate message using schema
            sentiment_result = SentimentResultMessageSchema(**message)

            logger.info(
                f"Processing sentiment result for article: {sentiment_result.article_id} | "
                f"Title: '{sentiment_result.title[:50] if sentiment_result.title else 'N/A'}...' | "
                f"Sentiment: {sentiment_result.sentiment_label} | "
                f"Confidence: {sentiment_result.confidence:.2f} | "
                f"URL: {sentiment_result.url}"
            )

            # Log detailed sentiment analysis results
            logger.info(
                f"Sentiment Analysis Results:\n"
                f"  Article ID: {sentiment_result.article_id}\n"
                f"  URL: {sentiment_result.url}\n"
                f"  Title: {sentiment_result.title}\n"
                f"  Search Query: {sentiment_result.search_query}\n"
                f"  Sentiment Label: {sentiment_result.sentiment_label}\n"
                f"  Confidence: {sentiment_result.confidence:.2f}\n"
                f"  Scores: Positive={sentiment_result.scores.get('positive', 0):.2f}, "
                f"Negative={sentiment_result.scores.get('negative', 0):.2f}, "
                f"Neutral={sentiment_result.scores.get('neutral', 0):.2f}\n"
                f"  Reasoning: {sentiment_result.reasoning}\n"
                f"  Processed At: {sentiment_result.processed_at}\n"
                f"  Content Preview: {sentiment_result.content_preview[:100] if sentiment_result.content_preview else 'N/A'}..."
            )

            # Here you could add additional processing logic such as:
            # - Storing aggregated sentiment data
            # - Triggering notifications for high-confidence negative sentiment
            # - Updating search result rankings based on sentiment
            # - Publishing to other services or dashboards

            # For now, we just log the successful processing
            logger.info(
                f"Successfully processed sentiment for article: {sentiment_result.article_id}"
            )

        except Exception as e:
            logger.error(f"Failed to process sentiment result message: {str(e)}")
            # Don't raise exception to avoid requeue loops

    def is_running(self) -> bool:
        """Check if consumer is running."""
        return self._running
