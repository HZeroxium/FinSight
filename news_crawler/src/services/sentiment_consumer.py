# service/sentiment_consumer.py

"""
Sentiment consumer service for processing sentiment analysis results from RabbitMQ.
"""

from typing import Dict, Any

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
        """Start consuming sentiment analysis results from RabbitMQ."""
        try:
            logger.info("Starting sentiment consumer service")

            # Connect to message broker
            await self.message_broker.connect()

            # Declare necessary exchanges and queues
            await self._setup_message_infrastructure()

            # Start consuming from processed sentiment queue
            await self.message_broker.consume(
                queue=settings.rabbitmq_queue_processed,
                callback=self._process_sentiment_result_message,
                auto_ack=False,
            )

            self._running = True
            logger.info("Sentiment consumer service started successfully")

        except Exception as e:
            logger.error(f"Failed to start sentiment consumer: {str(e)}")
            raise

    async def stop_consuming(self) -> None:
        """Stop consuming messages."""
        try:
            self._running = False
            await self.message_broker.disconnect()
            logger.info("Sentiment consumer service stopped")
        except Exception as e:
            logger.error(f"Error stopping sentiment consumer: {str(e)}")

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

            # Bind processed sentiments queue to sentiment analysis exchange
            await self.message_broker.bind_queue(
                settings.rabbitmq_queue_processed,
                "sentiment_analysis_exchange",
                "sentiment.processed",
            )

            logger.info("Sentiment consumer infrastructure setup completed")

        except Exception as e:
            logger.error(f"Failed to setup sentiment consumer infrastructure: {str(e)}")
            raise

    async def _process_sentiment_result_message(self, message: Dict[str, Any]) -> None:
        """
        Process a sentiment analysis result message.

        Args:
            message: Sentiment analysis result message from sentiment service
        """
        try:
            article_id = message.get("article_id")
            url = message.get("url")
            title = message.get("title", "")
            sentiment_label = message.get("sentiment_label")
            scores = message.get("scores", {})
            confidence = message.get("confidence", 0.0)
            reasoning = message.get("reasoning", "")
            processed_at = message.get("processed_at", "")
            search_query = message.get("search_query", "")

            logger.info(
                f"Processing sentiment result for article: {article_id} | "
                f"Title: '{title[:50]}...' | "
                f"Sentiment: {sentiment_label} | "
                f"Confidence: {confidence:.2f} | "
                f"URL: {url}"
            )

            # Log detailed sentiment analysis results
            logger.info(
                f"Sentiment Analysis Results:\n"
                f"  Article ID: {article_id}\n"
                f"  URL: {url}\n"
                f"  Title: {title}\n"
                f"  Search Query: {search_query}\n"
                f"  Sentiment Label: {sentiment_label}\n"
                f"  Confidence: {confidence:.2f}\n"
                f"  Scores: Positive={scores.get('positive', 0):.2f}, "
                f"Negative={scores.get('negative', 0):.2f}, "
                f"Neutral={scores.get('neutral', 0):.2f}\n"
                f"  Reasoning: {reasoning}\n"
                f"  Processed At: {processed_at}\n"
                f"  Content Preview: {message.get('content_preview', '')[:100]}..."
            )

            # Here you could add additional processing logic such as:
            # - Storing aggregated sentiment data
            # - Triggering notifications for high-confidence negative sentiment
            # - Updating search result rankings based on sentiment
            # - Publishing to other services or dashboards

            # For now, we just log the successful processing
            logger.info(f"Successfully processed sentiment for article: {article_id}")

        except Exception as e:
            logger.error(f"Failed to process sentiment result message: {str(e)}")
            # Don't raise exception to avoid requeue loops
            # In production, might want to send to dead letter queue
