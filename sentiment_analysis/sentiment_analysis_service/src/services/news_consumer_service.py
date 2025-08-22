# services/news_consumer_service.py

"""
Consolidated news message consumer service for sentiment analysis.
"""

import asyncio
from typing import Dict, Any

from ..interfaces.message_broker import MessageBroker, MessageBrokerError
from .sentiment_service import SentimentService
from ..core.config import settings
from ..schemas.message_schemas import NewsMessageSchema
from common.logger import LoggerFactory, LoggerType, LogLevel


class NewsConsumerService:
    """News message consumer for sentiment analysis processing."""

    def __init__(
        self,
        message_broker: MessageBroker,
        sentiment_service: SentimentService,
    ):
        """
        Initialize news consumer service.

        Args:
            message_broker: Message broker instance
            sentiment_service: Sentiment service for processing
        """
        self.message_broker = message_broker
        self.sentiment_service = sentiment_service
        self._running = False

        self.logger = LoggerFactory.get_logger(
            name="news-consumer",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            log_file="logs/news_message_consumer.log",
        )
        self.logger.info("NewsConsumerService initialized")

    async def start_consuming(self) -> None:
        """Start consuming news messages from RabbitMQ with resilient error handling."""
        try:
            self.logger.info("Starting news consumer service")

            # Connect to message broker with retry logic
            await self._connect_with_retry()

            # Setup infrastructure
            await self._setup_infrastructure()

            # Start consuming
            await self.message_broker.consume(
                queue=settings.rabbitmq_queue_news_to_sentiment,
                callback=self._process_news_message,
                auto_ack=False,
            )

            self._running = True
            self.logger.info("News consumer service started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start news consumer: {e}")
            self._running = False
            # Don't re-raise to prevent startup failure

    async def stop_consuming(self) -> None:
        """Stop consuming messages."""
        try:
            self._running = False
            if self.message_broker:
                await self.message_broker.disconnect()
            self.logger.info("News consumer service stopped")
        except Exception as e:
            self.logger.error(f"Error stopping news consumer: {e}")

    async def _connect_with_retry(self, max_retries: int = 3) -> None:
        """Connect to message broker with retry logic and exponential backoff."""
        for attempt in range(max_retries):
            try:
                await self.message_broker.connect()
                self.logger.info("Connected to message broker")
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise MessageBrokerError(
                        f"Failed to connect after {max_retries} attempts: {e}"
                    )

                wait_time = 2**attempt  # Exponential backoff
                self.logger.warning(
                    f"Connection attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                )
                await asyncio.sleep(wait_time)

    async def _setup_infrastructure(self) -> None:
        """Setup RabbitMQ infrastructure using centralized config."""
        try:
            # Declare exchange
            await self.message_broker.declare_exchange(
                settings.rabbitmq_exchange, "topic", durable=True
            )

            # Declare news-to-sentiment queue
            await self.message_broker.declare_queue(
                settings.rabbitmq_queue_news_to_sentiment, durable=True
            )

            # Bind news-to-sentiment queue
            await self.message_broker.bind_queue(
                settings.rabbitmq_queue_news_to_sentiment,
                settings.rabbitmq_exchange,
                settings.rabbitmq_routing_key_news_to_sentiment,
            )

            self.logger.info("Message broker infrastructure setup completed")

        except Exception as e:
            self.logger.error(f"Failed to setup infrastructure: {e}")
            raise

    async def _process_news_message(self, message: Dict[str, Any]) -> None:
        """
        Process a news message using the sentiment service.

        Args:
            message: Raw message data received from RabbitMQ as dictionary
        """
        try:
            # Get message ID for logging
            message_id = message.get("id", "unknown")

            self.logger.debug(f"Processing news message: {message_id}")

            # Log message structure for debugging
            self.logger.info(
                f"Received message: {message_id} - {message.get('title', 'No title')[:50]}..."
            )

            # Convert dictionary to NewsMessageSchema for validation and processing
            try:
                news_message = NewsMessageSchema(**message)
            except Exception as validation_error:
                self.logger.error(
                    f"Failed to validate message schema for {message_id}: {validation_error}"
                )
                return

            # Use sentiment service to process the validated message
            success = await self.sentiment_service.process_news_message(news_message)

            if success:
                self.logger.info(f"Successfully processed news message: {message_id}")
            else:
                self.logger.error(f"Failed to process news message: {message_id}")

        except Exception as e:
            self.logger.error(f"Error processing news message: {e}")
            # Don't re-raise to prevent message requeue loops

    def is_running(self) -> bool:
        """Check if consumer is running."""
        return self._running

    async def health_check(self) -> bool:
        """Check consumer service health."""
        try:
            # Check if running
            if not self._running:
                return False

            # Check message broker health
            if not await self.message_broker.health_check():
                return False

            # Check sentiment service health
            if not await self.sentiment_service.health_check():
                return False

            return True

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
