# services/news_message_consumer_service.py

"""
News message consumer service for processing news messages from RabbitMQ.
"""

import json
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import asyncio

from ..interfaces.message_broker import MessageBroker, MessageBrokerError
from ..interfaces.news_repository_interface import NewsRepositoryInterface
from ..services.sentiment_service import SentimentService
from ..services.sentiment_message_producer_service import (
    SentimentMessageProducerService,
)
from ..core.config import settings
from common.logger import LoggerFactory
from ..schemas.message_schemas import NewsMessageSchema, SentimentResultMessageSchema
from ..models.sentiment import SentimentAnalysisResult

logger = LoggerFactory.get_logger(
    name="news-message-consumer", log_file="logs/news_message_consumer.log"
)


class NewsMessageConsumerService:
    """
    Service for consuming and processing news messages from RabbitMQ.
    """

    def __init__(
        self,
        message_broker: Optional[MessageBroker] = None,
        sentiment_service: Optional[SentimentService] = None,
        news_repository: Optional[NewsRepositoryInterface] = None,
        sentiment_producer: Optional[SentimentMessageProducerService] = None,
    ):
        """
        Initialize news message consumer service.

        Args:
            message_broker: Message broker instance
            sentiment_service: Sentiment service instance
            news_repository: News repository for database operations
            sentiment_producer: Sentiment message producer for publishing results
        """
        self.message_broker = message_broker
        self.sentiment_service = sentiment_service
        self.news_repository = news_repository
        self.sentiment_producer = sentiment_producer
        self._running = False
        self._queue_name = getattr(
            settings, "rabbitmq_queue_news_sentiment", "sentiment.analysis.news"
        )
        self._exchange_name = getattr(
            settings, "rabbitmq_news_exchange", "news_exchange"
        )

    async def start_consuming(self) -> None:
        """Start consuming news messages from RabbitMQ."""
        if not self.message_broker:
            logger.warning("Message broker not available, cannot start consuming")
            return

        try:
            logger.info("Starting news message consumer service")

            # Connect to message broker
            await self.message_broker.connect()

            # Setup message infrastructure
            await self._setup_message_infrastructure()

            # Start consuming news messages
            await self.message_broker.consume(
                queue=self._queue_name,
                callback=self._process_news_message,
                auto_ack=False,
            )

            self._running = True
            logger.info("News message consumer service started successfully")

        except Exception as e:
            logger.error(f"Failed to start news message consumer: {str(e)}")
            self._running = False

    async def stop_consuming(self) -> None:
        """Stop consuming messages."""
        try:
            self._running = False
            if self.message_broker:
                await self.message_broker.disconnect()
            logger.info("News message consumer service stopped")
        except Exception as e:
            logger.error(f"Error stopping news message consumer: {str(e)}")

    async def _setup_message_infrastructure(self) -> None:
        """Setup RabbitMQ exchanges and queues."""
        try:
            # Declare exchange
            await self.message_broker.declare_exchange(
                exchange=self._exchange_name, exchange_type="topic", durable=True
            )

            # Declare queue for news sentiment analysis
            await self.message_broker.declare_queue(
                queue=self._queue_name, durable=True
            )

            # Bind queue to exchange
            await self.message_broker.bind_queue(
                queue=self._queue_name,
                exchange=self._exchange_name,
                routing_key="sentiment.analysis.news",
            )

            logger.info("Message infrastructure setup completed")

        except Exception as e:
            logger.error(f"Failed to setup message infrastructure: {e}")
            raise

    async def _process_news_message(
        self, message_body: bytes, delivery_tag: int
    ) -> None:
        """
        Process a news message from the queue.

        Args:
            message_body: Raw message body
            delivery_tag: Message delivery tag for acknowledgment
        """
        start_time = time.time()

        try:
            # Parse message
            message_data = json.loads(message_body.decode("utf-8"))
            news_message = NewsMessageSchema(**message_data)

            logger.info(
                f"Processing news message: id={news_message.id}, title='{news_message.title[:50]}...'"
            )

            # Process sentiment analysis
            await self._analyze_news_sentiment(news_message)

            # Acknowledge message
            await self.message_broker.ack_message(delivery_tag)

            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"Successfully processed news message {news_message.id} in {processing_time:.2f}ms"
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message JSON: {e}")
            await self.message_broker.nack_message(delivery_tag, requeue=False)

        except Exception as e:
            logger.error(f"Error processing news message: {e}")
            # Requeue message for retry (could add retry logic here)
            await self.message_broker.nack_message(delivery_tag, requeue=True)

    async def _analyze_news_sentiment(self, news_message: NewsMessageSchema) -> None:
        """
        Analyze sentiment for news message and update database.

        Args:
            news_message: News message to analyze
        """
        try:
            if not self.sentiment_service:
                logger.warning("Sentiment service not available, skipping analysis")
                return

            # Prepare text for analysis (combine title and description)
            analysis_text = news_message.title
            if news_message.description:
                analysis_text += f" {news_message.description}"

            # Perform sentiment analysis
            start_time = time.time()
            result = await self.sentiment_service.analyze_text(
                text=analysis_text,
                title=news_message.title,
                article_id=news_message.id,
                source_url=news_message.url,
                save_result=False,  # We'll save manually
                publish_result=False,  # We'll publish manually
                metadata={
                    "source": news_message.source,
                    "author": news_message.author,
                    "tags": news_message.tags,
                    "published_at": news_message.published_at.isoformat(),
                    "fetched_at": news_message.fetched_at.isoformat(),
                },
            )

            processing_time = (time.time() - start_time) * 1000

            # Update news item with sentiment results
            if self.news_repository and result:
                await self.news_repository.update_news_sentiment(
                    item_id=news_message.id,
                    sentiment_label=result.label.value,
                    sentiment_scores={
                        score.label.value: score.score for score in result.scores
                    },
                    sentiment_confidence=result.confidence,
                    sentiment_reasoning=result.reasoning,
                    analyzer_version=getattr(result, "analyzer_version", None),
                )

            # Publish sentiment result message
            if self.sentiment_producer and result:
                await self._publish_sentiment_result(
                    news_message, result, processing_time
                )

            logger.info(
                f"Sentiment analysis completed for news {news_message.id}: {result.label.value}"
            )

        except Exception as e:
            logger.error(f"Failed to analyze sentiment for news {news_message.id}: {e}")

    async def _publish_sentiment_result(
        self,
        news_message: NewsMessageSchema,
        sentiment_result: SentimentAnalysisResult,
        processing_time_ms: float,
    ) -> None:
        """
        Publish sentiment analysis result.

        Args:
            news_message: Original news message
            sentiment_result: Sentiment analysis result
            processing_time_ms: Processing time in milliseconds
        """
        try:
            # Create sentiment result message
            result_message = SentimentResultMessageSchema(
                news_id=news_message.id,
                url=news_message.url,
                title=news_message.title,
                sentiment_label=sentiment_result.label.value,
                sentiment_scores={
                    score.label.value: score.score for score in sentiment_result.scores
                },
                confidence=sentiment_result.confidence,
                reasoning=sentiment_result.reasoning,
                processed_at=datetime.now(timezone.utc),
                processing_time_ms=processing_time_ms,
                analyzer_version=getattr(sentiment_result, "analyzer_version", None),
                metadata={"source": news_message.source, "tags": news_message.tags},
            )

            # Publish the result
            await self.sentiment_producer.publish_sentiment_result(result_message)

            logger.debug(f"Published sentiment result for news {news_message.id}")

        except Exception as e:
            logger.error(
                f"Failed to publish sentiment result for news {news_message.id}: {e}"
            )

    def is_running(self) -> bool:
        """
        Check if consumer is running.

        Returns:
            bool: True if consumer is running, False otherwise
        """
        return self._running

    def set_dependencies(
        self,
        message_broker: Optional[MessageBroker] = None,
        sentiment_service: Optional[SentimentService] = None,
        news_repository: Optional[NewsRepositoryInterface] = None,
        sentiment_producer: Optional[SentimentMessageProducerService] = None,
    ) -> None:
        """
        Set or update service dependencies.

        Args:
            message_broker: Message broker instance
            sentiment_service: Sentiment service instance
            news_repository: News repository instance
            sentiment_producer: Sentiment message producer instance
        """
        if message_broker:
            self.message_broker = message_broker
        if sentiment_service:
            self.sentiment_service = sentiment_service
        if news_repository:
            self.news_repository = news_repository
        if sentiment_producer:
            self.sentiment_producer = sentiment_producer

        logger.info("Dependencies updated for NewsMessageConsumerService")
