# services/sentiment_message_producer_service.py

"""
Sentiment message producer service for publishing sentiment results.
"""

import json
from typing import Optional, Dict, Any
from datetime import datetime

from ..interfaces.message_broker import MessageBroker, MessageBrokerError
from ..schemas.message_schemas import SentimentResultMessageSchema
from ..core.config import settings
from common.logger import LoggerFactory

logger = LoggerFactory.get_logger(
    name="sentiment-message-producer", log_file="logs/sentiment_message_producer.log"
)


class SentimentMessageProducerService:
    """Service for publishing sentiment analysis results."""

    def __init__(self, message_broker: Optional[MessageBroker] = None):
        """
        Initialize sentiment message producer service.

        Args:
            message_broker: Message broker instance for publishing
        """
        self.message_broker = message_broker
        self._routing_key = getattr(
            settings, "rabbitmq_routing_key_sentiment_results", "sentiment.results"
        )
        self._exchange_name = getattr(
            settings, "rabbitmq_sentiment_exchange", "sentiment_exchange"
        )
        logger.info("SentimentMessageProducerService initialized")

    async def publish_sentiment_result(
        self,
        result_message: SentimentResultMessageSchema,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Publish sentiment analysis result to message queue.

        Args:
            result_message: Sentiment result message to publish
            additional_metadata: Additional metadata to include

        Returns:
            bool: True if published successfully, False otherwise
        """
        if not self.message_broker:
            logger.warning("Message broker not available, skipping message publishing")
            return False

        try:
            # Add additional metadata if provided
            if additional_metadata:
                result_message.metadata.update(additional_metadata)

            # Publish message
            await self.message_broker.publish_message(
                exchange_name=self._exchange_name,
                routing_key=self._routing_key,
                message=result_message.model_dump(mode="json"),
                message_type="sentiment_result",
            )

            logger.info(
                f"Successfully published sentiment result: "
                f"news_id={result_message.news_id}, label={result_message.sentiment_label}"
            )
            return True

        except MessageBrokerError as e:
            logger.error(f"Message broker error when publishing sentiment result: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error when publishing sentiment result: {e}")
            return False

    async def publish_batch_sentiment_results(
        self,
        result_messages: list[SentimentResultMessageSchema],
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Publish multiple sentiment analysis results to message queue.

        Args:
            result_messages: List of sentiment result messages to publish
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
        for result_message in result_messages:
            if await self.publish_sentiment_result(result_message, additional_metadata):
                successful_count += 1

        logger.info(
            f"Batch publishing completed: {successful_count}/{len(result_messages)} sentiment results published"
        )
        return successful_count

    def set_message_broker(self, message_broker: MessageBroker) -> None:
        """
        Set or update the message broker instance.

        Args:
            message_broker: New message broker instance
        """
        self.message_broker = message_broker
        logger.info("Message broker updated for SentimentMessageProducerService")

    def is_available(self) -> bool:
        """
        Check if message producer is available.

        Returns:
            bool: True if message broker is available, False otherwise
        """
        return self.message_broker is not None
