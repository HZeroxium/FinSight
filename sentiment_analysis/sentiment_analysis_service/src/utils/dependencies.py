# utils/dependencies.py

"""
Dependency injection utilities for sentiment analysis service.
"""

from functools import lru_cache

from ..adapters.openai_sentiment_analyzer import OpenAISentimentAnalyzer
from ..adapters.rabbitmq_broker import RabbitMQBroker
from ..repositories.mongo_news_repository import MongoNewsRepository
from ..services.sentiment_service import SentimentService
from ..services.news_message_consumer_service import NewsMessageConsumerService
from ..services.sentiment_message_producer_service import (
    SentimentMessageProducerService,
)
from ..core.config import settings
from common.logger import LoggerFactory, LoggerType, LogLevel

# Create logger for dependencies
logger = LoggerFactory.get_logger(
    name="dependencies", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


@lru_cache()
def get_sentiment_analyzer() -> OpenAISentimentAnalyzer:
    """Get sentiment analyzer instance."""
    logger.info("Creating OpenAI sentiment analyzer instance")
    return OpenAISentimentAnalyzer(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        max_tokens=settings.openai_max_tokens,
        max_retries=settings.analysis_retry_attempts,
    )


@lru_cache()
def get_news_repository() -> MongoNewsRepository:
    """Get news repository instance."""
    logger.info("Creating MongoDB news repository instance")
    return MongoNewsRepository(
        mongo_url=settings.mongodb_url,
        database_name=settings.mongodb_database,
    )


@lru_cache()
def get_message_broker() -> RabbitMQBroker:
    """Get message broker instance."""
    logger.info("Creating RabbitMQ broker instance")
    return RabbitMQBroker(connection_url=settings.rabbitmq_url)


@lru_cache()
def get_sentiment_service() -> SentimentService:
    """Get sentiment service instance."""
    logger.info("Creating sentiment service instance")
    analyzer = get_sentiment_analyzer()
    news_repository = get_news_repository()
    message_broker = get_message_broker()

    return SentimentService(
        analyzer=analyzer,
        news_repository=news_repository,
        message_broker=message_broker,
    )


@lru_cache()
def get_sentiment_message_producer() -> SentimentMessageProducerService:
    """Get sentiment message producer instance."""
    logger.info("Creating sentiment message producer instance")
    message_broker = get_message_broker()
    return SentimentMessageProducerService(message_broker=message_broker)


@lru_cache()
def get_news_message_consumer() -> NewsMessageConsumerService:
    """Get news message consumer instance."""
    logger.info("Creating news message consumer instance")
    message_broker = get_message_broker()
    sentiment_service = get_sentiment_service()
    news_repository = get_news_repository()
    sentiment_producer = get_sentiment_message_producer()

    return NewsMessageConsumerService(
        message_broker=message_broker,
        sentiment_service=sentiment_service,
        news_repository=news_repository,
        sentiment_producer=sentiment_producer,
    )
