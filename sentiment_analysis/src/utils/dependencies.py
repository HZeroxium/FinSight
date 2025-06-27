# utils/dependencies.py

"""
Dependency injection utilities for sentiment analysis service.
"""

from functools import lru_cache

from ..adapters.openai_sentiment_analyzer import OpenAISentimentAnalyzer
from ..adapters.rabbitmq_broker import RabbitMQBroker
from ..repositories.mongodb_sentiment_repository import MongoDBSentimentRepository
from ..services.sentiment_service import SentimentService
from ..core.config import settings
from ..common.logger import LoggerFactory, LoggerType, LogLevel

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
def get_sentiment_repository() -> MongoDBSentimentRepository:
    """Get sentiment repository instance."""
    logger.info("Creating MongoDB sentiment repository instance")
    return MongoDBSentimentRepository(
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
    return SentimentService(
        analyzer=get_sentiment_analyzer(),
        repository=get_sentiment_repository(),
    )


@lru_cache()
def get_sentiment_service() -> SentimentService:
    """
    Get singleton sentiment service instance.

    Returns:
        SentimentService: Configured sentiment service
    """
    logger.info("Creating sentiment service instance")
    analyzer = get_sentiment_analyzer()
    repository = get_sentiment_repository()

    return SentimentService(
        analyzer=analyzer,
        repository=repository,
    )
