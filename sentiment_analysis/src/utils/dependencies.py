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
    """
    Get singleton sentiment analyzer instance.

    Returns:
        OpenAISentimentAnalyzer: Configured sentiment analyzer
    """
    logger.info("Creating OpenAI sentiment analyzer instance")
    return OpenAISentimentAnalyzer(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        max_tokens=settings.openai_max_tokens,
    )


@lru_cache()
def get_message_broker() -> RabbitMQBroker:
    """
    Get singleton message broker instance.

    Returns:
        RabbitMQBroker: Configured message broker
    """
    logger.info("Creating RabbitMQ broker instance")
    return RabbitMQBroker(connection_url=settings.rabbitmq_url)


@lru_cache()
def get_sentiment_repository() -> MongoDBSentimentRepository:
    """
    Get singleton sentiment repository instance.

    Returns:
        MongoDBSentimentRepository: Configured sentiment repository
    """
    logger.info("Creating MongoDB sentiment repository instance")
    return MongoDBSentimentRepository(
        mongo_url=settings.mongodb_url, database_name=settings.mongodb_database
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
