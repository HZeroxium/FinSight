# utils/dependencies.py

"""
Dependency injection utilities for sentiment analysis service using dependency_injector.
"""

from dependency_injector import containers, providers

from ..adapters.openai_sentiment_analyzer import OpenAISentimentAnalyzer
from ..adapters.rabbitmq_broker import RabbitMQBroker
from ..repositories.mongo_news_repository import MongoNewsRepository
from ..services.sentiment_service import SentimentService
from ..services.news_consumer_service import NewsConsumerService
from ..core.config import settings
from common.logger import LoggerFactory, LoggerType, LogLevel

# Create logger for dependencies
logger = LoggerFactory.get_logger(
    name="dependencies", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


class Container(containers.DeclarativeContainer):
    """Dependency injection container for sentiment analysis service."""

    # Configuration
    config = providers.Configuration()

    # Logger
    logger = providers.Singleton(
        LoggerFactory.get_logger,
        name="sentiment-service",
        logger_type=LoggerType.STANDARD,
        level=LogLevel.INFO,
    )

    # Sentiment Analyzer
    sentiment_analyzer = providers.Singleton(
        OpenAISentimentAnalyzer,
        api_key=config.openai_api_key.as_(str),
        model=config.openai_model.as_(str),
        temperature=config.openai_temperature.as_(float),
        max_tokens=config.openai_max_tokens.as_(int),
        max_retries=config.analysis_retry_attempts.as_(int),
    )

    # Message Broker
    message_broker = providers.Singleton(
        RabbitMQBroker,
        connection_url=config.rabbitmq_url.as_(str),
    )

    # News Repository
    news_repository = providers.Singleton(
        MongoNewsRepository,
        mongo_url=config.mongodb_url.as_(str),
        database_name=config.mongodb_database.as_(str),
    )

    # Sentiment Service
    sentiment_service = providers.Singleton(
        SentimentService,
        analyzer=sentiment_analyzer,
        news_repository=news_repository,
        message_broker=message_broker,
    )

    # News Consumer Service
    news_consumer_service = providers.Singleton(
        NewsConsumerService,
        message_broker=message_broker,
        sentiment_service=sentiment_service,
    )


# Global container instance
container = Container()

# Configure container with settings
container.config.openai_api_key.from_value(settings.openai_api_key)
container.config.openai_model.from_value(settings.openai_model)
container.config.openai_temperature.from_value(settings.openai_temperature)
container.config.openai_max_tokens.from_value(settings.openai_max_tokens)
container.config.analysis_retry_attempts.from_value(settings.analysis_retry_attempts)
container.config.rabbitmq_url.from_value(settings.rabbitmq_url)
container.config.mongodb_url.from_value(settings.mongodb_url)
container.config.mongodb_database.from_value(settings.mongodb_database)


async def initialize_services() -> None:
    """Initialize all async services with proper error handling."""
    logger.info("Initializing sentiment analysis services...")

    try:
        # Initialize news repository
        news_repo = container.news_repository()
        await news_repo.initialize()
        logger.info("✅ News repository initialized successfully")

        logger.info("✅ All sentiment analysis services initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize sentiment analysis services: {e}")
        # Don't raise - allow service to start without message broker
        logger.warning(
            "Service will continue without full message broker functionality"
        )


async def cleanup_services() -> None:
    """Cleanup all async services."""
    logger.info("Cleaning up sentiment analysis services...")

    try:
        # Close repository connections
        news_repo = container.news_repository()
        await news_repo.close()
        logger.info("✅ Repository connections closed")

        logger.info("✅ All sentiment analysis services cleaned up successfully")

    except Exception as e:
        logger.error(f"❌ Error during sentiment analysis service cleanup: {e}")


# Dependency provider functions
def get_sentiment_analyzer() -> OpenAISentimentAnalyzer:
    """Get sentiment analyzer instance."""
    return container.sentiment_analyzer()


def get_news_repository() -> MongoNewsRepository:
    """Get news repository instance."""
    return container.news_repository()


def get_message_broker() -> RabbitMQBroker:
    """Get message broker instance."""
    return container.message_broker()


def get_sentiment_service() -> SentimentService:
    """Get sentiment service instance."""
    return container.sentiment_service()


def get_news_consumer_service() -> NewsConsumerService:
    """Get news consumer service instance."""
    return container.news_consumer_service()
