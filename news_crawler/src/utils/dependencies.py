# utils/dependencies.py

"""
Dependency injection utilities for news crawler service.
"""

from typing import Dict, Any, Optional
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from ..repositories.mongo_news_repository import MongoNewsRepository
from ..services.news_service import NewsService
from ..services.search_service import SearchService
from ..services.news_collector_service import NewsCollectorService
from ..adapters.rabbitmq_broker import RabbitMQBroker
from ..adapters.tavily_search_engine import TavilySearchEngine
from ..common.logger import LoggerFactory, LoggerType, LogLevel
from ..core.config import settings


class Container(containers.DeclarativeContainer):
    """Dependency injection container using dependency-injector"""

    # Configuration
    config = providers.Configuration()

    # Logger
    logger = providers.Singleton(
        LoggerFactory.get_logger,
        name="dependency-container",
        logger_type=LoggerType.STANDARD,
        level=LogLevel.INFO,
    )

    # MongoDB Repository
    mongo_repository = providers.Singleton(
        MongoNewsRepository,
        mongo_url=config.mongo_url.as_(str),
        database_name=config.database_name.as_(str),
    )

    # News Service
    news_service = providers.Singleton(
        NewsService,
        repository=mongo_repository,
    )

    # News Collector Service
    news_collector_service = providers.Singleton(
        NewsCollectorService,
        news_service=news_service,
        use_cache=config.use_cache.as_(bool),
    )

    # Message Broker (RabbitMQ)
    message_broker = providers.Singleton(
        RabbitMQBroker,
        connection_url=config.rabbitmq_url.as_(str),
    )

    # Search Engine (Tavily)
    search_engine = providers.Singleton(
        TavilySearchEngine,
        api_key=config.tavily_api_key.as_(str),
    )

    # Search Service
    search_service = providers.Singleton(
        SearchService,
        search_engine=search_engine,
        message_broker=message_broker,
    )


# Global container instance
container = Container()

# Configure default values from settings
container.config.mongo_url.from_value(settings.mongodb_url)
container.config.database_name.from_value(settings.mongodb_database)
container.config.use_cache.from_value(settings.enable_caching)
container.config.rabbitmq_url.from_value(settings.rabbitmq_url)
container.config.tavily_api_key.from_value(settings.tavily_api_key or "")


async def initialize_services() -> None:
    """Initialize all async services with improved error handling"""
    logger = container.logger()
    logger.info("Initializing services...")

    try:
        # Initialize repository first
        repository = container.mongo_repository()
        await repository.initialize()
        logger.info("MongoDB repository initialized successfully")

        # Initialize message broker with timeout and error handling
        try:
            message_broker = container.message_broker()

            # Add timeout to prevent hanging
            import asyncio

            await asyncio.wait_for(message_broker.connect(), timeout=10.0)
            logger.info("Message broker initialized successfully")

        except asyncio.TimeoutError:
            logger.warning(
                "Message broker connection timed out - service will continue without full messaging"
            )
        except Exception as broker_error:
            logger.warning(
                f"Message broker initialization failed: {broker_error} - service will continue in degraded mode"
            )

        logger.info("All services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


async def cleanup_services() -> None:
    """Cleanup all services and resources"""
    logger = container.logger()
    logger.info("Cleaning up services...")

    try:
        # Check if services are instantiated and clean them up
        if hasattr(container.mongo_repository, "_provided_value"):
            repository = container.mongo_repository()
            await repository.close()
            logger.info("MongoDB repository closed")

        if hasattr(container.message_broker, "_provided_value"):
            message_broker = container.message_broker()
            await message_broker.disconnect()
            logger.info("Message broker disconnected")

        logger.info("Services cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


# Context manager for proper lifecycle management
class ServiceContext:
    """Context manager for proper service lifecycle"""

    def __init__(
        self,
        mongo_url: str = "mongodb://localhost:27017",
        database_name: str = "finsight_news",
        use_cache: bool = True,
        rabbitmq_url: str = "amqp://localhost",
        tavily_api_key: str = "",
    ):
        """
        Initialize service context with configuration

        Args:
            mongo_url: MongoDB connection URL
            database_name: Database name for news storage
            use_cache: Whether to enable caching
            rabbitmq_url: RabbitMQ connection URL
            tavily_api_key: Tavily API key for search
        """
        container.config.mongo_url.from_value(mongo_url)
        container.config.database_name.from_value(database_name)
        container.config.use_cache.from_value(use_cache)
        container.config.rabbitmq_url.from_value(rabbitmq_url)
        container.config.tavily_api_key.from_value(tavily_api_key)

    async def __aenter__(self):
        """Async context manager entry"""
        await initialize_services()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await cleanup_services()

    def get_news_collector_service(self) -> NewsCollectorService:
        """Get news collector service"""
        return container.news_collector_service()

    def get_news_service(self) -> NewsService:
        """Get news service"""
        return container.news_service()

    def get_repository(self) -> MongoNewsRepository:
        """Get repository"""
        return container.mongo_repository()

    def get_search_service(self) -> SearchService:
        """Get search service"""
        return container.search_service()

    def get_message_broker(self) -> RabbitMQBroker:
        """Get message broker"""
        return container.message_broker()


# Dependency injection functions for FastAPI
def get_news_collector_service() -> NewsCollectorService:
    """
    Get news collector service (for FastAPI dependency injection)

    Returns:
        NewsCollectorService: Configured news collector service
    """
    return container.news_collector_service()


def get_news_service() -> NewsService:
    """
    Get news service (for FastAPI dependency injection)

    Returns:
        NewsService: Configured news service
    """
    return container.news_service()


def get_repository() -> MongoNewsRepository:
    """
    Get repository (for FastAPI dependency injection)

    Returns:
        MongoNewsRepository: Configured MongoDB repository
    """
    return container.mongo_repository()


def get_search_service() -> SearchService:
    """
    Get search service (for FastAPI dependency injection)

    Returns:
        SearchService: Configured search service
    """
    return container.search_service()


def get_message_broker() -> RabbitMQBroker:
    """
    Get message broker (for FastAPI dependency injection)

    Returns:
        RabbitMQBroker: Configured RabbitMQ message broker
    """
    return container.message_broker()


def get_search_engine() -> TavilySearchEngine:
    """
    Get search engine (for FastAPI dependency injection)

    Returns:
        TavilySearchEngine: Configured Tavily search engine
    """
    return container.search_engine()


# Convenience functions with async support
@inject
async def get_news_collector_service_async(
    service: NewsCollectorService = Provide[Container.news_collector_service],
) -> NewsCollectorService:
    """
    Get news collector service (async version for injection)

    Args:
        service: Injected news collector service

    Returns:
        NewsCollectorService: Configured news collector service
    """
    return service


@inject
async def get_news_service_async(
    service: NewsService = Provide[Container.news_service],
) -> NewsService:
    """
    Get news service (async version for injection)

    Args:
        service: Injected news service

    Returns:
        NewsService: Configured news service
    """
    return service


@inject
async def get_repository_async(
    repository: MongoNewsRepository = Provide[Container.mongo_repository],
) -> MongoNewsRepository:
    """
    Get repository (async version for injection)

    Args:
        repository: Injected MongoDB repository

    Returns:
        MongoNewsRepository: Configured MongoDB repository
    """
    return repository


@inject
async def get_search_service_async(
    service: SearchService = Provide[Container.search_service],
) -> SearchService:
    """
    Get search service (async version for injection)

    Args:
        service: Injected search service

    Returns:
        SearchService: Configured search service
    """
    return service


def configure_container(config_overrides: Optional[Dict[str, Any]] = None) -> None:
    """
    Configure container with custom settings

    Args:
        config_overrides: Dictionary of configuration overrides
    """
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(container.config, key):
                getattr(container.config, key).from_value(value)


def reset_container() -> None:
    """Reset container state (useful for testing)"""
    container.reset_singletons()


def get_container_info() -> Dict[str, Any]:
    """
    Get information about the current container state

    Returns:
        Dict[str, Any]: Container state information
    """
    logger = container.logger()

    info = {
        "container_type": type(container).__name__,
        "services_initialized": {},
        "config": {
            "mongo_url": container.config.mongo_url(),
            "database_name": container.config.database_name(),
            "use_cache": container.config.use_cache(),
            "rabbitmq_url": container.config.rabbitmq_url(),
            "tavily_api_key_configured": bool(container.config.tavily_api_key()),
        },
    }

    # Check which services are initialized
    services = [
        "mongo_repository",
        "news_service",
        "news_collector_service",
        "message_broker",
        "search_engine",
        "search_service",
    ]

    for service_name in services:
        try:
            service_provider = getattr(container, service_name)
            info["services_initialized"][service_name] = hasattr(
                service_provider, "_provided_value"
            )
        except Exception as e:
            logger.warning(f"Could not check service {service_name}: {e}")
            info["services_initialized"][service_name] = False

    return info
