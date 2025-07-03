# utils/dependencies.py

"""
Dependency injection utilities for news crawler service.
"""

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from ..repositories.mongo_news_repository import MongoNewsRepository
from ..services.news_service import NewsService
from ..services.news_collector_service import NewsCollectorService
from ..common.logger import LoggerFactory, LoggerType, LogLevel


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


# Global container instance
container = Container()

# Configure default values
container.config.mongo_url.from_value("mongodb://localhost:27017")
container.config.database_name.from_value("finsight_news")
container.config.use_cache.from_value(True)


async def initialize_services() -> None:
    """Initialize all async services"""
    logger = container.logger()
    logger.info("Initializing services...")

    # Initialize repository
    repository = container.mongo_repository()
    await repository.initialize()

    logger.info("All services initialized successfully")


async def cleanup_services() -> None:
    """Cleanup all services and resources"""
    logger = container.logger()
    logger.info("Cleaning up services...")

    try:
        # # Get services if they exist
        # if container.mongo_repository.provided._is_singleton_provided():
        #     repository = container.mongo_repository()
        #     await repository.close()

        # if container.news_collector_service.provided._is_singleton_provided():
        #     collector_service = container.news_collector_service()
        #     await collector_service.close()

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
    ):
        container.config.mongo_url.from_value(mongo_url)
        container.config.database_name.from_value(database_name)
        container.config.use_cache.from_value(use_cache)

    async def __aenter__(self):
        await initialize_services()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
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


# Convenience functions
@inject
async def get_news_collector_service(
    service: NewsCollectorService = Provide[Container.news_collector_service],
) -> NewsCollectorService:
    """Get news collector service (for injection)"""
    return service


@inject
async def get_news_service(
    service: NewsService = Provide[Container.news_service],
) -> NewsService:
    """Get news service (for injection)"""
    return service


@inject
async def get_repository(
    repository: MongoNewsRepository = Provide[Container.mongo_repository],
) -> MongoNewsRepository:
    """Get repository (for injection)"""
    return repository
