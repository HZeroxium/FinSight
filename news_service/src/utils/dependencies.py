# utils/dependencies.py

"""
Dependency injection utilities for news crawler service.
"""

from typing import Dict, Any, Optional
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from ..repositories.mongo_news_repository import MongoNewsRepository
from ..services.news_service import NewsService
from ..services.search_service import SearchService
from ..services.news_collector_service import NewsCollectorService
from ..services.job_management_service import JobManagementService
from ..services.eureka_client_service import EurekaClientService
from ..adapters.rabbitmq_broker import RabbitMQBroker
from ..adapters.tavily_search_engine import TavilySearchEngine
from ..utils.cache_utils import get_cache_manager, CacheManager
from common.logger import LoggerFactory, LoggerType, LogLevel
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

    # Cache Manager
    cache_manager = providers.Singleton(
        get_cache_manager,
    )

    # Eureka Client Service
    eureka_client_service = providers.Singleton(
        EurekaClientService,
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

    # Job Management Service
    job_service = providers.Singleton(
        JobManagementService,
        mongo_url=config.mongo_url.as_(str),
        database_name=config.database_name.as_(str),
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
        # Initialize cache manager first
        logger.info("🚀 Initializing cache manager...")
        cache_manager = await get_cache_manager()
        logger.info("✅ Cache manager initialized successfully")

        # Initialize Eureka client service
        eureka_service = container.eureka_client_service()
        if settings.enable_eureka_client:
            logger.info("🚀 Initializing Eureka client service...")
            success = await eureka_service.start()
            if success:
                logger.info("✅ Eureka client service initialized successfully")
            else:
                logger.warning("⚠️ Eureka client service initialization failed")
        else:
            logger.info("🔄 Eureka client service is disabled")

        # Initialize other services...
        logger.info("✅ All services initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise


async def cleanup_services() -> None:
    """Cleanup all async services"""
    logger = container.logger()
    logger.info("Cleaning up services...")

    try:
        # Cleanup Eureka client service
        eureka_service = container.eureka_client_service()
        if settings.enable_eureka_client and eureka_service.is_registered():
            logger.info("🛑 Stopping Eureka client service...")
            await eureka_service.stop()
            logger.info("✅ Eureka client service stopped")

        # Cleanup other services...
        logger.info("✅ All services cleaned up successfully")

    except Exception as e:
        logger.error(f"❌ Error during service cleanup: {e}")


class ServiceContext:
    """Context manager for service lifecycle management"""

    def __init__(
        self,
        mongo_url: str = "mongodb://localhost:27017",
        database_name: str = "finsight_news",
        use_cache: bool = True,
        rabbitmq_url: str = "amqp://localhost",
        tavily_api_key: str = "",
    ):
        self.mongo_url = mongo_url
        self.database_name = database_name
        self.use_cache = use_cache
        self.rabbitmq_url = rabbitmq_url
        self.tavily_api_key = tavily_api_key

    async def __aenter__(self):
        configure_container(
            {
                "mongo_url": self.mongo_url,
                "database_name": self.database_name,
                "use_cache": self.use_cache,
                "rabbitmq_url": self.rabbitmq_url,
                "tavily_api_key": self.tavily_api_key,
            }
        )
        await initialize_services()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await cleanup_services()

    def get_news_collector_service(self) -> NewsCollectorService:
        return container.news_collector_service()

    def get_news_service(self) -> NewsService:
        return container.news_service()

    def get_repository(self) -> MongoNewsRepository:
        return container.mongo_repository()

    def get_search_service(self) -> SearchService:
        return container.search_service()

    def get_message_broker(self) -> RabbitMQBroker:
        return container.message_broker()

    def get_job_service(self) -> JobManagementService:
        return container.job_service()

    def get_eureka_client_service(self) -> EurekaClientService:
        return container.eureka_client_service()

    async def get_cache_manager(self) -> CacheManager:
        return await get_cache_manager()


def get_news_collector_service() -> NewsCollectorService:
    """Get news collector service instance"""
    try:
        return container.news_collector_service()
    except Exception as e:
        logger = container.logger()
        logger.error(f"Failed to get news collector service: {e}")
        raise


def get_news_service() -> NewsService:
    """Get news service instance"""
    try:
        return container.news_service()
    except Exception as e:
        logger = container.logger()
        logger.error(f"Failed to get news service: {e}")
        raise


def get_repository() -> MongoNewsRepository:
    """Get MongoDB repository instance"""
    try:
        return container.mongo_repository()
    except Exception as e:
        logger = container.logger()
        logger.error(f"Failed to get repository: {e}")
        raise


def get_search_service() -> SearchService:
    """Get search service instance"""
    try:
        return container.search_service()
    except Exception as e:
        logger = container.logger()
        logger.error(f"Failed to get search service: {e}")
        raise


def get_message_broker() -> RabbitMQBroker:
    """Get message broker instance"""
    try:
        return container.message_broker()
    except Exception as e:
        logger = container.logger()
        logger.error(f"Failed to get message broker: {e}")
        raise


def get_search_engine() -> TavilySearchEngine:
    """Get search engine instance"""
    try:
        return container.search_engine()
    except Exception as e:
        logger = container.logger()
        logger.error(f"Failed to get search engine: {e}")
        raise


def get_job_service() -> JobManagementService:
    """Get job management service instance"""
    try:
        return container.job_service()
    except Exception as e:
        logger = container.logger()
        logger.error(f"Failed to get job service: {e}")
        raise


def get_eureka_client_service() -> EurekaClientService:
    """Get Eureka client service instance"""
    try:
        return container.eureka_client_service()
    except Exception as e:
        logger = container.logger()
        logger.error(f"Failed to get Eureka client service: {e}")
        raise


async def get_cache_manager_dependency() -> CacheManager:
    """Get cache manager as FastAPI dependency"""
    try:
        return await get_cache_manager()
    except Exception as e:
        logger = container.logger()
        logger.error(f"Failed to get cache manager: {e}")
        raise


# Security
security = HTTPBearer(auto_error=False)


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> bool:
    """
    Verify API key for admin endpoints.

    Args:
        credentials: HTTP authorization credentials

    Returns:
        bool: True if API key is valid, False otherwise

    Raises:
        HTTPException: If API key is invalid or missing
    """
    if not settings.secret_api_key:
        # If no API key is configured, allow access (for development)
        return True

    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials != settings.secret_api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


def require_admin_access(api_key_verified: bool = Depends(verify_api_key)) -> bool:
    """
    Require admin access for protected endpoints.

    Args:
        api_key_verified: Whether API key is verified

    Returns:
        bool: True if admin access is granted
    """
    return api_key_verified


# Async dependency injection functions
@inject
async def get_news_collector_service_async(
    service: NewsCollectorService = Provide[Container.news_collector_service],
) -> NewsCollectorService:
    """Get news collector service as async dependency"""
    return service


@inject
async def get_news_service_async(
    service: NewsService = Provide[Container.news_service],
) -> NewsService:
    """Get news service as async dependency"""
    return service


@inject
async def get_repository_async(
    repository: MongoNewsRepository = Provide[Container.mongo_repository],
) -> MongoNewsRepository:
    """Get repository as async dependency"""
    return repository


@inject
async def get_search_service_async(
    service: SearchService = Provide[Container.search_service],
) -> SearchService:
    """Get search service as async dependency"""
    return service


@inject
async def get_eureka_client_service_async(
    service: EurekaClientService = Provide[Container.eureka_client_service],
) -> EurekaClientService:
    """Get Eureka client service as async dependency"""
    return service


# Configuration functions
def configure_container(config_overrides: Optional[Dict[str, Any]] = None) -> None:
    """Configure container with optional overrides"""
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(container.config, key):
                getattr(container.config, key).from_value(value)


def reset_container() -> None:
    """Reset container to default configuration"""
    container.reset_singletons()


def get_container_info() -> Dict[str, Any]:
    """Get container configuration information"""
    return {
        "mongo_url": settings.mongodb_url,
        "database_name": settings.mongodb_database,
        "use_cache": settings.enable_caching,
        "rabbitmq_url": settings.rabbitmq_url,
        "eureka_enabled": settings.enable_eureka_client,
        "eureka_server_url": settings.eureka_server_url,
        "eureka_app_name": settings.eureka_app_name,
        "cache_config": settings.cache_config,
    }
