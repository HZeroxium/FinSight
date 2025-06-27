# utils/dependencies.py

"""
Dependency injection utilities for news crawler service.
"""

from functools import lru_cache

from ..adapters.tavily_search_engine import TavilySearchEngine
from ..adapters.rabbitmq_broker import RabbitMQBroker
from ..repositories.article_repository import ArticleRepository
from ..services.search_service import SearchService
from ..services.crawler_service import CrawlerService
from ..core.config import settings
from ..common.logger import LoggerFactory, LoggerType, LogLevel

# Create logger for dependencies
logger = LoggerFactory.get_logger(
    name="dependencies", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


@lru_cache()
def get_search_engine() -> TavilySearchEngine:
    """
    Get singleton search engine instance.

    Returns:
        TavilySearchEngine: Configured search engine
    """
    logger.info("Creating Tavily search engine instance")
    return TavilySearchEngine(api_key=settings.tavily_api_key)


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
def get_article_repository() -> ArticleRepository:
    """
    Get singleton article repository instance.

    Returns:
        ArticleRepository: Configured article repository
    """
    logger.info("Creating article repository instance")
    return ArticleRepository(
        mongo_url=settings.mongodb_url,
        database_name=settings.mongodb_database,
    )


@lru_cache()
def get_crawler_service() -> CrawlerService:
    """
    Get singleton crawler service instance.

    Returns:
        CrawlerService: Configured crawler service
    """
    logger.info("Creating crawler service instance")
    return CrawlerService()


@lru_cache()
def get_search_service() -> SearchService:
    """
    Get singleton search service instance.

    Returns:
        SearchService: Configured search service
    """
    logger.info("Creating search service instance")
    return SearchService(
        search_engine=get_search_engine(),
        message_broker=get_message_broker(),
        article_repository=get_article_repository(),
        crawler_service=get_crawler_service(),
    )
    message_broker = get_message_broker()
    article_repository = get_article_repository()
    crawler_service = get_crawler_service()

    return SearchService(
        search_engine=search_engine,
        message_broker=message_broker,
        article_repository=article_repository,
        crawler_service=crawler_service,
    )
