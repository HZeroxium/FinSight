# news_collector_service_usage.py

"""
News Collector Service Usage Examples

This module demonstrates how to use the NewsCollectorService for collecting
and storing crypto news from various sources.
"""

import asyncio
from datetime import datetime, timezone, timedelta

from .services.news_collector_service import (
    NewsCollectorService,
    CollectionRequest,
    BatchCollectionRequest,
)
from .services.news_service import NewsService
from .repositories.mongo_news_repository import MongoNewsRepository
from .schemas.news_schemas import NewsSource
from .core.news_collector_factory import CollectorType
from .common.logger import LoggerFactory, LoggerType, LogLevel


# Initialize logging
logger = LoggerFactory.get_logger(
    name="news-collector-usage",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    file_level=LogLevel.DEBUG,
    log_file="logs/news_collector_usage.log",
)


async def initialize_service() -> NewsCollectorService:
    """
    Initialize the NewsCollectorService with all dependencies

    Returns:
        NewsCollectorService: Initialized service instance
    """
    logger.info("Initializing NewsCollectorService...")

    # Initialize MongoDB repository
    repository = MongoNewsRepository(
        mongo_url="mongodb://localhost:27017", database_name="finsight_news_demo"
    )
    await repository.initialize()

    # Initialize news service
    news_service = NewsService(repository)

    # Initialize collector service
    collector_service = NewsCollectorService(
        news_service=news_service, use_cache=True, enable_fallback=True
    )

    logger.info("NewsCollectorService initialized successfully")
    return collector_service


async def demo_single_source_collection(service: NewsCollectorService) -> None:
    """
    Demonstrate collecting news from a single source with specific collector type

    Args:
        service: NewsCollectorService instance
    """
    logger.info("=== Demo: Single Source Collection ===")

    # Example 1: Collect from CoinDesk using API collector
    request = CollectionRequest(
        source=NewsSource.COINDESK,
        collector_type=CollectorType.API_REST,
        max_items=20,
        enable_fallback=True,
    )

    result = await service.collect_and_store(request)

    logger.info(f"CoinDesk Collection Result:")
    logger.info(f"  - Success: {result['collection_success']}")
    logger.info(f"  - Items Collected: {result['items_collected']}")
    logger.info(f"  - Items Stored: {result['items_stored']}")
    logger.info(f"  - Duplicates: {result['items_duplicated']}")

    # Example 2: Collect from CoinTelegraph using auto-selection
    request = CollectionRequest(
        source=NewsSource.COINTELEGRAPH,
        collector_type=None,  # Auto-select best collector
        max_items=15,
        enable_fallback=True,
    )

    result = await service.collect_and_store(request)

    logger.info(f"CoinTelegraph Collection Result:")
    logger.info(f"  - Success: {result['collection_success']}")
    logger.info(f"  - Collector Used: {result['collector_type']}")
    logger.info(f"  - Items Collected: {result['items_collected']}")
    logger.info(f"  - Items Stored: {result['items_stored']}")


async def demo_batch_collection(service: NewsCollectorService) -> None:
    """
    Demonstrate batch collection from multiple sources

    Args:
        service: NewsCollectorService instance
    """
    logger.info("=== Demo: Batch Collection ===")

    # Batch collection with specific collector preferences
    request = BatchCollectionRequest(
        sources=[NewsSource.COINDESK, NewsSource.COINTELEGRAPH],
        collector_preferences={
            NewsSource.COINDESK.value: CollectorType.API_REST.value,
            NewsSource.COINTELEGRAPH.value: CollectorType.API_GRAPHQL.value,
        },
        max_items_per_source=25,
        enable_fallback=True,
    )

    result = await service.collect_and_store_batch(request)

    logger.info(f"Batch Collection Result:")
    logger.info(f"  - Sources: {result['sources']}")
    logger.info(f"  - Total Collected: {result['total_items_collected']}")
    logger.info(f"  - Total Stored: {result['total_items_stored']}")
    logger.info(f"  - Total Duplicated: {result['total_items_duplicated']}")

    # Show per-source results
    for source, source_result in result["source_results"].items():
        logger.info(f"  - {source}: {source_result['items_collected']} items")


async def demo_all_sources_collection(service: NewsCollectorService) -> None:
    """
    Demonstrate collecting from all available sources with best adapters

    Args:
        service: NewsCollectorService instance
    """
    logger.info("=== Demo: All Sources Collection ===")

    result = await service.collect_all_with_best_adapters(max_items_per_source=30)

    logger.info(f"All Sources Collection Result:")
    logger.info(f"  - Sources: {result['sources']}")
    logger.info(f"  - Total Collected: {result['total_items_collected']}")
    logger.info(f"  - Total Stored: {result['total_items_stored']}")
    logger.info(f"  - Total Duplicated: {result['total_items_duplicated']}")


async def demo_search_and_filtering(service: NewsCollectorService) -> None:
    """
    Demonstrate searching and filtering stored news

    Args:
        service: NewsCollectorService instance
    """
    logger.info("=== Demo: Search and Filtering ===")

    # Search recent news
    recent_news = await service.get_recent_news(
        source=None, hours=24, limit=10  # All sources
    )
    logger.info(f"Found {len(recent_news)} recent news items")

    # Search by keywords
    bitcoin_news = await service.search_stored_news(
        keywords=["bitcoin", "BTC"], limit=5
    )
    logger.info(f"Found {len(bitcoin_news)} Bitcoin-related articles")

    # Search by date range
    last_week = datetime.now(timezone.utc) - timedelta(days=7)
    recent_coindesk = await service.search_stored_news(
        source=NewsSource.COINDESK, start_date=last_week, limit=10
    )
    logger.info(f"Found {len(recent_coindesk)} CoinDesk articles from last week")

    # Get news by specific source
    cointelegraph_news = await service.get_news_by_source(
        source=NewsSource.COINTELEGRAPH, limit=8
    )
    logger.info(f"Found {len(cointelegraph_news)} CoinTelegraph articles")


async def demo_service_information(service: NewsCollectorService) -> None:
    """
    Demonstrate getting service information and statistics

    Args:
        service: NewsCollectorService instance
    """
    logger.info("=== Demo: Service Information ===")

    # Get available adapters
    adapters = service.get_available_adapters()
    logger.info("Available Adapters:")
    for source, adapter_types in adapters.items():
        logger.info(f"  - {source}: {adapter_types}")

    # Get repository statistics
    stats = await service.get_repository_stats()
    logger.info("Repository Statistics:")
    logger.info(f"  - Total Articles: {stats.get('total_articles', 'N/A')}")
    logger.info(f"  - Articles by Source: {stats.get('articles_by_source', {})}")
    logger.info(f"  - Recent Articles (24h): {stats.get('recent_articles_24h', 'N/A')}")


async def demo_error_handling(service: NewsCollectorService) -> None:
    """
    Demonstrate error handling and fallback mechanisms

    Args:
        service: NewsCollectorService instance
    """
    logger.info("=== Demo: Error Handling ===")

    # Try collection with invalid configuration
    try:
        request = CollectionRequest(
            source=NewsSource.COINDESK,
            collector_type=CollectorType.API_REST,
            max_items=5,
            config_overrides={"invalid_param": "invalid_value"},
            enable_fallback=True,
        )

        result = await service.collect_and_store(request)

        if not result["collection_success"]:
            logger.warning(
                f"Collection failed as expected: {result['collection_error']}"
            )
        else:
            logger.info("Collection succeeded with fallback mechanism")

    except Exception as e:
        logger.error(f"Unexpected error during collection: {e}")


async def demo_performance_monitoring(service: NewsCollectorService) -> None:
    """
    Demonstrate performance monitoring during collection

    Args:
        service: NewsCollectorService instance
    """
    logger.info("=== Demo: Performance Monitoring ===")

    start_time = datetime.now()

    # Perform a batch collection and measure time
    request = BatchCollectionRequest(
        sources=[NewsSource.COINDESK, NewsSource.COINTELEGRAPH],
        max_items_per_source=10,
        enable_fallback=True,
    )

    result = await service.collect_and_store_batch(request)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info(f"Performance Metrics:")
    logger.info(f"  - Duration: {duration:.2f} seconds")
    logger.info(f"  - Items/Second: {result['total_items_collected'] / duration:.2f}")
    logger.info(
        f"  - Success Rate: {(result['total_items_stored'] / max(result['total_items_collected'], 1)) * 100:.1f}%"
    )


async def main():
    """
    Main function demonstrating comprehensive NewsCollectorService usage
    """
    logger.info("Starting NewsCollectorService Usage Demo")

    try:
        # Initialize service
        service = await initialize_service()

        # Run all demos
        await demo_single_source_collection(service)
        # await demo_batch_collection(service)
        # await demo_all_sources_collection(service)
        # await demo_search_and_filtering(service)
        # await demo_service_information(service)
        # await demo_error_handling(service)
        # await demo_performance_monitoring(service)

        logger.info("All demos completed successfully!")

        # Final statistics
        final_stats = await service.get_repository_stats()
        logger.info(f"Final Repository State:")
        logger.info(f"  - Total Articles: {final_stats.get('total_articles', 'N/A')}")
        logger.info(
            f"  - Articles by Source: {final_stats.get('articles_by_source', {})}"
        )

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise
    finally:
        # Cleanup
        if "service" in locals():
            await service.close()
        logger.info("NewsCollectorService usage demo completed")


async def quick_demo():
    """
    Quick demo for testing basic functionality
    """
    logger.info("Starting Quick Demo")

    service = await initialize_service()

    try:
        # Quick single source collection
        request = CollectionRequest(
            source=NewsSource.COINDESK, max_items=5, enable_fallback=True
        )

        result = await service.collect_and_store(request)

        logger.info(f"Quick Demo Result:")
        logger.info(f"  - Success: {result['collection_success']}")
        logger.info(f"  - Items: {result['items_collected']}")
        logger.info(f"  - Stored: {result['items_stored']}")

    finally:
        await service.close()


if __name__ == "__main__":
    # Run the comprehensive demo
    # asyncio.run(main())

    # Uncomment to run quick demo instead
    asyncio.run(quick_demo())
