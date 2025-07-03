"""
News Collector Service Usage Examples

This module demonstrates how to use the NewsCollectorService for collecting
and storing crypto news from various sources.
"""

import asyncio

from .services.news_collector_service import CollectionRequest
from .schemas.news_schemas import NewsSource
from .common.logger import LoggerFactory, LoggerType, LogLevel
from .utils.dependencies import ServiceContext
from .repositories.mongo_news_repository import MongoNewsRepository
from .services.news_collector_service import NewsCollectorService


async def demo_basic_collection():
    """Demonstrate basic news collection from a single source"""
    print("\n=== Basic Collection Demo ===")

    async with ServiceContext() as ctx:
        try:
            service = ctx.get_news_collector_service()

            # Create collection request
            request = CollectionRequest(source=NewsSource.COINTELEGRAPH, max_items=10)

            # Collect and store news
            result = await service.collect_and_store_from_source(request)

            print(f"Collection Result:")
            print(f"  Source: {result.source}")
            print(f"  Items Collected: {result.items_collected}")
            print(f"  Items Stored: {result.items_stored}")
            print(f"  Duplicates: {result.items_duplicated}")
            print(f"  Failed: {result.items_failed}")

            if result.collection_error:
                print(f"  Collection Error: {result.collection_error}")

            if result.storage_errors:
                print(f"  Storage Errors: {result.storage_errors}")

        except Exception as e:
            print(f"Error in basic collection: {e}")


async def demo_multi_source_collection():
    """Demonstrate collecting from multiple sources"""
    print("\n=== Multi-Source Collection Demo ===")

    async with ServiceContext() as ctx:
        try:
            service = ctx.get_news_collector_service()

            from .services.news_collector_service import MultiSourceCollectionRequest

            # Collect from multiple sources
            request = MultiSourceCollectionRequest(
                sources=[NewsSource.COINDESK, NewsSource.COINTELEGRAPH],
                max_items_per_source=15,
            )

            result = await service.collect_and_store_from_multiple_sources(request)

            print(f"Multi-Source Collection Result:")
            print(f"  Total Sources: {len(result['sources'])}")
            print(f"  Total Collected: {result['total_items_collected']}")
            print(f"  Total Stored: {result['total_items_stored']}")
            print(f"  Total Duplicates: {result['total_items_duplicated']}")

            print(f"\nPer-Source Results:")
            for source, stats in result["source_results"].items():
                print(f"  {source}:")
                print(f"    Collected: {stats['items_collected']}")
                print(f"    Success: {stats['collection_success']}")

        except Exception as e:
            print(f"Error in multi-source collection: {e}")


async def demo_news_search_and_retrieval():
    """Demonstrate searching and retrieving stored news"""
    print("\n=== News Search and Retrieval Demo ===")

    async with ServiceContext() as ctx:
        try:
            service = ctx.get_news_collector_service()

            # First, collect some news
            print("Collecting some news first...")
            await service.collect_and_store_all_supported(max_items_per_source=10)

            # Get recent news
            print("\n--- Recent News (Last 24 hours) ---")
            recent_news = await service.get_recent_news(hours=24, limit=5)

            for i, item in enumerate(recent_news, 1):
                print(f"{i}. {item.title} ({item.source.value})")
                print(f"   Published: {item.published_at}")
                print(f"   URL: {item.url}")
                print()

            # Search for specific keywords
            print("--- Search Results for 'Bitcoin' ---")
            bitcoin_news = await service.search_stored_news(
                keywords=["Bitcoin"], limit=3
            )

            for i, item in enumerate(bitcoin_news, 1):
                print(f"{i}. {item.title}")
                print(f"   Source: {item.source.value}")
                print(f"   Tags: {', '.join(item.tags[:3])}")
                print()

        except Exception as e:
            print(f"Error in search demo: {e}")


async def demo_repository_statistics():
    """Demonstrate repository statistics"""
    print("\n=== Repository Statistics Demo ===")

    async with ServiceContext() as ctx:
        try:
            service = ctx.get_news_collector_service()

            # Get repository stats
            stats = await service.get_repository_stats()

            print("Repository Statistics:")
            print(f"  Database: {stats.get('database_name')}")
            print(f"  Total Articles: {stats.get('total_articles', 0)}")
            print(f"  Recent Articles (24h): {stats.get('recent_articles_24h', 0)}")

            if stats.get("oldest_article"):
                print(f"  Oldest Article: {stats['oldest_article']}")
            if stats.get("newest_article"):
                print(f"  Newest Article: {stats['newest_article']}")

            print(f"\nArticles by Source:")
            articles_by_source = stats.get("articles_by_source", {})
            for source, count in articles_by_source.items():
                print(f"  {source}: {count}")

        except Exception as e:
            print(f"Error getting stats: {e}")


async def main():
    """Run all demo functions"""
    print("News Collector Service Usage Examples")
    print("=" * 50)

    # Set up logging
    logger = LoggerFactory.get_logger(
        name="news-collector-demo",
        logger_type=LoggerType.STANDARD,
        level=LogLevel.INFO,
    )

    logger.info("Starting news collector service demos")

    demos = [
        demo_basic_collection,
        demo_multi_source_collection,
        demo_news_search_and_retrieval,
        demo_repository_statistics,
    ]

    for demo in demos:
        try:
            await demo()
            await asyncio.sleep(1)  # Brief pause between demos
        except Exception as e:
            print(f"Demo {demo.__name__} failed: {e}")
            logger.error(f"Demo {demo.__name__} failed: {e}")

    print("\n" + "=" * 50)
    print("All demos completed!")
    logger.info("All news collector service demos completed")


async def demo_custom_collection_config():
    """Demonstrate custom collection configuration"""
    print("\n=== Custom Collection Configuration Demo ===")

    repository = MongoNewsRepository(
        mongo_url="mongodb://localhost:27017", database_name="finsight_news_demo"
    )

    await repository.initialize()
    service = NewsCollectorService(repository)

    try:
        # Custom configuration for specific sources
        config_overrides = {
            NewsSource.COINDESK: {
                "timeout": 60,
                "max_items": 25,
                "retry_attempts": 5,
            },
            NewsSource.COINTELEGRAPH: {
                "timeout": 45,
                "max_items": 20,
            },
        }

        result = await service.collect_and_store_from_multiple_sources(
            sources=[NewsSource.COINDESK, NewsSource.COINTELEGRAPH],
            config_overrides=config_overrides,
        )

        print("Custom Configuration Collection Result:")
        print(f"  Total Collected: {result['total_items_collected']}")
        print(f"  Total Stored: {result['total_items_stored']}")

        for source, stats in result["source_results"].items():
            print(f"  {source}: {stats['items_collected']} items")

    except Exception as e:
        print(f"Error in custom config demo: {e}")

    finally:
        await service.close()
        await repository.close()


async def main():
    """Run all demo functions"""
    print("News Collector Service Usage Examples")
    print("=" * 50)

    # Set up logging
    logger = LoggerFactory.get_logger(
        name="news-collector-demo",
        logger_type=LoggerType.STANDARD,
        level=LogLevel.INFO,
    )

    logger.info("Starting news collector service demos")

    demos = [
        demo_basic_collection,
        # demo_multi_source_collection,
        # demo_all_sources_collection,
        # demo_news_search_and_retrieval,
        # demo_repository_statistics,
        # demo_custom_collection_config,
    ]

    for demo in demos:
        try:
            await demo()
            await asyncio.sleep(1)  # Brief pause between demos
        except Exception as e:
            print(f"Demo {demo.__name__} failed: {e}")
            logger.error(f"Demo {demo.__name__} failed: {e}")

    print("\n" + "=" * 50)
    print("All demos completed!")
    logger.info("All news collector service demos completed")


if __name__ == "__main__":
    # Run the demos
    asyncio.run(main())
