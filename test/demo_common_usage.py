#!/usr/bin/env python3
"""
Demo script showing how to use the common module from any service

This script demonstrates how to use the FinSight common module
from any location in the project after it has been installed.
"""


def demo_logger_usage():
    """Demonstrate logger usage"""
    print("=== Logger Demo ===")

    from common.logger import LoggerFactory, LoggerType, LogLevel

    # Create different types of loggers
    console_logger = LoggerFactory.create_logger(
        name="demo_service", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
    )

    print_logger = LoggerFactory.create_logger(
        name="demo_print", logger_type=LoggerType.PRINT, level=LogLevel.DEBUG
    )

    # Test logging
    console_logger.info("This is an info message from console logger")
    console_logger.warning("This is a warning message")
    console_logger.error("This is an error message")

    print_logger.info("This is from print logger")
    print_logger.debug("This is a debug message")

    print("✅ Logger demo completed\n")


def demo_cache_usage():
    """Demonstrate cache usage"""
    print("=== Cache Demo ===")

    from common.cache import CacheFactory, CacheType

    # Create a cache instance
    cache = CacheFactory.create_cache(
        name="demo_cache", cache_type=CacheType.MEMORY, max_size=1000
    )

    # Use the cache
    cache.set("user:123", {"name": "John Doe", "email": "john@example.com"}, ttl=300)
    cache.set("config:app", {"debug": True, "version": "1.0.0"})

    # Retrieve values
    user_data = cache.get("user:123")
    config_data = cache.get("config:app")
    non_existent = cache.get("non_existent", "default_value")

    print(f"User data: {user_data}")
    print(f"Config data: {config_data}")
    print(f"Non-existent key: {non_existent}")

    # Check existence
    print(f"User exists: {cache.exists('user:123')}")
    print(f"Unknown key exists: {cache.exists('unknown')}")

    # Get cache stats
    stats = cache.get_stats()
    print(
        f"Cache stats - Hits: {stats.hits}, Misses: {stats.misses}, Hit rate: {stats.get_hit_rate():.2%}"
    )

    print("✅ Cache demo completed\n")


def demo_llm_usage():
    """Demonstrate LLM usage (basic import test)"""
    print("=== LLM Demo ===")

    try:
        from common.llm import LLMFacade

        print("✅ LLM module imported successfully")
        print("Note: LLM functionality requires API keys to be configured")
    except ImportError as e:
        print(f"❌ LLM import failed: {e}")

    print("✅ LLM demo completed\n")


def demo_cross_service_usage():
    """Demonstrate how the common module can be used across services"""
    print("=== Cross-Service Usage Demo ===")

    # This simulates how different services can use the common module
    from common.logger import LoggerFactory, LoggerType
    from common.cache import CacheFactory, CacheType

    # Service A: News Crawler
    news_logger = LoggerFactory.create_logger("news_crawler", LoggerType.STANDARD)
    news_cache = CacheFactory.create_cache("news_cache", CacheType.MEMORY)

    news_logger.info("News crawler service started")
    news_cache.set("latest_articles", ["article1", "article2", "article3"])

    # Service B: Sentiment Analysis
    sentiment_logger = LoggerFactory.create_logger(
        "sentiment_analysis", LoggerType.STANDARD
    )
    sentiment_cache = CacheFactory.create_cache("sentiment_cache", CacheType.MEMORY)

    sentiment_logger.info("Sentiment analysis service started")
    # Get articles from news cache
    articles = news_cache.get("latest_articles", [])
    sentiment_cache.set("processed_articles", f"Processed {len(articles)} articles")

    # Service C: Model Builder
    model_logger = LoggerFactory.create_logger("model_builder", LoggerType.STANDARD)

    model_logger.info("Model builder service started")
    processed_info = sentiment_cache.get("processed_articles")
    model_logger.info(f"Found sentiment data: {processed_info}")

    print("✅ Cross-service demo completed")
    print("All services can use the same common components!")
    print()


def main():
    """Run all demos"""
    print("FinSight Common Module Usage Demo")
    print("=" * 50)
    print()

    try:
        demo_logger_usage()
        demo_cache_usage()
        demo_llm_usage()
        demo_cross_service_usage()

        print("🎉 All demos completed successfully!")
        print()
        print("How to use in your services:")
        print("1. From any service directory (e.g., ai_prediction/src/), import:")
        print("   from common.logger import LoggerFactory")
        print("   from common.cache import CacheFactory")
        print("   from common.llm import LLMFacade")
        print()
        print("2. No need to copy common files or worry about paths!")
        print("3. The common module is installed as a package in your environment.")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
