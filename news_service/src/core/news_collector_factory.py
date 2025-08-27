# core/news_collector_factory.py

from enum import Enum
from typing import Any, Dict, List, Optional

from common.logger import LoggerFactory, LoggerType, LogLevel

from ..adapters.api_coindesk_news_collector import APICoinDeskNewsCollector
from ..adapters.api_cointelegraph_news_collector import \
    APICoinTelegraphNewsCollector
from ..adapters.rss_news_collector import RSSNewsCollector
from ..interfaces.news_collector_interface import NewsCollectorInterface
from ..schemas.news_schemas import NewsCollectorConfig, NewsSource


class CollectorType(Enum):
    """Available collector types"""

    RSS = "rss"
    API_REST = "api_rest"
    API_GRAPHQL = "api_graphql"


class NewsCollectorFactory:
    """Factory for creating news collector instances with support for multiple adapter types"""

    _instances: Dict[str, NewsCollectorInterface] = {}

    # Enhanced configurations for all sources and collector types
    _default_configs = {
        NewsSource.COINDESK: {
            CollectorType.RSS: {
                "url": "https://www.coindesk.com/arc/outboundfeeds/rss",
                "timeout": 30,
                "max_items": 50,
            },
            CollectorType.API_REST: {
                "url": "https://data-api.coindesk.com/news/v1/article/list",
                "timeout": 30,
                "max_items": 100,
                "retry_attempts": 5,
                "retry_delay": 2,
            },
        },
        NewsSource.COINTELEGRAPH: {
            CollectorType.RSS: {
                "url": "https://cointelegraph.com/rss",
                "timeout": 30,
                "max_items": 50,
            },
            CollectorType.API_GRAPHQL: {
                "url": "https://conpletus.cointelegraph.com/v1/",
                "timeout": 45,
                "max_items": 20,
                "retry_attempts": 5,
                "retry_delay": 3,
            },
        },
    }

    @classmethod
    def get_collector(
        cls,
        source: NewsSource,
        collector_type: CollectorType = CollectorType.RSS,
        config_overrides: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> NewsCollectorInterface:
        """
        Get or create a news collector instance with specified type

        Args:
            source: News source to collect from
            collector_type: Type of collector to create
            config_overrides: Configuration overrides
            use_cache: Whether to use cached instances

        Returns:
            NewsCollectorInterface instance
        """
        logger = LoggerFactory.get_logger(
            name="news-collector-factory",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file="logs/news_collector_factory.log",
        )

        cache_key = f"{source.value}_{collector_type.value}"

        if use_cache and cache_key in cls._instances:
            logger.debug(f"Returning cached collector: {cache_key}")
            return cls._instances[cache_key]

        try:
            logger.debug(f"Creating new collector: {cache_key}")

            # Build configuration
            config = cls._build_config(source, collector_type, config_overrides)

            # Create collector instance
            collector = cls._create_collector(collector_type, config)

            if use_cache:
                cls._instances[cache_key] = collector
                logger.info(f"Created and cached collector: {cache_key}")
            else:
                logger.info(f"Created collector (no caching): {cache_key}")

            return collector

        except Exception as e:
            logger.error(f"Failed to create collector {cache_key}: {e}")
            raise

    @classmethod
    def _build_config(
        cls,
        source: NewsSource,
        collector_type: CollectorType,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> NewsCollectorConfig:
        """Build configuration for collector"""

        # Get default config for source and type
        source_configs = cls._default_configs.get(source, {})
        default_config = source_configs.get(collector_type, {})

        if not default_config:
            raise ValueError(
                f"No default configuration for {source.value} with {collector_type.value}"
            )

        # Apply overrides
        if overrides:
            default_config.update(overrides)

        # Ensure source is set
        default_config["source"] = source

        return NewsCollectorConfig(**default_config)

    @classmethod
    def _create_collector(
        cls, collector_type: CollectorType, config: NewsCollectorConfig
    ) -> NewsCollectorInterface:
        """Create collector instance based on type"""

        if collector_type == CollectorType.RSS:
            return RSSNewsCollector(config)
        elif collector_type == CollectorType.API_REST:
            return APICoinDeskNewsCollector(config)
        elif collector_type == CollectorType.API_GRAPHQL:
            return APICoinTelegraphNewsCollector(config)
        else:
            raise ValueError(f"Unknown collector type: {collector_type}")

    @classmethod
    def create_coindesk_collector(
        cls, config_overrides: Optional[Dict[str, Any]] = None
    ) -> NewsCollectorInterface:
        """
        Convenience method to create CoinDesk collector

        Args:
            config_overrides: Configuration overrides

        Returns:
            CoinDesk news collector
        """
        return cls.create_collector(
            source=NewsSource.COINDESK,
            collector_type=CollectorType.RSS,
            config_overrides=config_overrides,
        )

    @classmethod
    def create_cointelegraph_collector(
        cls, config_overrides: Optional[Dict[str, Any]] = None
    ) -> NewsCollectorInterface:
        """
        Convenience method to create CoinTelegraph collector

        Args:
            config_overrides: Configuration overrides

        Returns:
            CoinTelegraph news collector
        """
        return cls.create_collector(
            source=NewsSource.COINTELEGRAPH,
            collector_type=CollectorType.RSS,
            config_overrides=config_overrides,
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached collector instances"""
        logger = LoggerFactory.get_logger(
            name="news-collector-factory",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
        )

        logger.info(f"Clearing {len(cls._instances)} cached collector instances")
        cls._instances.clear()

    @classmethod
    def get_cached_instances(cls) -> Dict[str, str]:
        """Get information about cached instances"""
        return {
            key: type(instance).__name__ for key, instance in cls._instances.items()
        }

    @classmethod
    def get_supported_sources(cls) -> List[NewsSource]:
        """Get list of supported news sources"""
        return list(cls._default_configs.keys())

    @classmethod
    def get_best_collector_for_source(
        cls, source: NewsSource, config_overrides: Optional[Dict[str, Any]] = None
    ) -> NewsCollectorInterface:
        """
        Get the best/preferred collector type for a given source

        Args:
            source: News source
            config_overrides: Configuration overrides

        Returns:
            Best collector for the source
        """
        # Define preferred collector types per source
        preferred_types = {
            NewsSource.COINDESK: CollectorType.API_REST,
            NewsSource.COINTELEGRAPH: CollectorType.API_GRAPHQL,
        }

        preferred_type = preferred_types.get(source, CollectorType.RSS)
        return cls.get_collector(source, preferred_type, config_overrides)

    @classmethod
    def get_all_collectors_for_source(
        cls, source: NewsSource, config_overrides: Optional[Dict[str, Any]] = None
    ) -> List[NewsCollectorInterface]:
        """
        Get all available collectors for a source

        Args:
            source: News source
            config_overrides: Configuration overrides

        Returns:
            List of all available collectors for the source
        """
        collectors = []
        source_configs = cls._default_configs.get(source, {})

        for collector_type in source_configs.keys():
            try:
                collector = cls.get_collector(
                    source, collector_type, config_overrides, use_cache=False
                )
                collectors.append(collector)
            except Exception:
                continue  # Skip unavailable collectors

        return collectors

    @classmethod
    def get_supported_types_for_source(cls, source: NewsSource) -> List[CollectorType]:
        """Get supported collector types for a source"""
        source_configs = cls._default_configs.get(source, {})
        return list(source_configs.keys())
