from typing import Dict, Optional, Any, List
from enum import Enum

from ..interfaces.news_collector_interface import NewsCollectorInterface
from ..adapters.rss_news_collector import RSSNewsCollector
from ..schemas.news_schemas import NewsCollectorConfig, NewsSource
from common.logger import LoggerFactory, LoggerType, LogLevel


class CollectorType(Enum):
    """Available collector types"""

    RSS = "rss"


class NewsCollectorFactory:
    """Factory for creating news collector instances"""

    _instances: Dict[str, NewsCollectorInterface] = {}

    # Default configurations for known sources
    _default_configs = {
        NewsSource.COINDESK: {
            "url": "https://www.coindesk.com/arc/outboundfeeds/rss",
            "timeout": 30,
            "max_items": 50,
        },
        NewsSource.COINTELEGRAPH: {
            "url": "https://cointelegraph.com/rss",
            "timeout": 30,
            "max_items": 50,
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
        Get or create a news collector instance

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
            config = cls._build_config(source, config_overrides)

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
    def create_collector(
        cls,
        source: NewsSource,
        collector_type: CollectorType = CollectorType.RSS,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> NewsCollectorInterface:
        """
        Create a new collector instance (not cached)

        Args:
            source: News source to collect from
            collector_type: Type of collector to create
            config_overrides: Configuration overrides

        Returns:
            New NewsCollectorInterface instance
        """
        return cls.get_collector(
            source=source,
            collector_type=collector_type,
            config_overrides=config_overrides,
            use_cache=False,
        )

    @classmethod
    def _build_config(
        cls, source: NewsSource, overrides: Optional[Dict[str, Any]] = None
    ) -> NewsCollectorConfig:
        """Build configuration for collector"""

        # Start with default config
        default_config = cls._default_configs.get(source, {})

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
