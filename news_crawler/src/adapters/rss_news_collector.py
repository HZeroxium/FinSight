# adapters/rss_news_collector.py

import asyncio
import aiohttp
import feedparser
from typing import Optional

from ..interfaces.news_collector_interface import NewsCollectorInterface
from ..schemas.news_schemas import (
    NewsCollectionResult,
    NewsCollectorConfig,
)
from ..core.rss_parsing_strategies import get_parsing_strategy
from ..common.logger import LoggerFactory, LoggerType, LogLevel


class RSSNewsCollector(NewsCollectorInterface):
    """RSS-based news collector implementation"""

    def __init__(self, config: NewsCollectorConfig):
        """
        Initialize RSS news collector

        Args:
            config: Configuration for the collector
        """
        super().__init__(config)
        self.logger = LoggerFactory.get_logger(
            name=f"rss-collector-{config.source.value}",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file=f"logs/news_collector_{config.source.value}.log",
        )
        self.parsing_strategy = get_parsing_strategy(config.source)

        self.logger.info(f"Initialized RSS collector for {config.source.value}")

    async def collect_news(
        self, max_items: Optional[int] = None
    ) -> NewsCollectionResult:
        """
        Collect news items from RSS feed

        Args:
            max_items: Maximum number of items to collect

        Returns:
            NewsCollectionResult with collected items
        """
        max_items = max_items or self.config.max_items

        self.logger.info(f"Starting news collection from {self.config.source.value}")

        try:
            # Fetch RSS feed content
            feed_content = await self._fetch_rss_content()

            # Parse RSS feed
            feed = feedparser.parse(feed_content)

            if feed.bozo:
                self.logger.warning(
                    f"RSS feed has parsing issues: {feed.bozo_exception}"
                )

            # Parse entries
            items = []
            entries_to_process = feed.entries

            if max_items:
                entries_to_process = entries_to_process[:max_items]

            for entry in entries_to_process:
                try:
                    news_item = self.parsing_strategy.parse_item(
                        entry, self.config.source
                    )
                    items.append(news_item)
                    self.logger.debug(f"Parsed item: {news_item.title}")
                except Exception as e:
                    self.logger.error(f"Failed to parse RSS entry: {e}")
                    continue

            self.logger.info(f"Successfully collected {len(items)} news items")

            self.logger.debug(items)

            return NewsCollectionResult(
                source=self.config.source, items=items, success=True
            )

        except Exception as e:
            self.logger.error(f"Failed to collect news: {e}")
            return NewsCollectionResult(
                source=self.config.source, items=[], success=False, error_message=str(e)
            )

    async def _fetch_rss_content(self) -> str:
        """
        Fetch RSS feed content with retries

        Returns:
            RSS feed content as string
        """
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "application/rss+xml, application/xml, text/xml",
        }

        for attempt in range(self.config.retry_attempts):
            try:
                timeout = aiohttp.ClientTimeout(total=self.config.timeout)

                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(
                        str(self.config.url), headers=headers
                    ) as response:
                        response.raise_for_status()
                        content = await response.text()

                        self.logger.debug(f"Fetched RSS content ({len(content)} bytes)")
                        return content

            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    raise e

    def is_available(self) -> bool:
        """
        Check if RSS feed is available (synchronous check)

        Returns:
            True if feed is accessible
        """
        try:
            import requests

            headers = {
                "User-Agent": self.config.user_agent,
                "Accept": "application/rss+xml, application/xml, text/xml",
            }

            response = requests.head(str(self.config.url), headers=headers, timeout=10)

            return response.status_code == 200

        except Exception as e:
            self.logger.error(f"Availability check failed: {e}")
            return False

    def get_source_info(self) -> dict:
        """
        Get information about the RSS source

        Returns:
            Dictionary with source information
        """
        return {
            "source": self.config.source.value,
            "url": str(self.config.url),
            "timeout": self.config.timeout,
            "max_items": self.config.max_items,
            "user_agent": self.config.user_agent,
            "retry_attempts": self.config.retry_attempts,
            "retry_delay": self.config.retry_delay,
        }
