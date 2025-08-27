# core/news_collector_facade.py

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from common.logger import LoggerFactory, LoggerType, LogLevel

from ..schemas.news_schemas import NewsCollectionResult, NewsItem, NewsSource
from .news_collector_factory import CollectorType, NewsCollectorFactory


class NewsCollectorFacade:
    """
    Enhanced facade providing flexible API for news collection with multiple collector types
    """

    def __init__(self, use_cache: bool = True, enable_fallback: bool = True):
        """
        Initialize the news collector facade

        Args:
            use_cache: Whether to cache collector instances
            enable_fallback: Whether to try alternative collectors on failure
        """
        self.use_cache = use_cache
        self.enable_fallback = enable_fallback
        self.logger = LoggerFactory.get_logger(
            name="news-collector-facade",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file="logs/news_collector_facade.log",
        )

        self.logger.info("Enhanced NewsCollectorFacade initialized")

    async def collect_from_source(
        self,
        source: NewsSource,
        collector_type: Optional[CollectorType] = None,
        max_items: Optional[int] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> NewsCollectionResult:
        """
        Collect news from a specific source with specified or best collector type

        Args:
            source: News source to collect from
            collector_type: Specific collector type (uses best if None)
            max_items: Maximum number of items to collect
            config_overrides: Configuration overrides for the collector

        Returns:
            NewsCollectionResult with collected items
        """
        self.logger.info(
            f"Collecting news from {source.value} with collector type: {collector_type}"
        )

        try:
            # Get primary collector
            if collector_type:
                collector = NewsCollectorFactory.get_collector(
                    source=source,
                    collector_type=collector_type,
                    config_overrides=config_overrides,
                    use_cache=self.use_cache,
                )
            else:
                collector = NewsCollectorFactory.get_best_collector_for_source(
                    source=source,
                    config_overrides=config_overrides,
                )

            # Try primary collector
            result = await collector.collect_news(max_items=max_items)

            if result.success:
                self.logger.info(
                    f"Successfully collected {result.total_items} items from {source.value}"
                )
                return result

            # Fallback mechanism if enabled and primary failed
            if self.enable_fallback and not result.success:
                self.logger.warning(
                    f"Primary collector failed for {source.value}, trying fallback options"
                )
                return await self._try_fallback_collectors(
                    source, max_items, config_overrides
                )

            return result

        except Exception as e:
            self.logger.error(f"Failed to collect from {source.value}: {e}")

            # Try fallback on exception if enabled
            if self.enable_fallback:
                try:
                    return await self._try_fallback_collectors(
                        source, max_items, config_overrides
                    )
                except Exception as fallback_error:
                    self.logger.error(
                        f"All fallback collectors failed: {fallback_error}"
                    )

            return NewsCollectionResult(
                source=source, items=[], success=False, error_message=str(e)
            )

    async def _try_fallback_collectors(
        self,
        source: NewsSource,
        max_items: Optional[int],
        config_overrides: Optional[Dict[str, Any]],
    ) -> NewsCollectionResult:
        """Try alternative collectors for the source"""

        collectors = NewsCollectorFactory.get_all_collectors_for_source(
            source=source, config_overrides=config_overrides
        )

        for collector in collectors:
            try:
                self.logger.info(
                    f"Trying fallback collector: {type(collector).__name__}"
                )
                result = await collector.collect_news(max_items=max_items)

                if result.success:
                    self.logger.info(
                        f"Fallback collector succeeded with {result.total_items} items"
                    )
                    return result

            except Exception as e:
                self.logger.warning(f"Fallback collector failed: {e}")
                continue

        # All collectors failed
        return NewsCollectionResult(
            source=source,
            items=[],
            success=False,
            error_message="All collectors failed",
        )

    async def collect_from_multiple_sources(
        self,
        sources: List[NewsSource],
        collector_preferences: Optional[Dict[NewsSource, CollectorType]] = None,
        max_items_per_source: Optional[int] = None,
        config_overrides: Optional[Dict[NewsSource, Dict[str, Any]]] = None,
    ) -> Dict[NewsSource, NewsCollectionResult]:
        """
        Collect news from multiple sources with flexible collector selection

        Args:
            sources: List of news sources to collect from
            collector_preferences: Preferred collector type per source
            max_items_per_source: Maximum items per source
            config_overrides: Configuration overrides per source

        Returns:
            Dictionary mapping sources to their collection results
        """
        self.logger.info(f"Collecting news from {len(sources)} sources")

        # Prepare tasks for concurrent execution
        tasks = []
        for source in sources:
            collector_type = (
                collector_preferences.get(source) if collector_preferences else None
            )
            source_config = config_overrides.get(source) if config_overrides else None

            task = self.collect_from_source(
                source=source,
                collector_type=collector_type,
                max_items=max_items_per_source,
                config_overrides=source_config,
            )
            tasks.append((source, task))

        # Execute tasks concurrently
        results = {}
        completed_tasks = await asyncio.gather(
            *[task for _, task in tasks], return_exceptions=True
        )

        for (source, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                self.logger.error(f"Exception collecting from {source.value}: {result}")
                results[source] = NewsCollectionResult(
                    source=source, items=[], success=False, error_message=str(result)
                )
            else:
                results[source] = result

        # Log summary
        total_items = sum(r.total_items for r in results.values())
        successful_sources = sum(1 for r in results.values() if r.success)

        self.logger.info(
            f"Collection complete: {total_items} total items from "
            f"{successful_sources}/{len(sources)} sources"
        )

        return results

    async def collect_all_supported_sources(
        self,
        max_items_per_source: Optional[int] = None,
        use_best_collectors: bool = True,
    ) -> Dict[NewsSource, NewsCollectionResult]:
        """
        Collect news from all supported sources using best available collectors

        Args:
            max_items_per_source: Maximum items per source
            use_best_collectors: Whether to use best collectors for each source

        Returns:
            Dictionary mapping sources to their collection results
        """
        supported_sources = list(NewsSource)

        if use_best_collectors:
            return await self.collect_from_multiple_sources(
                sources=supported_sources, max_items_per_source=max_items_per_source
            )
        else:
            # Use RSS as fallback for all
            collector_preferences = {
                source: CollectorType.RSS for source in supported_sources
            }
            return await self.collect_from_multiple_sources(
                sources=supported_sources,
                collector_preferences=collector_preferences,
                max_items_per_source=max_items_per_source,
            )

    def get_available_collectors(self, source: NewsSource) -> List[str]:
        """Get available collector types for a source"""
        try:
            supported_types = NewsCollectorFactory.get_supported_types_for_source(
                source
            )
            return [t.value for t in supported_types]
        except Exception as e:
            self.logger.error(
                f"Failed to get available collectors for {source.value}: {e}"
            )
            return []

    def aggregate_results(
        self, results: Dict[NewsSource, NewsCollectionResult], sort_by_date: bool = True
    ) -> List[NewsItem]:
        """
        Aggregate news items from multiple collection results

        Args:
            results: Collection results from multiple sources
            sort_by_date: Whether to sort items by publication date

        Returns:
            Aggregated list of news items
        """
        all_items = []

        for source, result in results.items():
            if result.success:
                all_items.extend(result.items)
                self.logger.debug(
                    f"Added {len(result.items)} items from {source.value}"
                )

        if sort_by_date:
            all_items.sort(key=lambda x: x.published_at, reverse=True)

        self.logger.info(f"Aggregated {len(all_items)} total news items")
        return all_items

    def filter_items(
        self,
        items: List[NewsItem],
        keywords: Optional[List[str]] = None,
        sources: Optional[List[NewsSource]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[NewsItem]:
        """
        Filter news items based on criteria

        Args:
            items: List of news items to filter
            keywords: Keywords to search in title/description
            sources: Sources to include
            start_date: Minimum publication date
            end_date: Maximum publication date

        Returns:
            Filtered list of news items
        """
        filtered_items = items

        # Filter by sources
        if sources:
            filtered_items = [item for item in filtered_items if item.source in sources]

        # Filter by date range
        if start_date:
            filtered_items = [
                item for item in filtered_items if item.published_at >= start_date
            ]

        if end_date:
            filtered_items = [
                item for item in filtered_items if item.published_at <= end_date
            ]

        # Filter by keywords
        if keywords:
            keyword_items = []
            for item in filtered_items:
                text_to_search = f"{item.title} {item.description or ''} {' '.join(item.tags)}".lower()
                if any(keyword.lower() in text_to_search for keyword in keywords):
                    keyword_items.append(item)
            filtered_items = keyword_items

        self.logger.debug(f"Filtered {len(items)} items to {len(filtered_items)}")
        return filtered_items

    def is_source_available(self, source: NewsSource) -> bool:
        """
        Check if a news source is currently available

        Args:
            source: News source to check

        Returns:
            True if source is available
        """
        try:
            collector = NewsCollectorFactory.get_collector(
                source=source, use_cache=self.use_cache
            )
            return collector.is_available()
        except Exception as e:
            self.logger.error(f"Failed to check availability for {source.value}: {e}")
            return False

    def get_source_info(self, source: NewsSource) -> Dict[str, Any]:
        """
        Get information about a news source

        Args:
            source: News source to get info for

        Returns:
            Dictionary with source information
        """
        try:
            collector = NewsCollectorFactory.get_collector(
                source=source, use_cache=self.use_cache
            )
            return collector.get_source_info()
        except Exception as e:
            self.logger.error(f"Failed to get info for {source.value}: {e}")
            return {"source": source.value, "error": str(e)}

    def clear_cache(self) -> None:
        """Clear cached collector instances"""
        NewsCollectorFactory.clear_cache()
        self.logger.info("Cleared collector cache")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the facade

        Returns:
            Dictionary with usage statistics
        """
        cached_instances = NewsCollectorFactory.get_cached_instances()
        supported_sources = NewsCollectorFactory.get_supported_sources()

        return {
            "supported_sources": [s.value for s in supported_sources],
            "cached_instances": cached_instances,
            "cache_enabled": self.use_cache,
            "total_cached": len(cached_instances),
        }


# Convenience functions for quick access
async def collect_crypto_news(
    max_items_per_source: int = 20, sources: Optional[List[NewsSource]] = None
) -> List[NewsItem]:
    """
    Quick function to collect crypto news from all or specified sources

    Args:
        max_items_per_source: Maximum items per source
        sources: Sources to collect from (all if None)

    Returns:
        Aggregated list of news items
    """
    facade = NewsCollectorFacade()

    if sources:
        results = await facade.collect_from_multiple_sources(
            sources=sources, max_items_per_source=max_items_per_source
        )
    else:
        results = await facade.collect_all_supported_sources(
            max_items_per_source=max_items_per_source
        )

    return facade.aggregate_results(results)


async def collect_latest_news(
    keywords: Optional[List[str]] = None, max_items: int = 50
) -> List[NewsItem]:
    """
    Quick function to collect latest crypto news with optional keyword filtering

    Args:
        keywords: Keywords to filter by
        max_items: Maximum total items to return

    Returns:
        Latest filtered news items
    """
    facade = NewsCollectorFacade()
    results = await facade.collect_all_supported_sources(max_items_per_source=max_items)
    items = facade.aggregate_results(results)

    if keywords:
        items = facade.filter_items(items, keywords=keywords)

    return items[:max_items]
