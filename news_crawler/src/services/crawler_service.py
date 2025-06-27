# services/crawler_service.py

"""
Crawler service layer for managing news crawling operations.
"""

import asyncio
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
from urllib.parse import urlparse

from ..interfaces.message_broker import MessageBroker
from ..adapters.default_crawler import DefaultNewsCrawler
from ..models.crawler import (
    CrawlerConfig,
    CrawlerStats,
    CrawlResult,
    CrawlerStatus,
)
from ..models.article import CrawledArticle
from ..repositories.article_repository import ArticleRepository
from ..common.logger import LoggerFactory, LoggerType, LogLevel
from ..common.cache import CacheFactory, CacheType, cache_result

logger = LoggerFactory.get_logger(
    name="crawler-service", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


class CrawlerService:
    """
    Service layer for crawler operations and business logic.
    Similar to SearchService but focused on crawling operations.
    """

    def __init__(
        self,
        article_repository: ArticleRepository,
        message_broker: MessageBroker,
        max_concurrent: int = 10,
        timeout: int = 30,
        retry_attempts: int = 3,
    ):
        """
        Initialize crawler service.

        Args:
            article_repository: Repository for storing articles
            message_broker: Message broker for async communication
            max_concurrent: Maximum concurrent crawls
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
        """
        self.article_repository = article_repository
        self.message_broker = message_broker
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.crawlers: Dict[str, DefaultNewsCrawler] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Initialize cache for crawler results
        self._cache = CacheFactory.get_cache(
            name="crawler_cache",
            cache_type=CacheType.MEMORY,
            max_size=1000,
            default_ttl=3600,  # 1 hour
        )

        # Initialize default crawlers
        self._initialize_default_crawlers()

        logger.info("Crawler service initialized successfully")

    def _initialize_default_crawlers(self) -> None:
        """Initialize default crawler configurations."""
        default_configs = [
            CrawlerConfig(
                name="bbc_news",
                base_url="https://www.bbc.com",
                listing_url="https://www.bbc.com/news/business",
                listing_selector="a[href*='/news/']",
                title_selector="h1",
                content_selector="div[data-component='text-block'] p",
                date_selector="time",
                author_selector="span[data-component='byline'] span",
                date_format="%Y-%m-%dT%H:%M:%S.%fZ",
                category="business",
                credibility_score=0.9,
            ),
            CrawlerConfig(
                name="reuters",
                base_url="https://www.reuters.com",
                listing_url="https://www.reuters.com/markets/",
                listing_selector="a[href*='/markets/']",
                title_selector="h1",
                content_selector="div[data-testid='paragraph'] p",
                date_selector="time",
                date_format="%Y-%m-%dT%H:%M:%SZ",
                category="finance",
                credibility_score=0.95,
            ),
        ]

        for config in default_configs:
            self.add_crawler(config)

    def add_crawler(self, config: CrawlerConfig) -> None:
        """
        Add a new crawler configuration.

        Args:
            config: Crawler configuration
        """
        if not config.enabled:
            logger.info(f"Skipping disabled crawler: {config.name}")
            return

        crawler = DefaultNewsCrawler(
            name=config.name,
            base_url=config.base_url,
            listing_url=config.listing_url,
            listing_selector=config.listing_selector,
            title_selector=config.title_selector,
            content_selector=config.content_selector,
            date_selector=config.date_selector,
            author_selector=config.author_selector,
            date_format=config.date_format,
            category=config.category,
            credibility_score=config.credibility_score,
            enabled=config.enabled,
        )

        self.crawlers[config.name] = crawler
        logger.info(f"Added crawler for {config.name}")

    async def crawl_single_url(self, url: str) -> Optional[CrawledArticle]:
        """
        Crawl a single URL using the most appropriate crawler.

        Args:
            url: URL to crawl

        Returns:
            Optional[CrawledArticle]: Crawled article or None
        """
        logger.info(f"Crawling single URL: {url}")

        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        # Find the best crawler for this domain
        best_crawler = None
        for name, crawler in self.crawlers.items():
            crawler_domain = urlparse(crawler.base_url).netloc.lower()
            if domain == crawler_domain or domain.endswith(f".{crawler_domain}"):
                best_crawler = crawler
                break

        if not best_crawler:
            logger.warning(f"No suitable crawler found for {url}")
            return None

        try:
            async with self._semaphore:
                article = await best_crawler.fetch_article(url)

                # If article was successfully crawled, publish event
                if article:
                    await self._publish_crawl_event(article, "single_url_crawled")

                return article

        except Exception as e:
            logger.error(f"Failed to crawl single URL {url}: {str(e)}")
            return None

    @cache_result(ttl=300, key_prefix="crawl_all_")
    async def crawl_all_sources(self) -> Dict[str, CrawlResult]:
        """
        Crawl all configured sources.

        Returns:
            Dict[str, CrawlResult]: Results keyed by source name
        """
        logger.info("Starting crawl of all sources")

        results = {}
        tasks = []

        for name, crawler in self.crawlers.items():
            task = asyncio.create_task(self._crawl_source_with_result(name, crawler))
            tasks.append(task)

        completed_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(completed_results):
            source_name = list(self.crawlers.keys())[i]
            if isinstance(result, Exception):
                logger.error(f"Crawler {source_name} failed: {str(result)}")
                results[source_name] = CrawlResult(
                    job_id=f"crawl_all_{source_name}_{int(time.time())}",
                    source_name=source_name,
                    articles_found=0,
                    articles_crawled=0,
                    articles_saved=0,
                    duration=0.0,
                    status="failed",
                    errors=[str(result)],
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                )
            else:
                results[source_name] = result

        total_crawled = sum(r.articles_saved for r in results.values())
        logger.info(f"Crawl completed. Total articles: {total_crawled}")

        # Publish aggregated crawl event
        await self._publish_crawl_summary_event(results)

        return results

    async def crawl_source(self, source_name: str) -> CrawlResult:
        """
        Crawl specific source by name.

        Args:
            source_name: Name of the source to crawl

        Returns:
            CrawlResult: Crawl operation result
        """
        if source_name not in self.crawlers:
            logger.error(f"Unknown crawler source: {source_name}")
            return CrawlResult(
                job_id=f"crawl_{source_name}_{int(time.time())}",
                source_name=source_name,
                articles_found=0,
                articles_crawled=0,
                articles_saved=0,
                duration=0.0,
                status="failed",
                errors=[f"Unknown source: {source_name}"],
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
            )

        crawler = self.crawlers[source_name]
        return await self._crawl_source_with_result(source_name, crawler)

    async def _crawl_source_with_result(
        self, source_name: str, crawler: DefaultNewsCrawler
    ) -> CrawlResult:
        """
        Internal method to crawl a source and return detailed results.

        Args:
            source_name: Source name
            crawler: Crawler instance

        Returns:
            CrawlResult: Detailed crawl results
        """
        start_time = time.time()
        started_at = datetime.utcnow()
        errors = []

        try:
            logger.info(f"Starting crawl for {source_name}")

            # Fetch article URLs
            urls = await crawler.fetch_listings()
            articles_found = len(urls)

            if not urls:
                logger.warning(f"No URLs found for {source_name}")
                return CrawlResult(
                    job_id=f"crawl_{source_name}_{int(start_time)}",
                    source_name=source_name,
                    articles_found=0,
                    articles_crawled=0,
                    articles_saved=0,
                    duration=time.time() - start_time,
                    status="completed",
                    errors=["No URLs found"],
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                )

            # Filter out already crawled articles
            new_urls = []
            for url in urls:
                if not await self.article_repository.article_exists(url):
                    new_urls.append(url)

            logger.info(f"Found {len(new_urls)} new articles for {source_name}")

            # Crawl new articles with concurrency control
            crawl_tasks = []
            for url in new_urls:
                task = asyncio.create_task(self._crawl_article_with_retry(crawler, url))
                crawl_tasks.append(task)

            articles = await asyncio.gather(*crawl_tasks, return_exceptions=True)

            # Save successful articles and track errors
            saved_count = 0
            crawled_count = 0

            for i, article in enumerate(articles):
                if isinstance(article, Exception):
                    errors.append(f"Failed to crawl {new_urls[i]}: {str(article)}")
                elif isinstance(article, CrawledArticle):
                    crawled_count += 1
                    try:
                        await self.article_repository.save_crawled_article(article)
                        saved_count += 1

                        # Publish individual article event
                        await self._publish_crawl_event(article, "article_crawled")

                    except Exception as e:
                        error_msg = f"Failed to save article {article.url}: {str(e)}"
                        errors.append(error_msg)
                        logger.error(error_msg)

            duration = time.time() - start_time
            status = "completed" if saved_count > 0 else "completed_with_errors"

            result = CrawlResult(
                job_id=f"crawl_{source_name}_{int(start_time)}",
                source_name=source_name,
                articles_found=articles_found,
                articles_crawled=crawled_count,
                articles_saved=saved_count,
                duration=duration,
                status=status,
                errors=errors[:10],  # Limit errors to avoid huge payloads
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

            logger.info(
                f"Crawled {saved_count} articles from {source_name} in {duration:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to crawl {source_name}: {str(e)}")
            return CrawlResult(
                job_id=f"crawl_{source_name}_{int(start_time)}",
                source_name=source_name,
                articles_found=0,
                articles_crawled=0,
                articles_saved=0,
                duration=time.time() - start_time,
                status="failed",
                errors=[str(e)],
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

    async def _crawl_article_with_retry(
        self, crawler: DefaultNewsCrawler, url: str
    ) -> Optional[CrawledArticle]:
        """Crawl article with retry logic."""
        for attempt in range(self.retry_attempts):
            try:
                async with self._semaphore:
                    article = await crawler.fetch_article(url)
                    if article:
                        return article
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    logger.error(
                        f"Failed to crawl {url} after {self.retry_attempts} attempts: {str(e)}"
                    )
                    raise e
                else:
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {url}: {str(e)}, retrying..."
                    )
                    await asyncio.sleep(2**attempt)  # Exponential backoff

        return None

    async def _publish_crawl_event(
        self, article: CrawledArticle, event_type: str
    ) -> None:
        """
        Publish crawl event for analytics and downstream processing.

        Args:
            article: Crawled article
            event_type: Type of crawl event
        """
        try:
            event = {
                "event_type": event_type,
                "article_id": str(article.id),
                "url": str(article.url),
                "title": article.title,
                "source": article.source.name,
                "category": article.source.category,
                "content_length": len(article.content),
                "published_at": (
                    article.published_at.isoformat() if article.published_at else None
                ),
                "crawled_at": article.created_at.isoformat(),
                "timestamp": datetime.utcnow().isoformat(),
            }

            await self.message_broker.publish(
                exchange="news_crawler_exchange",
                routing_key="article.crawled",
                message=event,
            )

            logger.debug(f"Published crawl event for article: {article.title}")

        except Exception as e:
            logger.warning(f"Failed to publish crawl event: {str(e)}")

    async def _publish_crawl_summary_event(
        self, results: Dict[str, CrawlResult]
    ) -> None:
        """
        Publish summary event for all crawl operations.

        Args:
            results: Dictionary of crawl results by source name
        """
        try:
            total_found = sum(r.articles_found for r in results.values())
            total_crawled = sum(r.articles_crawled for r in results.values())
            total_saved = sum(r.articles_saved for r in results.values())

            event = {
                "event_type": "crawl_summary",
                "total_sources": len(results),
                "total_articles_found": total_found,
                "total_articles_crawled": total_crawled,
                "total_articles_saved": total_saved,
                "success_rate": (
                    total_saved / total_crawled if total_crawled > 0 else 0.0
                ),
                "source_results": {
                    name: result.dict() for name, result in results.items()
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

            await self.message_broker.publish(
                exchange="news_crawler_exchange",
                routing_key="crawl.summary",
                message=event,
            )

        except Exception as e:
            logger.warning(f"Failed to publish crawl summary: {str(e)}")

    def get_crawler_stats(self) -> Dict[str, Any]:
        """
        Get crawler statistics.

        Returns:
            Dict[str, Any]: Crawler statistics
        """
        enabled_crawlers = [
            name for name, crawler in self.crawlers.items() if crawler.enabled
        ]

        return {
            "total_crawlers": len(self.crawlers),
            "enabled_crawlers": len(enabled_crawlers),
            "crawler_names": list(self.crawlers.keys()),
            "enabled_crawler_names": enabled_crawlers,
            "max_concurrent": self.max_concurrent,
            "timeout": self.timeout,
            "retry_attempts": self.retry_attempts,
        }

    def get_detailed_crawler_stats(self) -> List[CrawlerStats]:
        """
        Get detailed statistics for each crawler.

        Returns:
            List[CrawlerStats]: Detailed statistics per crawler
        """
        stats = []
        for name, crawler in self.crawlers.items():
            status = CrawlerStatus.ACTIVE if crawler.enabled else CrawlerStatus.DISABLED

            # In a production system, these would be tracked in the database
            crawler_stat = CrawlerStats(
                name=name,
                status=status,
                total_articles_found=0,  # Would track in production
                total_articles_crawled=0,  # Would track in production
                total_articles_saved=0,  # Would track in production
                success_rate=0.95,  # Would calculate in production
                average_crawl_time=2.5,  # Would track in production
                last_crawl_time=None,  # Would track in production
                errors=[],
            )
            stats.append(crawler_stat)

        return stats

    async def health_check(self) -> bool:
        """Check service health."""
        try:
            # Check if we have active crawlers
            active_crawlers = sum(
                1 for crawler in self.crawlers.values() if crawler.enabled
            )

            # Check message broker health
            broker_healthy = await self.message_broker.health_check()

            # Check repository health
            repository_healthy = True
            try:
                # Simple test to see if we can access the repository
                await self.article_repository.article_exists("http://test.example.com")
            except Exception:
                repository_healthy = False

            is_healthy = active_crawlers > 0 and broker_healthy and repository_healthy
            logger.debug(f"Crawler service health check: {is_healthy}")
            return is_healthy

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

    async def close(self):
        """Close all crawlers."""
        for crawler in self.crawlers.values():
            await crawler.close()
        logger.info("All crawlers closed")
