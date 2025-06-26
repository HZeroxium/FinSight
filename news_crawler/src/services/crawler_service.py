"""
Advanced crawler service for deep content extraction.
"""

import asyncio
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup, Tag
from pydantic import HttpUrl

from ..interfaces.crawler import NewsCrawler
from ..models.article import CrawledArticle, ArticleSource, ArticleMetadata
from ..repositories.article_repository import ArticleRepository
from ..common.logger import LoggerFactory, LoggerType, LogLevel
from ..common.cache import CacheFactory, CacheType

logger = LoggerFactory.get_logger(
    name="crawler-service", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


@dataclass
class CrawlerConfig:
    """Configuration for website crawler."""

    name: str
    base_url: str
    listing_url: str
    listing_selector: str
    title_selector: str
    content_selector: str
    date_selector: str
    author_selector: Optional[str] = None
    date_format: str = "%Y-%m-%d"
    category: Optional[str] = None
    credibility_score: float = 0.8
    enabled: bool = True


class EnhancedNewsCrawler(NewsCrawler):
    """Enhanced news crawler with smart extraction."""

    def __init__(self, config: CrawlerConfig):
        """
        Initialize crawler with configuration.

        Args:
            config: Crawler configuration
        """
        self.config = config
        self.session = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )

        # Initialize cache for this crawler
        self._cache = CacheFactory.get_cache(
            name=f"crawler_{config.name}",
            cache_type=CacheType.MEMORY,
            max_size=500,
            default_ttl=3600,  # 1 hour
        )

    async def fetch_listings(self) -> List[str]:
        """Fetch article URLs from listing page."""
        try:
            logger.info(f"Fetching listings from {self.config.name}")

            # Check cache first
            cache_key = f"listings_{self.config.name}"
            cached_urls = self._cache.get(cache_key)
            if cached_urls:
                logger.debug(f"Using cached listings for {self.config.name}")
                return cached_urls

            response = await self.session.get(self.config.listing_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            links = soup.select(self.config.listing_selector)

            urls = []
            for link in links:
                href = link.get("href")
                if href:
                    full_url = urljoin(self.config.base_url, href)
                    urls.append(full_url)

            # Cache the results
            self._cache.set(cache_key, urls, ttl=1800)  # 30 minutes

            logger.info(f"Found {len(urls)} article URLs from {self.config.name}")
            return urls

        except Exception as e:
            logger.error(f"Failed to fetch listings from {self.config.name}: {str(e)}")
            return []

    async def fetch_article(self, url: str) -> Optional[CrawledArticle]:
        """Fetch and parse single article."""
        try:
            logger.debug(f"Crawling article: {url}")

            # Check cache first
            cache_key = f"article_{url}"
            cached_article = self._cache.get(cache_key)
            if cached_article:
                return cached_article

            response = await self.session.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract title
            title_elem = soup.select_one(self.config.title_selector)
            title = title_elem.get_text(strip=True) if title_elem else ""

            # Extract content
            content_parts = []
            for elem in soup.select(self.config.content_selector):
                if isinstance(elem, Tag):
                    text = elem.get_text(strip=True)
                    if text:
                        content_parts.append(text)
            content = "\n\n".join(content_parts)

            # Extract publication date
            published_at = None
            if self.config.date_selector:
                date_elem = soup.select_one(self.config.date_selector)
                if date_elem:
                    date_str = date_elem.get("datetime") or date_elem.get_text(
                        strip=True
                    )
                    published_at = self._parse_date(date_str)

            # Extract author
            author = None
            if self.config.author_selector:
                author_elem = soup.select_one(self.config.author_selector)
                if author_elem:
                    author = author_elem.get_text(strip=True)

            # Create article source
            parsed_url = urlparse(url)
            source = ArticleSource(
                name=self.config.name,
                domain=parsed_url.netloc,
                url=HttpUrl(self.config.base_url),
                credibility_score=self.config.credibility_score,
                category=self.config.category,
            )

            # Create metadata
            metadata = ArticleMetadata(
                content_length=len(content),
                language=self._detect_language(soup),
                tags=self._extract_tags(soup),
                entities=self._extract_entities(soup),
                summary=self._generate_summary(content),
            )

            article = CrawledArticle(
                url=HttpUrl(url),
                title=title,
                content=content,
                published_at=published_at,
                author=author,
                source=source,
                metadata=metadata,
                raw_html=str(soup) if len(str(soup)) < 1000000 else None,  # Limit size
            )

            # Cache the article
            self._cache.set(cache_key, article, ttl=3600)  # 1 hour

            logger.debug(f"Successfully crawled article: {title}")
            return article

        except Exception as e:
            logger.error(f"Failed to crawl article {url}: {str(e)}")
            return None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string using various formats."""
        formats = [
            self.config.date_format,
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        logger.warning(f"Could not parse date: {date_str}")
        return None

    def _detect_language(self, soup: BeautifulSoup) -> Optional[str]:
        """Detect article language."""
        # Check html lang attribute
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            return html_tag.get("lang")

        # Check meta tags
        meta_lang = soup.find("meta", attrs={"name": "language"})
        if meta_lang and meta_lang.get("content"):
            return meta_lang.get("content")

        return None

    def _extract_tags(self, soup: BeautifulSoup) -> List[str]:
        """Extract article tags/keywords."""
        tags = []

        # Check meta keywords
        meta_keywords = soup.find("meta", attrs={"name": "keywords"})
        if meta_keywords and meta_keywords.get("content"):
            keywords = meta_keywords.get("content").split(",")
            tags.extend([kw.strip() for kw in keywords if kw.strip()])

        # Check article tags (common selectors)
        tag_selectors = [".tags a", ".categories a", ".labels a", '[rel="tag"]']
        for selector in tag_selectors:
            for elem in soup.select(selector):
                tag_text = elem.get_text(strip=True)
                if tag_text and tag_text not in tags:
                    tags.append(tag_text)

        return tags[:20]  # Limit to 20 tags

    def _extract_entities(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract structured data and entities."""
        entities = {}

        # Extract JSON-LD structured data
        json_ld_scripts = soup.find_all("script", type="application/ld+json")
        if json_ld_scripts:
            import json

            for script in json_ld_scripts:
                try:
                    data = json.loads(script.string)
                    entities["structured_data"] = data
                    break
                except:
                    continue

        # Extract Open Graph data
        og_data = {}
        for meta in soup.find_all("meta", property=lambda x: x and x.startswith("og:")):
            property_name = meta.get("property")
            content = meta.get("content")
            if property_name and content:
                og_data[property_name] = content
        if og_data:
            entities["open_graph"] = og_data

        return entities

    def _generate_summary(self, content: str, max_length: int = 300) -> Optional[str]:
        """Generate simple summary from content."""
        if not content:
            return None

        sentences = content.split(". ")
        summary = ""

        for sentence in sentences:
            if len(summary + sentence) <= max_length:
                summary += sentence + ". "
            else:
                break

        return summary.strip() if summary else None

    async def close(self):
        """Close HTTP session."""
        await self.session.aclose()


class CrawlerService:
    """Service for managing multiple crawlers."""

    def __init__(
        self,
        article_repository: ArticleRepository,
        max_concurrent: int = 10,
        timeout: int = 30,
        retry_attempts: int = 3,
    ):
        """
        Initialize crawler service.

        Args:
            article_repository: Repository for storing articles
            max_concurrent: Maximum concurrent crawls
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
        """
        self.article_repository = article_repository
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.crawlers: Dict[str, EnhancedNewsCrawler] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Initialize default crawlers
        self._initialize_default_crawlers()

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
        """Add a new crawler configuration."""
        if not config.enabled:
            logger.info(f"Skipping disabled crawler: {config.name}")
            return

        crawler = EnhancedNewsCrawler(config)
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
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        # Find the best crawler for this domain
        best_crawler = None
        for name, crawler in self.crawlers.items():
            crawler_domain = urlparse(crawler.config.base_url).netloc.lower()
            if domain == crawler_domain or domain.endswith(f".{crawler_domain}"):
                best_crawler = crawler
                break

        if not best_crawler:
            logger.warning(f"No suitable crawler found for {url}")
            return None

        try:
            async with self._semaphore:
                return await best_crawler.fetch_article(url)
        except Exception as e:
            logger.error(f"Failed to crawl single URL {url}: {str(e)}")
            return None

    async def crawl_all_sources(self) -> Dict[str, int]:
        """Crawl all configured sources."""
        logger.info("Starting crawl of all sources")
        results = {}

        tasks = []
        for name, crawler in self.crawlers.items():
            task = asyncio.create_task(self._crawl_source(name, crawler))
            tasks.append(task)

        completed_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(completed_results):
            source_name = list(self.crawlers.keys())[i]
            if isinstance(result, Exception):
                logger.error(f"Crawler {source_name} failed: {str(result)}")
                results[source_name] = 0
            else:
                results[source_name] = result

        total_crawled = sum(results.values())
        logger.info(f"Crawl completed. Total articles: {total_crawled}")
        return results

    async def crawl_source(self, source_name: str) -> int:
        """Crawl specific source by name."""
        if source_name not in self.crawlers:
            logger.error(f"Unknown crawler source: {source_name}")
            return 0

        crawler = self.crawlers[source_name]
        return await self._crawl_source(source_name, crawler)

    async def _crawl_source(
        self, source_name: str, crawler: EnhancedNewsCrawler
    ) -> int:
        """Internal method to crawl a source."""
        async with self._semaphore:
            try:
                logger.info(f"Starting crawl for {source_name}")
                start_time = time.time()

                # Fetch article URLs
                urls = await crawler.fetch_listings()
                if not urls:
                    logger.warning(f"No URLs found for {source_name}")
                    return 0

                # Filter out already crawled articles
                new_urls = []
                for url in urls:
                    if not await self.article_repository.article_exists(url):
                        new_urls.append(url)

                logger.info(f"Found {len(new_urls)} new articles for {source_name}")

                # Crawl new articles
                crawl_tasks = []
                for url in new_urls:
                    task = asyncio.create_task(
                        self._crawl_article_with_retry(crawler, url)
                    )
                    crawl_tasks.append(task)

                articles = await asyncio.gather(*crawl_tasks, return_exceptions=True)

                # Save successful articles
                saved_count = 0
                for article in articles:
                    if isinstance(article, CrawledArticle):
                        try:
                            await self.article_repository.save_crawled_article(article)
                            saved_count += 1
                        except Exception as e:
                            logger.error(f"Failed to save article: {str(e)}")

                duration = time.time() - start_time
                logger.info(
                    f"Crawled {saved_count} articles from {source_name} in {duration:.2f}s"
                )

                return saved_count

            except Exception as e:
                logger.error(f"Failed to crawl {source_name}: {str(e)}")
                return 0

    async def _crawl_article_with_retry(
        self, crawler: EnhancedNewsCrawler, url: str
    ) -> Optional[CrawledArticle]:
        """Crawl article with retry logic."""
        for attempt in range(self.retry_attempts):
            try:
                article = await crawler.fetch_article(url)
                if article:
                    return article
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    logger.error(
                        f"Failed to crawl {url} after {self.retry_attempts} attempts: {str(e)}"
                    )
                else:
                    await asyncio.sleep(2**attempt)  # Exponential backoff

        return None

    def get_crawler_stats(self) -> Dict[str, Any]:
        """Get crawler statistics."""
        enabled_crawlers = [
            name for name, crawler in self.crawlers.items() if crawler.config.enabled
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

    async def close(self):
        """Close all crawlers."""
        for crawler in self.crawlers.values():
            await crawler.close()
        logger.info("All crawlers closed")
