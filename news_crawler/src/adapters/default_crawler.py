# adapters/default_crawler.py

"""
Default NewsCrawler implementation using HTTPX + BeautifulSoup.
"""

from typing import List, Optional
from datetime import datetime
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag
from pydantic import HttpUrl

from ..interfaces.crawler import NewsCrawler
from ..models.article import CrawledArticle, ArticleSource, ArticleMetadata
from ..common.logger import LoggerFactory, LoggerType, LogLevel
from ..common.cache import CacheFactory, CacheType

logger = LoggerFactory.get_logger(
    name="default-crawler", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


class DefaultNewsCrawler(NewsCrawler):
    """
    Default crawler implementation with configurable selectors.
    Uses HTTPX for async HTTP requests and BeautifulSoup for HTML parsing.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        listing_url: str,
        listing_selector: str,
        title_selector: str,
        content_selector: str,
        date_selector: str,
        author_selector: Optional[str] = None,
        date_format: str = "%Y-%m-%dT%H:%M:%S",
        category: Optional[str] = None,
        credibility_score: float = 0.8,
        enabled: bool = True,
    ):
        """
        Initialize crawler with configuration.

        Args:
            name: Crawler name identifier
            base_url: Base URL for the website
            listing_url: URL of the page containing article links
            listing_selector: CSS selector to locate <a> tags for articles
            title_selector: CSS selector to locate the title element
            content_selector: CSS selector to locate the article body
            date_selector: CSS selector to locate the datetime element
            author_selector: CSS selector for author (optional)
            date_format: strptime format to parse the published_at string
            category: Article category
            credibility_score: Source credibility score (0.0-1.0)
            enabled: Whether this crawler is enabled
        """
        self.name = name
        self.base_url = base_url
        self.listing_url = listing_url
        self.listing_selector = listing_selector
        self.title_selector = title_selector
        self.content_selector = content_selector
        self.date_selector = date_selector
        self.author_selector = author_selector
        self.date_format = date_format
        self.category = category
        self.credibility_score = credibility_score
        self.enabled = enabled

        # Initialize HTTP client
        self.session = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )

        # Initialize cache for this crawler
        self._cache = CacheFactory.get_cache(
            name=f"crawler_{name}",
            cache_type=CacheType.MEMORY,
            max_size=500,
            default_ttl=3600,  # 1 hour
        )

        logger.info(f"Initialized {name} crawler for {base_url}")

    async def fetch_listings(self) -> List[str]:
        """
        Retrieve a list of absolute article URLs from the listing page.

        Returns:
            List[str]: Fully qualified URLs to crawl
        """
        if not self.enabled:
            logger.info(f"Crawler {self.name} is disabled")
            return []

        try:
            logger.info(f"Fetching listings from {self.name}")

            # Check cache first
            cache_key = f"listings_{self.name}"
            cached_urls = self._cache.get(cache_key)
            if cached_urls:
                logger.debug(f"Using cached listings for {self.name}")
                return cached_urls

            response = await self.session.get(self.listing_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            links = soup.select(self.listing_selector)

            urls = []
            for link in links:
                href = link.get("href")
                if href:
                    full_url = urljoin(self.base_url, href)
                    urls.append(full_url)

            # Cache the results
            self._cache.set(cache_key, urls, ttl=1800)  # 30 minutes

            logger.info(f"Found {len(urls)} article URLs from {self.name}")
            return urls

        except Exception as e:
            logger.error(f"Failed to fetch listings from {self.name}: {str(e)}")
            return []

    async def fetch_article(self, url: str) -> Optional[CrawledArticle]:
        """
        Fetch and parse a single article page.

        Args:
            url: The article's URL

        Returns:
            Optional[CrawledArticle]: Parsed article with metadata and content
        """
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
            title_elem = soup.select_one(self.title_selector)
            title = title_elem.get_text(strip=True) if title_elem else ""

            # Extract content
            content_parts = []
            for elem in soup.select(self.content_selector):
                if isinstance(elem, Tag):
                    text = elem.get_text(strip=True)
                    if text:
                        content_parts.append(text)
            content = "\n\n".join(content_parts)

            # Extract publication date
            published_at = None
            if self.date_selector:
                date_elem = soup.select_one(self.date_selector)
                if date_elem:
                    date_str = date_elem.get("datetime") or date_elem.get_text(
                        strip=True
                    )
                    published_at = self._parse_date(date_str)

            # Extract author
            author = None
            if self.author_selector:
                author_elem = soup.select_one(self.author_selector)
                if author_elem:
                    author = author_elem.get_text(strip=True)

            # Create article source
            parsed_url = urlparse(url)
            source = ArticleSource(
                name=self.name,
                domain=parsed_url.netloc,
                url=HttpUrl(self.base_url),
                credibility_score=self.credibility_score,
                category=self.category,
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
            self.date_format,
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

    def _extract_entities(self, soup: BeautifulSoup) -> dict:
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
