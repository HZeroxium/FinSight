# misc/crawl4ai_collector.py

"""
Comprehensive Crawl4AI Example for Crypto News Crawling

This module demonstrates the capabilities of crawl4ai library for advanced
web crawling and content extraction, specifically tailored for crypto news
and financial content gathering for sentiment analysis.

Features demonstrated:
- Basic content extraction
- Advanced CSS/XPath selectors
- JavaScript-heavy site handling
- Multi-page crawling
- Content filtering and cleaning
- Structured data extraction
- Performance optimization
- Error handling and retries

Requirements:
    pip install crawl4ai playwright
    playwright install

Author: Expert Software Architect
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urljoin

from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.extraction_strategy import (
    JsonCssExtractionStrategy,
)
from crawl4ai.chunking_strategy import RegexChunking, NlpSentenceChunking
from crawl4ai.content_filter_strategy import PruningContentFilter

from common.logger import LoggerFactory, LoggerType, LogLevel


class Crawl4AICryptoNewsCollector:
    """
    Advanced crypto news collector using Crawl4AI for enhanced content extraction
    """

    def __init__(
        self,
        headless: bool = True,
        browser_type: str = "chromium",
        logger_name: str = "crawl4ai_collector",
    ):
        """
        Initialize Crawl4AI collector

        Args:
            headless: Whether to run browser in headless mode
            browser_type: Browser type (chromium, firefox, webkit)
            logger_name: Name for the logger instance
        """
        self.headless = headless
        self.browser_type = browser_type

        # Initialize logger
        self.logger = LoggerFactory.get_logger(
            name=logger_name,
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            use_colors=True,
        )

        # Common crypto news sites for demonstration
        self.crypto_news_sites = {
            "coindesk": {
                "base_url": "https://www.coindesk.com",
                "news_page": "https://www.coindesk.com/latest-crypto-news",
                "article_selector": "a[href*='/latest-crypto-news/']",
                "title_selector": "h1, .headline, .article-title",
                "content_selector": ".at-content, .entry-content, .article-content",
                "date_selector": "time, .date, .published-date",
                "author_selector": ".author, .byline",
            },
            "cointelegraph": {
                "base_url": "https://cointelegraph.com",
                "news_page": "https://cointelegraph.com/news",
                "article_selector": "a[href*='/news/']",
                "title_selector": "h1, .post-title",
                "content_selector": ".post-content, .post-body",
                "date_selector": "time, .post-date",
                "author_selector": ".author-name",
            },
            # "cryptonews": {
            #     "base_url": "https://cryptonews.com",
            #     "news_page": "https://cryptonews.com/news/",
            #     "article_selector": "a[href*='/news/']",
            #     "title_selector": "h1, .article-title",
            #     "content_selector": ".article-body, .content",
            #     "date_selector": "time, .date",
            #     "author_selector": ".author",
            # },
        }

        self.logger.info("Initialized Crawl4AI crypto news collector")

    async def basic_content_extraction(self, url: str) -> Dict[str, Any]:
        """
        Basic content extraction using Crawl4AI

        Args:
            url: URL to crawl

        Returns:
            Dict containing extracted content
        """
        self.logger.info(f"Performing basic content extraction for: {url}")

        async with AsyncWebCrawler(
            headless=self.headless,
            browser_type=self.browser_type,
        ) as crawler:
            try:
                result = await crawler.arun(
                    url=url,
                    cache_mode=CacheMode.BYPASS,
                    word_count_threshold=10,
                    bypass_cache=True,
                )

                if result.success:
                    extracted_data = {
                        "url": url,
                        "title": result.metadata.get("title", ""),
                        "description": result.metadata.get("description", ""),
                        "keywords": result.metadata.get("keywords", ""),
                        "content": result.markdown,
                        "cleaned_html": result.cleaned_html,
                        "links": result.links,
                        "media": result.media,
                        "metadata": result.metadata,
                        "crawl_timestamp": datetime.now().isoformat(),
                        "success": True,
                    }

                    self.logger.info(
                        f"Successfully extracted {len(result.markdown)} chars of content"
                    )
                    return extracted_data
                else:
                    self.logger.error(f"Failed to crawl {url}: {result.error_message}")
                    return {"url": url, "success": False, "error": result.error_message}

            except Exception as e:
                self.logger.error(f"Exception during basic extraction: {e}")
                return {"url": url, "success": False, "error": str(e)}

    async def advanced_css_extraction(
        self, url: str, selectors: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Advanced content extraction using CSS selectors

        Args:
            url: URL to crawl
            selectors: Dictionary of CSS selectors for different content types

        Returns:
            Dict containing structured extracted content
        """
        self.logger.info(f"Performing advanced CSS extraction for: {url}")

        # Create extraction strategy using CSS selectors
        extraction_strategy = JsonCssExtractionStrategy(
            schema={
                "title": selectors.get("title_selector", "h1"),
                "content": selectors.get("content_selector", ".content"),
                "date": selectors.get("date_selector", "time"),
                "author": selectors.get("author_selector", ".author"),
                "tags": ".tags a, .categories a",
                "related_links": "a[href*='news'], a[href*='article']",
            }
        )

        async with AsyncWebCrawler(
            headless=self.headless,
            browser_type=self.browser_type,
        ) as crawler:
            try:
                result = await crawler.arun(
                    url=url,
                    extraction_strategy=extraction_strategy,
                    cache_mode=CacheMode.ENABLED,
                    word_count_threshold=10,
                )

                if result.success:
                    # Parse extracted JSON
                    try:
                        structured_data = json.loads(result.extracted_content)
                    except json.JSONDecodeError:
                        structured_data = {"raw_extraction": result.extracted_content}

                    extracted_data = {
                        "url": url,
                        "structured_content": structured_data,
                        "markdown": result.markdown,
                        "links": result.links,
                        "media": result.media,
                        "metadata": result.metadata,
                        "crawl_timestamp": datetime.now().isoformat(),
                        "success": True,
                    }

                    self.logger.info(f"Successfully extracted structured content")
                    return extracted_data
                else:
                    self.logger.error(f"Failed to crawl {url}: {result.error_message}")
                    return {"url": url, "success": False, "error": result.error_message}

            except Exception as e:
                self.logger.error(f"Exception during CSS extraction: {e}")
                return {"url": url, "success": False, "error": str(e)}

    async def javascript_heavy_site_crawling(
        self, url: str, wait_time: int = 3
    ) -> Dict[str, Any]:
        """
        Crawl JavaScript-heavy sites with dynamic content loading

        Args:
            url: URL to crawl
            wait_time: Time to wait for JavaScript execution

        Returns:
            Dict containing extracted content from JS-rendered page
        """
        self.logger.info(f"Crawling JavaScript-heavy site: {url}")

        async with AsyncWebCrawler(
            headless=self.headless,
            browser_type=self.browser_type,
        ) as crawler:
            try:
                # JavaScript to execute for crypto sites
                js_code = """
                // Wait for common crypto news site elements to load
                const waitForElement = (selector, timeout = 5000) => {
                    return new Promise((resolve) => {
                        if (document.querySelector(selector)) {
                            return resolve(document.querySelector(selector));
                        }
                        
                        const observer = new MutationObserver(() => {
                            if (document.querySelector(selector)) {
                                resolve(document.querySelector(selector));
                                observer.disconnect();
                            }
                        });
                        
                        observer.observe(document.body, {
                            childList: true,
                            subtree: true
                        });
                        
                        setTimeout(() => {
                            observer.disconnect();
                            resolve(null);
                        }, timeout);
                    });
                };

                // Wait for article content or main content area
                await waitForElement('article, .article, .post, .news-item');
                
                // Scroll to load lazy-loaded content
                window.scrollTo(0, document.body.scrollHeight / 2);
                await new Promise(resolve => setTimeout(resolve, 1000));
                window.scrollTo(0, document.body.scrollHeight);
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                // Return some additional data
                return {
                    totalArticles: document.querySelectorAll('article, .article, .post').length,
                    hasComments: !!document.querySelector('.comments, #comments'),
                    socialShares: document.querySelectorAll('.share, .social').length
                };
                """

                result = await crawler.arun(
                    url=url,
                    js_code=js_code,
                    wait_for_js=True,
                    wait_for=wait_time,
                    cache_mode=CacheMode.BYPASS,
                    word_count_threshold=10,
                )

                if result.success:
                    extracted_data = {
                        "url": url,
                        "title": result.metadata.get("title", ""),
                        "content": result.markdown,
                        "html": result.cleaned_html,
                        "links": result.links,
                        "media": result.media,
                        "js_execution_result": result.js_execution_result,
                        "metadata": result.metadata,
                        "crawl_timestamp": datetime.now().isoformat(),
                        "success": True,
                    }

                    self.logger.info(f"Successfully crawled JS-heavy site")
                    return extracted_data
                else:
                    self.logger.error(f"Failed to crawl {url}: {result.error_message}")
                    return {"url": url, "success": False, "error": result.error_message}

            except Exception as e:
                self.logger.error(f"Exception during JS site crawling: {e}")
                return {"url": url, "success": False, "error": str(e)}

    async def multi_page_news_crawling(
        self, site_config: Dict[str, str], max_pages: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Crawl multiple pages from a crypto news site

        Args:
            site_config: Configuration for the news site
            max_pages: Maximum number of pages to crawl

        Returns:
            List of extracted articles
        """
        self.logger.info(f"Starting multi-page crawling for {site_config['base_url']}")

        articles = []
        async with AsyncWebCrawler(
            headless=self.headless,
            browser_type=self.browser_type,
        ) as crawler:
            try:
                # First, get the main news page to find article links
                result = await crawler.arun(
                    url=site_config["news_page"],
                    cache_mode=CacheMode.ENABLED,
                )

                if not result.success:
                    self.logger.error(
                        f"Failed to load news page: {result.error_message}"
                    )
                    return articles

                # Extract article URLs from the news page
                article_urls = []
                for link in result.links:
                    if "/news/" in link["href"] or "/article/" in link["href"]:
                        full_url = urljoin(site_config["base_url"], link["href"])
                        if full_url not in article_urls:
                            article_urls.append(full_url)

                # Limit the number of articles to crawl
                article_urls = article_urls[: max_pages * 10]  # 10 articles per "page"

                self.logger.info(f"Found {len(article_urls)} article URLs")

                # Crawl each article
                for i, article_url in enumerate(
                    article_urls[: max_pages * 5]
                ):  # Limit for demo
                    try:
                        self.logger.info(
                            f"Crawling article {i+1}/{len(article_urls)}: {article_url}"
                        )

                        article_result = await crawler.arun(
                            url=article_url,
                            cache_mode=CacheMode.ENABLED,
                            word_count_threshold=50,
                        )

                        if article_result.success:
                            article_data = {
                                "url": article_url,
                                "title": article_result.metadata.get("title", ""),
                                "content": article_result.markdown,
                                "links": article_result.links,
                                "metadata": article_result.metadata,
                                "crawl_timestamp": datetime.now().isoformat(),
                                "source_site": site_config["base_url"],
                            }
                            articles.append(article_data)

                        # Add delay between requests
                        await asyncio.sleep(1)

                    except Exception as e:
                        self.logger.error(f"Failed to crawl article {article_url}: {e}")
                        continue

            except Exception as e:
                self.logger.error(f"Exception during multi-page crawling: {e}")

        self.logger.info(
            f"Completed multi-page crawling. Extracted {len(articles)} articles"
        )
        return articles

    async def content_filtering_and_cleaning(self, url: str) -> Dict[str, Any]:
        """
        Advanced content filtering and cleaning for crypto news

        Args:
            url: URL to crawl

        Returns:
            Dict containing filtered and cleaned content
        """
        self.logger.info(f"Performing content filtering and cleaning for: {url}")

        # Create content filter strategy
        content_filter = PruningContentFilter(
            threshold=0.48,
            threshold_type="fixed",
            min_word_threshold=10,
            # Remove common non-content elements
            excluded_tags=["nav", "footer", "aside", "advertisement", "sidebar"],
            excluded_classnames=[
                "ad",
                "advertisement",
                "sidebar",
                "navigation",
                "menu",
                "footer",
                "header",
                "popup",
                "modal",
            ],
            excluded_ids=["ads", "sidebar", "navigation", "footer"],
        )

        # Create chunking strategy for better content organization
        chunking_strategy = NlpSentenceChunking(max_length=1000)

        async with AsyncWebCrawler(
            headless=self.headless,
            browser_type=self.browser_type,
        ) as crawler:
            try:
                result = await crawler.arun(
                    url=url,
                    content_filter=content_filter,
                    chunking_strategy=chunking_strategy,
                    cache_mode=CacheMode.ENABLED,
                    word_count_threshold=50,
                )

                if result.success:
                    # Additional crypto-specific content filtering
                    filtered_content = self._filter_crypto_content(result.markdown)

                    extracted_data = {
                        "url": url,
                        "title": result.metadata.get("title", ""),
                        "original_content": result.markdown,
                        "filtered_content": filtered_content,
                        "content_chunks": (
                            result.chunks if hasattr(result, "chunks") else []
                        ),
                        "word_count": len(filtered_content.split()),
                        "metadata": result.metadata,
                        "crawl_timestamp": datetime.now().isoformat(),
                        "success": True,
                    }

                    self.logger.info(f"Content filtered and cleaned successfully")
                    return extracted_data
                else:
                    self.logger.error(f"Failed to crawl {url}: {result.error_message}")
                    return {"url": url, "success": False, "error": result.error_message}

            except Exception as e:
                self.logger.error(f"Exception during content filtering: {e}")
                return {"url": url, "success": False, "error": str(e)}

    async def structured_data_extraction(self, url: str) -> Dict[str, Any]:
        """
        Extract structured data (JSON-LD, OpenGraph, etc.) from crypto news

        Args:
            url: URL to crawl

        Returns:
            Dict containing structured data
        """
        self.logger.info(f"Extracting structured data from: {url}")

        # JavaScript to extract structured data
        structured_data_js = """
        const extractStructuredData = () => {
            const data = {
                jsonLd: [],
                openGraph: {},
                twitter: {},
                meta: {},
                breadcrumbs: [],
                schema: {}
            };
            
            // Extract JSON-LD
            const jsonLdScripts = document.querySelectorAll('script[type="application/ld+json"]');
            jsonLdScripts.forEach(script => {
                try {
                    data.jsonLd.push(JSON.parse(script.textContent));
                } catch (e) {
                    console.log('Failed to parse JSON-LD:', e);
                }
            });
            
            // Extract OpenGraph data
            const ogTags = document.querySelectorAll('meta[property^="og:"]');
            ogTags.forEach(tag => {
                const property = tag.getAttribute('property');
                const content = tag.getAttribute('content');
                if (property && content) {
                    data.openGraph[property] = content;
                }
            });
            
            // Extract Twitter Card data
            const twitterTags = document.querySelectorAll('meta[name^="twitter:"]');
            twitterTags.forEach(tag => {
                const name = tag.getAttribute('name');
                const content = tag.getAttribute('content');
                if (name && content) {
                    data.twitter[name] = content;
                }
            });
            
            // Extract other meta tags
            const metaTags = document.querySelectorAll('meta[name]');
            metaTags.forEach(tag => {
                const name = tag.getAttribute('name');
                const content = tag.getAttribute('content');
                if (name && content) {
                    data.meta[name] = content;
                }
            });
            
            // Extract breadcrumbs
            const breadcrumbSelectors = [
                '.breadcrumb a', '.breadcrumbs a', '[data-breadcrumb] a',
                'nav[aria-label="breadcrumb"] a', '.breadcrumb-item'
            ];
            
            breadcrumbSelectors.forEach(selector => {
                const elements = document.querySelectorAll(selector);
                if (elements.length > 0) {
                    data.breadcrumbs = Array.from(elements).map(el => ({
                        text: el.textContent.trim(),
                        href: el.href || null
                    }));
                }
            });
            
            return data;
        };
        
        return extractStructuredData();
        """

        async with AsyncWebCrawler(
            headless=self.headless,
            browser_type=self.browser_type,
        ) as crawler:
            try:
                result = await crawler.arun(
                    url=url,
                    js_code=structured_data_js,
                    wait_for_js=True,
                    cache_mode=CacheMode.ENABLED,
                )

                if result.success:
                    extracted_data = {
                        "url": url,
                        "title": result.metadata.get("title", ""),
                        "content": result.markdown,
                        "structured_data": result.js_execution_result,
                        "metadata": result.metadata,
                        "links": result.links,
                        "media": result.media,
                        "crawl_timestamp": datetime.now().isoformat(),
                        "success": True,
                    }

                    self.logger.info("Successfully extracted structured data")
                    return extracted_data
                else:
                    self.logger.error(f"Failed to crawl {url}: {result.error_message}")
                    return {"url": url, "success": False, "error": result.error_message}

            except Exception as e:
                self.logger.error(f"Exception during structured data extraction: {e}")
                return {"url": url, "success": False, "error": str(e)}

    async def sentiment_focused_extraction(self, url: str) -> Dict[str, Any]:
        """
        Extract content specifically optimized for sentiment analysis

        Args:
            url: URL to crawl

        Returns:
            Dict containing sentiment-focused extracted content
        """
        self.logger.info(f"Performing sentiment-focused extraction for: {url}")

        # Create extraction strategy focused on sentiment analysis
        sentiment_extraction = JsonCssExtractionStrategy(
            schema={
                "headline": "h1, .headline, .title",
                "summary": ".summary, .excerpt, .lead",
                "main_content": ".content, .article-body, .post-content",
                "quotes": "blockquote, .quote",
                "author_opinion": ".author-note, .editorial-note",
                "market_data": ".price, .market-cap, .volume, .change",
                "related_tags": ".tags a, .categories a",
                "social_signals": ".share-count, .likes, .comments-count",
            }
        )

        # Chunking strategy optimized for sentiment analysis
        sentiment_chunking = RegexChunking(
            patterns=[
                r"\. (?=[A-Z])",  # Split on sentences
                r"\n\n",  # Split on paragraphs
                r"(?<=\.)\s+(?=[A-Z])",  # Split on sentence boundaries
            ]
        )

        async with AsyncWebCrawler(
            headless=self.headless,
            browser_type=self.browser_type,
        ) as crawler:
            try:
                result = await crawler.arun(
                    url=url,
                    extraction_strategy=sentiment_extraction,
                    chunking_strategy=sentiment_chunking,
                    cache_mode=CacheMode.ENABLED,
                    word_count_threshold=20,
                )

                if result.success:
                    # Parse extracted JSON for sentiment analysis
                    try:
                        sentiment_data = json.loads(result.extracted_content)
                    except json.JSONDecodeError:
                        sentiment_data = {"raw_content": result.extracted_content}

                    # Additional processing for sentiment analysis
                    processed_content = self._process_for_sentiment(
                        sentiment_data, result.markdown
                    )

                    extracted_data = {
                        "url": url,
                        "sentiment_optimized_content": processed_content,
                        "content_chunks": (
                            result.chunks if hasattr(result, "chunks") else []
                        ),
                        "raw_markdown": result.markdown,
                        "metadata": result.metadata,
                        "crawl_timestamp": datetime.now().isoformat(),
                        "success": True,
                    }

                    self.logger.info("Successfully extracted sentiment-focused content")
                    return extracted_data
                else:
                    self.logger.error(f"Failed to crawl {url}: {result.error_message}")
                    return {"url": url, "success": False, "error": result.error_message}

            except Exception as e:
                self.logger.error(f"Exception during sentiment extraction: {e}")
                return {"url": url, "success": False, "error": str(e)}

    async def performance_optimized_crawling(
        self, urls: List[str], max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Performance-optimized crawling for multiple URLs

        Args:
            urls: List of URLs to crawl
            max_concurrent: Maximum concurrent crawlers

        Returns:
            List of extracted content from all URLs
        """
        self.logger.info(
            f"Starting performance-optimized crawling for {len(urls)} URLs"
        )

        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def crawl_with_semaphore(url: str) -> Dict[str, Any]:
            async with semaphore:
                async with AsyncWebCrawler(
                    headless=self.headless,
                    browser_type=self.browser_type,
                    # Performance optimizations
                    max_depth=1,
                    ignore_images=True,
                    ignore_stylesheets=True,
                ) as crawler:
                    try:
                        result = await crawler.arun(
                            url=url,
                            cache_mode=CacheMode.ENABLED,
                            word_count_threshold=30,
                            # Performance settings
                            bypass_cache=False,
                        )

                        if result.success:
                            return {
                                "url": url,
                                "title": result.metadata.get("title", ""),
                                "content": result.markdown[
                                    :5000
                                ],  # Limit content for performance
                                "word_count": len(result.markdown.split()),
                                "links_count": len(result.links),
                                "crawl_timestamp": datetime.now().isoformat(),
                                "success": True,
                            }
                        else:
                            return {
                                "url": url,
                                "success": False,
                                "error": result.error_message,
                            }
                    except Exception as e:
                        return {"url": url, "success": False, "error": str(e)}

        # Execute concurrent crawling
        tasks = [crawl_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log results
        successful_results = []
        for result in results:
            if isinstance(result, dict):
                successful_results.append(result)
            else:
                self.logger.error(f"Exception in concurrent crawling: {result}")

        successful_count = sum(1 for r in successful_results if r.get("success"))
        self.logger.info(
            f"Performance crawling completed: {successful_count}/{len(urls)} successful"
        )

        return successful_results

    def _filter_crypto_content(self, content: str) -> str:
        """
        Filter content to focus on crypto-relevant information

        Args:
            content: Raw content to filter

        Returns:
            Filtered content
        """
        # Crypto-related keywords to prioritize
        crypto_keywords = [
            "bitcoin",
            "ethereum",
            "crypto",
            "blockchain",
            "defi",
            "nft",
            "trading",
            "price",
            "market",
            "investment",
            "mining",
            "wallet",
            "exchange",
            "token",
            "coin",
            "altcoin",
            "hodl",
            "bull",
            "bear",
        ]

        lines = content.split("\n")
        filtered_lines = []

        for line in lines:
            line_lower = line.lower()
            # Keep lines that contain crypto keywords or are substantial
            if (
                any(keyword in line_lower for keyword in crypto_keywords)
                or len(line.split()) > 10
            ):
                filtered_lines.append(line)

        return "\n".join(filtered_lines)

    def _process_for_sentiment(
        self, structured_data: Dict, raw_content: str
    ) -> Dict[str, Any]:
        """
        Process extracted data for optimal sentiment analysis

        Args:
            structured_data: Structured extracted data
            raw_content: Raw markdown content

        Returns:
            Processed data optimized for sentiment analysis
        """
        return {
            "headline": structured_data.get("headline", ""),
            "summary": structured_data.get("summary", ""),
            "main_content": structured_data.get("main_content", ""),
            "quotes": structured_data.get("quotes", []),
            "sentiment_text": f"{structured_data.get('headline', '')} {structured_data.get('summary', '')} {structured_data.get('main_content', '')}",
            "market_mentions": structured_data.get("market_data", ""),
            "author_stance": structured_data.get("author_opinion", ""),
            "tags": structured_data.get("related_tags", []),
            "full_content_preview": raw_content[:1000],  # First 1000 chars for context
        }
