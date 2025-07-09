# adapters/api_cointelegraph_news_collector.py

import asyncio
import random
from typing import Optional, Dict, Any
import aiohttp
from datetime import datetime, timezone

from ..interfaces.news_collector_interface import NewsCollectorInterface
from ..schemas.news_schemas import NewsCollectionResult, NewsCollectorConfig
from ..core.api_parsing_strategies import get_api_parsing_strategy
from common.logger import LoggerFactory, LoggerType, LogLevel
from ..utils.browser_session import BrowserSession


class APICoinTelegraphNewsCollector(NewsCollectorInterface):
    """Advanced API-based news collector with anti-detection features"""

    def __init__(self, config: NewsCollectorConfig):
        super().__init__(config)
        self.logger = LoggerFactory.get_logger(
            name=f"api-collector-{config.source.value}",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file=f"logs/api_collector_{config.source.value}.log",
        )
        self.parsing_strategy = get_api_parsing_strategy(config.source)
        self.browser_session = BrowserSession()
        self.request_count = 0
        self.last_request_time = 0

        self.logger.info(f"Initialized API collector for {config.source.value}")

    async def collect_news(
        self,
        max_items: Optional[int] = None,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> NewsCollectionResult:
        """
        Collect news items from API endpoint with anti-detection measures

        Args:
            max_items: Maximum number of items to collect (takes precedence)
            offset: Starting offset for pagination
            limit: Number of items per request (fallback, usually for API pagination)

        Returns:
            NewsCollectionResult with collected items
        """
        # Priority: max_items > config.max_items > limit > default
        actual_limit = (
            max_items
            or self.config.max_items
            or limit
            or 10  # Default fallback for API pagination
        )

        self.logger.info(
            f"Starting news collection from {self.config.source.value} "
            f"(offset: {offset}, limit: {actual_limit})"
        )

        try:
            # Apply rate limiting
            await self._apply_rate_limiting()

            # Fetch data from API endpoint
            api_data = await self._fetch_api_data(offset, actual_limit)

            # Parse API response
            items = self.parsing_strategy.parse_response(api_data, self.config.source)

            self.logger.info(f"Successfully collected {len(items)} news items")
            self.logger.debug(f"Items: {[item.title[:50] + '...' for item in items]}")

            return NewsCollectionResult(
                source=self.config.source, items=items, success=True
            )

        except Exception as e:
            self.logger.error(f"Failed to collect news: {e}")
            return NewsCollectionResult(
                source=self.config.source, items=[], success=False, error_message=str(e)
            )

    async def _fetch_api_data(self, offset: int, limit: int) -> Dict[str, Any]:
        """
        Fetch data from API endpoint with advanced anti-detection

        Args:
            offset: Starting offset for pagination
            limit: Number of items to fetch

        Returns:
            API response data
        """
        # Rotate session periodically
        if self.request_count > 0 and self.request_count % 10 == 0:
            self.browser_session = BrowserSession()
            self.logger.debug("Rotated browser session")

        # Get realistic headers
        headers = self.browser_session.get_headers(is_graphql=True)

        # Prepare request payload
        payload = self._prepare_request_payload(offset, limit)

        for attempt in range(self.config.retry_attempts):
            try:
                # Add jitter to timeout
                timeout_with_jitter = self.config.timeout + random.uniform(-2, 5)
                timeout = aiohttp.ClientTimeout(total=timeout_with_jitter)

                # Use connector with realistic settings
                connector = aiohttp.TCPConnector(
                    limit=10,
                    limit_per_host=2,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                )

                async with aiohttp.ClientSession(
                    timeout=timeout, connector=connector, cookie_jar=aiohttp.CookieJar()
                ) as session:

                    # Pre-flight request to establish session (optional)
                    if attempt == 0 and self.request_count == 0:
                        await self._warmup_session(session, headers)

                    async with session.post(
                        str(self.config.url),
                        json=payload,
                        headers=headers,
                        ssl=False,  # Disable SSL verification if needed
                    ) as response:

                        # Log response details
                        self.logger.debug(
                            f"Response: {response.status} from {response.url} "
                            f"(attempt {attempt + 1})"
                        )

                        if response.status == 200:
                            data = await response.json()
                            self.request_count += 1
                            self.last_request_time = datetime.now(
                                timezone.utc
                            ).timestamp()
                            return data

                        elif response.status == 403:
                            # Handle rate limiting / blocking
                            self.logger.warning(
                                f"Access forbidden (403) - rotating session"
                            )
                            self.browser_session = BrowserSession()
                            headers = self.browser_session.get_headers(is_graphql=True)

                        elif response.status == 429:
                            # Rate limited
                            retry_after = int(response.headers.get("Retry-After", 60))
                            self.logger.warning(
                                f"Rate limited - waiting {retry_after} seconds"
                            )
                            await asyncio.sleep(retry_after)

                        response.raise_for_status()

            except aiohttp.ClientError as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < self.config.retry_attempts - 1:
                    # Exponential backoff with jitter
                    base_delay = self.config.retry_delay * (2**attempt)
                    jitter = random.uniform(0.5, 1.5)
                    wait_time = base_delay * jitter

                    self.logger.info(f"Retrying in {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)

                    # Rotate session on retry
                    self.browser_session = BrowserSession()
                else:
                    raise Exception(
                        f"Failed to fetch API data after {self.config.retry_attempts} attempts: {e}"
                    )

    async def _warmup_session(
        self, session: aiohttp.ClientSession, headers: Dict[str, str]
    ) -> None:
        """
        Perform warmup request to establish realistic session

        Args:
            session: aiohttp session
            headers: Request headers
        """
        try:
            # Visit main page first to establish session
            main_headers = self.browser_session.get_headers(is_graphql=False)
            async with session.get(
                "https://cointelegraph.com/",
                headers=main_headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                self.logger.debug(f"Warmup request: {response.status}")
                # Small delay to simulate user behavior
                await asyncio.sleep(random.uniform(1, 3))

        except Exception as e:
            self.logger.debug(f"Warmup request failed (non-critical): {e}")

    async def _apply_rate_limiting(self) -> None:
        """Apply intelligent rate limiting"""
        current_time = datetime.now(timezone.utc).timestamp()

        if self.last_request_time > 0:
            time_since_last = current_time - self.last_request_time
            min_delay = 1.0  # Minimum 1 second between requests

            if time_since_last < min_delay:
                wait_time = min_delay - time_since_last
                # Add random jitter
                wait_time += random.uniform(0.5, 2.0)
                self.logger.debug(f"Rate limiting: waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)

    def _prepare_request_payload(self, offset: int, limit: int) -> Dict[str, Any]:
        """
        Prepare request payload for specific news source

        Args:
            offset: Starting offset for pagination
            limit: Number of items to fetch

        Returns:
            Request payload dictionary
        """
        if self.config.source.value == "cointelegraph":
            return {
                "query": """
                query CategoryPagePostsQuery(
                  $short: String,
                  $slug: String!,
                  $offset: Int = 0,
                  $length: Int = 10,
                  $hideFromMainPage: Boolean = null
                ) {
                  locale(short: $short) {
                    category(slug: $slug) {
                      id
                      posts(
                        order: "postPublishedTime"
                        offset: $offset
                        length: $length
                        hideFromMainPage: $hideFromMainPage
                      ) {
                        data {
                          id
                          slug
                          views
                          postTranslate {
                            id
                            title
                            avatar
                            published
                            publishedHumanFormat
                            leadText
                            author {
                              id
                              slug
                              innovationCircleUrl
                              authorTranslates {
                                id
                                name
                              }
                            }
                          }
                          category {
                            id
                            slug
                            categoryTranslates {
                              id
                              title
                            }
                          }
                          author {
                            id
                            slug
                            authorTranslates {
                              id
                              name
                            }
                          }
                          postBadge {
                            id
                            label
                            postBadgeTranslates {
                              id
                              title
                            }
                          }
                          showShares
                          showStats
                        }
                        postsCount
                      }
                    }
                  }
                }
                """,
                "variables": {
                    "cacheTimeInMS": 300000,
                    "hideFromMainPage": False,
                    "length": limit,
                    "offset": offset,
                    "short": "en",
                    "slug": "latest-news",
                },
            }
        else:
            raise ValueError(
                f"Unsupported source for API collection: {self.config.source.value}"
            )

    def is_available(self) -> bool:
        """Check if API endpoint is available with realistic request"""
        try:
            import requests

            headers = self.browser_session.get_headers(is_graphql=True)
            payload = self._prepare_request_payload(0, 1)

            response = requests.post(
                str(self.config.url),
                json=payload,
                headers=headers,
                timeout=10,
                verify=False,
            )

            return response.status_code == 200

        except Exception as e:
            self.logger.error(f"Availability check failed: {e}")
            return False

    def get_source_info(self) -> Dict[str, Any]:
        """Get information about the API source"""
        return {
            "source": self.config.source.value,
            "url": str(self.config.url),
            "timeout": self.config.timeout,
            "max_items": self.config.max_items,
            "user_agent": self.browser_session.user_agent,
            "retry_attempts": self.config.retry_attempts,
            "retry_delay": self.config.retry_delay,
            "type": "api",
            "session_id": self.browser_session.session_id,
            "request_count": self.request_count,
        }
