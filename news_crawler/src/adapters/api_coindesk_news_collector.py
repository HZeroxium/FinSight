# adapters/api_coindesk_news_collector.py

import asyncio
import random
from typing import Optional, Dict, Any, List
import aiohttp
from datetime import datetime, timezone

from ..interfaces.news_collector_interface import NewsCollectorInterface
from ..schemas.news_schemas import NewsCollectionResult, NewsCollectorConfig
from ..core.api_parsing_strategies import get_api_parsing_strategy
from common.logger import LoggerFactory, LoggerType, LogLevel
from ..utils.browser_session import BrowserSession


class APICoinDeskNewsCollector(NewsCollectorInterface):
    """CoinDesk API news collector with time-based pagination and anti-detection features"""

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

        self.logger.info(
            f"Initialized CoinDesk API collector for {config.source.value}"
        )

    async def collect_news(
        self,
        max_items: Optional[int] = None,
        to_timestamp: Optional[int] = None,
        limit: int = 10,
        lang: str = "EN",
        source_ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        exclude_categories: Optional[List[str]] = None,
    ) -> NewsCollectionResult:
        """
        Collect news items from CoinDesk API endpoint

        Args:
            max_items: Maximum number of items to collect
            to_timestamp: Unix timestamp for pagination (get articles before this time)
            limit: Number of items per request
            lang: Language filter (EN, ES, TR, FR, JP, PT)
            source_ids: List of source keys to include
            categories: List of categories to include
            exclude_categories: List of categories to exclude

        Returns:
            NewsCollectionResult with collected items
        """
        actual_limit = (
            min(max_items or self.config.max_items or limit, limit)
            if max_items
            else limit
        )

        self.logger.info(
            f"Starting news collection from {self.config.source.value} "
            f"(to_timestamp: {to_timestamp}, limit: {actual_limit})"
        )

        try:
            # Apply rate limiting
            await self._apply_rate_limiting()

            # Fetch data from API endpoint
            api_data = await self._fetch_api_data(
                to_timestamp=to_timestamp,
                limit=actual_limit,
                lang=lang,
                source_ids=source_ids,
                categories=categories,
                exclude_categories=exclude_categories,
            )

            self.logger.debug(
                f"API response len: {len(api_data["Data"])} (to_timestamp: {to_timestamp}, "
                f"limit: {actual_limit})"
            )

            self.logger.debug(
                f"API Response First 5 Items: {api_data.get('Data', [])[:5]}..."  # Log first 5 items for brevity
            )

            # Parse API response
            items = self.parsing_strategy.parse_response(api_data, self.config.source)

            self.logger.info(f"Successfully fetched {len(api_data["Data"])} news items")
            self.logger.info(f"Successfully processed {len(items)} news items")
            self.logger.debug(f"Items: {[item.title[:50] + '...' for item in items]}")

            return NewsCollectionResult(
                source=self.config.source, items=items, success=True
            )

        except Exception as e:
            self.logger.error(f"Failed to collect news: {e}")
            return NewsCollectionResult(
                source=self.config.source, items=[], success=False, error_message=str(e)
            )

    async def _fetch_api_data(
        self,
        to_timestamp: Optional[int] = None,
        limit: int = 10,
        lang: str = "EN",
        source_ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        exclude_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fetch data from CoinDesk API endpoint with anti-detection

        Args:
            to_timestamp: Unix timestamp for pagination
            limit: Number of items to fetch
            lang: Language filter
            source_ids: List of source keys to include
            categories: List of categories to include
            exclude_categories: List of categories to exclude

        Returns:
            API response data
        """
        # Rotate session periodically
        if self.request_count > 0 and self.request_count % 15 == 0:
            self.browser_session = BrowserSession()
            self.logger.debug("Rotated browser session")

        # Get realistic headers for REST API
        headers = self.browser_session.get_headers(is_graphql=False)
        headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json; charset=UTF-8",
            }
        )

        # Prepare query parameters
        params = self._prepare_request_params(
            to_timestamp=to_timestamp,
            limit=limit,
            lang=lang,
            source_ids=source_ids,
            categories=categories,
            exclude_categories=exclude_categories,
        )

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

                    async with session.get(
                        str(self.config.url),
                        params=params,
                        headers=headers,
                        ssl=False,
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
                            headers = self.browser_session.get_headers(is_graphql=False)
                            headers.update(
                                {
                                    "Accept": "application/json",
                                    "Content-Type": "application/json; charset=UTF-8",
                                }
                            )

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

    def _prepare_request_params(
        self,
        to_timestamp: Optional[int] = None,
        limit: int = 10,
        lang: str = "EN",
        source_ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        exclude_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare request parameters for CoinDesk API

        Args:
            to_timestamp: Unix timestamp for pagination
            limit: Number of items to fetch
            lang: Language filter
            source_ids: List of source keys to include
            categories: List of categories to include
            exclude_categories: List of categories to exclude

        Returns:
            Request parameters dictionary
        """
        params = {
            "lang": lang,
            "limit": limit,
        }

        if to_timestamp:
            params["to_ts"] = to_timestamp

        if source_ids:
            params["source_ids"] = ",".join(source_ids)

        if categories:
            params["categories"] = ",".join(categories)

        if exclude_categories:
            params["exclude_categories"] = ",".join(exclude_categories)

        return params

    async def _apply_rate_limiting(self) -> None:
        """Apply intelligent rate limiting"""
        current_time = datetime.now(timezone.utc).timestamp()

        if self.last_request_time > 0:
            time_since_last = current_time - self.last_request_time
            min_delay = 2.0  # Minimum 2 seconds between requests for CoinDesk

            if time_since_last < min_delay:
                wait_time = min_delay - time_since_last
                # Add random jitter
                wait_time += random.uniform(0.5, 2.0)
                self.logger.debug(f"Rate limiting: waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)

    def is_available(self) -> bool:
        """Check if CoinDesk API endpoint is available"""
        try:
            import requests

            headers = self.browser_session.get_headers(is_graphql=False)
            headers.update(
                {
                    "Accept": "application/json",
                    "Content-Type": "application/json; charset=UTF-8",
                }
            )

            response = requests.get(
                str(self.config.url),
                params={"lang": "EN", "limit": 1},
                headers=headers,
                timeout=10,
                verify=False,
            )

            return response.status_code == 200

        except Exception as e:
            self.logger.error(f"Availability check failed: {e}")
            return False

    def get_source_info(self) -> Dict[str, Any]:
        """Get information about the CoinDesk API source"""
        return {
            "source": self.config.source.value,
            "url": str(self.config.url),
            "timeout": self.config.timeout,
            "max_items": self.config.max_items,
            "user_agent": self.browser_session.user_agent,
            "retry_attempts": self.config.retry_attempts,
            "retry_delay": self.config.retry_delay,
            "type": "api_rest",
            "session_id": self.browser_session.session_id,
            "request_count": self.request_count,
            "pagination_type": "timestamp_based",
        }
