"""
Search service layer for business logic.
"""

from typing import Optional, List
from datetime import datetime, timedelta

from ..interfaces.search_engine import SearchEngine, SearchEngineError
from ..interfaces.message_broker import MessageBroker
from ..models.search import SearchRequest, SearchResponse, SearchResult
from ..services.crawler_service import CrawlerService
from ..repositories.article_repository import ArticleRepository
from ..common.logger import LoggerFactory, LoggerType, LogLevel
from ..common.cache import CacheFactory, CacheType, cache_result

logger = LoggerFactory.get_logger(
    name="search-service", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


class SearchService:
    """
    Service layer for search operations and business logic.
    """

    def __init__(
        self,
        search_engine: SearchEngine,
        message_broker: MessageBroker,
        article_repository: ArticleRepository,
        crawler_service: CrawlerService,
    ):
        """
        Initialize search service.

        Args:
            search_engine: Search engine implementation
            message_broker: Message broker for async communication
            article_repository: Article repository for storage
            crawler_service: Crawler service for deep content extraction
        """
        self.search_engine = search_engine
        self.message_broker = message_broker
        self.article_repository = article_repository
        self.crawler_service = crawler_service

        # Initialize cache for search results
        self._cache = CacheFactory.get_cache(
            name="search_cache",
            cache_type=CacheType.MEMORY,
            max_size=1000,
            default_ttl=300,  # 5 minutes
        )

        logger.info("Search service initialized successfully")

    @cache_result(ttl=300, key_prefix="search_")
    async def search_news(self, request: SearchRequest) -> SearchResponse:
        """
        Search for news articles with business logic applied.

        Args:
            request: Search parameters

        Returns:
            SearchResponse: Search results
        """
        logger.info(
            f"Processing search request: {request.query} (crawler: {request.enable_crawler})"
        )

        # Validate and enhance request
        enhanced_request = self._enhance_search_request(request)

        try:
            # Perform initial search
            response = await self.search_engine.search(enhanced_request)
            logger.info(f"Initial search returned {len(response.results)} results")

            # Apply business logic filters
            filtered_response = self._filter_results(response)

            # Deep crawl if enabled and needed
            if (
                request.enable_crawler
                and len(filtered_response.results) < request.max_results
            ):
                logger.info("Enabling deep crawler for additional results")
                crawled_results = await self._deep_crawl_search(
                    enhanced_request, filtered_response
                )
                filtered_response = self._merge_search_results(
                    filtered_response, crawled_results
                )

            # Publish search event for analytics
            await self._publish_search_event(request, filtered_response)

            logger.info(
                f"Search completed with {len(filtered_response.results)} final results"
            )
            return filtered_response

        except SearchEngineError as e:
            logger.error(f"Search failed: {e.message}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in search service: {str(e)}")
            raise SearchEngineError(f"Search service error: {str(e)}")

    async def search_financial_sentiment(
        self, symbol: str, days: int = 7
    ) -> SearchResponse:
        """
        Search for financial sentiment about a specific symbol.

        Args:
            symbol: Financial symbol (e.g., "BTC", "AAPL")
            days: Number of days to look back

        Returns:
            SearchResponse: Sentiment-related search results
        """
        logger.info(f"Searching financial sentiment for {symbol} ({days} days)")

        # Construct financial sentiment query
        time_range = self._get_time_range(days)
        query = f"{symbol} market sentiment analysis price movement"

        request = SearchRequest(
            query=query,
            topic="finance",
            search_depth="advanced",
            time_range=time_range,
            include_answer=True,
            max_results=20,
            chunks_per_source=5,
            enable_crawler=True,  # Enable deep crawling for financial data
        )

        return await self.search_news(request)

    async def get_trending_topics(self, topic: str = "finance") -> SearchResponse:
        """
        Get trending topics in a specific domain.

        Args:
            topic: Topic category

        Returns:
            SearchResponse: Trending topics
        """
        logger.info(f"Fetching trending topics for: {topic}")

        query = f"trending {topic} news today market analysis"

        request = SearchRequest(
            query=query,
            topic=topic,
            search_depth="advanced",
            time_range="day",
            include_answer=True,
            max_results=15,
            enable_crawler=False,  # Disable crawler for trending topics
        )

        return await self.search_news(request)

    async def _deep_crawl_search(
        self, request: SearchRequest, initial_response: SearchResponse
    ) -> List[SearchResult]:
        """
        Perform deep crawling on search results for additional content.

        Args:
            request: Original search request
            initial_response: Initial search response

        Returns:
            List[SearchResult]: Additional crawled results
        """
        logger.info("Starting deep crawl for additional content")

        try:
            # Extract URLs from initial results for crawling
            urls_to_crawl = [result.url for result in initial_response.results]

            # Perform deep crawling
            crawled_articles = []
            for url in urls_to_crawl[:5]:  # Limit to top 5 for performance
                try:
                    # Check if already crawled
                    if await self.article_repository.article_exists(str(url)):
                        continue

                    # Use crawler service to get enhanced content
                    article = await self.crawler_service.crawl_single_url(str(url))
                    if article:
                        crawled_articles.append(article)

                except Exception as e:
                    logger.warning(f"Failed to crawl {url}: {str(e)}")
                    continue

            # Convert crawled articles to search results
            search_results = []
            for article in crawled_articles:
                result = SearchResult(
                    url=article.url,
                    title=article.title,
                    content=article.content[:1000],  # Truncate for response
                    score=0.8,  # High score for crawled content
                    published_at=article.published_at,
                    source=article.source.domain,
                    is_crawled=True,
                )
                search_results.append(result)

            logger.info(
                f"Deep crawl completed with {len(search_results)} additional results"
            )
            return search_results

        except Exception as e:
            logger.error(f"Deep crawl failed: {str(e)}")
            return []

    def _merge_search_results(
        self, original_response: SearchResponse, crawled_results: List[SearchResult]
    ) -> SearchResponse:
        """
        Merge original search results with crawled results.

        Args:
            original_response: Original search response
            crawled_results: Additional crawled results

        Returns:
            SearchResponse: Merged response
        """
        # Combine results and remove duplicates
        all_results = list(original_response.results) + crawled_results

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if str(result.url) not in seen_urls:
                seen_urls.add(str(result.url))
                unique_results.append(result)

        # Sort by score and recency
        unique_results.sort(
            key=lambda x: (x.score, x.published_at or datetime.min), reverse=True
        )

        # Update response
        original_response.results = unique_results
        original_response.total_results = len(unique_results)
        original_response.crawler_used = len(crawled_results) > 0

        return original_response

    async def _publish_search_event(
        self, request: SearchRequest, response: SearchResponse
    ) -> None:
        """
        Publish search event for analytics.

        Args:
            request: Search request
            response: Search response
        """
        try:
            event = {
                "event_type": "search_performed",
                "query": request.query,
                "topic": request.topic,
                "results_count": len(response.results),
                "response_time": response.response_time,
                "crawler_used": getattr(response, "crawler_used", False),
                "timestamp": datetime.utcnow().isoformat(),
            }

            await self.message_broker.publish(
                exchange="analytics_exchange", routing_key="search.event", message=event
            )

        except Exception as e:
            logger.warning(f"Failed to publish search event: {str(e)}")

    def _enhance_search_request(self, request: SearchRequest) -> SearchRequest:
        """
        Enhance search request with defaults and optimizations.

        Args:
            request: Original search request

        Returns:
            SearchRequest: Enhanced request
        """
        # Set default topic for financial queries
        if not request.topic and any(
            keyword in request.query.lower()
            for keyword in ["btc", "bitcoin", "stock", "market", "trading"]
        ):
            request.topic = "finance"

        # Enhance query for better financial results
        if request.topic == "finance" and "sentiment" not in request.query.lower():
            request.query = f"{request.query} market analysis"

        return request

    def _filter_results(self, response: SearchResponse) -> SearchResponse:
        """
        Apply business logic filters to search results.

        Args:
            response: Original search response

        Returns:
            SearchResponse: Filtered response
        """
        # Filter out low-quality results
        min_score = 0.7
        filtered_results = [
            result
            for result in response.results
            if result.score >= min_score and len(result.content) > 100
        ]

        # Sort by score and recency
        filtered_results.sort(
            key=lambda x: (x.score, x.published_at or datetime.min), reverse=True
        )

        # Update response
        response.results = filtered_results
        response.total_results = len(filtered_results)

        return response

    def _get_time_range(self, days: int) -> str:
        """Convert days to time range string."""
        if days <= 1:
            return "day"
        elif days <= 7:
            return "week"
        elif days <= 30:
            return "month"
        else:
            return "year"

    async def health_check(self) -> bool:
        """Check service health."""
        try:
            # Check search engine health
            search_engine_healthy = await self.search_engine.health_check()

            # Check message broker health
            broker_healthy = await self.message_broker.health_check()

            # Check crawler service health
            crawler_healthy = (
                self.crawler_service.get_crawler_stats()["total_crawlers"] >= 0
            )

            is_healthy = search_engine_healthy and broker_healthy and crawler_healthy

            logger.debug(f"Health check completed: {is_healthy}")
            return is_healthy

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
