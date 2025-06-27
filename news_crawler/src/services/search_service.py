# services/search_service.py

"""
Search service layer for business logic.
"""

from typing import List
from datetime import datetime, timezone

from ..interfaces.search_engine import SearchEngine, SearchEngineError
from ..interfaces.message_broker import MessageBroker
from ..schemas.search_schemas import (
    SearchRequestSchema,
    SearchResponseSchema,
    SearchResultSchema,
)
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

        # Initialize cache for search results with enhanced configuration
        try:
            self._cache = CacheFactory.get_cache(
                name="search_cache",
                cache_type=CacheType.REDIS,
                host="localhost",
                port=6379,
                db=0,
                key_prefix="search:",
                serialization="json",
                max_size=1000,
                default_ttl=300,  # 5 minutes
            )
            logger.info("Search service initialized with Redis cache")
        except Exception as cache_error:
            logger.warning(
                f"Failed to initialize Redis cache: {cache_error}, falling back to memory cache"
            )
            self._cache = CacheFactory.get_cache(
                name="search_cache_memory",
                cache_type=CacheType.REDIS,
                max_size=1000,
                default_ttl=300,
            )
            logger.info("Search service initialized with memory cache fallback")

        logger.info("Search service initialized successfully")

    @cache_result(ttl=300, cache_type=CacheType.REDIS, key_prefix="search_results:")
    async def search_news(self, request: SearchRequestSchema) -> SearchResponseSchema:
        """
        Search for news articles with business logic applied.

        Args:
            request: Search parameters

        Returns:
            SearchResponseSchema: Search results
        """
        logger.info(
            f"Processing search request: {request.query} (crawler: {request.enable_crawler})"
        )

        # Generate cache key for this request
        cache_key = self._generate_cache_key(request)

        # Try to get from cache
        # try:
        #     cached_result = self._cache.get(cache_key)
        #     if cached_result is not None:
        #         logger.info(f"Cache hit for search request: {request.query}")
        #         # Convert dict back to schema
        #         return SearchResponseSchema(**cached_result)
        # except Exception as e:
        #     logger.warning(f"Cache get failed: {e}")

        # Validate and enhance request
        enhanced_request = self._enhance_search_request(request)

        try:
            # Perform initial search - Convert schema to dict for search engine
            search_engine_request = enhanced_request.model_dump()
            response = await self.search_engine.search(search_engine_request)
            logger.info(
                f"Initial search returned {response.get('total_results', 0)} results"
            )

            # Convert search engine response to schema
            schema_response = self._convert_to_schema_response(
                response, enhanced_request
            )

            # Apply business logic filters
            filtered_response = self._filter_results(schema_response)

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

            # Cache the result
            # try:
            #     cache_data = filtered_response.model_dump()
            #     self._cache.set(cache_key, cache_data, ttl=300)
            #     logger.debug(f"Cached search result for key: {cache_key}")
            # except Exception as e:
            #     logger.warning(f"Failed to cache result: {e}")

            # Publish search event for analytics - with better error handling
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

    def _generate_cache_key(self, request: SearchRequestSchema) -> str:
        """Generate cache key for search request"""
        import hashlib
        import json

        # Create a normalized representation of the request
        cache_data = {
            "query": request.query,
            "topic": request.topic,
            "search_depth": request.search_depth,
            "time_range": request.time_range,
            "max_results": request.max_results,
            "enable_crawler": request.enable_crawler,
            "include_answer": request.include_answer,
            "chunks_per_source": request.chunks_per_source,
        }

        # Sort keys for consistent hashing
        normalized_str = json.dumps(cache_data, sort_keys=True)
        cache_hash = hashlib.md5(normalized_str.encode()).hexdigest()
        return f"search_result:{cache_hash}"

    async def search_financial_sentiment(
        self, symbol: str, days: int = 7
    ) -> SearchResponseSchema:
        """
        Search for financial sentiment about a specific symbol.

        Args:
            symbol: Financial symbol (e.g., "BTC", "AAPL")
            days: Number of days to look back

        Returns:
            SearchResponseSchema: Sentiment-related search results
        """
        logger.info(f"Searching financial sentiment for {symbol} ({days} days)")

        # Construct financial sentiment query
        time_range = self._get_time_range(days)
        query = f"{symbol} market sentiment analysis price movement"

        request = SearchRequestSchema(
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

    async def get_trending_topics(self, topic: str = "finance") -> SearchResponseSchema:
        """
        Get trending topics in a specific domain.

        Args:
            topic: Topic category

        Returns:
            SearchResponseSchema: Trending topics
        """
        logger.info(f"Fetching trending topics for: {topic}")

        query = f"trending {topic} news today market analysis"

        request = SearchRequestSchema(
            query=query,
            topic=topic,
            search_depth="advanced",
            time_range="day",
            include_answer=True,
            max_results=15,
            enable_crawler=False,  # Disable crawler for trending topics
        )

        return await self.search_news(request)

    def _convert_to_schema_response(
        self, search_engine_response: dict, request: SearchRequestSchema
    ) -> SearchResponseSchema:
        """
        Convert search engine response to our schema format.

        Args:
            search_engine_response: Raw response from search engine
            request: Original request for context

        Returns:
            SearchResponseSchema: Converted response
        """
        # Convert results to schema format
        results = []
        for result_data in search_engine_response.get("results", []):
            result = SearchResultSchema(
                url=result_data["url"],
                title=result_data["title"],
                content=result_data["content"],
                score=result_data["score"],
                published_at=result_data.get("published_at"),
                source=result_data.get("source"),
                is_crawled=result_data.get("is_crawled", False),
                metadata=result_data.get("metadata"),
            )
            results.append(result)

        return SearchResponseSchema(
            query=search_engine_response["query"],
            total_results=search_engine_response["total_results"],
            results=results,
            answer=search_engine_response.get("answer"),
            follow_up_questions=search_engine_response.get("follow_up_questions"),
            response_time=search_engine_response["response_time"],
            search_depth=request.search_depth,
            topic=request.topic,
            time_range=request.time_range,
            crawler_used=search_engine_response.get("crawler_used", False),
        )

    async def _deep_crawl_search(
        self, request: SearchRequestSchema, initial_response: SearchResponseSchema
    ) -> List[SearchResultSchema]:
        """
        Perform deep crawling on search results for additional content.

        Args:
            request: Original search request
            initial_response: Initial search response

        Returns:
            List[SearchResultSchema]: Additional crawled results
        """
        logger.info("Starting deep crawl for additional content")

        try:
            # Extract URLs from initial results for crawling
            urls_to_crawl = [str(result.url) for result in initial_response.results]

            # Perform deep crawling
            crawled_articles = []
            for url in urls_to_crawl[:5]:  # Limit to top 5 for performance
                try:
                    # Check if already crawled
                    if await self.article_repository.article_exists(url):
                        continue

                    # Use crawler service to get enhanced content
                    article = await self.crawler_service.crawl_single_url(url)
                    if article:
                        crawled_articles.append(article)

                except Exception as e:
                    logger.warning(f"Failed to crawl {url}: {str(e)}")
                    continue

            # Convert crawled articles to search results
            search_results = []
            for article in crawled_articles:
                result = SearchResultSchema(
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
        self,
        original_response: SearchResponseSchema,
        crawled_results: List[SearchResultSchema],
    ) -> SearchResponseSchema:
        """
        Merge original search results with crawled results.

        Args:
            original_response: Original search response
            crawled_results: Additional crawled results

        Returns:
            SearchResponseSchema: Merged response
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
        self, request: SearchRequestSchema, response: SearchResponseSchema
    ) -> None:
        """
        Publish search event for analytics with improved error handling.

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
                "crawler_used": response.crawler_used,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # First ensure exchange exists
            exchange_name = "analytics_exchange"
            try:
                logger.debug(f"Creating exchange: {exchange_name}")
                await self.message_broker.create_exchange(exchange_name, "topic")
                logger.debug(f"Exchange {exchange_name} created/verified successfully")
            except Exception as exchange_error:
                logger.warning(
                    f"Failed to create exchange {exchange_name}: {exchange_error}"
                )
                # Continue with publish attempt anyway

            # Try to publish the event
            try:
                success = await self.message_broker.publish(
                    exchange=exchange_name,
                    routing_key="search.event",
                    message=event,
                )
                if success:
                    logger.debug("Search event published successfully")
                else:
                    logger.warning("Search event publish returned False")

            except Exception as pub_error:
                logger.warning(f"Failed to publish search event: {str(pub_error)}")
                # Don't fail the search operation for analytics issues

        except Exception as e:
            logger.warning(f"Failed to prepare/publish search event: {str(e)}")
            # Analytics publishing should never fail the main search operation

    def _enhance_search_request(
        self, request: SearchRequestSchema
    ) -> SearchRequestSchema:
        """
        Enhance search request with defaults and optimizations.

        Args:
            request: Original search request

        Returns:
            SearchRequestSchema: Enhanced request
        """
        # Create a copy to avoid mutating the original
        enhanced_data = request.model_dump()

        # Set default topic for financial queries
        if not enhanced_data.get("topic") and any(
            keyword in request.query.lower()
            for keyword in ["btc", "bitcoin", "stock", "market", "trading"]
        ):
            enhanced_data["topic"] = "finance"

        # Enhance query for better financial results
        if (
            enhanced_data.get("topic") == "finance"
            and "sentiment" not in request.query.lower()
        ):
            enhanced_data["query"] = f"{request.query} market analysis"

        return SearchRequestSchema(**enhanced_data)

    def _filter_results(self, response: SearchResponseSchema) -> SearchResponseSchema:
        """
        Apply business logic filters to search results.

        Args:
            response: Original search response

        Returns:
            SearchResponseSchema: Filtered response
        """
        # Apply more lenient filtering to avoid empty results
        min_score = 0.3  # Reduced from 0.7 to be more inclusive
        min_content_length = 50  # Reduced from 100

        filtered_results = [
            result
            for result in response.results
            if result.score >= min_score
            and len(result.content.strip()) > min_content_length
        ]

        # If filtering results in too few results, be more lenient
        if len(filtered_results) < 3 and len(response.results) > 0:
            # Use even more lenient criteria
            filtered_results = [
                result
                for result in response.results
                if result.score >= 0.1 and len(result.content.strip()) > 20
            ]

        # Sort by score and recency
        filtered_results.sort(
            key=lambda x: (
                x.score,
                x.published_at or datetime.min.replace(tzinfo=timezone.utc),
            ),
            reverse=True,
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
        """Check service health with cache health check."""
        try:
            # Check search engine health
            search_engine_healthy = await self.search_engine.health_check()
            logger.debug(f"Search engine health: {search_engine_healthy}")

            # Check message broker health
            broker_healthy = await self.message_broker.health_check()
            logger.debug(f"Message broker health: {broker_healthy}")

            # Check crawler service health
            crawler_healthy = (
                self.crawler_service.get_crawler_stats()["total_crawlers"] >= 0
            )
            logger.debug(f"Crawler service health: {crawler_healthy}")

            # Check cache health
            cache_healthy = True
            try:
                # Test cache operations
                test_key = "health_check_test"
                test_value = {"timestamp": datetime.now().isoformat()}

                # Test set
                set_result = self._cache.set(test_key, test_value, ttl=60)
                if not set_result:
                    cache_healthy = False
                    logger.warning("Cache set operation failed")

                # Test get
                get_result = self._cache.get(test_key)
                if get_result != test_value:
                    cache_healthy = False
                    logger.warning("Cache get operation failed")

                # Test delete
                delete_result = self._cache.delete(test_key)
                if not delete_result:
                    logger.warning("Cache delete operation failed (non-critical)")

                logger.debug(f"Cache health: {cache_healthy}")

            except Exception as cache_error:
                logger.warning(f"Cache health check failed: {cache_error}")
                cache_healthy = False

            is_healthy = (
                search_engine_healthy
                and broker_healthy
                and crawler_healthy
                and cache_healthy
            )

            logger.debug(f"Overall health check completed: {is_healthy}")
            return is_healthy

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
