# services/sentiment_service.py

"""
Sentiment analysis service layer for business logic.
"""

from typing import List, Optional
from datetime import datetime, timezone
from urllib.parse import urlparse

from ..interfaces.sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisError
from ..interfaces.sentiment_repository import SentimentRepository
from ..interfaces.message_broker import MessageBroker
from ..models.sentiment import (
    SentimentRequest,
    SentimentAnalysisResult,
    ProcessedSentiment,
    SentimentQueryFilter,
    SentimentAggregation,
    SentimentLabel,
    SentimentScore,
)
from ..schemas.message_schemas import SentimentResultMessageSchema
from common.logger import LoggerFactory, LoggerType, LogLevel
from common.cache import CacheFactory, CacheType
from ..core.config import settings

logger = LoggerFactory.get_logger(
    name="sentiment-service", logger_type=LoggerType.STANDARD, level=LogLevel.DEBUG
)


class SentimentService:
    """
    Service layer for sentiment analysis operations and business logic.
    """

    def __init__(
        self,
        analyzer: SentimentAnalyzer,
        repository: SentimentRepository,
        message_broker: Optional[MessageBroker] = None,
    ):
        """
        Initialize sentiment service.

        Args:
            analyzer: Sentiment analyzer implementation
            repository: Sentiment repository for storage
            message_broker: Optional message broker for publishing results
        """
        self.analyzer = analyzer
        self.repository = repository
        self.message_broker = message_broker

        # Initialize cache for sentiment results
        try:
            self._cache = CacheFactory.get_cache(
                name="sentiment_cache",
                cache_type=CacheType.REDIS,
                host="localhost",
                port=6379,
                db=0,  # Different DB from news crawler
                key_prefix="sentiment:",
                serialization="json",
            )
            logger.info("Sentiment service initialized with Redis cache")
        except Exception as cache_error:
            logger.warning(
                f"Failed to initialize Redis cache: {cache_error}, falling back to memory cache"
            )
            self._cache = CacheFactory.get_cache(
                name="sentiment_cache_memory",
                cache_type=CacheType.MEMORY,
                max_size=1000,
                default_ttl=settings.cache_ttl_seconds,
            )
            logger.info("Sentiment service initialized with memory cache fallback")

        logger.info("Sentiment service initialized successfully")

    async def analyze_text(
        self,
        text: str,
        title: Optional[str] = None,
        article_id: Optional[str] = None,
        source_url: Optional[str] = None,
        save_result: bool = True,
        publish_result: Optional[bool] = None,  # Changed to Optional
        search_query: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> SentimentAnalysisResult:
        """
        Analyze sentiment of a single text with proper caching and optional message publishing.

        Args:
            text: Text content to analyze
            title: Optional title for context
            article_id: Optional article ID for storage
            source_url: Optional source URL
            save_result: Whether to save the result
            publish_result: Whether to publish result to message broker (None = use config default)
            search_query: Optional search query for context
            metadata: Optional metadata

        Returns:
            SentimentAnalysisResult: Analysis results
        """
        logger.info(f"Analyzing sentiment for text length: {len(text)}")

        try:
            import time
            import hashlib

            start_time = time.time()

            # Create cache key from text content and title
            cache_content = f"{text}||{title or ''}"
            cache_key = (
                f"sentiment_analysis:{hashlib.md5(cache_content.encode()).hexdigest()}"
            )

            # Try to get from cache first
            try:
                cached_result = self._cache.get(cache_key)
                if cached_result:
                    logger.debug(f"Cache hit for sentiment analysis: {cache_key}")
                    # Reconstruct SentimentAnalysisResult from cached dict
                    if isinstance(cached_result, dict):
                        return SentimentAnalysisResult.from_dict(cached_result)
                    return cached_result
                else:
                    logger.debug(f"Cache miss for sentiment analysis: {cache_key}")
            except Exception as cache_error:
                logger.warning(f"Cache get failed: {cache_error}")

            # Perform sentiment analysis
            result = await self.analyzer.analyze(text, title)

            processing_time = (time.time() - start_time) * 1000

            # Cache the result (serialize to dict)
            try:
                cache_data = result.model_dump()
                self._cache.set(cache_key, cache_data, ttl=3600)
                logger.debug(f"Cached sentiment result: {cache_key}")
            except Exception as cache_error:
                logger.warning(f"Cache set failed: {cache_error}")

            # Save result if requested and article_id is provided
            if save_result and article_id:
                await self._save_sentiment_result(
                    result=result,
                    article_id=article_id,
                    text=text,
                    title=title,
                    source_url=source_url,
                    processing_time=processing_time,
                )

            # Determine if should publish based on parameter or config
            should_publish = publish_result
            if should_publish is None:
                should_publish = (
                    settings.enable_analyze_text_publishing
                    and settings.enable_message_publishing
                )

            # Publish result if enabled and configured
            if should_publish and self.message_broker and article_id:
                try:
                    await self._publish_sentiment_result(
                        result=result,
                        article_id=article_id,
                        title=title,
                        source_url=source_url,
                        search_query=search_query,
                        text=text,
                        metadata=metadata,
                    )
                    logger.debug(
                        f"Successfully published sentiment result for article {article_id}"
                    )
                except Exception as publish_error:
                    logger.warning(
                        f"Failed to publish sentiment result: {publish_error}"
                    )
            elif not should_publish:
                logger.debug(f"Message publishing disabled for article {article_id}")
            elif not self.message_broker:
                logger.debug(
                    f"No message broker available for publishing article {article_id}"
                )

            logger.info(
                f"Sentiment analysis completed: {result.label} (confidence: {result.confidence:.2f})"
            )
            return result

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            raise SentimentAnalysisError(f"Analysis failed: {str(e)}")

    async def analyze_batch(
        self, requests: List[SentimentRequest], save_results: bool = True
    ) -> List[SentimentAnalysisResult]:
        """
        Analyze sentiment for multiple texts in batch.

        Args:
            requests: List of sentiment requests
            save_results: Whether to save results

        Returns:
            List[SentimentAnalysisResult]: Analysis results
        """
        logger.info(f"Starting batch sentiment analysis for {len(requests)} items")

        try:
            # Use the analyzer's batch processing capability
            results = await self.analyzer.analyze_batch(requests)

            # Save results if requested
            if save_results:
                for i, (request, result) in enumerate(zip(requests, results)):
                    try:
                        article_id = f"batch_{i}_{int(datetime.now().timestamp())}"
                        await self._save_sentiment_result(
                            result=result,
                            article_id=article_id,
                            text=request.text,
                            title=request.title,
                            source_url=(
                                str(request.source_url) if request.source_url else None
                            ),
                            processing_time=0.0,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save batch result {i}: {str(e)}")

            logger.info(f"Batch sentiment analysis completed: {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Batch sentiment analysis failed: {str(e)}")
            raise SentimentAnalysisError(f"Batch analysis failed: {str(e)}")

    async def analyze_batch_simple(
        self, requests: List[dict], save_results: bool = True
    ) -> List[SentimentAnalysisResult]:
        """
        Simplified batch analysis for simple dictionary requests.

        Args:
            requests: List of simple dictionary requests
            save_results: Whether to save results

        Returns:
            List[SentimentAnalysisResult]: Analysis results
        """
        logger.info(
            f"Starting simple batch sentiment analysis for {len(requests)} items"
        )

        # Convert simple dicts to SentimentRequest objects
        sentiment_requests = []
        for req in requests:
            sentiment_req = SentimentRequest(
                text=req["text"],
                title=req.get("title"),
                source_url=req.get("source_url"),
            )
            sentiment_requests.append(sentiment_req)

        return await self.analyze_batch(sentiment_requests, save_results)

    async def get_sentiment_by_article_id(
        self, article_id: str
    ) -> Optional[ProcessedSentiment]:
        """
        Retrieve stored sentiment by article ID.

        Args:
            article_id: Article ID

        Returns:
            Optional[ProcessedSentiment]: Stored sentiment or None
        """
        try:
            sentiment = await self.repository.get_sentiment_by_article_id(article_id)
            if sentiment:
                logger.debug(f"Retrieved sentiment for article: {article_id}")
            else:
                logger.debug(f"No sentiment found for article: {article_id}")
            return sentiment

        except Exception as e:
            logger.error(
                f"Failed to retrieve sentiment for article {article_id}: {str(e)}"
            )
            return None

    async def search_sentiments(
        self, filter_params: SentimentQueryFilter
    ) -> List[ProcessedSentiment]:
        """
        Search stored sentiments with filters.

        Args:
            filter_params: Filter parameters

        Returns:
            List[ProcessedSentiment]: Matching sentiments
        """
        try:
            sentiments = await self.repository.search_sentiments(filter_params)
            logger.info(f"Found {len(sentiments)} sentiments matching filters")
            return sentiments

        except Exception as e:
            logger.error(f"Failed to search sentiments: {str(e)}")
            return []

    async def get_sentiment_aggregation(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        source_domain: Optional[str] = None,
    ) -> SentimentAggregation:
        """
        Get aggregated sentiment statistics.

        Args:
            date_from: Start date filter
            date_to: End date filter
            source_domain: Source domain filter

        Returns:
            SentimentAggregation: Aggregated statistics
        """
        try:
            aggregation = await self.repository.get_sentiment_aggregation(
                date_from=date_from,
                date_to=date_to,
                source_domain=source_domain,
            )
            logger.info(
                f"Retrieved sentiment aggregation: {aggregation.total_count} total items"
            )
            return aggregation

        except Exception as e:
            logger.error(f"Failed to get sentiment aggregation: {str(e)}")
            # Return empty aggregation as fallback
            return SentimentAggregation(
                total_count=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                average_confidence=0.0,
                sentiment_distribution={},
            )

    async def _save_sentiment_result(
        self,
        result: SentimentAnalysisResult,
        article_id: str,
        text: str,
        title: Optional[str] = None,
        source_url: Optional[str] = None,
        processing_time: float = 0.0,
    ) -> None:
        """
        Save sentiment analysis result to repository.

        Args:
            result: Sentiment analysis result
            article_id: Article ID
            text: Original text
            title: Optional title
            source_url: Optional source URL
            processing_time: Processing time in milliseconds
        """
        try:
            # Extract source domain from URL
            source_domain = None
            if source_url:
                parsed_url = urlparse(source_url)
                source_domain = parsed_url.netloc

            # Create processed sentiment object
            processed_sentiment = ProcessedSentiment(
                article_id=article_id,
                url=source_url,  # Store as string
                title=title,
                content_preview=text[:200] if text else None,
                sentiment_label=result.label,
                sentiment_scores=result.scores,
                confidence=result.confidence,
                reasoning=result.reasoning,
                processed_at=datetime.now(timezone.utc),
                processing_time_ms=processing_time,
                model_version=settings.openai_model,
                source_domain=source_domain,
            )

            # Save to repository
            sentiment_id = await self.repository.save_sentiment(processed_sentiment)
            logger.debug(f"Saved sentiment result with ID: {sentiment_id}")

        except Exception as e:
            logger.error(f"Failed to save sentiment result: {str(e)}")
            # Don't raise exception - saving failure shouldn't break analysis

    async def _publish_sentiment_result(
        self,
        result: SentimentAnalysisResult,
        article_id: str,
        title: Optional[str] = None,
        source_url: Optional[str] = None,
        search_query: Optional[str] = None,
        text: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Publish sentiment analysis result to message broker.

        Args:
            result: Sentiment analysis result
            article_id: Article ID
            title: Optional title
            source_url: Optional source URL
            search_query: Optional search query
            text: Original text for preview
            metadata: Optional metadata
        """
        try:
            if not self.message_broker:
                logger.warning("Message broker not available for publishing")
                return

            # Check if message broker is healthy
            is_healthy = await self.message_broker.health_check()
            if not is_healthy:
                logger.warning("Message broker is not healthy, skipping publish")
                return

            # Create response message using schema
            sentiment_message = SentimentResultMessageSchema(
                article_id=article_id,
                url=source_url,
                title=title,
                content_preview=(text[:200] if text else ""),
                search_query=search_query,
                sentiment_label=result.label.value,
                scores={
                    "positive": result.scores.positive,
                    "negative": result.scores.negative,
                    "neutral": result.scores.neutral,
                },
                confidence=result.confidence,
                reasoning=result.reasoning,
                processed_at=datetime.now(timezone.utc).isoformat(),
                metadata=metadata or {},
            )

            # Publish to processed sentiments queue using config
            success = await self.message_broker.publish(
                exchange=settings.rabbitmq_sentiment_exchange,
                routing_key=settings.rabbitmq_routing_key_sentiment_processed,
                message=sentiment_message.model_dump(),
            )

            if success:
                logger.debug(
                    f"Published sentiment result for article {article_id} from analyze_text"
                )
            else:
                logger.warning(
                    f"Failed to publish sentiment result for article {article_id}"
                )

        except Exception as e:
            logger.error(
                f"Failed to publish sentiment result from analyze_text: {str(e)}"
            )
            # Don't raise - publishing failure shouldn't break analysis

    async def health_check(self) -> bool:
        """
        Check sentiment service health.

        Returns:
            bool: True if service is healthy
        """
        try:
            # Check analyzer health
            analyzer_healthy = await self.analyzer.health_check()
            logger.debug(f"Analyzer health: {analyzer_healthy}")

            # Check repository health (try a simple operation)
            repository_healthy = True
            try:
                # Try to check if a non-existent sentiment exists
                await self.repository.sentiment_exists("health_check_test")
            except Exception:
                repository_healthy = False

            logger.debug(f"Repository health: {repository_healthy}")

            # Check cache health
            cache_healthy = True
            try:
                test_key = "health_check_test"
                test_value = {"test": True}
                self._cache.set(test_key, test_value, ttl=60)
                get_result = self._cache.get(test_key)
                cache_healthy = get_result == test_value
                self._cache.delete(test_key)
            except Exception:
                cache_healthy = False

            logger.debug(f"Cache health: {cache_healthy}")

            is_healthy = analyzer_healthy and repository_healthy and cache_healthy
            logger.debug(f"Overall health: {is_healthy}")

            return is_healthy

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
