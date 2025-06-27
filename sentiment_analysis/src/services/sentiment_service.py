# services/sentiment_service.py

"""
Sentiment analysis service layer for business logic.
"""

from typing import List, Optional
from datetime import datetime
from urllib.parse import urlparse

from ..interfaces.sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisError
from ..interfaces.sentiment_repository import SentimentRepository
from ..models.sentiment import (
    SentimentRequest,
    SentimentAnalysisResult,
    ProcessedSentiment,
    SentimentQueryFilter,
    SentimentAggregation,
    SentimentLabel,
    SentimentScore,
)
from ..common.logger import LoggerFactory, LoggerType, LogLevel
from ..common.cache import CacheFactory, CacheType, cache_result
from ..core.config import settings

logger = LoggerFactory.get_logger(
    name="sentiment-service", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


class SentimentService:
    """
    Service layer for sentiment analysis operations and business logic.
    """

    def __init__(
        self,
        analyzer: SentimentAnalyzer,
        repository: SentimentRepository,
    ):
        """
        Initialize sentiment service.

        Args:
            analyzer: Sentiment analyzer implementation
            repository: Sentiment repository for storage
        """
        self.analyzer = analyzer
        self.repository = repository

        # Initialize cache for sentiment results
        try:
            self._cache = CacheFactory.get_cache(
                name="sentiment_cache",
                cache_type=CacheType.REDIS,
                host="localhost",
                port=6379,
                db=1,  # Different DB from news crawler
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

    @cache_result(ttl=3600, key_prefix="sentiment_")
    async def analyze_text(
        self,
        text: str,
        title: Optional[str] = None,
        article_id: Optional[str] = None,
        source_url: Optional[str] = None,
        save_result: bool = True,
    ) -> SentimentAnalysisResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text content to analyze
            title: Optional title for context
            article_id: Optional article ID for storage
            source_url: Optional source URL
            save_result: Whether to save the result

        Returns:
            SentimentAnalysisResult: Analysis results
        """
        logger.info(f"Analyzing sentiment for text length: {len(text)}")

        try:
            import time

            start_time = time.time()

            # Perform sentiment analysis
            result = await self.analyzer.analyze(text, title)

            processing_time = (time.time() - start_time) * 1000

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
                url=source_url,
                title=title,
                content_preview=text[:200] if text else None,
                sentiment_label=result.label,
                sentiment_scores=result.scores,
                confidence=result.confidence,
                reasoning=result.reasoning,
                processed_at=datetime.utcnow(),
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
