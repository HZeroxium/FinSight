"""
Sentiment analysis service layer for business logic.
"""

import time
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
)
from ..common.logger import LoggerFactory, LoggerType, LogLevel
from ..common.cache import CacheFactory, CacheType, cache_result

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
        self._cache = CacheFactory.get_cache(
            name="sentiment_cache",
            cache_type=CacheType.MEMORY,
            max_size=1000,
            default_ttl=3600,  # 1 hour
        )

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
        Analyze sentiment of text with optional saving.

        Args:
            text: Text to analyze
            title: Optional title for context
            article_id: Optional article ID for storage
            source_url: Optional source URL
            save_result: Whether to save the result

        Returns:
            SentimentAnalysisResult: Analysis results
        """
        logger.info(f"Analyzing sentiment for text (length: {len(text)})")

        try:
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

        except SentimentAnalysisError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in sentiment analysis: {str(e)}")
            raise SentimentAnalysisError(f"Sentiment analysis failed: {str(e)}")

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
            # Perform batch analysis
            results = await self.analyzer.analyze_batch(requests)

            # Save results if requested
            if save_results:
                save_tasks = []
                for i, (request, result) in enumerate(zip(requests, results)):
                    if hasattr(request, "article_id") and request.article_id:
                        save_tasks.append(
                            self._save_sentiment_result(
                                result=result,
                                article_id=getattr(request, "article_id", f"batch_{i}"),
                                text=request.text,
                                title=request.title,
                                source_url=(
                                    str(request.source_url)
                                    if request.source_url
                                    else None
                                ),
                                processing_time=0.0,  # Not tracked for batch
                            )
                        )

                # Execute save tasks concurrently
                if save_tasks:
                    import asyncio

                    await asyncio.gather(*save_tasks, return_exceptions=True)

            logger.info(f"Batch sentiment analysis completed: {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Batch sentiment analysis failed: {str(e)}")
            raise SentimentAnalysisError(f"Batch analysis failed: {str(e)}")

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
            return await self.repository.get_sentiment_by_article_id(article_id)
        except Exception as e:
            logger.error(f"Failed to get sentiment by article ID: {str(e)}")
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
            return await self.repository.search_sentiments(filter_params)
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
            return await self.repository.get_sentiment_aggregation(
                date_from=date_from, date_to=date_to, source_domain=source_domain
            )
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
                try:
                    parsed_url = urlparse(source_url)
                    source_domain = parsed_url.netloc
                except Exception:
                    pass

            # Create processed sentiment
            processed_sentiment = ProcessedSentiment(
                article_id=article_id,
                url=source_url,
                title=title,
                content_preview=text[:200] if text else None,
                sentiment_label=result.label,
                sentiment_scores=result.scores,
                confidence=result.confidence,
                reasoning=result.reasoning,
                processing_time_ms=processing_time,
                model_version=getattr(self.analyzer, "model", "unknown"),
                source_domain=source_domain,
            )

            # Save to repository
            sentiment_id = await self.repository.save_sentiment(processed_sentiment)
            logger.debug(f"Saved sentiment result with ID: {sentiment_id}")

        except Exception as e:
            logger.error(f"Failed to save sentiment result: {str(e)}")
            # Don't raise exception here to avoid breaking the main analysis flow

    async def health_check(self) -> bool:
        """Check service health."""
        try:
            # Check analyzer health
            analyzer_healthy = await self.analyzer.health_check()

            # Check repository health (simple test)
            repository_healthy = True
            try:
                test_filter = SentimentQueryFilter(limit=1)
                await self.repository.search_sentiments(test_filter)
            except Exception:
                repository_healthy = False

            is_healthy = analyzer_healthy and repository_healthy
            logger.debug(f"Sentiment service health check: {is_healthy}")
            return is_healthy

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
