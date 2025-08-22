# services/sentiment_service.py

"""
Consolidated sentiment analysis service for handling news sentiment analysis.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone

from ..interfaces.sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisError
from ..interfaces.news_repository_interface import NewsRepositoryInterface
from ..interfaces.message_broker import MessageBroker, MessageBrokerError
from ..models.sentiment import SentimentAnalysisResult
from ..core.config import settings
from ..schemas.message_schemas import SentimentResultMessageSchema, NewsMessageSchema
from common.logger import LoggerFactory, LoggerType, LogLevel


class SentimentService:
    """Consolidated sentiment analysis service for news sentiment processing."""

    def __init__(
        self,
        analyzer: SentimentAnalyzer,
        news_repository: NewsRepositoryInterface,
        message_broker: Optional[MessageBroker] = None,
    ):
        """
        Initialize sentiment service.

        Args:
            analyzer: Sentiment analyzer implementation
            news_repository: News repository for data access
            message_broker: Optional message broker for publishing results
        """
        self.analyzer = analyzer
        self.news_repository = news_repository
        self.message_broker = message_broker

        self.logger = LoggerFactory.get_logger(
            name="sentiment-service",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            log_file="logs/sentiment_analysis.log",
        )
        self.logger.info("SentimentService initialized")

    async def analyze_news_sentiment(
        self, news_id: str, title: str, content: Optional[str] = None
    ) -> Optional[SentimentAnalysisResult]:
        """
        Analyze sentiment for a news item.

        Args:
            news_id: News item ID
            title: News title
            content: Optional news content

        Returns:
            SentimentAnalysisResult: Analysis result or None if failed
        """
        try:
            # Combine title and content for analysis
            text_to_analyze = title
            if content:
                text_to_analyze = f"{title}\n\n{content}"

            # Perform sentiment analysis
            result = await self.analyzer.analyze(text_to_analyze)

            self.logger.info(
                f"Sentiment analysis completed for news {news_id}: {result.label}"
            )

            return result

        except SentimentAnalysisError as e:
            self.logger.error(f"Sentiment analysis error for news {news_id}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to analyze sentiment for news {news_id}: {e}")
            return None

    async def update_news_with_sentiment(
        self, news_id: str, result: SentimentAnalysisResult
    ) -> bool:
        """
        Update news item with sentiment analysis results using the repository.

        Args:
            news_id: News item ID
            result: Sentiment analysis result

        Returns:
            bool: True if successful
        """
        try:
            # Convert SentimentScore to dictionary for MongoDB compatibility
            sentiment_scores_dict = {
                "positive": result.scores.positive,
                "negative": result.scores.negative,
                "neutral": result.scores.neutral,
            }

            # Update sentiment data directly using repository
            success = await self.news_repository.update_news_sentiment(
                item_id=news_id,
                sentiment_label=result.label.value,  # Convert enum to string
                sentiment_scores=sentiment_scores_dict,
                sentiment_confidence=result.confidence,
                sentiment_reasoning=result.reasoning,
                analyzer_version=settings.analyzer_version,
            )

            if success:
                self.logger.info(
                    f"Updated news {news_id} with sentiment: {result.label}"
                )
            else:
                self.logger.warning(
                    f"Failed to update news item {news_id} - item not found"
                )

            return success

        except Exception as e:
            self.logger.error(f"Failed to update news {news_id} with sentiment: {e}")
            return False

    async def publish_sentiment_result(
        self,
        news_id: str,
        result: SentimentAnalysisResult,
        original_title: Optional[str] = None,
        original_url: Optional[str] = None,
        search_query: Optional[str] = None,
        content_preview: Optional[str] = None,
    ) -> bool:
        """
        Publish sentiment analysis result back to news service.

        Args:
            news_id: Original news ID
            result: Sentiment analysis result
            original_title: Original news title
            original_url: Original news URL
            search_query: Original search query
            content_preview: Content preview

        Returns:
            bool: True if published successfully
        """
        if not self.message_broker:
            self.logger.warning(
                "Message broker not available, skipping result publishing"
            )
            return False

        try:
            # Ensure connection
            if not await self._ensure_broker_connection():
                return False

            # Prepare result message matching SentimentResultMessageSchema
            result_message = {
                "article_id": news_id,
                "title": original_title,
                "url": original_url,
                "search_query": search_query,
                "sentiment_label": result.label,
                "confidence": result.confidence,
                "scores": result.scores,
                "reasoning": result.reasoning,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "content_preview": content_preview,
                "analyzer_version": settings.analyzer_version,
            }

            # Publish to sentiment results queue
            await self.message_broker.publish(
                exchange=settings.rabbitmq_exchange,
                routing_key=settings.rabbitmq_routing_key_sentiment_results,
                message=result_message,
            )

            self.logger.info(f"Published sentiment result for news {news_id}")
            return True

        except MessageBrokerError as e:
            self.logger.error(
                f"Message broker error publishing result for {news_id}: {e}"
            )
            return False
        except Exception as e:
            self.logger.error(f"Failed to publish sentiment result for {news_id}: {e}")
            return False

    async def process_news_message(self, message: NewsMessageSchema) -> bool:
        """
        Process a news message from the queue - main processing logic.

        Args:
            message: News message data from NewsMessageSchema

        Returns:
            bool: True if processed successfully
        """
        try:
            # Extract message fields using NewsMessageSchema structure
            news_id = message.id
            title = message.title
            description = message.description or ""
            url = message.url
            # Note: search_query is not part of NewsMessageSchema, use metadata if needed
            search_query = (
                message.metadata.get("search_query") if message.metadata else None
            )

            if not news_id or not title:
                self.logger.warning("Invalid news message: missing article_id or title")
                return False

            self.logger.info(f"Processing news message: {news_id} - '{title[:50]}...'")

            # Combine content sources for analysis
            text_content = description

            # Analyze sentiment
            result = await self.analyze_news_sentiment(news_id, title, text_content)
            if not result:
                self.logger.error(f"Failed to analyze sentiment for {news_id}")
                return False

            # Update news item with sentiment in database
            if not await self.update_news_with_sentiment(news_id, result):
                self.logger.error(f"Failed to update news {news_id} with sentiment")
                # Continue to publish even if DB update fails

            # Publish result back to news service
            # content_preview = (
            #     (text_content[:200] + "...")
            #     if len(text_content) > 200
            #     else text_content
            # )
            # if not await self.publish_sentiment_result(
            #     news_id, result, title, url, search_query, content_preview
            # ):
            #     self.logger.warning(f"Failed to publish sentiment result for {news_id}")
            #     # Don't return False here as the analysis was successful

            self.logger.info(
                f"Successfully processed sentiment for {news_id}: {result.label}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to process news message: {e}")
            return False

    async def health_check(self) -> bool:
        """
        Check service health.

        Returns:
            bool: True if healthy
        """
        try:
            # Check analyzer
            if not self.analyzer:
                return False

            # Check repository
            if not await self.news_repository.health_check():
                return False

            # Check message broker (optional - don't fail if not available)
            if self.message_broker:
                try:
                    if not await self.message_broker.health_check():
                        self.logger.warning(
                            "Message broker unhealthy - continuing without it"
                        )
                except Exception:
                    self.logger.warning(
                        "Message broker check failed - continuing without it"
                    )

            return True

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    async def _ensure_broker_connection(self) -> bool:
        """Ensure message broker is connected and configured."""
        try:
            if not self.message_broker:
                return False

            # Check and reconnect if needed
            if not await self.message_broker.health_check():
                await self.message_broker.connect()

            # Setup infrastructure
            await self._setup_broker_infrastructure()
            return True

        except Exception as e:
            self.logger.error(f"Failed to ensure broker connection: {e}")
            return False

    async def _setup_broker_infrastructure(self) -> None:
        """Setup message broker infrastructure."""
        try:
            # Declare exchange
            await self.message_broker.declare_exchange(
                settings.rabbitmq_exchange, "topic", durable=True
            )

            # Declare result queue
            await self.message_broker.declare_queue(
                settings.rabbitmq_queue_sentiment_results, durable=True
            )

            # Bind result queue
            await self.message_broker.bind_queue(
                settings.rabbitmq_queue_sentiment_results,
                settings.rabbitmq_exchange,
                settings.rabbitmq_routing_key_sentiment_results,
            )

            self.logger.debug("Message broker infrastructure setup completed")

        except Exception as e:
            self.logger.error(f"Failed to setup broker infrastructure: {e}")
            raise
