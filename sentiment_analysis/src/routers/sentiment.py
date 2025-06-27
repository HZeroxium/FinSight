"""
REST API routes for sentiment analysis operations.
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from ..schemas.sentiment_schemas import (
    SentimentAnalysisRequestSchema,
    SentimentBatchRequestSchema,
    SentimentAnalysisResponseSchema,
    SentimentBatchResponseSchema,
    ProcessedSentimentSchema,
    SentimentSearchRequestSchema,
    SentimentSearchResponseSchema,
    SentimentAggregationSchema,
    SentimentErrorSchema,
)
from ..schemas import HealthCheckSchema, ErrorResponseSchema
from ..services.sentiment_service import SentimentService
from ..models.sentiment import SentimentQueryFilter
from ..interfaces.sentiment_analyzer import SentimentAnalysisError
from ..utils.dependencies import get_sentiment_service
from ..common.logger import LoggerFactory, LoggerType, LogLevel

logger = LoggerFactory.get_logger(
    name="sentiment-router", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)

router = APIRouter(prefix="/api/v1/sentiment", tags=["sentiment"])


@router.post("/analyze", response_model=SentimentAnalysisResponseSchema)
async def analyze_sentiment(
    request: SentimentAnalysisRequestSchema,
    sentiment_service: SentimentService = Depends(get_sentiment_service),
) -> SentimentAnalysisResponseSchema:
    """
    Analyze sentiment of a single text.

    Args:
        request: Sentiment analysis request
        sentiment_service: Injected sentiment service

    Returns:
        SentimentAnalysisResponseSchema: Sentiment analysis result
    """
    try:
        logger.info(
            f"Single sentiment analysis request for text length: {len(request.text)}"
        )

        result = await sentiment_service.analyze_text(
            text=request.text,
            title=request.title,
            article_id=request.article_id,
            source_url=str(request.source_url) if request.source_url else None,
            save_result=request.save_result,
        )

        return SentimentAnalysisResponseSchema(
            sentiment_label=result.label,
            scores=result.scores,
            confidence=result.confidence,
            reasoning=result.reasoning,
            processing_time_ms=None,  # Would track in production
        )

    except SentimentAnalysisError as e:
        logger.error(f"Sentiment analysis error: {e.message}")
        raise HTTPException(status_code=502, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analyze/batch", response_model=SentimentBatchResponseSchema)
async def analyze_sentiment_batch(
    request: SentimentBatchRequestSchema,
    background_tasks: BackgroundTasks,
    sentiment_service: SentimentService = Depends(get_sentiment_service),
) -> SentimentBatchResponseSchema:
    """
    Analyze sentiment of multiple texts in batch.

    Args:
        request: Batch sentiment analysis request
        background_tasks: FastAPI background tasks
        sentiment_service: Injected sentiment service

    Returns:
        SentimentBatchResponseSchema: Batch analysis results
    """
    try:
        logger.info(f"Batch sentiment analysis request for {len(request.items)} items")

        # Convert schema requests to domain models
        from ..models.sentiment import SentimentRequest

        domain_requests = [
            SentimentRequest(
                text=item.text,
                title=item.title,
                source_url=item.source_url,
            )
            for item in request.items
        ]

        results = await sentiment_service.analyze_batch(
            requests=domain_requests, save_results=request.save_results
        )

        # Convert domain results to schema
        response_results = [
            SentimentAnalysisResponseSchema(
                sentiment_label=result.label,
                scores=result.scores,
                confidence=result.confidence,
                reasoning=result.reasoning,
            )
            for result in results
        ]

        success_count = len([r for r in results if r.confidence > 0.0])
        error_count = len(results) - success_count

        return SentimentBatchResponseSchema(
            results=response_results,
            total_processed=len(results),
            success_count=success_count,
            error_count=error_count,
        )

    except SentimentAnalysisError as e:
        logger.error(f"Batch sentiment analysis error: {e.message}")
        raise HTTPException(status_code=502, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/article/{article_id}", response_model=ProcessedSentimentSchema)
async def get_sentiment_by_article(
    article_id: str,
    sentiment_service: SentimentService = Depends(get_sentiment_service),
) -> ProcessedSentimentSchema:
    """
    Get stored sentiment by article ID.

    Args:
        article_id: Article ID
        sentiment_service: Injected sentiment service

    Returns:
        ProcessedSentimentSchema: Stored sentiment data
    """
    try:
        sentiment = await sentiment_service.get_sentiment_by_article_id(article_id)

        if not sentiment:
            raise HTTPException(status_code=404, detail="Sentiment not found")

        return ProcessedSentimentSchema(
            id=str(sentiment.id),
            article_id=sentiment.article_id,
            url=sentiment.url,
            title=sentiment.title,
            content_preview=sentiment.content_preview,
            sentiment_label=sentiment.sentiment_label,
            scores=sentiment.sentiment_scores,
            confidence=sentiment.confidence,
            reasoning=sentiment.reasoning,
            processed_at=sentiment.processed_at,
            processing_time_ms=sentiment.processing_time_ms,
            model_version=sentiment.model_version,
            source_domain=sentiment.source_domain,
            source_category=sentiment.source_category,
            published_at=sentiment.published_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/search", response_model=SentimentSearchResponseSchema)
async def search_sentiments(
    request: SentimentSearchRequestSchema,
    sentiment_service: SentimentService = Depends(get_sentiment_service),
) -> SentimentSearchResponseSchema:
    """
    Search stored sentiments with filters.

    Args:
        request: Search request parameters
        sentiment_service: Injected sentiment service

    Returns:
        SentimentSearchResponseSchema: Search results
    """
    try:
        # Convert schema to domain model
        filter_params = SentimentQueryFilter(
            sentiment_label=request.sentiment_label,
            min_confidence=request.min_confidence,
            max_confidence=request.max_confidence,
            date_from=request.date_from,
            date_to=request.date_to,
            source_domain=request.source_domain,
            source_category=request.source_category,
            limit=request.limit,
            offset=request.offset,
        )

        sentiments = await sentiment_service.search_sentiments(filter_params)

        # Convert to schema
        sentiment_schemas = [
            ProcessedSentimentSchema(
                id=str(sentiment.id),
                article_id=sentiment.article_id,
                url=sentiment.url,
                title=sentiment.title,
                content_preview=sentiment.content_preview,
                sentiment_label=sentiment.sentiment_label,
                scores=sentiment.sentiment_scores,
                confidence=sentiment.confidence,
                reasoning=sentiment.reasoning,
                processed_at=sentiment.processed_at,
                processing_time_ms=sentiment.processing_time_ms,
                model_version=sentiment.model_version,
                source_domain=sentiment.source_domain,
                source_category=sentiment.source_category,
                published_at=sentiment.published_at,
            )
            for sentiment in sentiments
        ]

        # Get total count (would implement in production)
        total_count = len(sentiment_schemas)

        return SentimentSearchResponseSchema(
            sentiments=sentiment_schemas,
            total_count=total_count,
            limit=request.limit,
            offset=request.offset,
        )

    except Exception as e:
        logger.error(f"Failed to search sentiments: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/aggregation", response_model=SentimentAggregationSchema)
async def get_sentiment_aggregation(
    date_from: datetime = Query(None, description="Start date filter"),
    date_to: datetime = Query(None, description="End date filter"),
    source_domain: str = Query(None, description="Source domain filter"),
    sentiment_service: SentimentService = Depends(get_sentiment_service),
) -> SentimentAggregationSchema:
    """
    Get aggregated sentiment statistics.

    Args:
        date_from: Start date filter
        date_to: End date filter
        source_domain: Source domain filter
        sentiment_service: Injected sentiment service

    Returns:
        SentimentAggregationSchema: Aggregated statistics
    """
    try:
        aggregation = await sentiment_service.get_sentiment_aggregation(
            date_from=date_from, date_to=date_to, source_domain=source_domain
        )

        # Determine time period
        time_period = None
        if date_from and date_to:
            time_period = (
                f"{date_from.strftime('%Y-%m-%d')} to {date_to.strftime('%Y-%m-%d')}"
            )

        return SentimentAggregationSchema(
            total_count=aggregation.total_count,
            positive_count=aggregation.positive_count,
            negative_count=aggregation.negative_count,
            neutral_count=aggregation.neutral_count,
            average_confidence=aggregation.average_confidence,
            sentiment_distribution=aggregation.sentiment_distribution,
            time_period=time_period,
        )

    except Exception as e:
        logger.error(f"Failed to get sentiment aggregation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health", response_model=HealthCheckSchema)
async def health_check(
    sentiment_service: SentimentService = Depends(get_sentiment_service),
) -> HealthCheckSchema:
    """
    Health check endpoint for sentiment service.

    Returns:
        HealthCheckSchema: Health status
    """
    try:
        is_healthy = await sentiment_service.health_check()

        dependencies = {
            "sentiment_analyzer": "healthy" if is_healthy else "unhealthy",
            "sentiment_repository": "healthy" if is_healthy else "unhealthy",
        }

        return HealthCheckSchema(
            status="healthy" if is_healthy else "unhealthy",
            service="sentiment-analysis-service",
            dependencies=dependencies,
        )

    except Exception as e:
        logger.error(f"Sentiment health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content=ErrorResponseSchema(
                error="ServiceUnavailable", message=f"Health check failed: {str(e)}"
            ).model_dump(),
        )
