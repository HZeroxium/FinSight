# routers/sentiment.py

"""
Simplified REST API routes for sentiment analysis testing.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any

from ..services.sentiment_service import SentimentService
from ..utils.dependencies import get_sentiment_service
from common.logger import LoggerFactory, LoggerType, LogLevel

logger = LoggerFactory.get_logger(
    name="sentiment-router", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)

router = APIRouter(prefix="/api/v1/sentiment", tags=["sentiment"])


@router.get("/health")
async def health_check(
    sentiment_service: SentimentService = Depends(get_sentiment_service),
) -> Dict[str, Any]:
    """
    Health check endpoint for sentiment analysis service.

    Returns:
        Dict[str, Any]: Health status information
    """
    try:
        is_healthy = await sentiment_service.health_check()

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "service": "sentiment-analysis",
            "timestamp": "2025-08-19T21:00:00Z",
            "version": "1.0.0",
            "components": {
                "sentiment_analyzer": "available",
                "news_repository": "connected",
                "message_broker": "optional",
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")


@router.post("/test")
async def test_sentiment_analysis(
    request: Dict[str, Any],
    sentiment_service: SentimentService = Depends(get_sentiment_service),
) -> Dict[str, Any]:
    """
    Test endpoint for manual sentiment analysis.

    Args:
        request: Test request with 'text' field

    Returns:
        Dict[str, Any]: Sentiment analysis result
    """
    try:
        text = request.get("text")
        if not text:
            raise HTTPException(
                status_code=400, detail="Missing 'text' field in request"
            )

        logger.info(f"Testing sentiment analysis for text: {text[:50]}...")

        # Analyze sentiment
        result = await sentiment_service.analyze_news_sentiment(
            news_id="test", title=text, content=None
        )

        if not result:
            raise HTTPException(status_code=500, detail="Sentiment analysis failed")

        return {
            "status": "success",
            "input_text": text,
            "sentiment_label": result.label,
            "confidence": result.confidence,
            "scores": result.scores,
            "reasoning": result.reasoning,
            "analyzer_version": "openai-gpt-4o-mini",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
