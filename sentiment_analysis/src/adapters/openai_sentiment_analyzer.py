# adapters/openai_sentiment_analyzer.py

"""
OpenAI-based sentiment analyzer using LangChain with structured output.
"""

import time
import asyncio
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel, Field

from ..interfaces.sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisError
from ..models.sentiment import (
    SentimentAnalysisResult,
    SentimentRequest,
    SentimentLabel,
    SentimentScore,
)
from ..common.logger import LoggerFactory, LoggerType, LogLevel

logger = LoggerFactory.get_logger(
    name="openai-sentiment-analyzer",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
)


class SentimentOutput(LangChainBaseModel):
    """Structured output for sentiment analysis."""

    sentiment_label: str = Field(
        description="Overall sentiment: positive, negative, or neutral"
    )
    positive_score: float = Field(description="Positive sentiment score (0.0 to 1.0)")
    negative_score: float = Field(description="Negative sentiment score (0.0 to 1.0)")
    neutral_score: float = Field(description="Neutral sentiment score (0.0 to 1.0)")
    confidence: float = Field(description="Confidence in the analysis (0.0 to 1.0)")
    reasoning: str = Field(
        description="Brief explanation of the sentiment classification"
    )


class OpenAISentimentAnalyzer(SentimentAnalyzer):
    """
    OpenAI-powered sentiment analyzer using LangChain with structured output.

    Uses GPT models for financial sentiment analysis with domain-specific prompting.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        max_retries: int = 3,
    ):
        """
        Initialize OpenAI sentiment analyzer.

        Args:
            api_key: OpenAI API key
            model: Model name to use
            temperature: Model temperature
            max_tokens: Maximum tokens for response
            max_retries: Maximum retry attempts
        """
        self.model = model
        self.max_retries = max_retries

        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Create structured LLM
        self.structured_llm = self.llm.with_structured_output(SentimentOutput)

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._get_system_prompt()),
                ("human", self._get_human_prompt()),
            ]
        )

        # Create the chain
        self.chain = self.prompt | self.structured_llm

        logger.info(f"OpenAI sentiment analyzer initialized with model: {model}")

    def _get_system_prompt(self) -> str:
        """Get system prompt for sentiment analysis."""
        return """You are an expert financial sentiment analyst. Your task is to analyze the sentiment of financial news articles, social media posts, and market commentary.

Focus on:
- Overall market sentiment (bullish/bearish)
- Impact on financial markets and investments
- Investor confidence and market mood
- Economic implications

Consider:
- Financial terminology and context
- Market-moving events and announcements
- Company performance and earnings
- Economic indicators and trends
- Geopolitical events affecting markets

Provide accurate sentiment classification with confidence scores that sum to 1.0.
Give clear reasoning for your classification focusing on financial implications."""

    def _get_human_prompt(self) -> str:
        """Get human prompt template."""
        return """Analyze the sentiment of the following financial text:

Title: {title}
Content: {content}

Classify the sentiment as positive, negative, or neutral from a financial/investment perspective.
Provide scores for each sentiment category and explain your reasoning."""

    async def analyze(self, text: str, title: str = None) -> SentimentAnalysisResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text content to analyze
            title: Optional title for context

        Returns:
            SentimentAnalysisResult: Analysis results
        """
        try:
            logger.debug(f"Analyzing sentiment for text: {text[:100]}...")
            start_time = time.time()

            # Prepare input
            input_data = {
                "title": title or "No title provided",
                "content": text,
            }

            # Run analysis with retry logic
            for attempt in range(self.max_retries):
                try:
                    result = await self.chain.ainvoke(input_data)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}, retrying..."
                    )
                    await asyncio.sleep(2**attempt)  # Exponential backoff

            # Convert to our model
            sentiment_result = self._convert_to_sentiment_result(result)

            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"Sentiment analysis completed in {processing_time:.2f}ms: {sentiment_result.label}"
            )

            return sentiment_result

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            raise SentimentAnalysisError(
                message=f"Failed to analyze sentiment: {str(e)}",
                details={"text_length": len(text), "has_title": title is not None},
            )

    async def analyze_batch(
        self, requests: List[SentimentRequest]
    ) -> List[SentimentAnalysisResult]:
        """
        Analyze sentiment of multiple texts in batch.

        Args:
            requests: List of sentiment requests

        Returns:
            List[SentimentAnalysisResult]: Analysis results
        """
        logger.info(f"Starting batch sentiment analysis for {len(requests)} items")

        try:
            # Create tasks for concurrent processing
            tasks = [self.analyze(request.text, request.title) for request in requests]

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch item {i} failed: {str(result)}")
                    # Create neutral result as fallback
                    fallback_result = SentimentAnalysisResult(
                        label=SentimentLabel.NEUTRAL,
                        scores=SentimentScore(
                            positive=0.33, negative=0.33, neutral=0.34
                        ),
                        confidence=0.0,
                        reasoning="Analysis failed - using neutral fallback",
                    )
                    processed_results.append(fallback_result)
                else:
                    processed_results.append(result)

            logger.info(
                f"Batch sentiment analysis completed: {len(processed_results)} results"
            )
            return processed_results

        except Exception as e:
            logger.error(f"Batch sentiment analysis failed: {str(e)}")
            raise SentimentAnalysisError(f"Batch analysis failed: {str(e)}")

    def _convert_to_sentiment_result(
        self, openai_result: SentimentOutput
    ) -> SentimentAnalysisResult:
        """Convert OpenAI result to our sentiment result model."""

        # Normalize sentiment label
        label_map = {
            "positive": SentimentLabel.POSITIVE,
            "negative": SentimentLabel.NEGATIVE,
            "neutral": SentimentLabel.NEUTRAL,
        }

        sentiment_label = label_map.get(
            openai_result.sentiment_label.lower(), SentimentLabel.NEUTRAL
        )

        # Ensure scores sum to 1.0 (normalize if needed)
        total_score = (
            openai_result.positive_score
            + openai_result.negative_score
            + openai_result.neutral_score
        )

        if total_score > 0:
            positive = openai_result.positive_score / total_score
            negative = openai_result.negative_score / total_score
            neutral = openai_result.neutral_score / total_score
        else:
            # Fallback to equal distribution
            positive = negative = neutral = 1.0 / 3.0

        scores = SentimentScore(
            positive=positive,
            negative=negative,
            neutral=neutral,
        )

        return SentimentAnalysisResult(
            label=sentiment_label,
            scores=scores,
            confidence=max(0.0, min(1.0, openai_result.confidence)),
            reasoning=openai_result.reasoning,
        )

    async def health_check(self) -> bool:
        """
        Check OpenAI service health.

        Returns:
            bool: True if service is healthy
        """
        try:
            logger.debug("Performing OpenAI sentiment analyzer health check")

            # Perform a simple test analysis
            test_result = await self.analyze(
                text="The market is performing well today.", title="Test Health Check"
            )

            is_healthy = test_result is not None and test_result.label in [
                SentimentLabel.POSITIVE,
                SentimentLabel.NEGATIVE,
                SentimentLabel.NEUTRAL,
            ]

            logger.debug(f"OpenAI sentiment analyzer health check result: {is_healthy}")
            return is_healthy

        except Exception as e:
            logger.error(f"OpenAI sentiment analyzer health check failed: {str(e)}")
            return False
