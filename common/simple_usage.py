# simple_usage.py

from typing import Optional
from pydantic import BaseModel, Field
from logger import LoggerFactory, LoggerType, LogLevel

# Import the LLM module
from llm import (
    generate_text,
    generate_structured,
    create_openai_llm,
    create_langchain_llm,
    create_google_adk_llm,
    LLMProvider,
    StrategyType,
    LLMInterface,
    configure_container,
    get_facade,
    LLMFactory,
)


# Define a simple Pydantic model for structured output
class StockAnalysis(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    analysis: str = Field(..., description="Detailed stock analysis")
    recommendation: str = Field(
        ..., description="Investment recommendation based on analysis"
    )


SIMPLE_TEXT_GENERATION_PROMPT = "Say hi to the user and tell them a joke about stocks."

STRUCTURED_STOCK_ANALYSIS_PROMPT = """
Analyze the following stock market data and provide a detailed analysis:
- Ticker: AAPL
- Current Price: $150
- Market Cap: $2.5 Trillion
- P/E Ratio: 28
- Dividend Yield: 0.6%
Please provide your analysis and recommendations.
"""

logger = LoggerFactory.get_logger(
    name="simple-usage", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)

result_openai = generate_text(
    prompt=SIMPLE_TEXT_GENERATION_PROMPT, provider="openai", model="gpt-4o-mini"
)
result_langchain = generate_text(
    prompt=SIMPLE_TEXT_GENERATION_PROMPT, provider="langchain", model="gpt-4o-mini"
)
result_google_adk = generate_text(
    prompt=SIMPLE_TEXT_GENERATION_PROMPT, provider="google-adk", model="gpt-4o-mini"
)

logger.info(f"OpenAI Text Generation Result: {result_openai}")
logger.info(f"LangChain Text Generation Result: {result_langchain}")
logger.info(f"Google ADK Text Generation Result: {result_google_adk}")

result_structured_openai = generate_structured(
    prompt=STRUCTURED_STOCK_ANALYSIS_PROMPT,
    provider="openai",
    model="gpt-4o-mini",
    schema=StockAnalysis,
)
result_structured_langchain = generate_structured(
    prompt=STRUCTURED_STOCK_ANALYSIS_PROMPT,
    provider="langchain",
    model="gpt-4o-mini",
    schema=StockAnalysis,
)
result_structured_google_adk = generate_structured(
    prompt=STRUCTURED_STOCK_ANALYSIS_PROMPT,
    provider="google-adk",
    model="gpt-4o-mini",
    schema=StockAnalysis,
)

logger.info(f"OpenAI Structured Result: {result_structured_openai}")
logger.info(f"LangChain Structured Result: {result_structured_langchain}")
logger.info(f"Google ADK Structured Result: {result_structured_google_adk}")
