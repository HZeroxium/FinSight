"""
Example usage of the FinSight LLM module

This file demonstrates various ways to use the LLM module in other parts of the application.
Focus on practical, real-world usage patterns for financial analysis and content generation.
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field

# Import the LLM module
from llm import (
    generate_text,
    generate_structured,
    create_openai_llm,
    create_langchain_llm,
    create_google_adk_llm,
    LLMProvider,
    StrategyType,
    configure_container,
    get_facade,
    LLMFactory,
)


# Example 1: Define structured output schemas for financial analysis
class StockAnalysis(BaseModel):
    """Stock analysis result schema"""

    symbol: str = Field(..., description="Stock symbol")
    recommendation: str = Field(..., description="Buy/Hold/Sell recommendation")
    target_price: float = Field(..., description="Target price prediction")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    reasoning: str = Field(..., description="Analysis reasoning")
    risks: List[str] = Field(default_factory=list, description="Key risks")


class MarketSummary(BaseModel):
    """Market summary schema"""

    market_sentiment: str = Field(..., description="Bullish/Bearish/Neutral")
    key_trends: List[str] = Field(..., description="Key market trends")
    economic_indicators: dict = Field(
        default_factory=dict, description="Key indicators"
    )
    summary: str = Field(..., description="Brief market summary")


class NewsAnalysis(BaseModel):
    """News sentiment analysis schema"""

    sentiment: str = Field(..., description="Positive/Negative/Neutral")
    impact_score: float = Field(
        ..., ge=0, le=10, description="Market impact score 0-10"
    )
    affected_sectors: List[str] = Field(
        default_factory=list, description="Affected sectors"
    )
    key_points: List[str] = Field(..., description="Key points from the news")


# Example 2: Simple text generation (most common use case)
def analyze_earnings_report(earnings_text: str) -> str:
    """
    Analyze earnings report and provide summary
    This is the most common pattern - just generate analysis text
    """
    prompt = f"""
    Please analyze the following earnings report and provide a concise summary:
    
    {earnings_text}
    
    Focus on:
    - Revenue and profit trends
    - Key performance metrics
    - Management guidance
    - Potential impact on stock price
    """

    # Simple one-line usage - this is what you'll use most often
    return generate_text(prompt, model="gpt-4o-mini", temperature=0.3)


# Example 3: Structured analysis (when you need parsed data)
def get_stock_recommendation(
    company_data: str, financial_metrics: dict
) -> StockAnalysis:
    """
    Get structured stock recommendation
    Use this when you need parsed, structured output for further processing
    """
    prompt = f"""
    Based on the following company information and financial metrics, provide a stock analysis:
    
    Company Data: {company_data}
    Financial Metrics: {financial_metrics}
    
    Provide a comprehensive analysis with recommendation, target price, and reasoning.
    """

    # Structured output - returns a Pydantic model instance
    return generate_structured(
        prompt=prompt,
        schema=StockAnalysis,
        model="gpt-4o-mini",  # Use better model for complex analysis
        temperature=0.2,  # Lower temperature for more consistent analysis
    )


# Example 4: Using different providers and strategies
def analyze_news_with_fallback(news_text: str) -> NewsAnalysis:
    """
    Analyze news with fallback strategy for reliability
    """
    # Configure the container for this use case
    configure_container(
        provider=LLMProvider.OPENAI,
        strategy=StrategyType.FALLBACK,
        model="gpt-4o-mini",
        fallback_models=["gpt-3.5-turbo", "gpt-4"],
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    facade = get_facade()

    prompt = f"""
    Analyze the following financial news and assess its market impact:
    
    {news_text}
    
    Provide sentiment analysis and impact assessment.
    """

    return facade.generate_structured(
        prompt=prompt, output_schema=NewsAnalysis, temperature=0.3
    )


# Example 5: Batch processing with custom LLM instance
def batch_analyze_companies(companies_data: List[dict]) -> List[StockAnalysis]:
    """
    Analyze multiple companies efficiently using a single LLM instance
    Use this pattern for batch processing to avoid recreating LLM instances
    """
    # Create LLM instance once for batch processing
    llm = create_openai_llm(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

    results = []
    for company in companies_data:
        prompt = f"""
        Analyze this company for investment potential:
        Name: {company.get('name')}
        Sector: {company.get('sector')}
        Financial Data: {company.get('financials')}
        Recent News: {company.get('news', 'No recent news')}
        """

        try:
            # Use the same LLM instance for all requests
            analysis = llm.generate_with_schema(
                prompt=prompt, output_schema=StockAnalysis
            )
            results.append(analysis)
        except Exception as e:
            print(f"Failed to analyze {company.get('name')}: {e}")
            continue

    return results


# Example 6: Real-world integration example
class FinancialAnalyzer:
    """
    Example class showing how to integrate LLM into your financial analysis workflow
    This is how you might use it in a real application
    """

    def __init__(self, api_key: Optional[str] = None):
        # Configure LLM on initialization
        configure_container(
            provider=LLMProvider.OPENAI,
            strategy=StrategyType.RETRY,  # Use retry for reliability
            model="gpt-4o-mini",
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            max_retries=3,
        )
        self.llm_facade = get_facade()

    def analyze_portfolio(self, portfolio_data: dict) -> dict:
        """Analyze entire portfolio"""
        prompt = f"""
        Analyze this investment portfolio and provide recommendations:
        
        Portfolio Holdings: {portfolio_data}
        
        Provide analysis on:
        - Overall portfolio performance
        - Risk assessment
        - Diversification recommendations
        - Rebalancing suggestions
        """

        analysis = self.llm_facade.generate_text(
            prompt=prompt, temperature=0.3, max_tokens=1000
        )

        return {"analysis": analysis, "timestamp": "2024-01-01"}  # Add metadata

    def get_market_insights(self, market_data: str) -> MarketSummary:
        """Get structured market insights"""
        prompt = f"""
        Based on current market data, provide a market summary:
        
        {market_data}
        
        Focus on overall sentiment, key trends, and important economic indicators.
        """

        return self.llm_facade.generate_structured(
            prompt=prompt, output_schema=MarketSummary, temperature=0.4
        )


# Example 7: Simple usage patterns for different modules
def demo_usage_patterns():
    """
    Demonstrate the most common usage patterns you'll use in other modules
    """

    print("=== FinSight LLM Module Usage Examples ===\n")

    # Pattern 1: Quick text generation (80% of use cases)
    print("1. Quick text generation:")
    try:
        result = generate_text(
            "Explain the concept of compound interest in simple terms",
            model="gpt-4o-mini",
        )
        print(f"Result: {result[:100]}...\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Pattern 2: Structured output (15% of use cases)
    print("2. Structured output generation:")
    try:
        analysis = generate_structured(
            prompt="Analyze AAPL stock: Current price $150, P/E ratio 25, growing revenue",
            schema=StockAnalysis,
            model="gpt-4o-mini",
        )
        print(f"Recommendation: {analysis.recommendation}")
        print(f"Target Price: ${analysis.target_price}")
        print(f"Confidence: {analysis.confidence}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Pattern 3: Provider comparison
    print("3. All providers demo:")
    demo_all_providers()

    # Pattern 4: Provider-specific features
    # demo_provider_specific_features()


# Example 8: Demonstrate all three providers
def demo_all_providers():
    """
    Demonstrate usage of all three LLM providers
    """
    print("=== Testing All LLM Providers ===\n")

    test_prompt = "Explain what is a stock market in 2 sentences."

    # Method 1: Using Factory
    # print("1. Using Factory:")

    # try:
    #     # OpenAI via Factory
    #     openai_llm = LLMFactory.get_llm(
    #         provider=LLMProvider.OPENAI,
    #         strategy=StrategyType.SIMPLE,
    #         default_model="gpt-4o-mini",
    #         api_key=os.getenv("OPENAI_API_KEY"),
    #     )
    #     result = openai_llm.simple_generate(test_prompt)
    #     print(f"OpenAI Factory: {result[:100]}...")
    # except Exception as e:
    #     print(f"OpenAI Factory Error: {e}")

    # try:
    #     # LangChain via Factory
    #     langchain_llm = LLMFactory.get_llm(
    #         provider=LLMProvider.LANGCHAIN,
    #         strategy=StrategyType.SIMPLE,
    #         model_name="gpt-4o-mini",
    #         api_key=os.getenv("OPENAI_API_KEY"),
    #         langchain_provider=LLMProvider.OPENAI,
    #     )
    #     result = langchain_llm.simple_generate(test_prompt)
    #     print(f"LangChain Factory: {result[:100]}...")
    # except Exception as e:
    #     print(f"LangChain Factory Error: {e}")

    try:
        # Google ADK via Factory
        google_adk_llm = LLMFactory.get_llm(
            provider=LLMProvider.GOOGLE_AGENT_DEVELOPMENT_KIT,
            strategy=StrategyType.SIMPLE,
            default_model="gemini-2.0-flash",
        )
        result = google_adk_llm.simple_generate(test_prompt)
        print(f"Google ADK Factory: {result[:100]}...")
    except Exception as e:
        print(f"Google ADK Factory Error: {e}")

    print("\n2. Using Direct Creation:")

    # try:
    #     # OpenAI Direct
    #     openai_direct = create_openai_llm(
    #         model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")
    #     )
    #     result = openai_direct.simple_generate(test_prompt)
    #     print(f"OpenAI Direct: {result[:100]}...")
    # except Exception as e:
    #     print(f"OpenAI Direct Error: {e}")

    # try:
    #     # LangChain Direct
    #     langchain_direct = create_langchain_llm(
    #         model="gpt-3.5-turbo",
    #         api_key=os.getenv("OPENAI_API_KEY"),
    #         langchain_provider=LLMProvider.OPENAI,
    #     )
    #     result = langchain_direct.simple_generate(test_prompt)
    #     print(f"LangChain Direct: {result[:100]}...")
    # except Exception as e:
    #     print(f"LangChain Direct Error: {e}")

    try:
        # Google ADK Direct
        google_adk_direct = create_google_adk_llm(model="gemini-2.0-flash")
        result = google_adk_direct.simple_generate(test_prompt)
        print(f"Google ADK Direct: {result[:100]}...")
    except Exception as e:
        print(f"Google ADK Direct Error: {e}")


def demo_provider_specific_features():
    """
    Demonstrate provider-specific features and use cases
    """
    print("\n=== Provider-Specific Features ===\n")

    # OpenAI - Best for general purpose and structured output
    print("1. OpenAI - Structured Analysis:")
    try:
        analysis = generate_structured(
            prompt="Analyze TSLA stock: Price $200, High volatility, EV market leader",
            schema=StockAnalysis,
            provider="openai",
            model="gpt-4o-mini",
        )
        print(f"   Recommendation: {analysis.recommendation}")
        print(f"   Confidence: {analysis.confidence}")
    except Exception as e:
        print(f"   Error: {e}")

    # LangChain - Good for complex workflows and multiple providers
    print("\n2. LangChain - Multi-step Analysis:")
    try:
        configure_container(
            provider=LLMProvider.LANGCHAIN,
            strategy=StrategyType.CHAIN_OF_THOUGHT,
            model="gpt-3.5-turbo",
        )
        facade = get_facade()
        result = facade.generate_text(
            "Should I invest in renewable energy stocks? Consider market trends.",
            temperature=0.4,
        )
        print(f"   Analysis: {result[:150]}...")
    except Exception as e:
        print(f"   Error: {e}")

    # Google ADK - Good for agent-based workflows
    print("\n3. Google ADK - Agent-based Generation:")
    try:
        google_llm = create_google_adk_llm(model="gemini-pro")
        result = google_llm.simple_generate(
            "What are the top 3 financial metrics for evaluating a company?"
        )
        print(f"   Metrics: {result[:150]}...")
    except Exception as e:
        print(f"   Error: {e}")


if __name__ == "__main__":
    # Set up API key (you would typically get this from environment or config)
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"

    # Run the demo
    demo_usage_patterns()

    # Example of what you'd typically do in other modules:
    """
    # In your trading module:
    from common.llm import generate_text, generate_structured
    
    def analyze_stock(symbol, data):
        analysis = generate_text(f"Analyze {symbol}: {data}")
        return analysis
    
    # In your reporting module:
    def create_report(portfolio_data):
        report = generate_text(f"Create portfolio report: {portfolio_data}")
        return report
    
    # In your risk module:
    def assess_risk(position_data):
        risk_analysis = generate_structured(
            f"Assess risk for: {position_data}",
            schema=RiskAssessment
        )
        return risk_analysis
    """
