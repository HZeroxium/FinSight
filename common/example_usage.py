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
    LLMProvider,
    StrategyType,
    configure_container,
    get_facade,
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
        model="gpt-4",  # Use better model for complex analysis
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
    # print("1. Quick text generation:")
    # try:
    #     result = generate_text(
    #         "Explain the concept of compound interest in simple terms",
    #         model="gpt-4o-mini",
    #     )
    #     print(f"Result: {result}...\n")
    # except Exception as e:
    #     print(f"Error: {e}\n")

    # # Pattern 2: Structured output (15% of use cases)
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

    # # Pattern 3: Using the analyzer class (5% of use cases - for complex workflows)
    # print("3. Using analyzer class:")
    # try:
    #     analyzer = FinancialAnalyzer()
    #     portfolio = {"AAPL": 0.3, "GOOGL": 0.2, "MSFT": 0.3, "BONDS": 0.2}
    #     insights = analyzer.analyze_portfolio(portfolio)
    #     print(f"Portfolio analysis: {insights['analysis'][:100]}...\n")
    # except Exception as e:
    #     print(f"Error: {e}\n")


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
