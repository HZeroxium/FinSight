# misc/crawl4ai_example.py

import asyncio
import json
import time

from common.logger import LoggerFactory, LoggerType, LogLevel
from crawl4ai_collector import Crawl4AICryptoNewsCollector

main_logger = LoggerFactory.get_logger(
    name="crawl4ai_demo",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    use_colors=True,
    log_file="logs/crawl4ai_demo.log",  # Specify log file path
    file_level=LogLevel.DEBUG,  # Log file will capture DEBUG level logs
)

# extracted_data = {
#                         "url": url,
#                         "title": result.metadata.get("title", ""),
#                         "description": result.metadata.get("description", ""),
#                         "keywords": result.metadata.get("keywords", ""),
#                         "content": result.markdown,
#                         "cleaned_html": result.cleaned_html,
#                         "links": result.links,
#                         "media": result.media,
#                         "metadata": result.metadata,
#                         "crawl_timestamp": datetime.now().isoformat(),
#                         "success": True,
#                     }

#                     self.logger.info(
#                         f"Successfully extracted {len(result.markdown)} chars of content"
#                     )
#                     return extracted_data


async def demonstrate_basic_extraction():
    """Demonstrate basic content extraction capabilities"""
    collector = Crawl4AICryptoNewsCollector()

    # Example crypto news URLs
    test_urls = [
        # "https://cointelegraph.com/category/latest-news",
        "https://www.coindesk.com/latest-crypto-news",
    ]

    for url in test_urls:
        try:
            result = await collector.basic_content_extraction(url)
            main_logger.info(f"Basic extraction for {url}:")
            main_logger.info(f"  Success: {result.get('success', False)}")
            if result.get("success"):
                main_logger.info(f"  Title: {result.get('title', 'N/A')}")
                main_logger.info(f" Description: {result.get('description', 'N/A')}")
                main_logger.info(f"  Keywords: {result.get('keywords', 'N/A')}")
                main_logger.info(f"  Media: {result.get('media', [])}")
                main_logger.info(f" Links: {result.get('links', [])}")
                # Save links to file with json format
                json_file = "extracted_links.json"
                with open(json_file, "w") as f:
                    json.dump(result.get("links", []), f, indent=2)
                main_logger.info(f" Metadata: {result.get('metadata', {})}")
                main_logger.info(f"  Content length: {len(result.get('content', ''))}")
                main_logger.debug(f"  Content preview: {result.get('content', '')}...")
                main_logger.debug(f"  Cleaned HTML: {result.get('cleaned_html', '')}")
            main_logger.info("-" * 50)
        except Exception as e:
            main_logger.error(f"Error with {url}: {e}")


async def demonstrate_advanced_css_extraction():
    """Demonstrate advanced CSS selector-based extraction"""
    collector = Crawl4AICryptoNewsCollector()

    # Use CoinDesk configuration
    coindesk_config = collector.crypto_news_sites["coindesk"]
    test_url = "https://www.coindesk.com/news/"

    try:
        result = await collector.advanced_css_extraction(test_url, coindesk_config)
        print("Advanced CSS extraction:")
        print(f"  Success: {result.get('success', False)}")
        if result.get("success"):
            structured = result.get("structured_content", {})
            print(f"  Extracted fields: {list(structured.keys())}")
        print("-" * 50)
    except Exception as e:
        print(f"Error in CSS extraction: {e}")


async def demonstrate_javascript_crawling():
    """Demonstrate JavaScript-heavy site crawling"""
    collector = Crawl4AICryptoNewsCollector()

    # Test with a JS-heavy crypto site
    test_url = "https://cointelegraph.com/news"

    try:
        result = await collector.javascript_heavy_site_crawling(test_url, wait_time=5)
        print("JavaScript crawling:")
        print(f"  Success: {result.get('success', False)}")
        if result.get("success"):
            print(f"  JS result: {result.get('js_execution_result', {})}")
        print("-" * 50)
    except Exception as e:
        print(f"Error in JS crawling: {e}")


async def demonstrate_multi_page_crawling():
    """Demonstrate multi-page news crawling"""
    collector = Crawl4AICryptoNewsCollector()

    # Use CoinDesk configuration
    coindesk_config = collector.crypto_news_sites["coindesk"]

    try:
        articles = await collector.multi_page_news_crawling(
            coindesk_config, max_pages=2
        )
        print("Multi-page crawling:")
        print(f"  Articles extracted: {len(articles)}")
        for i, article in enumerate(articles[:3]):  # Show first 3
            print(f"  Article {i+1}: {article.get('title', 'N/A')[:50]}...")
        print("-" * 50)
    except Exception as e:
        print(f"Error in multi-page crawling: {e}")


async def demonstrate_content_filtering():
    """Demonstrate content filtering and cleaning"""
    collector = Crawl4AICryptoNewsCollector()

    test_url = "https://www.coindesk.com/news/"

    try:
        result = await collector.content_filtering_and_cleaning(test_url)
        print("Content filtering:")
        print(f"  Success: {result.get('success', False)}")
        if result.get("success"):
            original_len = len(result.get("original_content", ""))
            filtered_len = len(result.get("filtered_content", ""))
            print(f"  Original content: {original_len} chars")
            print(f"  Filtered content: {filtered_len} chars")
            print(
                f"  Reduction: {((original_len - filtered_len) / original_len * 100):.1f}%"
            )
        print("-" * 50)
    except Exception as e:
        print(f"Error in content filtering: {e}")


async def demonstrate_structured_data_extraction():
    """Demonstrate structured data extraction"""
    collector = Crawl4AICryptoNewsCollector()

    test_url = "https://cointelegraph.com/news"

    try:
        result = await collector.structured_data_extraction(test_url)
        print("Structured data extraction:")
        print(f"  Success: {result.get('success', False)}")
        if result.get("success"):
            structured_data = result.get("structured_data", {})
            print(f"  JSON-LD objects: {len(structured_data.get('jsonLd', []))}")
            print(f"  OpenGraph fields: {len(structured_data.get('openGraph', {}))}")
            print(f"  Twitter fields: {len(structured_data.get('twitter', {}))}")
        print("-" * 50)
    except Exception as e:
        print(f"Error in structured data extraction: {e}")


async def demonstrate_sentiment_extraction():
    """Demonstrate sentiment-focused extraction"""
    collector = Crawl4AICryptoNewsCollector()

    test_url = "https://www.coindesk.com/news/"

    try:
        result = await collector.sentiment_focused_extraction(test_url)
        print("Sentiment-focused extraction:")
        print(f"  Success: {result.get('success', False)}")
        if result.get("success"):
            sentiment_content = result.get("sentiment_optimized_content", {})
            print(f"  Headline: {sentiment_content.get('headline', 'N/A')[:50]}...")
            print(
                f"  Sentiment text length: {len(sentiment_content.get('sentiment_text', ''))}"
            )
        print("-" * 50)
    except Exception as e:
        print(f"Error in sentiment extraction: {e}")


async def demonstrate_performance_crawling():
    """Demonstrate performance-optimized crawling"""
    collector = Crawl4AICryptoNewsCollector()

    # Test URLs for performance crawling
    test_urls = [
        "https://cointelegraph.com",
        "https://www.coindesk.com",
        "https://cryptonews.com",
    ]

    try:
        start_time = time.time()
        results = await collector.performance_optimized_crawling(
            test_urls, max_concurrent=3
        )
        end_time = time.time()

        print("Performance-optimized crawling:")
        print(f"  URLs processed: {len(results)}")
        print(f"  Time taken: {end_time - start_time:.2f} seconds")
        successful = sum(1 for r in results if r.get("success"))
        print(
            f"  Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)"
        )
        print("-" * 50)
    except Exception as e:
        print(f"Error in performance crawling: {e}")


async def main():
    """
    Main function to demonstrate all Crawl4AI capabilities
    """
    logger = LoggerFactory.get_logger(
        name="crawl4ai_demo",
        logger_type=LoggerType.STANDARD,
        level=LogLevel.INFO,
        use_colors=True,
    )

    logger.info("Starting Crawl4AI Comprehensive Demo for Crypto News")

    # Run all demonstrations
    demonstrations = [
        ("Basic Content Extraction", demonstrate_basic_extraction),
        # ("Advanced CSS Extraction", demonstrate_advanced_css_extraction),
        # ("JavaScript Site Crawling", demonstrate_javascript_crawling),
        # ("Multi-page Crawling", demonstrate_multi_page_crawling),
        # ("Content Filtering", demonstrate_content_filtering),
        # ("Structured Data Extraction", demonstrate_structured_data_extraction),
        # ("Sentiment-focused Extraction", demonstrate_sentiment_extraction),
        # ("Performance Crawling", demonstrate_performance_crawling),
    ]

    for demo_name, demo_func in demonstrations:
        try:
            logger.info(f"Running demonstration: {demo_name}")
            await demo_func()
            logger.info(f"Completed: {demo_name}")
        except Exception as e:
            logger.error(f"Failed demonstration {demo_name}: {e}")

        # Small delay between demonstrations
        await asyncio.sleep(2)

    logger.info("Crawl4AI Demo completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
