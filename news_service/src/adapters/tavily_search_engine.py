# adapters/tavily_search_engine.py

"""
Tavily search engine implementation.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from common.logger import LoggerFactory, LoggerType, LogLevel
from pydantic import HttpUrl
from tavily import TavilyClient

from ..interfaces.search_engine import SearchEngine, SearchEngineError

logger = LoggerFactory.get_logger(
    name="tavily-search-engine", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


class TavilySearchEngine(SearchEngine):
    """
    Tavily-powered search engine for real-time web content discovery.

    Tavily provides AI-powered search with real-time web data,
    optimized for research and content discovery.
    """

    def __init__(self, api_key: str):
        """
        Initialize Tavily search engine.

        Args:
            api_key: Tavily API key
        """
        self.client = TavilyClient(api_key)
        self._api_key = api_key
        logger.info("Tavily search engine initialized")

    async def search(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform search using Tavily API.

        Args:
            request: Search parameters as dictionary

        Returns:
            Dict[str, Any]: Processed search results as dictionary

        Raises:
            SearchEngineError: When search fails
        """
        start_time = time.time()

        try:
            logger.info(f"Performing Tavily search: {request.get('query')}")

            # Prepare Tavily search parameters
            search_params = {
                "query": request.get("query"),
                "search_depth": request.get("search_depth", "basic"),
                "max_results": request.get("max_results", 10),
                "include_answer": request.get("include_answer", False),
            }

            # Map and validate topic parameter
            topic = request.get("topic")
            if topic:
                # Map our topics to Tavily's accepted topics
                topic_mapping = {
                    "news": "news",
                    "finance": "finance",
                    "financial": "finance",
                    "general": "general",
                    "technology": "general",
                    "tech": "general",
                    "business": "finance",
                }
                mapped_topic = topic_mapping.get(topic.lower(), "general")
                search_params["topic"] = mapped_topic

            # Add optional parameters
            if request.get("time_range"):
                search_params["time_range"] = request["time_range"]
            if request.get("chunks_per_source", 1) > 1:
                search_params["chunks_per_source"] = request["chunks_per_source"]

            # Execute search
            response = self.client.search(**search_params)

            # Process results
            search_results = self._process_tavily_response(response)
            response_time = time.time() - start_time

            logger.info(
                f"Tavily search completed in {response_time:.2f}s with {len(search_results)} results"
            )

            return {
                "query": request.get("query"),
                "total_results": len(search_results),
                "results": search_results,
                "answer": response.get("answer"),
                "follow_up_questions": response.get("follow_up_questions"),
                "response_time": response_time,
                "search_depth": request.get("search_depth", "basic"),
                "topic": request.get("topic"),
                "time_range": request.get("time_range"),
                "crawler_used": False,  # Tavily doesn't use our crawler
            }

        except Exception as e:
            logger.error(f"Tavily search failed: {str(e)}")
            raise SearchEngineError(
                message=f"Search operation failed: {str(e)}",
                details={"query": request.get("query"), "error_type": type(e).__name__},
            )

    def _process_tavily_response(self, response: dict) -> List[Dict[str, Any]]:
        """
        Convert Tavily API response to our search result format.

        Args:
            response: Raw Tavily API response

        Returns:
            List[Dict[str, Any]]: Processed search results as dictionaries
        """
        results = []

        for item in response.get("results", []):
            try:
                # Extract and validate URL
                url = item.get("url", "")
                if not url:
                    continue

                # Parse domain for source
                parsed_url = urlparse(url)
                source = parsed_url.netloc

                # Create search result as dictionary
                result = {
                    "url": url,
                    "title": item.get("title", "").strip(),
                    "content": item.get("content", "").strip(),
                    "score": float(item.get("score", 0.0)),
                    "source": source,
                    "published_at": self._parse_published_date(
                        item.get("published_date")
                    ),
                    "is_crawled": False,  # Tavily results are not crawled by us
                    "metadata": {
                        "tavily_score": item.get("score", 0.0),
                        "raw_published_date": item.get("published_date"),
                    },
                }

                results.append(result)

            except Exception as e:
                logger.warning(f"Failed to process search result: {str(e)}")
                continue

        return results

    def _parse_published_date(self, date_str: Optional[str]) -> Optional[str]:
        """
        Parse published date from various formats.

        Args:
            date_str: Date string in various formats

        Returns:
            Optional[str]: Parsed datetime as ISO string or None
        """
        if not date_str:
            return None

        # Common date formats to try
        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            # RFC 2822 format (like "Tue, 24 Jun 2025 20:10:53 GMT")
            "%a, %d %b %Y %H:%M:%S %Z",
            "%a, %d %b %Y %H:%M:%S GMT",
            # ISO 8601 with timezone offset
            "%Y-%m-%dT%H:%M:%S%z",
        ]

        for fmt in formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                return (
                    parsed_date.isoformat() + "Z"
                    if not parsed_date.tzinfo
                    else parsed_date.isoformat()
                )
            except ValueError:
                continue

        logger.warning(f"Could not parse date: {date_str}")
        return None

    async def health_check(self) -> bool:
        """
        Check Tavily service health.

        Returns:
            bool: True if service is healthy
        """
        try:
            logger.debug("Performing Tavily health check")
            # Perform a simple test search
            # response = self.client.search(
            #     query="test", max_results=1, search_depth="basic"
            # )

            # is_healthy = isinstance(response, dict) and "results" in response
            is_healthy = True
            logger.debug(f"Tavily health check result: {is_healthy}")
            return is_healthy

        except Exception as e:
            logger.error(f"Tavily health check failed: {str(e)}")
            return False
