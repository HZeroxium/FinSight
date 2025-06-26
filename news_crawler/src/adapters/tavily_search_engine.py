"""
Tavily search engine implementation.
"""

import time
from typing import List, Optional
from datetime import datetime
from urllib.parse import urlparse

from tavily import TavilyClient
from pydantic import HttpUrl

from ..interfaces.search_engine import SearchEngine, SearchEngineError
from ..models.search import SearchRequest, SearchResponse, SearchResult
from ..common.logger import LoggerFactory, LoggerType, LogLevel

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

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Perform search using Tavily API.

        Args:
            request: Search parameters

        Returns:
            SearchResponse: Processed search results

        Raises:
            SearchEngineError: When search fails
        """
        start_time = time.time()

        try:
            logger.info(f"Performing Tavily search: {request.query}")

            # Prepare Tavily search parameters
            search_params = {
                "query": request.query,
                "search_depth": request.search_depth,
                "max_results": request.max_results,
                "include_answer": request.include_answer,
            }

            # Add optional parameters
            if request.topic:
                search_params["topic"] = request.topic
            if request.time_range:
                search_params["time_range"] = request.time_range
            if request.chunks_per_source > 1:
                search_params["chunks_per_source"] = request.chunks_per_source

            # Execute search
            response = self.client.search(**search_params)

            # Process results
            search_results = self._process_tavily_response(response)
            response_time = time.time() - start_time

            logger.info(
                f"Tavily search completed in {response_time:.2f}s with {len(search_results)} results"
            )

            return SearchResponse(
                query=request.query,
                total_results=len(search_results),
                results=search_results,
                answer=response.get("answer"),
                follow_up_questions=response.get("follow_up_questions"),
                response_time=response_time,
                search_depth=request.search_depth,
                topic=request.topic,
                time_range=request.time_range,
                crawler_used=False,  # Tavily doesn't use our crawler
            )

        except Exception as e:
            logger.error(f"Tavily search failed: {str(e)}")
            raise SearchEngineError(
                message=f"Search operation failed: {str(e)}",
                details={"query": request.query, "error_type": type(e).__name__},
            )

    def _process_tavily_response(self, response: dict) -> List[SearchResult]:
        """
        Convert Tavily API response to our SearchResult format.

        Args:
            response: Raw Tavily API response

        Returns:
            List[SearchResult]: Processed search results
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

                # Create search result
                result = SearchResult(
                    url=HttpUrl(url),
                    title=item.get("title", "").strip(),
                    content=item.get("content", "").strip(),
                    score=float(item.get("score", 0.0)),
                    source=source,
                    published_at=self._parse_published_date(item.get("published_date")),
                    is_crawled=False,  # Tavily results are not crawled by us
                    metadata={
                        "tavily_score": item.get("score", 0.0),
                        "raw_published_date": item.get("published_date"),
                    },
                )

                results.append(result)

            except Exception as e:
                logger.warning(f"Failed to process search result: {str(e)}")
                continue

        return results

    def _parse_published_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """
        Parse published date from various formats.

        Args:
            date_str: Date string in various formats

        Returns:
            Optional[datetime]: Parsed datetime or None
        """
        if not date_str:
            return None

        # Common date formats to try
        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
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
            response = self.client.search(
                query="test", max_results=1, search_depth="basic"
            )

            is_healthy = isinstance(response, dict) and "results" in response
            logger.debug(f"Tavily health check result: {is_healthy}")
            return is_healthy

        except Exception as e:
            logger.error(f"Tavily health check failed: {str(e)}")
            return False
