"""
Comprehensive gRPC test client for news service.

This script provides interactive testing capabilities for all gRPC endpoints,
serving as an alternative to Postman for gRPC testing.
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any


from grpc import aio

from common.logger import LoggerFactory, LoggerType, LogLevel

# Setup logging
logger = LoggerFactory.get_logger(
    name="grpc-test-client",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    console_level=LogLevel.INFO,
    use_colors=True,
)


class GrpcTestClient:
    """Interactive gRPC test client for news service."""

    def __init__(self, host: str = "localhost", port: int = 50051):
        self.host = host
        self.port = port
        self.channel = None
        self.stub = None

    async def connect(self) -> bool:
        """Connect to gRPC server."""
        try:
            self.channel = aio.insecure_channel(f"{self.host}:{self.port}")

            # Import generated gRPC modules
            from src.grpc_generated import (
                news_service_pb2_grpc,
                news_service_pb2,
            )

            self.stub = news_service_pb2_grpc.NewsServiceStub(self.channel)
            self.pb2 = news_service_pb2

            # Test connection with health check
            await self.health_check()
            logger.info(f"✅ Connected to gRPC server at {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to connect to gRPC server: {e}")
            return False

    async def disconnect(self):
        """Disconnect from gRPC server."""
        if self.channel:
            await self.channel.close()
            logger.info("✅ Disconnected from gRPC server")

    async def health_check(self) -> Dict[str, Any]:
        """Test health check endpoint."""
        try:
            request = self.pb2.HealthCheckRequest()
            response = await self.stub.HealthCheck(request, timeout=10.0)

            result = {
                "status": response.status,
                "service": response.service,
                "version": response.version,
                "timestamp": (
                    response.timestamp.ToDatetime() if response.timestamp else None
                ),
            }

            logger.info("✅ Health check successful")
            return result

        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            raise

    async def search_news(
        self,
        keywords: List[str],
        sources: Optional[List[str]] = None,
        limit: int = 10,
        skip: int = 0,
        sort_by: str = "published_at",
        sort_order: str = "desc",
    ) -> Dict[str, Any]:
        """Test search news endpoint."""
        try:
            request = self.pb2.SearchNewsRequest()
            request.keywords.extend(keywords)

            if sources:
                # Convert string sources to enum values
                source_mapping = {
                    "coindesk": self.pb2.NEWS_SOURCE_COINDESK,
                    "cointelegraph": self.pb2.NEWS_SOURCE_COINTELEGRAPH,
                    "tavily": self.pb2.NEWS_SOURCE_TAVILY,
                }

                for source in sources:
                    if source.lower() in source_mapping:
                        request.sources.append(source_mapping[source.lower()])

            request.limit = limit
            request.skip = skip
            request.sort_by = sort_by
            request.sort_order = sort_order

            response = await self.stub.SearchNews(request, timeout=30.0)

            result = {
                "total_count": response.total_count,
                "items_count": len(response.items),
                "skip": response.skip,
                "limit": response.limit,
                "has_more": response.has_more,
                "items": [],
            }

            # Convert items to dict format
            for item in response.items:
                item_dict = {
                    "id": item.id,
                    "title": item.title,
                    "content": item.content,
                    "url": item.url,
                    "source": item.source,
                    "published_at": (
                        item.published_at.ToDatetime() if item.published_at else None
                    ),
                    "fetched_at": (
                        item.fetched_at.ToDatetime() if item.fetched_at else None
                    ),
                    "author": item.author,
                    "tags": list(item.tags),
                }
                result["items"].append(item_dict)

            logger.info(f"✅ Search completed: found {result['total_count']} articles")
            return result

        except Exception as e:
            logger.error(f"❌ Search news failed: {e}")
            raise

    async def get_news_by_id(self, news_id: str) -> Dict[str, Any]:
        """Test get news by ID endpoint."""
        try:
            request = self.pb2.GetNewsByIdRequest()
            request.id = news_id

            response = await self.stub.GetNewsById(request, timeout=10.0)

            result = {
                "id": response.id,
                "title": response.title,
                "content": response.content,
                "url": response.url,
                "source": response.source,
                "published_at": (
                    response.published_at.ToDatetime()
                    if response.published_at
                    else None
                ),
                "fetched_at": (
                    response.fetched_at.ToDatetime() if response.fetched_at else None
                ),
                "author": response.author,
                "tags": list(response.tags),
            }

            logger.info(f"✅ Retrieved news item: {response.title}")
            return result

        except Exception as e:
            logger.error(f"❌ Get news by ID failed: {e}")
            raise

    async def get_latest_news(
        self,
        sources: Optional[List[str]] = None,
        limit: int = 10,
        skip: int = 0,
    ) -> Dict[str, Any]:
        """Test get latest news endpoint."""
        try:
            request = self.pb2.GetLatestNewsRequest()

            if sources:
                source_mapping = {
                    "coindesk": self.pb2.NEWS_SOURCE_COINDESK,
                    "cointelegraph": self.pb2.NEWS_SOURCE_COINTELEGRAPH,
                    "tavily": self.pb2.NEWS_SOURCE_TAVILY,
                }

                for source in sources:
                    if source.lower() in source_mapping:
                        request.sources.append(source_mapping[source.lower()])

            request.limit = limit
            request.skip = skip

            response = await self.stub.GetLatestNews(request, timeout=30.0)

            result = {
                "total_count": response.total_count,
                "items_count": len(response.items),
                "skip": response.skip,
                "limit": response.limit,
                "has_more": response.has_more,
                "items": [],
            }

            for item in response.items:
                item_dict = {
                    "id": item.id,
                    "title": item.title,
                    "content": (
                        item.content[:200] + "..."
                        if len(item.content) > 200
                        else item.content
                    ),
                    "url": item.url,
                    "source": item.source,
                    "published_at": (
                        item.published_at.ToDatetime() if item.published_at else None
                    ),
                    "author": item.author,
                    "tags": list(item.tags),
                }
                result["items"].append(item_dict)

            logger.info(f"✅ Retrieved {len(result['items'])} latest news items")
            return result

        except Exception as e:
            logger.error(f"❌ Get latest news failed: {e}")
            raise

    def print_json(self, data: Dict[str, Any], title: str = "Result"):
        """Pretty print JSON data."""
        print(f"\n📊 {title}:")
        print("=" * 50)
        print(json.dumps(data, indent=2, default=str))
        print("=" * 50)


async def interactive_test():
    """Interactive testing session."""
    client = GrpcTestClient()

    print("🚀 gRPC Test Client for News Service")
    print("=" * 50)

    # Connect to server
    if not await client.connect():
        return

    try:
        while True:
            print("\n📋 Available Commands:")
            print("1. Health Check")
            print("2. Search News")
            print("3. Get Latest News")
            print("4. Get News by ID")
            print("5. Exit")

            choice = input("\nEnter your choice (1-5): ").strip()

            if choice == "1":
                try:
                    result = await client.health_check()
                    client.print_json(result, "Health Check")
                except Exception as e:
                    print(f"❌ Error: {e}")

            elif choice == "2":
                keywords = (
                    input("Enter keywords (comma-separated): ").strip().split(",")
                )
                keywords = [k.strip() for k in keywords if k.strip()]

                sources = input(
                    "Enter sources (coindesk,cointelegraph,tavily) or press Enter for all: "
                ).strip()
                sources = (
                    [s.strip() for s in sources.split(",") if s.strip()]
                    if sources
                    else None
                )

                limit = int(input("Enter limit (default 10): ") or "10")

                try:
                    result = await client.search_news(keywords, sources, limit)
                    client.print_json(result, "Search Results")
                except Exception as e:
                    print(f"❌ Error: {e}")

            elif choice == "3":
                sources = input(
                    "Enter sources (coindesk,cointelegraph,tavily) or press Enter for all: "
                ).strip()
                sources = (
                    [s.strip() for s in sources.split(",") if s.strip()]
                    if sources
                    else None
                )

                limit = int(input("Enter limit (default 10): ") or "10")

                try:
                    result = await client.get_latest_news(sources, limit)
                    client.print_json(result, "Latest News")
                except Exception as e:
                    print(f"❌ Error: {e}")

            elif choice == "4":
                news_id = input("Enter news ID: ").strip()
                if news_id:
                    try:
                        result = await client.get_news_by_id(news_id)
                        client.print_json(result, "News Item")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    print("❌ News ID is required")

            elif choice == "5":
                break

            else:
                print("❌ Invalid choice. Please try again.")

    finally:
        await client.disconnect()


async def automated_test():
    """Run automated tests."""
    client = GrpcTestClient()

    print("🤖 Running Automated gRPC Tests")
    print("=" * 50)

    if not await client.connect():
        return False

    try:
        # Test 1: Health Check
        print("\n1️⃣ Testing Health Check...")
        health_result = await client.health_check()
        client.print_json(health_result, "Health Check")

        # Test 2: Search News
        print("\n2️⃣ Testing Search News...")
        search_result = await client.search_news(
            keywords=["bitcoin", "cryptocurrency"], sources=["coindesk"], limit=5
        )
        client.print_json(search_result, "Search Results")

        # Test 3: Get Latest News
        print("\n3️⃣ Testing Get Latest News...")
        latest_result = await client.get_latest_news(limit=3)
        client.print_json(latest_result, "Latest News")

        # Test 4: Get News by ID (if we have items from search)
        if search_result.get("items") and len(search_result["items"]) > 0:
            print("\n4️⃣ Testing Get News by ID...")
            news_id = search_result["items"][0]["id"]
            news_item = await client.get_news_by_id(news_id)
            client.print_json(news_item, "News Item Details")

        print("\n✅ All automated tests completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Automated test failed: {e}")
        return False

    finally:
        await client.disconnect()


async def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "auto":
        success = await automated_test()
        sys.exit(0 if success else 1)
    else:
        await interactive_test()


if __name__ == "__main__":
    asyncio.run(main())
