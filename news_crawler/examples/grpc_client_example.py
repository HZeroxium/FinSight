# examples/grpc_client_example.py

"""
Example gRPC client for the news service.

This demonstrates how to connect to and use the gRPC news service,
including various query types and error handling.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import grpc
from google.protobuf.timestamp_pb2 import Timestamp

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.logger import LoggerFactory, LoggerType, LogLevel

# Setup logging
logger = LoggerFactory.get_logger(
    name="grpc-client-example",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    console_level=LogLevel.INFO,
    use_colors=True,
    log_file="logs/grpc_client_example.log",
)


class NewsGrpcClient:
    """Example gRPC client for news service."""

    def __init__(self, host: str = "localhost", port: int = 50051):
        """
        Initialize gRPC client.

        Args:
            host: gRPC server host
            port: gRPC server port
        """
        self.host = host
        self.port = port
        self.channel = None
        self.stub = None

    async def connect(self):
        """Connect to gRPC server."""
        try:
            # Import generated gRPC code
            from src.grpc_generated import news_service_pb2_grpc

            # Create channel and stub
            self.channel = grpc.aio.insecure_channel(f"{self.host}:{self.port}")
            self.stub = news_service_pb2_grpc.NewsServiceStub(self.channel)

            logger.info(f"Connected to gRPC server at {self.host}:{self.port}")

        except ImportError:
            logger.error(
                "Failed to import generated gRPC code. Please run generate_grpc_code.py first."
            )
            raise
        except Exception as e:
            logger.error(f"Failed to connect to gRPC server: {e}")
            raise

    async def disconnect(self):
        """Disconnect from gRPC server."""
        if self.channel:
            await self.channel.close()
            logger.info("Disconnected from gRPC server")

    async def search_news_example(self):
        """Example: Search news with various filters."""
        try:
            from src.grpc_generated import news_service_pb2

            logger.info("=== Search News Example ===")

            # Create search request
            request = news_service_pb2.SearchNewsRequest()
            request.keywords.extend(["bitcoin", "cryptocurrency"])
            request.source = news_service_pb2.NEWS_SOURCE_COINDESK
            request.limit = 5

            # Execute search
            response = await self.stub.SearchNews(request)

            logger.info(f"Found {len(response.items)} news items")
            logger.info(f"Total count: {response.total_count}")
            logger.info(f"Has more: {response.has_more}")

            # Print items
            for i, item in enumerate(response.items, 1):
                logger.info(f"  {i}. {item.title[:80]}...")
                logger.info(f"     URL: {item.url}")
                logger.info(f"     Tags: {', '.join(item.tags)}")

        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()} - {e.details()}")
        except Exception as e:
            logger.error(f"Search news error: {e}")

    async def get_recent_news_example(self):
        """Example: Get recent news."""
        try:
            from src.grpc_generated import news_service_pb2

            logger.info("=== Get Recent News Example ===")

            # Create request
            request = news_service_pb2.GetRecentNewsRequest()
            request.hours = 24
            request.limit = 3

            # Execute request
            response = await self.stub.GetRecentNews(request)

            logger.info(f"Found {len(response.items)} recent news items")

            for i, item in enumerate(response.items, 1):
                logger.info(f"  {i}. {item.title[:60]}...")
                logger.info(
                    f"     Source: {news_service_pb2.NewsSource.Name(item.source)}"
                )

        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()} - {e.details()}")
        except Exception as e:
            logger.error(f"Get recent news error: {e}")

    async def get_news_by_tags_example(self):
        """Example: Get news by tags."""
        try:
            from src.grpc_generated import news_service_pb2

            logger.info("=== Get News by Tags Example ===")

            # Create request
            request = news_service_pb2.GetNewsByTagsRequest()
            request.tags.extend(["crypto", "blockchain"])
            request.limit = 3

            # Execute request
            response = await self.stub.GetNewsByTags(request)

            logger.info(f"Found {len(response.items)} items with specified tags")

            for i, item in enumerate(response.items, 1):
                logger.info(f"  {i}. {item.title[:60]}...")
                logger.info(f"     Tags: {', '.join(item.tags)}")

        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()} - {e.details()}")
        except Exception as e:
            logger.error(f"Get news by tags error: {e}")

    async def get_available_tags_example(self):
        """Example: Get available tags."""
        try:
            from src.grpc_generated import news_service_pb2

            logger.info("=== Get Available Tags Example ===")

            # Create request
            request = news_service_pb2.GetAvailableTagsRequest()
            request.limit = 10

            # Execute request
            response = await self.stub.GetAvailableTags(request)

            logger.info(f"Found {len(response.tags)} unique tags")
            logger.info(f"Tags: {', '.join(response.tags[:10])}")

        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()} - {e.details()}")
        except Exception as e:
            logger.error(f"Get available tags error: {e}")

    async def get_statistics_example(self):
        """Example: Get news statistics."""
        try:
            from google.protobuf.empty_pb2 import Empty

            logger.info("=== Get Statistics Example ===")

            # Execute request
            response = await self.stub.GetNewsStatistics(Empty())

            logger.info(f"Total articles: {response.total_articles}")
            logger.info(f"Recent articles (24h): {response.recent_articles_24h}")
            logger.info("Articles by source:")
            for source, count in response.articles_by_source.items():
                logger.info(f"  {source}: {count}")

        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()} - {e.details()}")
        except Exception as e:
            logger.error(f"Get statistics error: {e}")

    async def run_all_examples(self):
        """Run all example methods."""
        await self.connect()

        try:
            await self.search_news_example()
            await asyncio.sleep(1)

            await self.get_recent_news_example()
            await asyncio.sleep(1)

            await self.get_news_by_tags_example()
            await asyncio.sleep(1)

            await self.get_available_tags_example()
            await asyncio.sleep(1)

            await self.get_statistics_example()

        finally:
            await self.disconnect()


async def main():
    """Main entry point for gRPC client examples."""
    try:
        logger.info("🚀 Starting gRPC client examples...")

        # Create client
        client = NewsGrpcClient()

        # Run examples
        await client.run_all_examples()

        logger.info("✅ All examples completed successfully")

    except Exception as e:
        logger.error(f"❌ Client examples failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
