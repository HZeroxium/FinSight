#!/usr/bin/env python3
"""
Standalone gRPC server for news service.

This script runs the gRPC server independently from the FastAPI server,
useful for microservices deployment or when only gRPC access is needed.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.news_service import NewsService
from src.repositories.mongo_news_repository import MongoNewsRepository
from src.grpc_services import run_grpc_server_standalone
from src.core.config import settings
from common.logger import LoggerFactory, LoggerType, LogLevel

# Setup logging
logger = LoggerFactory.get_logger(
    name="grpc-standalone",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    console_level=LogLevel.INFO,
    use_colors=True,
    log_file="logs/grpc_standalone.log",
)


async def main():
    """Main entry point for standalone gRPC server."""
    try:
        logger.info("🚀 Starting standalone gRPC server...")
        logger.info(f"📊 Environment: {settings.environment}")
        logger.info(f"🔌 gRPC Host: {settings.grpc_host}:{settings.grpc_port}")

        # Initialize MongoDB repository
        logger.info("📋 Initializing MongoDB repository...")
        repository = MongoNewsRepository(
            mongo_url=settings.mongodb_url, database_name=settings.mongodb_database
        )
        await repository.initialize()
        logger.info("✅ MongoDB repository initialized")

        # Initialize news service
        logger.info("📋 Initializing news service...")
        news_service = NewsService(repository)
        logger.info("✅ News service initialized")

        # Run gRPC server
        await run_grpc_server_standalone(news_service)

    except KeyboardInterrupt:
        logger.info("👋 Received shutdown signal, exiting...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
