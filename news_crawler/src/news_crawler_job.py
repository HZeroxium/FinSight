# news_crawler_job.py

"""
News Crawler Job

Production-ready job runner for automated news collection with flexible configuration
and both cron job and manual execution support.
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

from .services.news_collector_service import (
    NewsCollectorService,
    BatchCollectionRequest,
)
from .services.news_service import NewsService
from .repositories.mongo_news_repository import MongoNewsRepository
from .schemas.news_schemas import NewsSource
from .core.news_collector_factory import CollectorType
from .common.logger import LoggerFactory, LoggerType, LogLevel


class NewsCrawlerJob:
    """
    Production news crawler job with comprehensive configuration and monitoring
    """

    def __init__(
        self,
        mongo_url: str = "mongodb://localhost:27017",
        database_name: str = "finsight_news",
        job_config_file: str = "news_crawler_config.json",
        log_file: str = "logs/news_crawler_job.log",
    ):
        """
        Initialize news crawler job

        Args:
            mongo_url: MongoDB connection URL
            database_name: Database name for storing news
            job_config_file: Configuration file path
            log_file: Log file path
        """
        self.mongo_url = mongo_url
        self.database_name = database_name
        self.job_config_file = Path(job_config_file)

        # Initialize logging
        self.logger = LoggerFactory.get_logger(
            name="news-crawler-job",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file=log_file,
        )

        # Initialize components
        self.repository: Optional[MongoNewsRepository] = None
        self.news_service: Optional[NewsService] = None
        self.collector_service: Optional[NewsCollectorService] = None

    async def initialize(self) -> None:
        """Initialize all job components"""
        try:
            # Initialize repository
            self.repository = MongoNewsRepository(
                mongo_url=self.mongo_url, database_name=self.database_name
            )
            await self.repository.initialize()

            # Initialize services
            self.news_service = NewsService(self.repository)
            self.collector_service = NewsCollectorService(
                news_service=self.news_service, use_cache=True, enable_fallback=True
            )

            self.logger.info("News crawler job initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize job: {e}")
            raise

    def load_job_config(self) -> Dict[str, Any]:
        """Load job configuration from file with defaults"""
        default_config = {
            "sources": [source.value for source in NewsSource],
            "collector_preferences": {
                NewsSource.COINDESK.value: CollectorType.API_REST.value,
                NewsSource.COINTELEGRAPH.value: CollectorType.API_GRAPHQL.value,
            },
            "max_items_per_source": 100,
            "enable_fallback": True,
            "config_overrides": {},
            "notification": {"enabled": False, "webhook_url": None, "email": None},
        }

        try:
            if self.job_config_file.exists():
                with open(self.job_config_file, "r") as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    self.logger.info(
                        f"Loaded configuration from {self.job_config_file}"
                    )
                    return config
            else:
                self.logger.info("Using default configuration")
                self._save_default_config(default_config)
                return default_config

        except Exception as e:
            self.logger.warning(f"Failed to load config file: {e}, using defaults")
            return default_config

    def _save_default_config(self, config: Dict[str, Any]) -> None:
        """Save default configuration to file"""
        try:
            self.job_config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.job_config_file, "w") as f:
                json.dump(config, f, indent=2)
            self.logger.info(f"Saved default configuration to {self.job_config_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save default config: {e}")

    async def run_collection_job(
        self, config_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the main news collection job

        Args:
            config_override: Optional configuration override

        Returns:
            Job execution results
        """
        if not all([self.repository, self.news_service, self.collector_service]):
            await self.initialize()

        start_time = datetime.now(timezone.utc)
        self.logger.info(f"Starting news collection job at {start_time}")

        try:
            # Load configuration
            config = self.load_job_config()
            if config_override:
                config.update(config_override)

            # Convert source strings to NewsSource enums
            sources = [NewsSource(source) for source in config["sources"]]

            # Create batch collection request
            request = BatchCollectionRequest(
                sources=sources,
                collector_preferences=config.get("collector_preferences"),
                max_items_per_source=config.get("max_items_per_source"),
                config_overrides=config.get("config_overrides"),
                enable_fallback=config.get("enable_fallback", True),
            )

            # Execute collection
            results = await self.collector_service.collect_and_store_batch(request)

            # Calculate execution time
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            # Prepare final results
            job_results = {
                "job_id": f"news_crawl_{int(start_time.timestamp())}",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "status": "success",
                "config_used": config,
                "collection_results": results,
                "summary": {
                    "total_sources": len(sources),
                    "successful_sources": sum(
                        1
                        for r in results["source_results"].values()
                        if r["collection_success"]
                    ),
                    "total_items_collected": results["total_items_collected"],
                    "total_items_stored": results["total_items_stored"],
                    "total_duplicates": results["total_items_duplicated"],
                },
            }

            self.logger.info(
                f"Job completed successfully - "
                f"Duration: {duration:.2f}s, "
                f"Sources: {job_results['summary']['successful_sources']}/{job_results['summary']['total_sources']}, "
                f"Items collected: {job_results['summary']['total_items_collected']}, "
                f"Items stored: {job_results['summary']['total_items_stored']}"
            )

            # Send notifications if configured
            await self._send_notifications(job_results, config)

            return job_results

        except Exception as e:
            error_time = datetime.now(timezone.utc)
            duration = (error_time - start_time).total_seconds()

            self.logger.error(f"Job failed after {duration:.2f}s: {e}")

            return {
                "job_id": f"news_crawl_{int(start_time.timestamp())}",
                "start_time": start_time.isoformat(),
                "end_time": error_time.isoformat(),
                "duration_seconds": duration,
                "status": "failed",
                "error": str(e),
                "summary": {
                    "total_sources": 0,
                    "successful_sources": 0,
                    "total_items_collected": 0,
                    "total_items_stored": 0,
                },
            }

    async def run_quick_collection(
        self, sources: Optional[List[str]] = None, max_items: int = 50
    ) -> Dict[str, Any]:
        """
        Run a quick collection for development/testing

        Args:
            sources: List of source names (default: all)
            max_items: Maximum items per source

        Returns:
            Quick collection results
        """
        self.logger.info("Running quick collection for development")

        if sources is None:
            sources = [source.value for source in NewsSource]

        config_override = {
            "sources": sources,
            "max_items_per_source": max_items,
            "enable_fallback": True,
        }

        return await self.run_collection_job(config_override)

    async def get_job_status(self) -> Dict[str, Any]:
        """Get current job and repository status"""
        status = {
            "job_initialized": all(
                [self.repository, self.news_service, self.collector_service]
            ),
            "available_adapters": {},
            "repository_stats": {},
            "config_file": str(self.job_config_file),
            "config_exists": self.job_config_file.exists(),
        }

        try:
            if self.collector_service:
                status["available_adapters"] = (
                    self.collector_service.get_available_adapters()
                )

            if self.news_service:
                status["repository_stats"] = (
                    await self.news_service.get_repository_stats()
                )

        except Exception as e:
            self.logger.warning(f"Failed to get complete status: {e}")
            status["status_error"] = str(e)

        return status

    async def _send_notifications(
        self, results: Dict[str, Any], config: Dict[str, Any]
    ) -> None:
        """Send job completion notifications if configured"""
        notification_config = config.get("notification", {})

        if not notification_config.get("enabled", False):
            return

        try:
            # Implementation would depend on your notification preferences
            # For now, just log the notification
            summary = results["summary"]
            message = (
                f"News crawl job completed:\n"
                f"- Sources: {summary['successful_sources']}/{summary['total_sources']}\n"
                f"- Items collected: {summary['total_items_collected']}\n"
                f"- Items stored: {summary['total_items_stored']}\n"
                f"- Duration: {results['duration_seconds']:.2f}s"
            )

            self.logger.info(f"Notification: {message}")

            # Add webhook/email notification logic here if needed

        except Exception as e:
            self.logger.warning(f"Failed to send notifications: {e}")

    async def close(self) -> None:
        """Close job and cleanup resources"""
        try:
            if self.repository:
                await self.repository.close()
            self.logger.info("News crawler job closed")
        except Exception as e:
            self.logger.error(f"Error closing job: {e}")


# Convenience functions for different use cases
async def run_daily_news_crawl(
    mongo_url: str = "mongodb://localhost:27017", database_name: str = "finsight_news"
) -> Dict[str, Any]:
    """
    Run daily news crawl - suitable for cron jobs

    Args:
        mongo_url: MongoDB connection URL
        database_name: Database name

    Returns:
        Crawl results
    """
    job = NewsCrawlerJob(mongo_url=mongo_url, database_name=database_name)
    try:
        return await job.run_collection_job()
    finally:
        await job.close()


async def run_dev_news_crawl(
    sources: Optional[List[str]] = None,
    max_items: int = 20,
    mongo_url: str = "mongodb://localhost:27017",
) -> Dict[str, Any]:
    """
    Run development news crawl - for manual testing

    Args:
        sources: List of source names
        max_items: Maximum items per source
        mongo_url: MongoDB connection URL

    Returns:
        Crawl results
    """
    job = NewsCrawlerJob(mongo_url=mongo_url, database_name="finsight_news_dev")
    try:
        return await job.run_quick_collection(sources=sources, max_items=max_items)
    finally:
        await job.close()


async def main():
    """Main function for CLI execution"""
    import argparse

    parser = argparse.ArgumentParser(description="News Crawler Job")
    parser.add_argument(
        "--mode",
        choices=["production", "dev", "status"],
        default="dev",
        help="Execution mode",
    )
    parser.add_argument("--sources", nargs="+", help="News sources to crawl")
    parser.add_argument(
        "--max-items", type=int, default=50, help="Max items per source"
    )
    parser.add_argument(
        "--mongo-url", default="mongodb://localhost:27017", help="MongoDB URL"
    )
    parser.add_argument("--database", default="finsight_news", help="Database name")

    args = parser.parse_args()

    if args.mode == "production":
        results = await run_daily_news_crawl(
            mongo_url=args.mongo_url, database_name=args.database
        )
    elif args.mode == "dev":
        results = await run_dev_news_crawl(
            sources=args.sources, max_items=args.max_items, mongo_url=args.mongo_url
        )
    elif args.mode == "status":
        job = NewsCrawlerJob(mongo_url=args.mongo_url, database_name=args.database)
        try:
            await job.initialize()
            results = await job.get_job_status()
        finally:
            await job.close()

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
