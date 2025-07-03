"""
CoinTelegraph Historical News Crawler

This module implements a specialized crawler for collecting historical news data
from CoinTelegraph's GraphQL API with comprehensive error handling and progress tracking.
"""

import asyncio
import json
import random
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path

from .adapters.api_news_collector import APINewsCollector
from .schemas.news_schemas import NewsCollectorConfig, NewsSource
from .services.news_service import NewsService
from .repositories.mongo_news_repository import MongoNewsRepository
from .common.logger import LoggerFactory, LoggerType, LogLevel


class CoinTelegraphCrawler:
    """
    Historical news crawler for CoinTelegraph with progress tracking and error recovery
    """

    def __init__(
        self,
        mongo_url: str = "mongodb://localhost:27017",
        database_name: str = "finsight_news",
        progress_file: str = "cointelegraph_progress.json",
    ):
        """
        Initialize CoinTelegraph crawler

        Args:
            mongo_url: MongoDB connection URL
            database_name: Database name for storing news
            progress_file: File to track crawling progress
        """
        self.mongo_url = mongo_url
        self.database_name = database_name
        self.progress_file = Path(progress_file)

        # Initialize logging
        self.logger = LoggerFactory.get_logger(
            name="cointelegraph-crawler",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file="logs/cointelegraph_crawler.log",
        )

        # Initialize components
        self.repository: Optional[MongoNewsRepository] = None
        self.news_service: Optional[NewsService] = None
        self.collector: Optional[APINewsCollector] = None

    async def initialize(self) -> None:
        """Initialize crawler components"""
        try:
            # Initialize repository
            self.repository = MongoNewsRepository(
                mongo_url=self.mongo_url, database_name=self.database_name
            )
            await self.repository.initialize()

            # Initialize news service
            self.news_service = NewsService(self.repository)

            # Initialize API collector with enhanced configuration
            config = NewsCollectorConfig(
                source=NewsSource.COINTELEGRAPH,
                url="https://conpletus.cointelegraph.com/v1/",
                timeout=45,  # Increased timeout
                max_items=10,
                retry_attempts=5,  # Increased retry attempts
                retry_delay=3,  # Increased base delay
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",  # Will be overridden by BrowserSession
            )
            self.collector = APINewsCollector(config)

            self.logger.info("CoinTelegraph crawler initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize crawler: {e}")
            raise

    async def crawl_historical_data(
        self,
        start_offset: int = 0,
        end_offset: int = 49390,
        batch_size: int = 10,
        delay_between_requests: float = 3.0,  # Increased delay
        checkpoint_interval: int = 50,  # Reduced checkpoint frequency
    ) -> Dict[str, Any]:
        """
        Crawl historical data with enhanced error handling and delays

        Args:
            start_offset: Starting offset for crawling
            end_offset: Ending offset for crawling
            batch_size: Number of items per request
            delay_between_requests: Delay between requests in seconds
            checkpoint_interval: Save progress every N requests

        Returns:
            Crawling results summary
        """
        if not all([self.repository, self.news_service, self.collector]):
            await self.initialize()

        # Load previous progress
        progress = self._load_progress()
        current_offset = max(start_offset, progress.get("last_offset", start_offset))

        self.logger.info(
            f"Starting historical crawl from offset {current_offset} to {end_offset}"
        )

        # Initialize counters
        total_collected = progress.get("total_collected", 0)
        total_stored = progress.get("total_stored", 0)
        total_duplicates = progress.get("total_duplicates", 0)
        total_failed = progress.get("total_failed", 0)
        total_errors = progress.get("total_errors", 0)
        start_time = datetime.now(timezone.utc)

        try:
            while current_offset <= end_offset:
                try:
                    self.logger.info(
                        f"Crawling offset {current_offset} (batch size: {batch_size})"
                    )

                    # Collect news batch
                    result = await self.collector.collect_news(
                        max_items=batch_size, offset=current_offset, limit=batch_size
                    )

                    if result.success and result.items:
                        # Store collected items
                        storage_result = await self.news_service.store_news_items_bulk(
                            result.items
                        )

                        # Update counters
                        batch_collected = len(result.items)
                        batch_stored = storage_result["stored_count"]
                        batch_duplicates = storage_result["duplicate_count"]
                        batch_failed = storage_result["failed_count"]

                        total_collected += batch_collected
                        total_stored += batch_stored
                        total_duplicates += batch_duplicates
                        total_failed += batch_failed

                        self.logger.info(
                            f"Batch completed - Collected: {batch_collected}, "
                            f"Stored: {batch_stored}, Duplicates: {batch_duplicates}, "
                            f"Failed: {batch_failed}"
                        )

                        # Check if we got fewer items than requested (end of data)
                        if len(result.items) < 2:
                            self.logger.info("Reached end of available data")
                            break

                    else:
                        self.logger.warning(
                            f"Failed to collect batch at offset {current_offset}: {result.error_message}"
                        )
                        total_errors += 1

                        # Special handling for 403 errors - longer backoff
                        if "403" in str(result.error_message):
                            backoff_time = random.uniform(
                                30, 60
                            )  # 30-60 second backoff
                            self.logger.info(
                                f"403 detected - backing off for {backoff_time:.1f} seconds"
                            )
                            await asyncio.sleep(backoff_time)

                    # Update offset
                    current_offset += batch_size

                    # Save progress at checkpoints
                    if (current_offset - start_offset) % (
                        checkpoint_interval * batch_size
                    ) == 0:
                        self._save_progress(
                            {
                                "last_offset": current_offset,
                                "total_collected": total_collected,
                                "total_stored": total_stored,
                                "total_duplicates": total_duplicates,
                                "total_failed": total_failed,
                                "total_errors": total_errors,
                                "last_update": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        self.logger.info(f"Progress saved at offset {current_offset}")

                    # Enhanced rate limiting with jitter
                    if delay_between_requests > 0:
                        # Add random jitter to avoid patterns
                        jitter = random.uniform(0.5, 1.5)
                        actual_delay = delay_between_requests * jitter
                        self.logger.debug(f"Sleeping for {actual_delay:.1f} seconds")
                        await asyncio.sleep(actual_delay)

                except Exception as e:
                    total_errors += 1
                    self.logger.error(f"Error processing offset {current_offset}: {e}")

                    # Save progress on error
                    self._save_progress(
                        {
                            "last_offset": current_offset,
                            "total_collected": total_collected,
                            "total_stored": total_stored,
                            "total_duplicates": total_duplicates,
                            "total_failed": total_failed,
                            "total_errors": total_errors,
                            "last_error": str(e),
                            "last_update": datetime.now(timezone.utc).isoformat(),
                        }
                    )

                    # Enhanced exponential backoff on errors
                    error_delay = min(
                        delay_between_requests * (2 ** min(total_errors, 5)), 300
                    )  # Max 5 minutes
                    jitter = random.uniform(0.8, 1.2)
                    actual_error_delay = error_delay * jitter

                    self.logger.info(
                        f"Waiting {actual_error_delay:.1f} seconds before retry..."
                    )
                    await asyncio.sleep(actual_error_delay)

                    current_offset += batch_size  # Move to next batch even on error

        except KeyboardInterrupt:
            self.logger.info("Crawling interrupted by user")
        except Exception as e:
            self.logger.error(f"Critical error during crawling: {e}")
            raise
        finally:
            # Final progress save
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            final_progress = {
                "last_offset": current_offset,
                "total_collected": total_collected,
                "total_stored": total_stored,
                "total_duplicates": total_duplicates,
                "total_failed": total_failed,
                "total_errors": total_errors,
                "duration_seconds": duration,
                "completed": current_offset > end_offset,
                "last_update": end_time.isoformat(),
            }

            self._save_progress(final_progress)

            self.logger.info(
                f"Crawling session completed - "
                f"Duration: {duration:.2f}s, "
                f"Collected: {total_collected}, "
                f"Stored: {total_stored}, "
                f"Duplicates: {total_duplicates}, "
                f"Errors: {total_errors}"
            )

        return {
            "start_offset": start_offset,
            "end_offset": end_offset,
            "final_offset": current_offset,
            "total_collected": total_collected,
            "total_stored": total_stored,
            "total_duplicates": total_duplicates,
            "total_failed": total_failed,
            "total_errors": total_errors,
            "duration_seconds": duration,
            "completed": current_offset > end_offset,
        }

    def _load_progress(self) -> Dict[str, Any]:
        """Load crawling progress from file"""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, "r") as f:
                    progress = json.load(f)
                    self.logger.info(f"Loaded progress from {self.progress_file}")
                    return progress
        except Exception as e:
            self.logger.warning(f"Failed to load progress file: {e}")

        return {}

    def _save_progress(self, progress: Dict[str, Any]) -> None:
        """Save crawling progress to file"""
        try:
            # Ensure directory exists
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.progress_file, "w") as f:
                json.dump(progress, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")

    async def get_crawl_status(self) -> Dict[str, Any]:
        """Get current crawling status"""
        progress = self._load_progress()

        # Get repository stats if available
        repository_stats = {}
        if self.news_service:
            try:
                repository_stats = await self.news_service.get_repository_stats()
            except Exception as e:
                self.logger.warning(f"Failed to get repository stats: {e}")

        return {
            "progress": progress,
            "repository_stats": repository_stats,
            "progress_file": str(self.progress_file),
        }

    async def close(self) -> None:
        """Close crawler and cleanup resources"""
        try:
            if self.repository:
                await self.repository.close()
            self.logger.info("CoinTelegraph crawler closed")
        except Exception as e:
            self.logger.error(f"Error closing crawler: {e}")


async def main():
    """Main function for running the crawler"""
    crawler = CoinTelegraphCrawler()

    try:
        await crawler.initialize()

        # Start crawling with more conservative settings
        result = await crawler.crawl_historical_data(
            start_offset=0,
            end_offset=49390,  # Reduced for testing
            batch_size=1000,  # Smaller batch size
            delay_between_requests=5.0,  # Longer delay (5 seconds)
            checkpoint_interval=20,  # More frequent checkpoints
        )

        print("Crawling completed!")
        print(f"Results: {result}")

    except Exception as e:
        print(f"Crawling failed: {e}")
    finally:
        await crawler.close()


if __name__ == "__main__":
    asyncio.run(main())
