# misc/coindesk_news_crawler.py

"""
CoinDesk Historical News Crawler

This module implements a specialized crawler for collecting historical news data
from CoinDesk's REST API with timestamp-based pagination and comprehensive error handling.
"""

import asyncio
import json
import random
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path

from ..adapters.api_coindesk_news_collector import APICoinDeskNewsCollector
from ..schemas.news_schemas import NewsCollectorConfig, NewsSource
from ..services.news_service import NewsService
from ..repositories.mongo_news_repository import MongoNewsRepository
from ..common.logger import LoggerFactory, LoggerType, LogLevel


class CoinDeskCrawler:
    """
    Historical news crawler for CoinDesk with timestamp-based pagination and error recovery
    """

    def __init__(
        self,
        mongo_url: str = "mongodb://localhost:27017",
        database_name: str = "finsight_coindesk_news",
        progress_file: str = "data/progress/sequential/coindesk/coindesk_progress.json",
    ):
        """
        Initialize CoinDesk crawler

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
            name="coindesk-crawler",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file="logs/coindesk_crawler.log",
        )

        # Initialize components
        self.repository: Optional[MongoNewsRepository] = None
        self.news_service: Optional[NewsService] = None
        self.collector: Optional[APICoinDeskNewsCollector] = None

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

            # Initialize API collector
            config = NewsCollectorConfig(
                source=NewsSource.COINDESK,
                url="https://data-api.coindesk.com/news/v1/article/list",
                timeout=30,
                max_items=50,
                retry_attempts=5,
                retry_delay=2,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )
            self.collector = APICoinDeskNewsCollector(config)

            self.logger.info("CoinDesk crawler initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize crawler: {e}")
            raise

    async def crawl_historical_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        batch_size: int = 50,
        delay_between_requests: float = 3.0,
        checkpoint_interval: int = 50,
        fallback_interval: timedelta = timedelta(days=1),
        lang: str = "EN",
        source_ids: Optional[list] = None,
        categories: Optional[list] = None,
        exclude_categories: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Crawl historical data with timestamp-based pagination

        Args:
            start_date: Starting date for crawling (defaults to 30 days ago)
            end_date: Ending date for crawling (defaults to now)
            batch_size: Number of items per request
            delay_between_requests: Delay between requests in seconds
            checkpoint_interval: Save progress every N requests
            fallback_interval: Minimum time to jump back when no data or small batches
            lang: Language filter
            source_ids: List of source keys to include
            categories: List of categories to include
            exclude_categories: List of categories to exclude

        Returns:
            Crawling results summary
        """
        if not all([self.repository, self.news_service, self.collector]):
            await self.initialize()

        # Set default date range if not provided
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        # Load previous progress
        progress = self._load_progress()
        current_timestamp = progress.get("last_timestamp", int(end_date.timestamp()))

        self.logger.info(
            f"Starting historical crawl from {start_date} to {end_date} "
            f"(current timestamp: {current_timestamp})"
        )

        # Initialize counters
        total_collected = progress.get("total_collected", 0)
        total_stored = progress.get("total_stored", 0)
        total_duplicates = progress.get("total_duplicates", 0)
        total_failed = progress.get("total_failed", 0)
        total_errors = progress.get("total_errors", 0)
        request_count = progress.get("request_count", 0)
        start_time = datetime.now(timezone.utc)

        end_timestamp = int(start_date.timestamp())
        fallback_secs = int(fallback_interval.total_seconds())

        try:
            while current_timestamp > end_timestamp:
                try:
                    request_count += 1
                    current_date = datetime.fromtimestamp(
                        current_timestamp, tz=timezone.utc
                    )

                    self.logger.info(
                        f"Crawling batch {request_count} - timestamp: {current_timestamp} "
                        f"({current_date.strftime('%Y-%m-%d %H:%M:%S UTC')}) - batch size: {batch_size}"
                    )

                    # Collect news batch
                    result = await self.collector.collect_news(
                        max_items=batch_size,
                        to_timestamp=current_timestamp,
                        limit=batch_size,
                        lang=lang,
                        source_ids=source_ids,
                        categories=categories,
                        exclude_categories=exclude_categories,
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

                        # Update timestamp for next batch (get older articles)
                        # Ensure we always jump back at least fallback_interval
                        oldest_timestamp = min(
                            int(item.published_at.timestamp()) for item in result.items
                        )
                        next_ts = oldest_timestamp - 1
                        jump = current_timestamp - next_ts
                        if jump < fallback_secs:
                            # nếu khoảng cách thực tế quá nhỏ, lùi bằng fallback
                            current_timestamp -= fallback_secs
                            self.logger.debug(
                                f"Jump too small ({jump}s), using fallback {fallback_secs}s"
                            )
                        else:
                            current_timestamp = next_ts

                        # Check if we got fewer items than requested (might be reaching end)
                        if len(result.items) < batch_size // 2:
                            self.logger.info(
                                "Getting fewer items, might be reaching data limits"
                            )

                    else:
                        self.logger.warning(
                            f"Failed to collect batch at timestamp {current_timestamp}: {result.error_message}"
                        )
                        total_errors += 1

                        # Move back in time even on failure by fallback interval
                        current_timestamp -= fallback_secs

                        # Special handling for 403 errors - longer backoff
                        if "403" in str(result.error_message):
                            backoff_time = random.uniform(30, 60)
                            self.logger.info(
                                f"403 detected - backing off for {backoff_time:.1f} seconds"
                            )
                            await asyncio.sleep(backoff_time)

                    # Save progress at checkpoints
                    if request_count % checkpoint_interval == 0:
                        self._save_progress(
                            {
                                "last_timestamp": current_timestamp,
                                "request_count": request_count,
                                "total_collected": total_collected,
                                "total_stored": total_stored,
                                "total_duplicates": total_duplicates,
                                "total_failed": total_failed,
                                "total_errors": total_errors,
                                "last_update": datetime.now(timezone.utc).isoformat(),
                                "current_date": datetime.fromtimestamp(
                                    current_timestamp, tz=timezone.utc
                                ).isoformat(),
                            }
                        )
                        self.logger.info(f"Progress saved at request {request_count}")

                    # Enhanced rate limiting with jitter
                    if delay_between_requests > 0:
                        jitter = random.uniform(0.5, 1.5)
                        actual_delay = delay_between_requests * jitter
                        self.logger.debug(f"Sleeping for {actual_delay:.1f} seconds")
                        await asyncio.sleep(actual_delay)

                except Exception as e:
                    total_errors += 1
                    self.logger.error(
                        f"Error processing timestamp {current_timestamp}: {e}"
                    )

                    # Save progress on error
                    self._save_progress(
                        {
                            "last_timestamp": current_timestamp,
                            "request_count": request_count,
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
                    )
                    jitter = random.uniform(0.8, 1.2)
                    actual_error_delay = error_delay * jitter

                    self.logger.info(
                        f"Waiting {actual_error_delay:.1f} seconds before retry..."
                    )
                    await asyncio.sleep(actual_error_delay)

                    # Move back in time even on error by fallback interval
                    current_timestamp -= fallback_secs

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
                "last_timestamp": current_timestamp,
                "request_count": request_count,
                "total_collected": total_collected,
                "total_stored": total_stored,
                "total_duplicates": total_duplicates,
                "total_failed": total_failed,
                "total_errors": total_errors,
                "duration_seconds": duration,
                "completed": current_timestamp <= end_timestamp,
                "last_update": end_time.isoformat(),
                "date_range": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "final_timestamp": current_timestamp,
                },
            }

            self._save_progress(final_progress)

            self.logger.info(
                f"Crawling session completed - "
                f"Duration: {duration:.2f}s, "
                f"Requests: {request_count}, "
                f"Collected: {total_collected}, "
                f"Stored: {total_stored}, "
                f"Duplicates: {total_duplicates}, "
                f"Errors: {total_errors}"
            )

        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "final_timestamp": current_timestamp,
            "request_count": request_count,
            "total_collected": total_collected,
            "total_stored": total_stored,
            "total_duplicates": total_duplicates,
            "total_failed": total_failed,
            "total_errors": total_errors,
            "duration_seconds": duration,
            "completed": current_timestamp <= end_timestamp,
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
            self.logger.info("CoinDesk crawler closed")
        except Exception as e:
            self.logger.error(f"Error closing crawler: {e}")


async def main():
    """Main function for running the CoinDesk crawler"""
    crawler = CoinDeskCrawler()

    try:
        await crawler.initialize()

        # Example: Crawl last 7 days of data
        end_date = datetime.now(timezone.utc)
        # start_date = end_date - timedelta(days=30)

        # Start date is 2013-03-01

        start_date = datetime(2013, 3, 1, tzinfo=timezone.utc)

        result = await crawler.crawl_historical_data(
            start_date=start_date,
            end_date=end_date,
            batch_size=100,
            delay_between_requests=3.0,
            checkpoint_interval=20,
            fallback_interval=timedelta(hours=24),
            lang="EN",
        )

        print("CoinDesk crawling completed!")
        print(f"Results: {result}")

    except Exception as e:
        print(f"Crawling failed: {e}")
    finally:
        await crawler.close()


if __name__ == "__main__":
    asyncio.run(main())
