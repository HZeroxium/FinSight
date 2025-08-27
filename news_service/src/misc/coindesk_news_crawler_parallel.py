# misc/coindesk_news_crawler_parallel.py

"""
CoinDesk Parallel Historical News Crawler

This module implements a parallel crawler for collecting historical news data
from CoinDesk's REST API with chunked date ranges and concurrent processing.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.logger import LoggerFactory, LoggerType, LogLevel

from .coindesk_news_crawler import CoinDeskCrawler


class CoinDeskParallelCrawler:
    """
    Parallel news crawler for CoinDesk with chunked date ranges and concurrent processing
    """

    def __init__(
        self,
        mongo_url: str = "mongodb://localhost:27017",
        database_name: str = "finsight_coindesk_news",
        progress_file: str = "data/progress/parallel/coindesk_parallel_progress.json",
        max_concurrent_chunks: int = 3,  # Conservative default to avoid overwhelming the API
    ):
        """
        Initialize CoinDesk parallel crawler

        Args:
            mongo_url: MongoDB connection URL
            database_name: Database name for storing news
            progress_file: File to track crawling progress
            max_concurrent_chunks: Maximum number of concurrent chunk processors
        """
        self.mongo_url = mongo_url
        self.database_name = database_name
        self.progress_file = Path(progress_file)
        self.max_concurrent_chunks = max_concurrent_chunks

        # Initialize logging
        self.logger = LoggerFactory.get_logger(
            name="coindesk-parallel-crawler",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file="logs/coindesk_parallel_crawler.log",
        )

        # Track chunk crawlers
        self.chunk_crawlers: Dict[int, CoinDeskCrawler] = {}
        self.semaphore: Optional[asyncio.Semaphore] = None

    async def crawl_historical_data_parallel(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        n_chunks: int = 5,
        batch_size: int = 50,
        delay_between_requests: float = 3.0,
        checkpoint_interval: int = 50,
        interval: timedelta = timedelta(hours=24),
        lang: str = "EN",
        source_ids: Optional[list] = None,
        categories: Optional[list] = None,
        exclude_categories: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Crawl historical data in parallel using chunked date ranges

        Args:
            start_date: Starting date for crawling (defaults to 30 days ago)
            end_date: Ending date for crawling (defaults to now)
            n_chunks: Number of date chunks to process in parallel
            batch_size: Number of items per request
            delay_between_requests: Delay between requests in seconds
            checkpoint_interval: Save progress every N requests
            interval: Minimum time interval for pagination fallback
            lang: Language filter
            source_ids: List of source keys to include
            categories: List of categories to include
            exclude_categories: List of categories to exclude

        Returns:
            Parallel crawling results summary
        """
        # Set default date range if not provided
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        # Load previous progress
        progress = self._load_progress()

        self.logger.info(
            f"Starting parallel historical crawl from {start_date} to {end_date} "
            f"with {n_chunks} chunks and max {self.max_concurrent_chunks} concurrent processors"
        )

        # Create date chunks
        date_chunks = self._create_date_chunks(start_date, end_date, n_chunks)

        # Initialize semaphore for concurrency control
        self.semaphore = asyncio.Semaphore(self.max_concurrent_chunks)

        # Initialize aggregate results
        total_results = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "n_chunks": n_chunks,
            "max_concurrent_chunks": self.max_concurrent_chunks,
            "total_collected": 0,
            "total_stored": 0,
            "total_duplicates": 0,
            "total_failed": 0,
            "total_errors": 0,
            "total_requests": 0,
            "chunk_results": {},
            "completed_chunks": 0,
            "failed_chunks": 0,
            "duration_seconds": 0,
        }

        start_time = datetime.now(timezone.utc)

        try:
            # Create tasks for parallel processing
            tasks = []
            for chunk_id, (chunk_start, chunk_end) in enumerate(date_chunks):
                task = self._process_chunk(
                    chunk_id=chunk_id,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    batch_size=batch_size,
                    delay_between_requests=delay_between_requests,
                    checkpoint_interval=checkpoint_interval,
                    interval=interval,
                    lang=lang,
                    source_ids=source_ids,
                    categories=categories,
                    exclude_categories=exclude_categories,
                    progress=progress,
                )
                tasks.append(task)

            # Execute chunks in parallel with progress tracking
            chunk_results = await self._execute_chunks_with_progress(
                tasks, total_results
            )

            # Aggregate results from all chunks
            for chunk_id, result in chunk_results.items():
                if result.get("success", False):
                    total_results["total_collected"] += result.get("total_collected", 0)
                    total_results["total_stored"] += result.get("total_stored", 0)
                    total_results["total_duplicates"] += result.get(
                        "total_duplicates", 0
                    )
                    total_results["total_failed"] += result.get("total_failed", 0)
                    total_results["total_errors"] += result.get("total_errors", 0)
                    total_results["total_requests"] += result.get("request_count", 0)
                    total_results["completed_chunks"] += 1
                else:
                    total_results["failed_chunks"] += 1

                total_results["chunk_results"][chunk_id] = result

        except Exception as e:
            self.logger.error(f"Critical error during parallel crawling: {e}")
            total_results["error"] = str(e)
        finally:
            # Cleanup chunk crawlers
            await self._cleanup_chunk_crawlers()

            # Calculate final duration
            end_time = datetime.now(timezone.utc)
            total_results["duration_seconds"] = (end_time - start_time).total_seconds()

            # Save final progress
            self._save_progress(total_results)

            self.logger.info(
                f"Parallel crawling session completed - "
                f"Duration: {total_results['duration_seconds']:.2f}s, "
                f"Completed chunks: {total_results['completed_chunks']}/{n_chunks}, "
                f"Total collected: {total_results['total_collected']}, "
                f"Total stored: {total_results['total_stored']}, "
                f"Total requests: {total_results['total_requests']}"
            )

        return total_results

    def _create_date_chunks(
        self, start_date: datetime, end_date: datetime, n_chunks: int
    ) -> List[Tuple[datetime, datetime]]:
        """
        Create date chunks for parallel processing

        Args:
            start_date: Starting date
            end_date: Ending date
            n_chunks: Number of chunks to create

        Returns:
            List of (chunk_start, chunk_end) tuples
        """
        total_duration = end_date - start_date
        chunk_duration = total_duration / n_chunks

        chunks = []
        current_start = start_date

        for i in range(n_chunks):
            if i == n_chunks - 1:
                # Last chunk goes to the end_date exactly
                chunk_end = end_date
            else:
                chunk_end = current_start + chunk_duration

            chunks.append((current_start, chunk_end))

            self.logger.info(
                f"Chunk {i}: {current_start.strftime('%Y-%m-%d %H:%M:%S')} to "
                f"{chunk_end.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            current_start = chunk_end

        return chunks

    async def _process_chunk(
        self,
        chunk_id: int,
        chunk_start: datetime,
        chunk_end: datetime,
        batch_size: int,
        delay_between_requests: float,
        checkpoint_interval: int,
        interval: timedelta,
        lang: str,
        source_ids: Optional[list],
        categories: Optional[list],
        exclude_categories: Optional[list],
        progress: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process a single date chunk

        Args:
            chunk_id: Unique identifier for the chunk
            chunk_start: Start date for the chunk
            chunk_end: End date for the chunk
            batch_size: Number of items per request
            delay_between_requests: Delay between requests
            checkpoint_interval: Checkpoint frequency
            interval: Fallback interval
            lang: Language filter
            source_ids: Source IDs filter
            categories: Categories filter
            exclude_categories: Exclude categories filter
            progress: Previous progress data

        Returns:
            Chunk processing results
        """
        async with self.semaphore:
            chunk_logger = LoggerFactory.get_logger(
                name=f"coindesk-chunk-{chunk_id}",
                logger_type=LoggerType.STANDARD,
                level=LogLevel.INFO,
                file_level=LogLevel.DEBUG,
                log_file=f"logs/coindesk_chunks/coindesk_chunk_{chunk_id}.log",
            )

            try:
                chunk_logger.info(
                    f"Starting chunk {chunk_id} processing: "
                    f"{chunk_start.strftime('%Y-%m-%d %H:%M:%S')} to "
                    f"{chunk_end.strftime('%Y-%m-%d %H:%M:%S')}"
                )

                # Create dedicated crawler for this chunk with unique database collection
                # chunk_db_name = f"{self.database_name}_chunk_{chunk_id}"
                chunk_db_name = self.database_name
                chunk_progress_file = (
                    f"data/progress/parallel/coindesk_chunk_{chunk_id}_progress.json"
                )

                crawler = CoinDeskCrawler(
                    mongo_url=self.mongo_url,
                    database_name=chunk_db_name,
                    progress_file=chunk_progress_file,
                )

                # Store crawler reference for cleanup
                self.chunk_crawlers[chunk_id] = crawler

                # Initialize crawler
                await crawler.initialize()

                # Add chunk-specific delay to avoid thundering herd
                initial_delay = chunk_id * 0.5  # Stagger chunk starts
                await asyncio.sleep(initial_delay)

                # Process the chunk
                result = await crawler.crawl_historical_data(
                    start_date=chunk_start,
                    end_date=chunk_end,
                    batch_size=batch_size,
                    delay_between_requests=delay_between_requests,
                    checkpoint_interval=checkpoint_interval,
                    fallback_interval=interval,
                    lang=lang,
                    source_ids=source_ids,
                    categories=categories,
                    exclude_categories=exclude_categories,
                )

                # Add chunk metadata
                result.update(
                    {
                        "chunk_id": chunk_id,
                        "chunk_start": chunk_start.isoformat(),
                        "chunk_end": chunk_end.isoformat(),
                        "success": True,
                    }
                )

                chunk_logger.info(
                    f"Completed chunk {chunk_id}: "
                    f"collected {result.get('total_collected', 0)}, "
                    f"stored {result.get('total_stored', 0)}"
                )

                return result

            except Exception as e:
                chunk_logger.error(f"Error processing chunk {chunk_id}: {e}")
                return {
                    "chunk_id": chunk_id,
                    "chunk_start": chunk_start.isoformat(),
                    "chunk_end": chunk_end.isoformat(),
                    "success": False,
                    "error": str(e),
                    "total_collected": 0,
                    "total_stored": 0,
                    "total_duplicates": 0,
                    "total_failed": 0,
                    "total_errors": 1,
                    "request_count": 0,
                }

    async def _execute_chunks_with_progress(
        self, tasks: List[asyncio.Task], total_results: Dict[str, Any]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Execute chunk tasks with progress tracking

        Args:
            tasks: List of chunk processing tasks
            total_results: Results dictionary to update

        Returns:
            Dictionary of chunk results by chunk_id
        """
        chunk_results = {}

        # Use asyncio.as_completed for real-time progress updates
        for completed_task in asyncio.as_completed(tasks):
            try:
                result = await completed_task
                chunk_id = result.get("chunk_id", -1)
                chunk_results[chunk_id] = result

                completed_count = len(chunk_results)
                total_chunks = len(tasks)

                self.logger.info(
                    f"Chunk {chunk_id} completed ({completed_count}/{total_chunks}). "
                    f"Status: {'SUCCESS' if result.get('success') else 'FAILED'}"
                )

                # Update progress periodically
                if completed_count % max(1, total_chunks // 10) == 0:
                    progress_percent = (completed_count / total_chunks) * 100
                    self.logger.info(f"Overall progress: {progress_percent:.1f}%")

            except Exception as e:
                self.logger.error(f"Task failed with error: {e}")

        return chunk_results

    async def _cleanup_chunk_crawlers(self) -> None:
        """Clean up all chunk crawlers"""
        cleanup_tasks = []
        for chunk_id, crawler in self.chunk_crawlers.items():
            cleanup_tasks.append(crawler.close())

        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                self.logger.info(f"Cleaned up {len(cleanup_tasks)} chunk crawlers")
            except Exception as e:
                self.logger.warning(f"Error during crawler cleanup: {e}")

        self.chunk_crawlers.clear()

    def _load_progress(self) -> Dict[str, Any]:
        """Load parallel crawling progress from file"""
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
        """Save parallel crawling progress to file"""
        try:
            # Ensure directory exists
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)

            # Add timestamp
            progress["last_update"] = datetime.now(timezone.utc).isoformat()

            with open(self.progress_file, "w") as f:
                json.dump(progress, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")

    async def get_crawl_status(self) -> Dict[str, Any]:
        """Get current parallel crawling status"""
        progress = self._load_progress()

        # Get individual chunk statuses
        chunk_statuses = {}
        for chunk_id, crawler in self.chunk_crawlers.items():
            try:
                chunk_status = await crawler.get_crawl_status()
                chunk_statuses[chunk_id] = chunk_status
            except Exception as e:
                chunk_statuses[chunk_id] = {"error": str(e)}

        return {
            "parallel_progress": progress,
            "chunk_statuses": chunk_statuses,
            "active_chunks": len(self.chunk_crawlers),
            "progress_file": str(self.progress_file),
        }

    async def close(self) -> None:
        """Close parallel crawler and cleanup resources"""
        try:
            await self._cleanup_chunk_crawlers()
            self.logger.info("CoinDesk parallel crawler closed")
        except Exception as e:
            self.logger.error(f"Error closing parallel crawler: {e}")


async def main():
    """Main function for running the parallel CoinDesk crawler"""
    crawler = CoinDeskParallelCrawler(
        max_concurrent_chunks=40  # Conservative for API rate limiting
    )

    try:
        # Example: Crawl last 30 days of data in 6 chunks
        end_date = datetime.now(timezone.utc)
        start_date = datetime(
            2013, 3, 1, tzinfo=timezone.utc
        )  # Or shorter range for testing

        result = await crawler.crawl_historical_data_parallel(
            start_date=start_date,
            end_date=end_date,
            n_chunks=40,  # Divide into 40 parallel chunks
            batch_size=100,
            delay_between_requests=4.0,  # Slightly longer delay for parallel processing
            checkpoint_interval=25,
            interval=timedelta(hours=24),  # Smaller interval for better granularity
            lang="EN",
        )

        print("Parallel CoinDesk crawling completed!")
        print(f"Results summary:")
        print(f"  Total collected: {result['total_collected']}")
        print(f"  Total stored: {result['total_stored']}")
        print(f"  Total requests: {result['total_requests']}")
        print(f"  Completed chunks: {result['completed_chunks']}/{result['n_chunks']}")
        print(f"  Duration: {result['duration_seconds']:.2f} seconds")

    except Exception as e:
        print(f"Parallel crawling failed: {e}")
    finally:
        await crawler.close()


if __name__ == "__main__":
    asyncio.run(main())
