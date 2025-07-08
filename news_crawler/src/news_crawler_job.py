# news_crawler_job.py

"""
News Crawler Background Job Service

A production-ready background job service for automated news collection with:
- Modern cron job scheduling using APScheduler
- Process management with PID files
- Graceful shutdown handling
- Comprehensive configuration management
- Health monitoring and status reporting
"""

import os
import sys
import json
import signal
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

from .services.news_collector_service import (
    NewsCollectorService,
    BatchCollectionRequest,
)
from .services.news_service import NewsService
from .repositories.mongo_news_repository import MongoNewsRepository
from .schemas.news_schemas import NewsSource
from .core.news_collector_factory import CollectorType
from .core.config import settings
from .common.logger import LoggerFactory, LoggerType, LogLevel


@dataclass
class JobConfig:
    """Configuration for news crawler job."""

    sources: List[str]
    collector_preferences: Dict[str, str]
    max_items_per_source: int
    enable_fallback: bool
    schedule: str
    config_overrides: Dict[str, Any]
    notification: Dict[str, Any]

    @classmethod
    def get_default(cls) -> "JobConfig":
        """Get default job configuration."""
        return cls(
            sources=[source.value for source in NewsSource],
            collector_preferences={
                NewsSource.COINDESK.value: CollectorType.API_REST.value,
                NewsSource.COINTELEGRAPH.value: CollectorType.API_GRAPHQL.value,
            },
            max_items_per_source=settings.cron_job_max_items_per_source,
            enable_fallback=True,
            schedule=settings.cron_job_schedule,
            config_overrides={},
            notification={
                "enabled": False,
                "webhook_url": None,
                "email": None,
            },
        )


class NewsCrawlerJobService:
    """
    Modern news crawler job service with APScheduler-based cron functionality.
    """

    def __init__(
        self,
        mongo_url: str = None,
        database_name: str = None,
        config_file: str = None,
        pid_file: str = None,
        log_file: str = None,
    ):
        """
        Initialize the news crawler job service.

        Args:
            mongo_url: MongoDB connection URL
            database_name: Database name for storing news
            config_file: Job configuration file path
            pid_file: Process ID file path
            log_file: Log file path
        """
        self.mongo_url = mongo_url or settings.mongodb_url
        self.database_name = database_name or settings.mongodb_database
        self.config_file = Path(config_file or settings.cron_job_config_file)
        self.pid_file = Path(pid_file or settings.cron_job_pid_file)
        self.log_file = log_file or settings.cron_job_log_file

        # Initialize logger
        self.logger = LoggerFactory.get_logger(
            name="news-crawler-job",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file=self.log_file,
        )

        # Components
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.repository: Optional[MongoNewsRepository] = None
        self.news_service: Optional[NewsService] = None
        self.collector_service: Optional[NewsCollectorService] = None
        self.is_running = False
        self.job_stats = {
            "total_jobs": 0,
            "successful_jobs": 0,
            "failed_jobs": 0,
            "last_run": None,
            "last_success": None,
            "last_error": None,
        }

    async def initialize(self) -> None:
        """Initialize all job components."""
        try:
            self.logger.info("🚀 Initializing News Crawler Job Service")

            # Initialize repository
            self.repository = MongoNewsRepository(
                mongo_url=self.mongo_url, database_name=self.database_name
            )
            await self.repository.initialize()
            self.logger.info("✅ MongoDB repository initialized")

            # Initialize services
            self.news_service = NewsService(self.repository)
            self.collector_service = NewsCollectorService(
                news_service=self.news_service,
                use_cache=settings.enable_caching,
                enable_fallback=True,
            )
            self.logger.info("✅ News services initialized")

            # Initialize scheduler
            self.scheduler = AsyncIOScheduler(timezone=timezone.utc)
            self.scheduler.add_listener(
                self._job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
            )
            self.logger.info("✅ Scheduler initialized")

            self.logger.info("🎉 News Crawler Job Service ready!")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize job service: {e}")
            raise

    def load_config(self) -> JobConfig:
        """Load job configuration from file or use defaults."""
        try:
            if self.config_file.exists():
                with open(self.config_file, "r") as f:
                    data = json.load(f)
                    config = JobConfig(**data)
                    self.logger.info(f"📋 Configuration loaded from {self.config_file}")
                    return config
            else:
                config = JobConfig.get_default()
                self.save_config(config)
                self.logger.info("📋 Using default configuration")
                return config

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load config: {e}, using defaults")
            return JobConfig.get_default()

    def save_config(self, config: JobConfig) -> None:
        """Save configuration to file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(asdict(config), f, indent=2)
            self.logger.info(f"💾 Configuration saved to {self.config_file}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save config: {e}")

    async def schedule_job(self, config: JobConfig = None) -> None:
        """Schedule the news crawling job."""
        if not self.scheduler:
            raise RuntimeError("Scheduler not initialized")

        config = config or self.load_config()

        # Parse cron expression
        cron_parts = config.schedule.split()
        if len(cron_parts) != 5:
            raise ValueError(f"Invalid cron expression: {config.schedule}")

        minute, hour, day, month, day_of_week = cron_parts

        # Create cron trigger
        trigger = CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
            timezone=timezone.utc,
        )

        # Schedule the job
        self.scheduler.add_job(
            self._execute_crawl_job,
            trigger=trigger,
            id="news_crawler_job",
            name="News Crawler Job",
            args=[config],
            replace_existing=True,
        )

        self.logger.info(f"📅 Job scheduled with cron: {config.schedule}")

    async def _execute_crawl_job(self, config: JobConfig) -> None:
        """Execute the news crawling job."""
        job_id = f"news_crawl_{int(datetime.now(timezone.utc).timestamp())}"
        start_time = datetime.now(timezone.utc)

        self.logger.info(f"🔄 Starting job {job_id}")
        self.job_stats["total_jobs"] += 1
        self.job_stats["last_run"] = start_time.isoformat()

        try:
            # Ensure services are initialized
            if not all([self.repository, self.news_service, self.collector_service]):
                await self.initialize()

            # Convert sources to NewsSource enums
            sources = [NewsSource(source) for source in config.sources]

            # Create batch collection request
            request = BatchCollectionRequest(
                sources=sources,
                collector_preferences=config.collector_preferences,
                max_items_per_source=config.max_items_per_source,
                config_overrides=config.config_overrides,
                enable_fallback=config.enable_fallback,
            )

            # Execute collection
            results = await self.collector_service.collect_and_store_batch(request)

            # Calculate execution time
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            # Update stats
            self.job_stats["successful_jobs"] += 1
            self.job_stats["last_success"] = end_time.isoformat()

            # Log results
            summary = {
                "total_sources": len(sources),
                "successful_sources": sum(
                    1
                    for r in results["source_results"].values()
                    if r["collection_success"]
                ),
                "total_items_collected": results["total_items_collected"],
                "total_items_stored": results["total_items_stored"],
                "duration": f"{duration:.2f}s",
            }

            self.logger.info(f"✅ Job {job_id} completed successfully: {summary}")

            # Send notifications if configured
            if config.notification.get("enabled", False):
                await self._send_notification(job_id, summary, "success")

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            self.job_stats["failed_jobs"] += 1
            self.job_stats["last_error"] = {
                "timestamp": end_time.isoformat(),
                "error": str(e),
            }

            self.logger.error(f"❌ Job {job_id} failed after {duration:.2f}s: {e}")

            # Send error notification
            if config.notification.get("enabled", False):
                await self._send_notification(job_id, {"error": str(e)}, "error")

    async def _send_notification(
        self, job_id: str, data: Dict[str, Any], status: str
    ) -> None:
        """Send job completion notification."""
        try:
            if status == "success":
                message = (
                    f"🎉 News Crawler Job {job_id} completed successfully!\n"
                    f"📊 Sources: {data.get('successful_sources', 0)}/{data.get('total_sources', 0)}\n"
                    f"📰 Items collected: {data.get('total_items_collected', 0)}\n"
                    f"💾 Items stored: {data.get('total_items_stored', 0)}\n"
                    f"⏱️ Duration: {data.get('duration', 'N/A')}"
                )
            else:
                message = (
                    f"❌ News Crawler Job {job_id} failed!\n"
                    f"🔥 Error: {data.get('error', 'Unknown error')}"
                )

            self.logger.info(f"📢 Notification: {message}")

            # Add webhook/email notification logic here if needed

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to send notification: {e}")

    def _job_listener(self, event):
        """APScheduler job event listener."""
        if hasattr(event, "exception") and event.exception:
            self.logger.error(f"📅 Scheduler job error: {event.exception}")
        else:
            self.logger.debug(f"📅 Scheduler job executed: {event.job_id}")

    async def start(self) -> None:
        """Start the job service."""
        try:
            if self.is_running:
                self.logger.warning("⚠️ Job service already running")
                return

            # Create PID file
            self.pid_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.pid_file, "w") as f:
                f.write(str(os.getpid()))

            self.logger.info(f"📝 PID file created: {self.pid_file}")

            # Initialize if not already done
            if not self.scheduler:
                await self.initialize()

            # Schedule the job
            await self.schedule_job()

            # Start scheduler
            self.scheduler.start()
            self.is_running = True

            self.logger.info("🚀 News Crawler Job Service started successfully!")

            # Keep the service running
            while self.is_running:
                await asyncio.sleep(1)

        except Exception as e:
            self.logger.error(f"❌ Failed to start job service: {e}")
            raise

    async def stop(self) -> None:
        """Stop the job service."""
        try:
            self.logger.info("🛑 Stopping News Crawler Job Service...")

            self.is_running = False

            if self.scheduler:
                self.scheduler.shutdown(wait=True)
                self.logger.info("✅ Scheduler stopped")

            if self.repository:
                await self.repository.close()
                self.logger.info("✅ Repository closed")

            # Remove PID file
            if self.pid_file.exists():
                self.pid_file.unlink()
                self.logger.info("✅ PID file removed")

            self.logger.info("👋 News Crawler Job Service stopped")

        except Exception as e:
            self.logger.error(f"❌ Error stopping job service: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current job service status."""
        return {
            "service": "news-crawler-job",
            "version": "1.0.0",
            "is_running": self.is_running,
            "pid": os.getpid(),
            "pid_file": str(self.pid_file),
            "config_file": str(self.config_file),
            "log_file": self.log_file,
            "scheduler_running": self.scheduler.running if self.scheduler else False,
            "next_run": (
                self.scheduler.get_jobs()[0].next_run_time.isoformat()
                if self.scheduler and self.scheduler.get_jobs()
                else None
            ),
            "stats": self.job_stats,
        }

    async def run_manual_job(
        self, sources: List[str] = None, max_items: int = None
    ) -> Dict[str, Any]:
        """Run a manual news collection job."""
        self.logger.info("🔧 Running manual news collection job")

        config = self.load_config()
        if sources:
            config.sources = sources
        if max_items:
            config.max_items_per_source = max_items

        await self._execute_crawl_job(config)

        return {
            "status": "completed",
            "sources": config.sources,
            "max_items": config.max_items_per_source,
        }


def setup_signal_handlers(job_service: NewsCrawlerJobService):
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        asyncio.create_task(job_service.stop())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main function for CLI execution."""
    import argparse

    parser = argparse.ArgumentParser(description="News Crawler Job Service")
    parser.add_argument(
        "command",
        choices=["start", "stop", "status", "run", "config"],
        help="Command to execute",
    )
    parser.add_argument("--sources", nargs="+", help="News sources to crawl")
    parser.add_argument("--max-items", type=int, help="Max items per source")
    parser.add_argument("--config-file", help="Configuration file path")
    parser.add_argument("--pid-file", help="PID file path")
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--schedule", help="Cron schedule expression")

    args = parser.parse_args()

    # Create job service
    job_service = NewsCrawlerJobService(
        config_file=args.config_file,
        pid_file=args.pid_file,
        log_file=args.log_file,
    )

    if args.command == "start":
        setup_signal_handlers(job_service)
        await job_service.start()

    elif args.command == "stop":
        # Send stop signal to running process
        pid_file = Path(args.pid_file or settings.cron_job_pid_file)
        if pid_file.exists():
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"🛑 Stop signal sent to process {pid}")
            except ProcessLookupError:
                print(f"⚠️ Process {pid} not found, removing stale PID file")
                pid_file.unlink()
        else:
            print("⚠️ No PID file found, service may not be running")

    elif args.command == "status":
        await job_service.initialize()
        status = job_service.get_status()
        print(json.dumps(status, indent=2, default=str))

    elif args.command == "run":
        await job_service.initialize()
        result = await job_service.run_manual_job(
            sources=args.sources, max_items=args.max_items
        )
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "config":
        config = job_service.load_config()
        if args.schedule:
            config.schedule = args.schedule
            job_service.save_config(config)
            print(f"✅ Schedule updated to: {args.schedule}")
        else:
            print(json.dumps(asdict(config), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
