# market_data_job.py

"""
Market Data Background Job Service

Production-ready background job service for automated market data collection with:
- Modern cron job scheduling using APScheduler
- Process management with PID files
- Graceful shutdown handling
- Comprehensive configuration management
- Health monitoring and status reporting
"""

import argparse
import asyncio
import signal
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from common.logger import LoggerFactory

from ..adapters.binance_market_data_collector import BinanceMarketDataCollector
from ..core.config import settings
from ..factories.market_data_repository_factory import create_repository
from ..schemas.enums import Exchange
from .market_data_collector_service import MarketDataCollectorService
from .market_data_service import MarketDataService


@dataclass
class JobConfig:
    """Configuration for market data cron job"""

    # Job scheduling
    cron_schedule: str = "*/15 * * * *"  # Every 15 seconds
    timezone: str = "UTC"

    # Collection parameters
    exchange: str = Exchange.BINANCE.value
    max_lookback_days: int = 30
    update_existing: bool = True
    max_concurrent_symbols: int = 5

    # Repository configuration
    repository_type: str = "csv"  # Will be overridden by settings
    repository_config: Optional[Dict[str, Any]] = None

    # Limits and filtering
    max_symbols_per_run: int = 50
    max_timeframes_per_run: int = 10
    priority_symbols: List[str] = None
    priority_timeframes: List[str] = None

    # Error handling
    max_retries: int = 3
    retry_delay_minutes: int = 5
    skip_failed_symbols: bool = True

    # Notifications
    enable_notifications: bool = False
    notification_webhook: Optional[str] = None
    notify_on_success: bool = False
    notify_on_error: bool = True

    def __post_init__(self):
        if self.priority_symbols is None:
            self.priority_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        if self.priority_timeframes is None:
            self.priority_timeframes = ["1h", "4h", "1d"]
        if self.repository_config is None:
            self.repository_config = {}


class MarketDataJobService:
    """
    Modern market data job service with APScheduler-based cron functionality
    """

    def __init__(
        self,
        config_file: str = "market_data_job_config.json",
        pid_file: str = "market_data_job.pid",
        log_file: str = "logs/market_data_job.log",
    ):
        """
        Initialize the market data job service

        Args:
            config_file: Path to job configuration file
            pid_file: Path to PID file for process management
            log_file: Path to log file
        """
        self.config_file = Path(config_file)
        self.pid_file = Path(pid_file)
        self.log_file = Path(log_file)

        # Initialize logger
        self.logger = LoggerFactory.get_logger(
            name="market_data_job",
            log_file=str(self.log_file),
            use_colors=True,
        )

        # Load settings
        self.settings = settings

        # Initialize scheduler
        self.scheduler = AsyncIOScheduler(
            timezone=self.config.timezone if hasattr(self, "config") else "UTC"
        )
        self.scheduler.add_listener(
            self._job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )

        # Job state
        self.is_running = False
        self.current_job_id = None
        self.job_stats = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "last_run_time": None,
            "last_success_time": None,
            "last_error_time": None,
            "last_error_message": None,
        }

        # Load configuration
        self.config = self.load_config()

        # Initialize services
        self._initialize_services()

        self.logger.info("Market Data Job Service initialized")

    def _initialize_services(self) -> None:
        """Initialize market data collection services"""
        try:
            # Setup repository
            if not self.config.repository_config:
                self.config.repository_config = self._get_default_repository_config()

            self.repository = create_repository(
                self.config.repository_type, self.config.repository_config
            )

            # Initialize services
            self.market_data_collector = BinanceMarketDataCollector()
            self.market_data_service = MarketDataService(self.repository)
            self.collector_service = MarketDataCollectorService(
                self.market_data_collector, self.market_data_service
            )

            self.logger.info(
                f"Services initialized with {self.config.repository_type} repository"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize services: {e}")
            raise

    def _get_default_repository_config(self) -> Dict[str, Any]:
        """Get default repository configuration"""
        # Use settings repository type if not explicitly set in config
        repo_type = self.config.repository_type
        if repo_type == "csv" and self.settings.repository_type != "csv":
            repo_type = self.settings.repository_type

        if repo_type == "csv":
            return {"base_directory": self.settings.storage_base_directory}
        elif repo_type == "mongodb":
            return {
                "connection_string": self.settings.mongodb_url,
                "database_name": self.settings.mongodb_database,
            }
        elif repo_type == "influxdb":
            return {
                "url": "http://localhost:8086",
                "token": "your-token",
                "org": "finsight",
                "bucket": "market_data",
            }
        else:
            return {}

    def load_config(self) -> JobConfig:
        """Load job configuration from file or create default"""
        if self.config_file.exists():
            try:
                import json

                with open(self.config_file, "r") as f:
                    config_data = json.load(f)
                config = JobConfig(**config_data)
                self.logger.info(f"Loaded configuration from {self.config_file}")
                return config
            except Exception as e:
                self.logger.warning(
                    f"Failed to load config file {self.config_file}: {e}"
                )
                self.logger.info("Using default configuration")

        # Create default configuration
        config = JobConfig()
        self.save_config(config)
        return config

    def save_config(self, config: JobConfig) -> None:
        """Save job configuration to file"""
        try:
            import json

            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(asdict(config), f, indent=2, default=str)
            self.logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")

    async def schedule_job(self, config: Optional[JobConfig] = None) -> None:
        """Schedule the market data collection job"""
        if config:
            self.config = config
            self.save_config(config)

        # Parse cron schedule
        cron_parts = self.config.cron_schedule.split()
        if len(cron_parts) != 5:
            raise ValueError(f"Invalid cron schedule: {self.config.cron_schedule}")

        minute, hour, day, month, day_of_week = cron_parts

        # Create cron trigger
        trigger = CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
            timezone=self.config.timezone,
        )

        # Schedule job
        self.scheduler.add_job(
            self._execute_collection_job,
            trigger=trigger,
            id="market_data_collection",
            name="Market Data Collection Job",
            misfire_grace_time=300,  # 5 minutes
            coalesce=True,
            max_instances=1,
            args=[self.config],
        )

        self.logger.info(f"Job scheduled with cron: {self.config.cron_schedule}")

    async def _execute_collection_job(self, config: JobConfig) -> None:
        """Execute the market data collection job"""
        job_id = f"job_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.current_job_id = job_id

        self.logger.info(f"🚀 Starting market data collection job {job_id}")

        start_time = datetime.now(timezone.utc)
        job_result = {
            "job_id": job_id,
            "start_time": start_time.isoformat(),
            "end_time": None,
            "success": False,
            "error": None,
            "collection_result": None,
        }

        try:
            # Execute scan and update
            collection_result = (
                await self.collector_service.scan_and_update_all_symbols(
                    exchange=config.exchange,
                    symbols=(
                        config.priority_symbols if config.priority_symbols else None
                    ),
                    timeframes=(
                        config.priority_timeframes
                        if config.priority_timeframes
                        else None
                    ),
                    max_lookback_days=config.max_lookback_days,
                    update_existing=config.update_existing,
                )
            )

            job_result["collection_result"] = collection_result
            job_result["success"] = True

            # Update job stats
            self.job_stats["successful_runs"] += 1
            self.job_stats["last_success_time"] = start_time.isoformat()

            # Log results
            self.logger.info(
                f"✅ Job {job_id} completed successfully: "
                f"{collection_result.get('successful_updates', 0)}/{collection_result.get('total_combinations', 0)} successful, "
                f"{collection_result.get('total_records_collected', 0)} records collected"
            )

            # Send success notification if enabled
            if config.enable_notifications and config.notify_on_success:
                await self._send_notification(job_id, job_result, "success")

        except Exception as e:
            error_msg = f"Job {job_id} failed: {str(e)}"
            job_result["error"] = error_msg

            # Update job stats
            self.job_stats["failed_runs"] += 1
            self.job_stats["last_error_time"] = start_time.isoformat()
            self.job_stats["last_error_message"] = error_msg

            self.logger.error(error_msg)

            # Send error notification if enabled
            if config.enable_notifications and config.notify_on_error:
                await self._send_notification(job_id, job_result, "error")

        finally:
            end_time = datetime.now(timezone.utc)
            job_result["end_time"] = end_time.isoformat()

            # Update job stats
            self.job_stats["total_runs"] += 1
            self.job_stats["last_run_time"] = start_time.isoformat()

            duration = (end_time - start_time).total_seconds()
            self.logger.info(f"Job {job_id} completed in {duration:.2f} seconds")

            self.current_job_id = None

    async def _send_notification(
        self, job_id: str, data: Dict[str, Any], status: str
    ) -> None:
        """Send notification about job status"""
        if not self.config.notification_webhook:
            return

        try:
            import aiohttp

            notification_data = {
                "job_id": job_id,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.notification_webhook,
                    json=notification_data,
                    timeout=10,
                ) as response:
                    if response.status == 200:
                        self.logger.info(
                            f"Notification sent successfully for job {job_id}"
                        )
                    else:
                        self.logger.warning(
                            f"Failed to send notification: HTTP {response.status}"
                        )

        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")

    def _job_listener(self, event):
        """Listen to job events"""
        if event.exception:
            self.logger.error(f"Job {event.job_id} crashed: {event.exception}")
        else:
            self.logger.info(f"Job {event.job_id} executed successfully")

    async def start(self) -> None:
        """Start the job scheduler"""
        if self.is_running:
            self.logger.warning("Job service is already running")
            return

        try:
            # Create PID file
            self.pid_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.pid_file, "w") as f:
                import os

                f.write(str(os.getpid()))

            # Schedule job
            await self.schedule_job()

            # Start scheduler
            self.scheduler.start()
            self.is_running = True

            self.logger.info(
                f"Market Data Job Service started with schedule: {self.config.cron_schedule}"
            )

            # Keep running
            try:
                while self.is_running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal")
                await self.stop()

        except Exception as e:
            self.logger.error(f"Failed to start job service: {e}")
            raise

    async def stop(self) -> None:
        """Stop the job scheduler"""
        if not self.is_running:
            return

        self.logger.info("Stopping Market Data Job Service...")

        try:
            # Shutdown scheduler
            self.scheduler.shutdown(wait=False)
            self.is_running = False

            # Remove PID file
            if self.pid_file.exists():
                self.pid_file.unlink()

            self.logger.info("Market Data Job Service stopped")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the job service"""
        return {
            "is_running": self.is_running,
            "current_job_id": self.current_job_id,
            "scheduler_state": (
                str(self.scheduler.state)
                if hasattr(self.scheduler, "state")
                else "unknown"
            ),
            "config": asdict(self.config),
            "stats": self.job_stats,
            "next_run_time": (
                self.scheduler.get_job(
                    "market_data_collection"
                ).next_run_time.isoformat()
                if self.scheduler.get_job("market_data_collection")
                else None
            ),
        }

    async def run_manual_job(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        max_lookback_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a manual collection job"""
        self.logger.info("🔧 Running manual market data collection job")

        # Use provided parameters or defaults from config
        symbols = symbols or self.config.priority_symbols
        timeframes = timeframes or self.config.priority_timeframes
        max_lookback_days = max_lookback_days or self.config.max_lookback_days

        try:
            result = await self.collector_service.scan_and_update_all_symbols(
                exchange=self.config.exchange,
                symbols=symbols,
                timeframes=timeframes,
                max_lookback_days=max_lookback_days,
                update_existing=self.config.update_existing,
            )

            self.logger.info(f"Manual job completed: {result}")
            return result

        except Exception as e:
            error_msg = f"Manual job failed: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}


def setup_signal_handlers(job_service: MarketDataJobService):
    """Setup signal handlers for graceful shutdown"""

    def signal_handler(signum, frame):
        job_service.logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(job_service.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main function for CLI execution"""

    parser = argparse.ArgumentParser(description="Market Data Job Service")
    parser.add_argument(
        "command",
        choices=["start", "stop", "status", "run", "config"],
        help="Command to execute",
    )
    parser.add_argument(
        "--config-file",
        default="market_data_job_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--pid-file",
        default="market_data_job.pid",
        help="Path to PID file",
    )
    parser.add_argument(
        "--log-file",
        default="logs/market_data_job.log",
        help="Path to log file",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols for manual run (e.g., BTCUSDT ETHUSDT)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        help="Timeframes for manual run (e.g., 1h 4h 1d)",
    )
    parser.add_argument(
        "--cron",
        help="Cron schedule for config command (e.g., '0 */1 * * *')",
    )

    args = parser.parse_args()

    # Initialize job service
    job_service = MarketDataJobService(
        config_file=args.config_file,
        pid_file=args.pid_file,
        log_file=args.log_file,
    )

    if args.command == "start":
        # Setup signal handlers
        setup_signal_handlers(job_service)

        # Start the service
        await job_service.start()

    elif args.command == "stop":
        # Stop the service
        await job_service.stop()

    elif args.command == "status":
        # Get status
        status = job_service.get_status()
        print(f"Job Service Status: {status}")

    elif args.command == "run":
        # Run manual job
        result = await job_service.run_manual_job(
            symbols=args.symbols,
            timeframes=args.timeframes,
        )
        print(f"Manual job result: {result}")

    elif args.command == "config":
        # Update configuration
        config = job_service.config
        if args.cron:
            config.cron_schedule = args.cron
        job_service.save_config(config)
        print(f"Configuration updated: {config}")


if __name__ == "__main__":
    asyncio.run(main())
