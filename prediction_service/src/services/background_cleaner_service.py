# services/background_cleaner_service.py

"""
Background Cleaner Service - Handles periodic cleanup of cache, datasets, and models

This service runs in the background and periodically cleans up:
- Cloud cache files in /tmp/cloud_cache
- Dataset files in /data
- Model files in /models

Features:
- Configurable cleanup intervals
- Background task execution
- Manual cleanup endpoints
- Comprehensive logging
"""

import asyncio
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.logger.logger_factory import LoggerFactory, LogLevel

from ..core.config import get_settings
from ..schemas.enums import CleanupInterval, CleanupTarget


class BackgroundCleanerService:
    """
    Background service for periodic cleanup operations.

    This service provides:
    - Automatic periodic cleanup based on configured intervals
    - Manual cleanup endpoints for immediate execution
    - Configurable cleanup targets and intervals
    - Comprehensive logging and monitoring
    """

    def __init__(self):
        """Initialize the background cleaner service."""
        self.logger = LoggerFactory.get_logger("BackgroundCleaner")
        self.settings = get_settings()

        # Service state
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._last_cleanup: Optional[datetime] = None

        # Cleanup configuration
        self.cleanup_interval = self._parse_cleanup_interval(
            getattr(self.settings, "cleanup_interval", "1d")
        )

        # Cleanup targets configuration
        self.enable_cloud_cache_cleanup = getattr(
            self.settings, "enable_cloud_cache_cleanup", True
        )
        self.enable_datasets_cleanup = getattr(
            self.settings, "enable_datasets_cleanup", True
        )
        self.enable_models_cleanup = getattr(
            self.settings, "enable_models_cleanup", True
        )

        # Cleanup age thresholds (files older than this will be deleted)
        self.cloud_cache_max_age_hours = getattr(
            self.settings, "cloud_cache_max_age_hours", 24
        )
        self.datasets_max_age_hours = getattr(
            self.settings, "datasets_max_age_hours", 168  # 7 days
        )
        self.models_max_age_hours = getattr(
            self.settings, "models_max_age_hours", 720  # 30 days
        )

        self.logger.info("Background cleaner service initialized")
        self.logger.info(f"Cleanup interval: {self.cleanup_interval}")
        self.logger.info(
            f"Cloud cache cleanup: {'enabled' if self.enable_cloud_cache_cleanup else 'disabled'}"
        )
        self.logger.info(
            f"Datasets cleanup: {'enabled' if self.enable_datasets_cleanup else 'disabled'}"
        )
        self.logger.info(
            f"Models cleanup: {'enabled' if self.enable_models_cleanup else 'disabled'}"
        )

    async def start(self) -> bool:
        """
        Start the background cleaner service.

        Returns:
            bool: True if service started successfully
        """
        if self._running:
            self.logger.info("Background cleaner service is already running")
            return True

        try:
            if self.cleanup_interval == CleanupInterval.DISABLED:
                self.logger.info("Background cleanup is disabled")
                return True

            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            self.logger.info("Background cleaner service started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start background cleaner service: {e}")
            self._running = False
            return False

    async def stop(self) -> bool:
        """
        Stop the background cleaner service.

        Returns:
            bool: True if service stopped successfully
        """
        if not self._running:
            self.logger.info("Background cleaner service is not running")
            return True

        try:
            self._running = False

            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            self.logger.info("Background cleaner service stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop background cleaner service: {e}")
            return False

    async def cleanup_cloud_cache(self) -> Dict[str, Any]:
        """
        Clean up cloud cache files.

        Returns:
            Dict containing cleanup results
        """
        try:
            cache_dir = Path(self.settings.base_dir) / "tmp" / "cloud_cache"
            if not cache_dir.exists():
                return {
                    "success": True,
                    "message": "Cloud cache directory does not exist",
                    "files_removed": 0,
                    "bytes_freed": 0,
                }

            cutoff_time = datetime.now() - timedelta(
                hours=self.cloud_cache_max_age_hours
            )
            files_removed = 0
            bytes_freed = 0

            for item in cache_dir.iterdir():
                try:
                    if item.is_file():
                        if datetime.fromtimestamp(item.stat().st_mtime) < cutoff_time:
                            file_size = item.stat().st_size
                            item.unlink()
                            files_removed += 1
                            bytes_freed += file_size
                            self.logger.debug(f"Removed old cache file: {item.name}")
                    elif item.is_dir():
                        # Remove empty directories
                        try:
                            item.rmdir()
                            self.logger.debug(
                                f"Removed empty cache directory: {item.name}"
                            )
                        except OSError:
                            # Directory not empty, skip
                            pass

                except Exception as e:
                    self.logger.warning(f"Failed to remove cache item {item}: {e}")
                    continue

            self.logger.info(
                f"Cloud cache cleanup completed: {files_removed} files removed, {bytes_freed} bytes freed"
            )

            return {
                "success": True,
                "message": f"Cloud cache cleanup completed",
                "files_removed": files_removed,
                "bytes_freed": bytes_freed,
                "cutoff_time": cutoff_time.isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Cloud cache cleanup failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "files_removed": 0,
                "bytes_freed": 0,
            }

    async def cleanup_datasets(self) -> Dict[str, Any]:
        """
        Clean up old dataset files.

        Returns:
            Dict containing cleanup results
        """
        try:
            data_dir = Path(self.settings.data_dir)
            if not data_dir.exists():
                return {
                    "success": True,
                    "message": "Data directory does not exist",
                    "files_removed": 0,
                    "bytes_freed": 0,
                }

            cutoff_time = datetime.now() - timedelta(hours=self.datasets_max_age_hours)
            files_removed = 0
            bytes_freed = 0

            # Clean up old CSV and parquet files
            for pattern in ["*.csv", "*.parquet"]:
                for file_path in data_dir.rglob(pattern):
                    try:
                        if (
                            datetime.fromtimestamp(file_path.stat().st_mtime)
                            < cutoff_time
                        ):
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            files_removed += 1
                            bytes_freed += file_size
                            self.logger.debug(
                                f"Removed old dataset file: {file_path.name}"
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to remove dataset file {file_path}: {e}"
                        )
                        continue

            self.logger.info(
                f"Datasets cleanup completed: {files_removed} files removed, {bytes_freed} bytes freed"
            )

            return {
                "success": True,
                "message": f"Datasets cleanup completed",
                "files_removed": files_removed,
                "bytes_freed": bytes_freed,
                "cutoff_time": cutoff_time.isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Datasets cleanup failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "files_removed": 0,
                "bytes_freed": 0,
            }

    async def cleanup_models(self) -> Dict[str, Any]:
        """
        Clean up old model files.

        Returns:
            Dict containing cleanup results
        """
        try:
            models_dir = Path(self.settings.models_dir)
            if not models_dir.exists():
                return {
                    "success": True,
                    "message": "Models directory does not exist",
                    "models_removed": 0,
                    "bytes_freed": 0,
                }

            cutoff_time = datetime.now() - timedelta(hours=self.models_max_age_hours)
            models_removed = 0
            bytes_freed = 0

            for model_dir in models_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                try:
                    # Check if model directory is old enough to be cleaned
                    if datetime.fromtimestamp(model_dir.stat().st_mtime) < cutoff_time:
                        # Calculate directory size before removal
                        dir_size = sum(
                            f.stat().st_size
                            for f in model_dir.rglob("*")
                            if f.is_file()
                        )

                        # Remove the entire model directory
                        shutil.rmtree(model_dir)
                        models_removed += 1
                        bytes_freed += dir_size

                        self.logger.info(
                            f"Removed old model: {model_dir.name} ({dir_size} bytes)"
                        )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to remove model directory {model_dir}: {e}"
                    )
                    continue

            self.logger.info(
                f"Models cleanup completed: {models_removed} models removed, {bytes_freed} bytes freed"
            )

            return {
                "success": True,
                "message": f"Models cleanup completed",
                "models_removed": models_removed,
                "bytes_freed": bytes_freed,
                "cutoff_time": cutoff_time.isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Models cleanup failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "models_removed": 0,
                "bytes_freed": 0,
            }

    async def cleanup_all(self) -> Dict[str, Any]:
        """
        Clean up all targets.

        Returns:
            Dict containing cleanup results for all targets
        """
        try:
            self.logger.info("Starting comprehensive cleanup of all targets")

            results = {}

            if self.enable_cloud_cache_cleanup:
                results["cloud_cache"] = await self.cleanup_cloud_cache()

            if self.enable_datasets_cleanup:
                results["datasets"] = await self.cleanup_datasets()

            if self.enable_models_cleanup:
                results["models"] = await self.cleanup_models()

            # Calculate totals
            total_files_removed = sum(
                result.get("files_removed", 0) + result.get("models_removed", 0)
                for result in results.values()
                if result.get("success", False)
            )

            total_bytes_freed = sum(
                result.get("bytes_freed", 0)
                for result in results.values()
                if result.get("success", False)
            )

            overall_success = all(
                result.get("success", False) for result in results.values()
            )

            self.logger.info(
                f"Comprehensive cleanup completed: {total_files_removed} items removed, {total_bytes_freed} bytes freed"
            )

            return {
                "success": overall_success,
                "message": "Comprehensive cleanup completed",
                "results": results,
                "summary": {
                    "total_items_removed": total_files_removed,
                    "total_bytes_freed": total_bytes_freed,
                    "timestamp": datetime.now().isoformat(),
                },
            }

        except Exception as e:
            self.logger.error(f"Comprehensive cleanup failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": {},
                "summary": {
                    "total_items_removed": 0,
                    "total_bytes_freed": 0,
                    "timestamp": datetime.now().isoformat(),
                },
            }

    async def get_cleanup_status(self) -> Dict[str, Any]:
        """
        Get the current status of the background cleaner service.

        Returns:
            Dict containing service status information
        """
        return {
            "running": self._running,
            "cleanup_interval": (
                self.cleanup_interval.value if self.cleanup_interval else None
            ),
            "last_cleanup": (
                self._last_cleanup.isoformat() if self._last_cleanup else None
            ),
            "next_cleanup": (
                self._get_next_cleanup_time().isoformat() if self._running else None
            ),
            "configuration": {
                "enable_cloud_cache_cleanup": self.enable_cloud_cache_cleanup,
                "enable_datasets_cleanup": self.enable_datasets_cleanup,
                "enable_models_cleanup": self.enable_models_cleanup,
                "cloud_cache_max_age_hours": self.cloud_cache_max_age_hours,
                "datasets_max_age_hours": self.datasets_max_age_hours,
                "models_max_age_hours": self.models_max_age_hours,
            },
        }

    # Private methods

    async def _cleanup_loop(self) -> None:
        """Main cleanup loop that runs in the background."""
        self.logger.info("Background cleanup loop started")

        while self._running:
            try:
                await self.cleanup_all()
                self._last_cleanup = datetime.now()

                # Wait for next cleanup cycle
                await asyncio.sleep(self._get_interval_seconds())

            except asyncio.CancelledError:
                self.logger.info("Background cleanup loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in background cleanup loop: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(60)  # 1 minute

        self.logger.info("Background cleanup loop stopped")

    def _parse_cleanup_interval(self, interval_str: str) -> CleanupInterval:
        """Parse cleanup interval string to enum value."""
        try:
            return CleanupInterval(interval_str)
        except ValueError:
            self.logger.warning(
                f"Invalid cleanup interval '{interval_str}', using default '1d'"
            )
            return CleanupInterval.DAY_1

    def _get_interval_seconds(self) -> int:
        """Get cleanup interval in seconds."""
        interval_map = {
            CleanupInterval.MINUTE_1: 60,
            CleanupInterval.MINUTE_5: 300,
            CleanupInterval.MINUTE_15: 900,
            CleanupInterval.HOUR_1: 3600,
            CleanupInterval.HOUR_4: 14400,
            CleanupInterval.HOUR_12: 43200,
            CleanupInterval.DAY_1: 86400,
            CleanupInterval.WEEK_1: 604800,
        }

        return interval_map.get(self.cleanup_interval, 86400)  # Default to 1 day

    def _get_next_cleanup_time(self) -> datetime:
        """Calculate the next cleanup time."""
        if not self._last_cleanup:
            return datetime.now()

        return self._last_cleanup + timedelta(seconds=self._get_interval_seconds())


# Global instance for easy access
_background_cleaner_service: Optional[BackgroundCleanerService] = None


def get_background_cleaner_service() -> BackgroundCleanerService:
    """Get the global background cleaner service instance."""
    global _background_cleaner_service
    if _background_cleaner_service is None:
        _background_cleaner_service = BackgroundCleanerService()
    return _background_cleaner_service


async def start_background_cleaner() -> bool:
    """Start the global background cleaner service."""
    service = get_background_cleaner_service()
    return await service.start()


async def stop_background_cleaner() -> bool:
    """Stop the global background cleaner service."""
    service = get_background_cleaner_service()
    return await service.stop()
