# utils/dependencies.py

"""Dependency injection utilities for the API."""

import time
from functools import lru_cache
from typing import Any, Dict, Optional

import psutil

from ..core.config import Config
from ..services.inference_service import SentimentInferenceService


class MetricsTracker:
    """Simple metrics tracker for API performance monitoring."""

    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.processing_times: list[float] = []
        self.max_processing_times = 1000  # Keep last 1000 processing times

    def increment_requests(self) -> None:
        """Increment total request counter."""
        self.total_requests += 1

    def increment_successful_requests(self) -> None:
        """Increment successful request counter."""
        self.successful_requests += 1

    def increment_failed_requests(self) -> None:
        """Increment failed request counter."""
        self.failed_requests += 1

    def add_processing_time(self, processing_time_ms: float) -> None:
        """Add processing time measurement."""
        self.processing_times.append(processing_time_ms)

        # Keep only the last N processing times
        if len(self.processing_times) > self.max_processing_times:
            self.processing_times = self.processing_times[-self.max_processing_times :]

    def get_uptime(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self.start_time

    def get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return round(memory_info.rss / 1024 / 1024, 2)
        except Exception:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        avg_processing_time = 0.0
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(
                self.processing_times
            )

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "average_processing_time_ms": round(avg_processing_time, 2),
            "uptime_seconds": self.get_uptime(),
            "memory_usage_mb": self.get_memory_usage(),
        }


# Global instances
_config: Optional[Config] = None
_inference_service: Optional[SentimentInferenceService] = None
_metrics_tracker: Optional[MetricsTracker] = None


@lru_cache()
def get_config() -> Config:
    """Get application configuration."""
    global _config
    if _config is None:
        _config = Config()
    return _config


async def get_inference_service() -> SentimentInferenceService:
    """Get sentiment inference service dependency."""
    global _inference_service

    if _inference_service is None:
        config = get_config()
        _inference_service = SentimentInferenceService(config.api)
        await _inference_service.initialize()

    return _inference_service


def get_metrics_tracker() -> MetricsTracker:
    """Get metrics tracker dependency."""
    global _metrics_tracker

    if _metrics_tracker is None:
        _metrics_tracker = MetricsTracker()

    return _metrics_tracker


async def cleanup_dependencies() -> None:
    """Cleanup all dependencies."""
    global _inference_service, _metrics_tracker

    if _inference_service is not None:
        await _inference_service.cleanup()
        _inference_service = None

    _metrics_tracker = None
