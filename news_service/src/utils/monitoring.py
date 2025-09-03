"""
Monitoring utilities for Prometheus metrics.

This module provides custom business metrics for the news service,
including news ingestion counters, processing duration histograms,
and cache statistics gauges.
"""

from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, Info
import time
from contextlib import contextmanager

# HTTP request duration buckets (in seconds) - optimized for web APIs
HTTP_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)

# Business metrics buckets (in seconds) - optimized for news processing
NEWS_PROCESSING_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)

# Custom business metrics
NEWS_INGESTED_TOTAL = Counter(
    name="news_ingested_total",
    documentation="Total number of news articles ingested",
    labelnames=["source", "status", "service"],
)

NEWS_PROCESSING_DURATION_SECONDS = Histogram(
    name="news_processing_duration_seconds",
    documentation="Time spent processing news articles",
    labelnames=["source", "operation", "service"],
    buckets=NEWS_PROCESSING_BUCKETS,
)

CACHE_ITEMS_TOTAL = Gauge(
    name="cache_items_total",
    documentation="Total number of items in cache",
    labelnames=["cache_type", "service"],
)

CACHE_HIT_RATIO = Gauge(
    name="cache_hit_ratio",
    documentation="Cache hit ratio (0.0 to 1.0)",
    labelnames=["cache_type", "service"],
)

DATABASE_CONNECTIONS_ACTIVE = Gauge(
    name="database_connections_active",
    documentation="Number of active database connections",
    labelnames=["database", "service"],
)

JOB_EXECUTIONS_TOTAL = Counter(
    name="job_executions_total",
    documentation="Total number of job executions",
    labelnames=["job_type", "status", "service"],
)

JOB_DURATION_SECONDS = Histogram(
    name="job_duration_seconds",
    documentation="Time spent executing jobs",
    labelnames=["job_type", "service"],
    buckets=NEWS_PROCESSING_BUCKETS,
)

# Service information
SERVICE_INFO = Info(
    name="news_service_info", documentation="Information about the news service"
)

# Initialize service info
SERVICE_INFO.info(
    {"service": "news_service", "version": "1.0.0", "environment": "production"}
)


class NewsMetrics:
    """Helper class for tracking news-related metrics."""

    def __init__(self, service_name: str = "news_service"):
        self.service_name = service_name

    def increment_news_ingested(self, source: str, status: str = "success") -> None:
        """Increment the news ingested counter."""
        NEWS_INGESTED_TOTAL.labels(
            source=source, status=status, service=self.service_name
        ).inc()

    def observe_news_processing_duration(
        self, source: str, operation: str, duration: float
    ) -> None:
        """Observe news processing duration."""
        NEWS_PROCESSING_DURATION_SECONDS.labels(
            source=source, operation=operation, service=self.service_name
        ).observe(duration)

    @contextmanager
    def track_news_processing(self, source: str, operation: str):
        """Context manager to track news processing duration."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.observe_news_processing_duration(source, operation, duration)

    def set_cache_items_total(self, cache_type: str, count: int) -> None:
        """Set the total number of items in cache."""
        CACHE_ITEMS_TOTAL.labels(cache_type=cache_type, service=self.service_name).set(
            count
        )

    def set_cache_hit_ratio(self, cache_type: str, ratio: float) -> None:
        """Set the cache hit ratio (0.0 to 1.0)."""
        CACHE_HIT_RATIO.labels(cache_type=cache_type, service=self.service_name).set(
            ratio
        )

    def set_database_connections_active(self, database: str, count: int) -> None:
        """Set the number of active database connections."""
        DATABASE_CONNECTIONS_ACTIVE.labels(
            database=database, service=self.service_name
        ).set(count)

    def increment_job_executions(self, job_type: str, status: str = "success") -> None:
        """Increment the job executions counter."""
        JOB_EXECUTIONS_TOTAL.labels(
            job_type=job_type, status=status, service=self.service_name
        ).inc()

    def observe_job_duration(self, job_type: str, duration: float) -> None:
        """Observe job execution duration."""
        JOB_DURATION_SECONDS.labels(
            job_type=job_type, service=self.service_name
        ).observe(duration)

    @contextmanager
    def track_job_execution(self, job_type: str):
        """Context manager to track job execution duration."""
        start_time = time.time()
        try:
            yield
            self.increment_job_executions(job_type, "success")
        except Exception:
            self.increment_job_executions(job_type, "error")
            raise
        finally:
            duration = time.time() - start_time
            self.observe_job_duration(job_type, duration)


class MetricsCollector:
    """Collector for gathering and updating metrics from various services."""

    def __init__(self, service_name: str = "news_service"):
        self.service_name = service_name
        self.news_metrics = NewsMetrics(service_name)

    async def collect_cache_metrics(self, cache_stats: Dict[str, Any]) -> None:
        """Collect and update cache-related metrics."""
        if not cache_stats:
            return

        # Set cache items total
        total_items = cache_stats.get("total_items", 0)
        self.news_metrics.set_cache_items_total("redis", total_items)

        # Set cache hit ratio
        hit_ratio = cache_stats.get("hit_ratio", 0.0)
        self.news_metrics.set_cache_hit_ratio("redis", hit_ratio)

    async def collect_database_metrics(self, db_stats: Dict[str, Any]) -> None:
        """Collect and update database-related metrics."""
        if not db_stats:
            return

        # Set active connections
        active_connections = db_stats.get("active_connections", 0)
        self.news_metrics.set_database_connections_active("mongodb", active_connections)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics for debugging/monitoring."""
        return {
            "service": self.service_name,
            "metrics_available": [
                "news_ingested_total",
                "news_processing_duration_seconds",
                "cache_items_total",
                "cache_hit_ratio",
                "database_connections_active",
                "job_executions_total",
                "job_duration_seconds",
            ],
            "http_duration_buckets": HTTP_DURATION_BUCKETS,
            "news_processing_buckets": NEWS_PROCESSING_BUCKETS,
        }


# Global metrics instance
news_metrics = NewsMetrics()
metrics_collector = MetricsCollector()


def get_news_metrics() -> NewsMetrics:
    """Get the global news metrics instance."""
    return news_metrics


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector


# Convenience functions for common operations
def track_news_ingestion(source: str, status: str = "success") -> None:
    """Track news ingestion with default metrics."""
    news_metrics.increment_news_ingested(source, status)


def track_news_processing(source: str, operation: str):
    """Track news processing with default metrics."""
    return news_metrics.track_news_processing(source, operation)


def track_job_execution(job_type: str):
    """Track job execution with default metrics."""
    return news_metrics.track_job_execution(job_type)
