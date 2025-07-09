# core/config.py

"""
Configuration management for the news crawler service.
"""

import os
from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""

    # Service configuration
    app_name: str = "news-crawler-service"
    debug: bool = False
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000

    # Tavily API configuration
    tavily_api_key: Optional[str] = Field(default=None)

    # MongoDB configuration
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "finsight_coindesk_news"
    mongodb_collection_articles: str = "articles"
    mongodb_collection_sources: str = "sources"

    # RabbitMQ configuration - centralized messaging settings
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    rabbitmq_article_exchange: str = "news_crawler_exchange"
    rabbitmq_analytics_exchange: str = "analytics_exchange"

    # Queue names
    rabbitmq_queue_raw_articles: str = "raw_articles_queue"
    rabbitmq_queue_processed_sentiments: str = "processed_sentiments_queue"

    # Routing keys
    rabbitmq_routing_key_article_sentiment: str = "article.sentiment_analysis"
    rabbitmq_routing_key_sentiment_processed: str = "sentiment.processed"
    rabbitmq_routing_key_search_event: str = "search.event"

    # Crawler configuration
    enable_advanced_crawling: bool = True
    max_concurrent_crawls: int = 10
    crawl_timeout: int = 30
    crawl_retry_attempts: int = 3

    # Cache configuration
    enable_caching: bool = True
    cache_ttl_seconds: int = 300

    # Rate limiting
    rate_limit_requests_per_minute: int = 100

    # Cron job configuration
    cron_job_enabled: bool = True
    cron_job_schedule: str = "0 */1 * * *"  # Every hour
    cron_job_max_items_per_source: int = 100
    cron_job_sources: List[str] = Field(
        default_factory=lambda: ["coindesk", "cointelegraph"]
    )
    cron_job_config_file: str = "news_crawler_config.json"
    cron_job_pid_file: str = "news_crawler_job.pid"
    cron_job_log_file: str = "logs/news_crawler_job.log"

    # Logging configuration
    log_level: str = "INFO"
    log_file_path: str = "logs/"
    enable_structured_logging: bool = True

    # Validators
    @field_validator("tavily_api_key", mode="before")
    @classmethod
    def validate_tavily_api_key(cls, v):
        if v is None:
            v = os.getenv("TAVILY_API_KEY")
        if not v:
            raise ValueError("TAVILY_API_KEY environment variable is required")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in levels:
            raise ValueError(f"log_level must be one of {sorted(levels)}")
        return v.upper()

    @field_validator("max_concurrent_crawls")
    @classmethod
    def validate_max_concurrent_crawls(cls, v):
        if v < 1 or v > 100:
            raise ValueError("max_concurrent_crawls must be between 1 and 100")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Global settings instance
settings = Settings()
