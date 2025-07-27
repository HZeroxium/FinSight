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

    # gRPC configuration
    enable_grpc: bool = True
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051
    grpc_max_workers: int = 10
    grpc_max_receive_message_length: int = 4 * 1024 * 1024  # 4MB
    grpc_max_send_message_length: int = 4 * 1024 * 1024  # 4MB

    # Tavily API configuration
    tavily_api_key: Optional[str] = Field(default=None)

    # Database Environment Configuration
    database_environment: str = Field(
        default="local", description="Database environment: 'local' or 'cloud'"
    )

    # MongoDB Local Configuration
    mongodb_local_url: str = Field(default="mongodb://localhost:27017")
    mongodb_local_database: str = "finsight_coindesk_news"

    # MongoDB Cloud Configuration
    mongodb_cloud_url: str = Field(
        default="", description="MongoDB Atlas connection string"
    )
    mongodb_cloud_database: str = "finsight_news"

    # MongoDB Collection Names (shared between local and cloud)
    mongodb_collection_news: str = "news_items"

    # Connection Options
    mongodb_connection_timeout: int = Field(
        default=10000, description="Connection timeout in milliseconds"
    )
    mongodb_server_selection_timeout: int = Field(
        default=5000, description="Server selection timeout in milliseconds"
    )
    mongodb_max_pool_size: int = Field(
        default=10, description="Maximum connection pool size"
    )
    mongodb_min_pool_size: int = Field(
        default=1, description="Minimum connection pool size"
    )

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
    # @field_validator("tavily_api_key", mode="before")
    # @classmethod
    # def validate_tavily_api_key(cls, v):
    #     if v is None:
    #         v = os.getenv("TAVILY_API_KEY")
    #     if not v:
    #         raise ValueError("TAVILY_API_KEY environment variable is required")
    #     return v

    @field_validator("database_environment")
    @classmethod
    def validate_database_environment(cls, v):
        valid_environments = {"local", "cloud"}
        if v.lower() not in valid_environments:
            raise ValueError(
                f"database_environment must be one of {sorted(valid_environments)}"
            )
        return v.lower()

    @field_validator("mongodb_cloud_url")
    @classmethod
    def validate_mongodb_cloud_url(cls, v, info):
        # Only validate cloud URL if database_environment is 'cloud'
        if info.data.get("database_environment") == "cloud" and not v:
            raise ValueError(
                "mongodb_cloud_url is required when database_environment is 'cloud'"
            )
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

    # Dynamic properties for current database configuration
    @property
    def mongodb_url(self) -> str:
        """Get the active MongoDB URL based on environment"""
        if self.database_environment == "cloud":
            return self.mongodb_cloud_url
        return self.mongodb_local_url

    @property
    def mongodb_database(self) -> str:
        """Get the active MongoDB database name based on environment"""
        if self.database_environment == "cloud":
            return self.mongodb_cloud_database
        return self.mongodb_local_database

    @property
    def database_info(self) -> dict:
        """Get current database configuration info"""
        return {
            "environment": self.database_environment,
            "url": self.mongodb_url,
            "database": self.mongodb_database,
            "connection_timeout": self.mongodb_connection_timeout,
            "server_selection_timeout": self.mongodb_server_selection_timeout,
            "max_pool_size": self.mongodb_max_pool_size,
            "min_pool_size": self.mongodb_min_pool_size,
        }

    def get_mongodb_connection_options(self) -> dict:
        """Get MongoDB connection options"""
        return {
            "connectTimeoutMS": self.mongodb_connection_timeout,
            "serverSelectionTimeoutMS": self.mongodb_server_selection_timeout,
            "maxPoolSize": self.mongodb_max_pool_size,
            "minPoolSize": self.mongodb_min_pool_size,
        }


# Global settings instance
settings = Settings()
