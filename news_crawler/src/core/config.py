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

    # Eureka Client configuration
    enable_eureka_client: bool = Field(
        default=True, description="Enable Eureka client registration"
    )
    eureka_server_url: str = Field(
        default="http://localhost:8761", description="Eureka server URL"
    )
    eureka_app_name: str = Field(
        default="news-crawler-service",
        description="Application name for Eureka registration",
    )
    eureka_instance_id: Optional[str] = Field(
        default=None, description="Instance ID for Eureka registration"
    )
    eureka_host_name: Optional[str] = Field(
        default=None, description="Host name for Eureka registration"
    )
    eureka_ip_address: Optional[str] = Field(
        default=None, description="IP address for Eureka registration"
    )
    eureka_port: int = Field(default=8000, description="Port for Eureka registration")
    eureka_secure_port: int = Field(
        default=8443, description="Secure port for Eureka registration"
    )
    eureka_secure_port_enabled: bool = Field(
        default=False, description="Enable secure port for Eureka registration"
    )
    eureka_home_page_url: Optional[str] = Field(
        default=None, description="Home page URL for Eureka registration"
    )
    eureka_status_page_url: Optional[str] = Field(
        default=None, description="Status page URL for Eureka registration"
    )
    eureka_health_check_url: Optional[str] = Field(
        default=None, description="Health check URL for Eureka registration"
    )
    eureka_vip_address: Optional[str] = Field(
        default=None, description="VIP address for Eureka registration"
    )
    eureka_secure_vip_address: Optional[str] = Field(
        default=None, description="Secure VIP address for Eureka registration"
    )
    eureka_prefer_ip_address: bool = Field(
        default=True,
        description="Prefer IP address over hostname for Eureka registration",
    )
    eureka_lease_renewal_interval_in_seconds: int = Field(
        default=30, description="Lease renewal interval in seconds"
    )
    eureka_lease_expiration_duration_in_seconds: int = Field(
        default=90, description="Lease expiration duration in seconds"
    )
    eureka_registry_fetch_interval_seconds: int = Field(
        default=30, description="Registry fetch interval in seconds"
    )
    eureka_instance_info_replication_interval_seconds: int = Field(
        default=30, description="Instance info replication interval in seconds"
    )
    eureka_initial_instance_info_replication_interval_seconds: int = Field(
        default=40, description="Initial instance info replication interval in seconds"
    )
    eureka_heartbeat_interval_seconds: int = Field(
        default=30, description="Heartbeat interval in seconds"
    )

    # Tavily API configuration
    tavily_api_key: Optional[str] = Field(default=None)

    # Admin API configuration
    secret_api_key: Optional[str] = Field(
        default=None, description="Secret API key for admin endpoints (job management)"
    )

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

    @field_validator("eureka_lease_renewal_interval_in_seconds")
    @classmethod
    def validate_eureka_lease_renewal_interval(cls, v):
        if v < 1 or v > 300:
            raise ValueError(
                "eureka_lease_renewal_interval_in_seconds must be between 1 and 300"
            )
        return v

    @field_validator("eureka_lease_expiration_duration_in_seconds")
    @classmethod
    def validate_eureka_lease_expiration_duration(cls, v):
        if v < 30 or v > 900:
            raise ValueError(
                "eureka_lease_expiration_duration_in_seconds must be between 30 and 900"
            )
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

    @property
    def eureka_config(self) -> dict:
        """Get Eureka client configuration"""
        return {
            "enable_eureka_client": self.enable_eureka_client,
            "eureka_server_url": self.eureka_server_url,
            "app_name": self.eureka_app_name,
            "instance_id": self.eureka_instance_id,
            "host_name": self.eureka_host_name,
            "ip_address": self.eureka_ip_address,
            "port": self.eureka_port,
            "secure_port": self.eureka_secure_port,
            "secure_port_enabled": self.eureka_secure_port_enabled,
            "home_page_url": self.eureka_home_page_url,
            "status_page_url": self.eureka_status_page_url,
            "health_check_url": self.eureka_health_check_url,
            "vip_address": self.eureka_vip_address,
            "secure_vip_address": self.eureka_secure_vip_address,
            "prefer_ip_address": self.eureka_prefer_ip_address,
            "lease_renewal_interval_in_seconds": self.eureka_lease_renewal_interval_in_seconds,
            "lease_expiration_duration_in_seconds": self.eureka_lease_expiration_duration_in_seconds,
            "registry_fetch_interval_seconds": self.eureka_registry_fetch_interval_seconds,
            "instance_info_replication_interval_seconds": self.eureka_instance_info_replication_interval_seconds,
            "initial_instance_info_replication_interval_seconds": self.eureka_initial_instance_info_replication_interval_seconds,
            "heartbeat_interval_seconds": self.eureka_heartbeat_interval_seconds,
        }


# Global settings instance
settings = Settings()
