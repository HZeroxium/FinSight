# core/config.py

"""
Configuration management for the news crawler service.
"""

import os
from typing import List, Optional

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
        default="news-service",
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

    # Eureka Retry Configuration
    eureka_registration_retry_attempts: int = Field(
        default=3, description="Number of retry attempts for registration"
    )
    eureka_registration_retry_delay_seconds: int = Field(
        default=5, description="Initial delay between registration retries in seconds"
    )
    eureka_heartbeat_retry_attempts: int = Field(
        default=3, description="Number of retry attempts for heartbeat"
    )
    eureka_heartbeat_retry_delay_seconds: int = Field(
        default=2, description="Initial delay between heartbeat retries in seconds"
    )
    eureka_retry_backoff_multiplier: float = Field(
        default=2.0, description="Multiplier for exponential backoff"
    )
    eureka_max_retry_delay_seconds: int = Field(
        default=60, description="Maximum delay between retries in seconds"
    )
    eureka_enable_auto_re_registration: bool = Field(
        default=True,
        description="Enable automatic re-registration after server restart",
    )
    eureka_re_registration_delay_seconds: int = Field(
        default=10, description="Delay before attempting re-registration in seconds"
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
    rabbitmq_url: str = Field(default="amqp://guest:guest@localhost:5672/")

    # Single exchange for all news events
    rabbitmq_exchange: str = Field(default="news.event")

    # Queue names
    rabbitmq_queue_news_to_sentiment: str = Field(default="news.sentiment_analysis")
    rabbitmq_queue_sentiment_results: str = Field(default="sentiment.results")

    # Routing keys
    rabbitmq_routing_key_news_to_sentiment: str = Field(
        default="news.sentiment.analyze"
    )
    rabbitmq_routing_key_sentiment_results: str = Field(
        default="sentiment.results.processed"
    )

    # Crawler configuration
    enable_advanced_crawling: bool = Field(default=True)
    max_concurrent_crawls: int = Field(default=10)
    crawl_timeout: int = Field(default=30)
    crawl_retry_attempts: int = Field(default=3)

    # Cache configuration
    enable_caching: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=300)

    # Redis Cache Configuration
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_key_prefix: str = Field(
        default="news-service:", description="Redis key prefix"
    )
    redis_connection_timeout: int = Field(
        default=5, description="Redis connection timeout"
    )
    redis_socket_timeout: int = Field(default=5, description="Redis socket timeout")
    redis_socket_connect_timeout: int = Field(
        default=5, description="Redis socket connect timeout"
    )
    redis_socket_keepalive: bool = Field(
        default=True, description="Redis socket keepalive"
    )
    redis_retry_on_timeout: bool = Field(
        default=True, description="Redis retry on timeout"
    )
    redis_max_connections: int = Field(default=10, description="Redis max connections")

    # Cache TTL Configuration for different endpoints
    cache_ttl_search_news: int = Field(
        default=1800, description="TTL for search news (30 minutes)"
    )
    cache_ttl_recent_news: int = Field(
        default=900, description="TTL for recent news (15 minutes)"
    )
    cache_ttl_news_by_source: int = Field(
        default=1800, description="TTL for news by source (30 minutes)"
    )
    cache_ttl_news_by_keywords: int = Field(
        default=1200, description="TTL for news by keywords (20 minutes)"
    )
    cache_ttl_news_by_tags: int = Field(
        default=1800, description="TTL for news by tags (30 minutes)"
    )
    cache_ttl_available_tags: int = Field(
        default=3600, description="TTL for available tags (1 hour)"
    )
    cache_ttl_repository_stats: int = Field(
        default=600, description="TTL for repository stats (10 minutes)"
    )
    cache_ttl_news_item: int = Field(
        default=7200, description="TTL for individual news item (2 hours)"
    )

    # Cache invalidation settings
    cache_invalidation_enabled: bool = Field(
        default=True, description="Enable cache invalidation"
    )
    cache_invalidation_pattern: str = Field(
        default="news-service:*", description="Cache invalidation pattern"
    )
    cache_invalidation_delay_seconds: int = Field(
        default=5, description="Delay before cache invalidation"
    )

    # Rate Limiting Configuration
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests_per_minute: int = Field(
        default=100, description="Default requests per minute"
    )
    rate_limit_requests_per_hour: int = Field(
        default=1000, description="Default requests per hour"
    )
    rate_limit_requests_per_day: int = Field(
        default=10000, description="Default requests per day"
    )

    # Rate limiting backend configuration
    rate_limit_storage_url: str = Field(
        default="redis://localhost:6379/1",
        description="Redis URL for rate limiting storage",
    )
    rate_limit_key_prefix: str = Field(
        default="rate-limit:", description="Redis key prefix for rate limiting"
    )

    # Rate limiting headers configuration
    rate_limit_include_headers: bool = Field(
        default=True, description="Include rate limit headers in responses"
    )
    rate_limit_retry_after_header: bool = Field(
        default=True, description="Include Retry-After header when rate limited"
    )

    # Rate limiting client identification
    rate_limit_by_api_key: bool = Field(
        default=True, description="Use API key for client identification when available"
    )
    rate_limit_by_ip: bool = Field(
        default=True, description="Use IP address for client identification"
    )
    rate_limit_trust_proxy: bool = Field(
        default=True, description="Trust X-Forwarded-For header from proxies"
    )

    # Rate limiting exempt endpoints
    rate_limit_exempt_endpoints: List[str] = Field(
        default_factory=lambda: [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ],
        description="Endpoints exempt from rate limiting",
    )

    # Rate limiting per-route configuration
    rate_limit_news_search_per_minute: int = Field(
        default=60, description="Rate limit for news search endpoints"
    )
    rate_limit_news_search_per_hour: int = Field(
        default=500, description="Hourly rate limit for news search endpoints"
    )

    rate_limit_admin_per_minute: int = Field(
        default=30, description="Rate limit for admin endpoints"
    )
    rate_limit_admin_per_hour: int = Field(
        default=200, description="Hourly rate limit for admin endpoints"
    )

    rate_limit_cache_per_minute: int = Field(
        default=20, description="Rate limit for cache management endpoints"
    )
    rate_limit_cache_per_hour: int = Field(
        default=100, description="Hourly rate limit for cache management endpoints"
    )

    # Cron job configuration
    cron_job_enabled: bool = True
    cron_job_schedule: str = "0 */6 * * *"  # Every 6 hours (changed from every hour)
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

    @field_validator("eureka_registration_retry_attempts")
    @classmethod
    def validate_eureka_registration_retry_attempts(cls, v):
        if v < 1 or v > 10:
            raise ValueError(
                "eureka_registration_retry_attempts must be between 1 and 10"
            )
        return v

    @field_validator("eureka_heartbeat_retry_attempts")
    @classmethod
    def validate_eureka_heartbeat_retry_attempts(cls, v):
        if v < 1 or v > 10:
            raise ValueError("eureka_heartbeat_retry_attempts must be between 1 and 10")
        return v

    @field_validator("eureka_retry_backoff_multiplier")
    @classmethod
    def validate_eureka_retry_backoff_multiplier(cls, v):
        if v < 1.0 or v > 5.0:
            raise ValueError(
                "eureka_retry_backoff_multiplier must be between 1.0 and 5.0"
            )
        return v

    @field_validator("cache_ttl_search_news")
    @classmethod
    def validate_cache_ttl_search_news(cls, v):
        if v < 60 or v > 7200:
            raise ValueError(
                "cache_ttl_search_news must be between 60 and 7200 seconds"
            )
        return v

    @field_validator("cache_ttl_recent_news")
    @classmethod
    def validate_cache_ttl_recent_news(cls, v):
        if v < 60 or v > 3600:
            raise ValueError(
                "cache_ttl_recent_news must be between 60 and 3600 seconds"
            )
        return v

    @field_validator("redis_max_connections")
    @classmethod
    def validate_redis_max_connections(cls, v):
        if v < 1 or v > 100:
            raise ValueError("redis_max_connections must be between 1 and 100")
        return v

    @field_validator("rate_limit_requests_per_minute")
    @classmethod
    def validate_rate_limit_requests_per_minute(cls, v):
        if v < 1 or v > 10000:
            raise ValueError(
                "rate_limit_requests_per_minute must be between 1 and 10000"
            )
        return v

    @field_validator("rate_limit_requests_per_hour")
    @classmethod
    def validate_rate_limit_requests_per_hour(cls, v):
        if v < 1 or v > 100000:
            raise ValueError(
                "rate_limit_requests_per_hour must be between 1 and 100000"
            )
        return v

    @field_validator("rate_limit_requests_per_day")
    @classmethod
    def validate_rate_limit_requests_per_day(cls, v):
        if v < 1 or v > 1000000:
            raise ValueError(
                "rate_limit_requests_per_day must be between 1 and 1000000"
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
            "registration_retry_attempts": self.eureka_registration_retry_attempts,
            "registration_retry_delay_seconds": self.eureka_registration_retry_delay_seconds,
            "heartbeat_retry_attempts": self.eureka_heartbeat_retry_attempts,
            "heartbeat_retry_delay_seconds": self.eureka_heartbeat_retry_delay_seconds,
            "retry_backoff_multiplier": self.eureka_retry_backoff_multiplier,
            "max_retry_delay_seconds": self.eureka_max_retry_delay_seconds,
            "enable_auto_re_registration": self.eureka_enable_auto_re_registration,
            "re_registration_delay_seconds": self.eureka_re_registration_delay_seconds,
        }

    @property
    def cache_config(self) -> dict:
        """Get cache configuration"""
        return {
            "enable_caching": self.enable_caching,
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "redis_db": self.redis_db,
            "redis_password": self.redis_password,
            "redis_key_prefix": self.redis_key_prefix,
            "redis_connection_timeout": self.redis_connection_timeout,
            "redis_socket_timeout": self.redis_socket_timeout,
            "redis_socket_connect_timeout": self.redis_socket_connect_timeout,
            "redis_socket_keepalive": self.redis_socket_keepalive,
            "redis_retry_on_timeout": self.redis_retry_on_timeout,
            "redis_max_connections": self.redis_max_connections,
            "cache_ttl_search_news": self.cache_ttl_search_news,
            "cache_ttl_recent_news": self.cache_ttl_recent_news,
            "cache_ttl_news_by_source": self.cache_ttl_news_by_source,
            "cache_ttl_news_by_keywords": self.cache_ttl_news_by_keywords,
            "cache_ttl_news_by_tags": self.cache_ttl_news_by_tags,
            "cache_ttl_available_tags": self.cache_ttl_available_tags,
            "cache_ttl_repository_stats": self.cache_ttl_repository_stats,
            "cache_ttl_news_item": self.cache_ttl_news_item,
            "cache_invalidation_enabled": self.cache_invalidation_enabled,
            "cache_invalidation_pattern": self.cache_invalidation_pattern,
            "cache_invalidation_delay_seconds": self.cache_invalidation_delay_seconds,
        }

    @property
    def rate_limit_config(self) -> dict:
        """Get rate limiting configuration"""
        return {
            "enabled": self.rate_limit_enabled,
            "default_limits": {
                "per_minute": self.rate_limit_requests_per_minute,
                "per_hour": self.rate_limit_requests_per_hour,
                "per_day": self.rate_limit_requests_per_day,
            },
            "storage": {
                "url": self.rate_limit_storage_url,
                "key_prefix": self.rate_limit_key_prefix,
            },
            "headers": {
                "include_headers": self.rate_limit_include_headers,
                "retry_after_header": self.rate_limit_retry_after_header,
            },
            "client_identification": {
                "by_api_key": self.rate_limit_by_api_key,
                "by_ip": self.rate_limit_by_ip,
                "trust_proxy": self.rate_limit_trust_proxy,
            },
            "exempt_endpoints": self.rate_limit_exempt_endpoints,
            "per_route_limits": {
                "news_search": {
                    "per_minute": self.rate_limit_news_search_per_minute,
                    "per_hour": self.rate_limit_news_search_per_hour,
                },
                "admin": {
                    "per_minute": self.rate_limit_admin_per_minute,
                    "per_hour": self.rate_limit_admin_per_hour,
                },
                "cache": {
                    "per_minute": self.rate_limit_cache_per_minute,
                    "per_hour": self.rate_limit_cache_per_hour,
                },
            },
        }


# Global settings instance
settings = Settings()
