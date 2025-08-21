# core/config.py

"""
Configuration management for the sentiment analysis service.
"""

import os
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""

    # Service configuration
    app_name: str = "sentiment-analysis-service"
    debug: bool = False
    environment: str = "development"

    # OpenAI API configuration
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.0
    openai_max_tokens: int = 1000

    # MongoDB configuration
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "finsight_news"  # Shared database with news service
    mongodb_collection_news: str = "news"

    # RabbitMQ configuration - synchronized with news service
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"

    # Single exchange for all news events
    rabbitmq_exchange: str = "news.event"

    # Queue names
    rabbitmq_queue_news_to_sentiment: str = "news.sentiment_analysis"
    rabbitmq_queue_sentiment_results: str = "sentiment.results"

    # Routing keys
    rabbitmq_routing_key_news_to_sentiment: str = "news.sentiment.analyze"
    rabbitmq_routing_key_sentiment_results: str = "sentiment.results.processed"

    # Processing configuration
    enable_batch_processing: bool = True
    max_concurrent_analysis: int = 5
    analysis_timeout: int = 30
    analysis_retry_attempts: int = 3
    analyzer_version: str = "openai-gpt-4o-mini"
    batch_size: int = 10

    # Cache configuration
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600

    # Rate limiting
    rate_limit_requests_per_minute: int = 50

    # Logging configuration
    log_level: str = "INFO"
    log_file_path: str = "logs/"
    enable_structured_logging: bool = True

    # Message Publishing Toggle
    enable_message_publishing: bool = Field(
        default=True, env="ENABLE_MESSAGE_PUBLISHING"
    )

    # Message Publishing for analyze_text method
    enable_analyze_text_publishing: bool = Field(
        default=True,
        env="ENABLE_ANALYZE_TEXT_PUBLISHING",
    )

    # RabbitMQ Connection Settings
    rabbitmq_connection_timeout: int = Field(
        default=10, env="RABBITMQ_CONNECTION_TIMEOUT"
    )

    rabbitmq_retry_attempts: int = Field(default=3, env="RABBITMQ_RETRY_ATTEMPTS")

    # Validators
    @field_validator("openai_api_key", mode="before")
    @classmethod
    def validate_openai_api_key(cls, v):
        if v is None:
            v = os.getenv("OPENAI_API_KEY")
        if not v:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in levels:
            raise ValueError(f"log_level must be one of {sorted(levels)}")
        return v.upper()

    @field_validator("max_concurrent_analysis")
    @classmethod
    def validate_max_concurrent_analysis(cls, v):
        if v < 1 or v > 50:
            raise ValueError("max_concurrent_analysis must be between 1 and 50")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Global settings instance
settings = Settings()
