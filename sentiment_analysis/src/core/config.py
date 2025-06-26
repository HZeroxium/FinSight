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
    mongodb_database: str = "sentiment_analysis"
    mongodb_collection_sentiments: str = "sentiments"
    mongodb_collection_articles: str = "processed_articles"

    # RabbitMQ configuration
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    rabbitmq_exchange: str = "sentiment_analysis_exchange"
    rabbitmq_queue_raw_articles: str = "raw_articles_queue"
    rabbitmq_queue_processed: str = "processed_sentiments_queue"

    # Processing configuration
    enable_batch_processing: bool = True
    max_concurrent_analysis: int = 5
    analysis_timeout: int = 30
    analysis_retry_attempts: int = 3
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
