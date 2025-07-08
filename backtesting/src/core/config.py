"""
Configuration management for the backtesting system.
Centralized configuration using Pydantic settings with environment variable support.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import SettingsConfigDict, BaseSettings

from ..common.logger import LoggerFactory, LoggerType, LogLevel


"""
Configuration management for the backtesting system.
Centralized configuration using Pydantic settings with environment variable support.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import SettingsConfigDict, BaseSettings

from ..common.logger import LoggerFactory, LoggerType, LogLevel


class Settings(BaseSettings):
    """Main configuration class for the backtesting system"""

    # Service configuration
    app_name: str = "backtesting-service"
    debug: bool = False
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000

    # Storage configuration
    storage_base_directory: str = Field(default="data", env="STORAGE_BASE_DIRECTORY")

    # MongoDB configuration
    mongodb_url: str = Field(default="mongodb://localhost:27017/", env="MONGODB_URL")
    mongodb_database: str = Field(
        default="finsight_market_data", env="MONGODB_DATABASE"
    )

    # Data collection configuration (environment variable support)
    default_symbols: List[str] = Field(
        default_factory=lambda: ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        env="DEFAULT_SYMBOLS",
    )
    default_timeframes: List[str] = Field(
        default_factory=lambda: ["1h", "4h", "1d"], env="DEFAULT_TIMEFRAMES"
    )

    # Rate limiting
    max_ohlcv_limit: int = Field(default=1000, env="MAX_OHLCV_LIMIT")
    max_trades_limit: int = Field(default=1000, env="MAX_TRADES_LIMIT")
    max_orderbook_limit: int = Field(default=100, env="MAX_ORDERBOOK_LIMIT")

    # Exchange configuration
    binance_requests_per_minute: int = Field(
        default=1200, env="BINANCE_REQUESTS_PER_MINUTE"
    )
    binance_orders_per_second: int = Field(default=10, env="BINANCE_ORDERS_PER_SECOND")
    binance_orders_per_day: int = Field(default=200000, env="BINANCE_ORDERS_PER_DAY")

    # Cross-repository configuration
    source_repository_type: str = Field(default="mongodb", env="SOURCE_REPOSITORY_TYPE")
    source_timeframe: str = Field(default="1h", env="SOURCE_TIMEFRAME")
    target_repository_type: str = Field(default="csv", env="TARGET_REPOSITORY_TYPE")
    target_timeframes: List[str] = Field(
        default_factory=lambda: ["2h", "4h", "12h", "1d"], env="TARGET_TIMEFRAMES"
    )
    enable_parallel_conversion: bool = Field(
        default=True, env="ENABLE_PARALLEL_CONVERSION"
    )
    max_concurrent_conversions: int = Field(default=3, env="MAX_CONCURRENT_CONVERSIONS")
    conversion_batch_size: int = Field(default=1000, env="CONVERSION_BATCH_SIZE")

    # Logging configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file_path: str = Field(default="logs/", env="LOG_FILE_PATH")
    enable_structured_logging: bool = Field(
        default=True, env="ENABLE_STRUCTURED_LOGGING"
    )

    # Cache configuration
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    cache_ttl_seconds: int = Field(default=300, env="CACHE_TTL_SECONDS")
    cache_max_size: int = Field(default=1000, env="CACHE_MAX_SIZE")

    # Admin API configuration
    admin_api_key: str = Field(
        default="admin-default-key-change-in-production", env="ADMIN_API_KEY"
    )

    # Cron job configuration
    cron_job_enabled: bool = Field(default=False, env="CRON_JOB_ENABLED")
    cron_job_schedule: str = Field(
        default="0 */4 * * *", env="CRON_JOB_SCHEDULE"
    )  # Every 4 hours
    cron_job_max_symbols_per_run: int = Field(
        default=10, env="CRON_JOB_MAX_SYMBOLS_PER_RUN"
    )
    cron_job_log_file: str = Field(
        default="logs/market_data_job.log", env="CRON_JOB_LOG_FILE"
    )
    cron_job_pid_file: str = Field(
        default="market_data_job.pid", env="CRON_JOB_PID_FILE"
    )

    @field_validator("default_symbols", mode="before")
    @classmethod
    def parse_symbols(cls, v):
        """Parse comma-separated symbols from environment variable"""
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("default_timeframes", mode="before")
    @classmethod
    def parse_timeframes(cls, v):
        """Parse comma-separated timeframes from environment variable"""
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("target_timeframes", mode="before")
    @classmethod
    def parse_target_timeframes(cls, v):
        """Parse comma-separated target timeframes from environment variable"""
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in levels:
            raise ValueError(f"log_level must be one of {sorted(levels)}")
        return v.upper()

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        allowed_envs = {"development", "staging", "production", "testing"}
        if v.lower() not in allowed_envs:
            raise ValueError(f"environment must be one of {sorted(allowed_envs)}")
        return v.lower()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Global settings instance
settings = Settings()


# For backward compatibility, create some helper config objects
class CrossRepositoryConfig(BaseModel):
    """Configuration for cross-repository operations"""

    source_repository: Dict[str, Any]
    target_repository: Dict[str, Any]
    source_timeframe: str
    target_timeframes: List[str]
    enable_parallel_conversion: bool
    max_concurrent_conversions: int
    conversion_batch_size: int


def get_cross_repository_config() -> CrossRepositoryConfig:
    """Get cross-repository configuration"""
    return CrossRepositoryConfig(
        source_repository={
            "type": settings.source_repository_type,
            "mongodb": {
                "connection_string": settings.mongodb_url,
                "database_name": settings.mongodb_database,
            },
        },
        target_repository={
            "type": settings.target_repository_type,
            "csv": {
                "base_directory": f"{settings.storage_base_directory}/converted_timeframes"
            },
        },
        source_timeframe=settings.source_timeframe,
        target_timeframes=settings.target_timeframes,
        enable_parallel_conversion=settings.enable_parallel_conversion,
        max_concurrent_conversions=settings.max_concurrent_conversions,
        conversion_batch_size=settings.conversion_batch_size,
    )


def get_symbols_for_exchange(exchange_name: str) -> List[str]:
    """Get default symbols for specific exchange"""
    # Exchange-specific symbol mappings
    symbol_mappings = {
        "binance": [s.replace("/", "") for s in settings.default_symbols],
        "ccxt": settings.default_symbols,
        "cryptofeed": [s.replace("/", "") for s in settings.default_symbols],
    }
    return symbol_mappings.get(exchange_name, settings.default_symbols)


def get_timeframes_for_exchange(exchange_name: str) -> List[str]:
    """Get supported timeframes for specific exchange"""
    # Exchange-specific timeframe mappings
    timeframe_mappings = {
        "binance": settings.default_timeframes,
        "ccxt": settings.default_timeframes,
        "cryptofeed": ["1m", "5m", "15m", "1h", "1d"],  # Cryptofeed naming
    }
    return timeframe_mappings.get(exchange_name, settings.default_timeframes)
