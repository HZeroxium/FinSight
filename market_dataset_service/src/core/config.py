# core/config.py

"""
Configuration management for the backtesting system.
Centralized configuration using Pydantic settings with environment variable support.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..schemas.enums import (CryptoSymbol, RepositoryType, StorageProviderType,
                             TimeFrame)


class Settings(BaseSettings):
    """Main configuration class for the backtesting system"""

    # Service configuration
    app_name: str = "market-dataset-service"
    debug: bool = False
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000

    # Storage configuration
    storage_base_directory: str = Field(
        default="data/market_data", env="STORAGE_BASE_DIRECTORY"
    )

    # Storage prefix configuration for object storage
    storage_prefix: str = Field(
        default="finsight/market_data/datasets", env="STORAGE_PREFIX"
    )
    storage_separator: str = Field(default="/", env="STORAGE_SEPARATOR")

    # MongoDB configuration
    mongodb_url: str = Field(default="mongodb://localhost:27017/", env="MONGODB_URL")
    mongodb_database: str = Field(
        default="finsight_market_data", env="MONGODB_DATABASE"
    )

    # Object Storage Configuration (S3-compatible: MinIO, DigitalOcean Spaces, AWS S3)
    # Storage provider selection
    storage_provider: str = Field(
        default=StorageProviderType.MINIO.value, env="STORAGE_PROVIDER"
    )  # minio, digitalocean, aws

    # S3-compatible storage settings
    s3_endpoint_url: str = Field(default="http://localhost:9000", env="S3_ENDPOINT_URL")
    s3_access_key: str = Field(default="minioadmin", env="S3_ACCESS_KEY")
    s3_secret_key: str = Field(default="minioadmin", env="S3_SECRET_KEY")
    s3_region_name: str = Field(default="us-east-1", env="S3_REGION_NAME")
    s3_bucket_name: str = Field(default="market-data", env="S3_BUCKET_NAME")
    s3_use_ssl: bool = Field(default=False, env="S3_USE_SSL")
    s3_verify_ssl: bool = Field(default=True, env="S3_VERIFY_SSL")
    s3_signature_version: str = Field(default="s3v4", env="S3_SIGNATURE_VERSION")
    s3_max_pool_connections: int = Field(default=50, env="S3_MAX_POOL_CONNECTIONS")

    # DigitalOcean Spaces specific settings
    spaces_endpoint_url: str = Field(
        default="https://nyc3.digitaloceanspaces.com", env="SPACES_ENDPOINT_URL"
    )
    spaces_access_key: str = Field(default="", env="SPACES_ACCESS_KEY")
    spaces_secret_key: str = Field(default="", env="SPACES_SECRET_KEY")
    spaces_region_name: str = Field(default="nyc3", env="SPACES_REGION_NAME")
    spaces_bucket_name: str = Field(
        default="finsight-market-data", env="SPACES_BUCKET_NAME"
    )

    # AWS S3 specific settings (if using AWS directly)
    aws_access_key_id: str = Field(default="", env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", env="AWS_SECRET_ACCESS_KEY")
    aws_region_name: str = Field(default="us-east-1", env="AWS_REGION_NAME")
    aws_bucket_name: str = Field(default="finsight-market-data", env="AWS_BUCKET_NAME")

    # Data collection configuration (environment variable support)
    default_symbols: List[str] = Field(
        default_factory=lambda: [
            CryptoSymbol.BTCUSDT.value,
            CryptoSymbol.ETHUSDT.value,
            CryptoSymbol.BNBUSDT.value,
        ],
        env="DEFAULT_SYMBOLS",
    )
    default_timeframes: List[str] = Field(
        default_factory=lambda: [
            TimeFrame.HOUR_1.value,
            TimeFrame.HOUR_4.value,
            TimeFrame.DAY_1.value,
        ],
        env="DEFAULT_TIMEFRAMES",
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

    binance_api_key: str = Field(default="", env="BINANCE_API_KEY")
    binance_secret_key: str = Field(default="", env="BINANCE_SECRET_KEY")

    # Repository configuration
    repository_type: str = Field(
        default=RepositoryType.CSV.value, env="REPOSITORY_TYPE"
    )

    # Cross-repository configuration
    source_repository_type: str = Field(
        default=RepositoryType.CSV.value, env="SOURCE_REPOSITORY_TYPE"
    )
    source_timeframe: str = Field(
        default=TimeFrame.HOUR_1.value, env="SOURCE_TIMEFRAME"
    )
    target_repository_type: str = Field(
        default=RepositoryType.CSV.value, env="TARGET_REPOSITORY_TYPE"
    )
    target_timeframes: List[str] = Field(
        default_factory=lambda: [
            TimeFrame.HOUR_2.value,
            TimeFrame.HOUR_4.value,
            TimeFrame.HOUR_12.value,
            TimeFrame.DAY_1.value,
        ],
        env="TARGET_TIMEFRAMES",
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
    api_key: str = Field(
        default="admin-default-key-change-in-production", env="API_KEY"
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

    # Demo configuration
    demo_max_symbols: int = Field(default=5, env="DEMO_MAX_SYMBOLS")
    demo_days_back: int = Field(default=7, env="DEMO_DAYS_BACK")

    # Eureka Client configuration
    enable_eureka_client: bool = Field(
        default=True,
        description="Enable Eureka client registration",
        env="ENABLE_EUREKA_CLIENT",
    )
    eureka_server_url: str = Field(
        default="http://localhost:8761", description="Eureka server URL"
    )
    eureka_app_name: str = Field(
        default="market-dataset-service",
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

    @field_validator("storage_provider")
    @classmethod
    def validate_storage_provider(cls, v):
        allowed_providers = {
            StorageProviderType.MINIO.value,
            StorageProviderType.DIGITALOCEAN.value,
            StorageProviderType.AWS.value,
            StorageProviderType.S3.value,
        }
        if v.lower() not in allowed_providers:
            raise ValueError(
                f"storage_provider must be one of {sorted(allowed_providers)}"
            )
        return v.lower()

    def get_storage_config(self) -> Dict[str, Any]:
        """
        Get storage configuration based on the selected provider.

        Returns:
            Dictionary with storage configuration parameters
        """
        if self.storage_provider == StorageProviderType.MINIO.value:
            return {
                "endpoint_url": self.s3_endpoint_url,
                "access_key": self.s3_access_key,
                "secret_key": self.s3_secret_key,
                "region_name": self.s3_region_name,
                "bucket_name": self.s3_bucket_name,
                "use_ssl": self.s3_use_ssl,
                "verify_ssl": self.s3_verify_ssl,
                "signature_version": self.s3_signature_version,
                "max_pool_connections": self.s3_max_pool_connections,
            }
        elif self.storage_provider == StorageProviderType.DIGITALOCEAN.value:
            return {
                "endpoint_url": self.spaces_endpoint_url,
                "access_key": self.spaces_access_key,
                "secret_key": self.spaces_secret_key,
                "region_name": self.spaces_region_name,
                "bucket_name": self.spaces_bucket_name,
                "use_ssl": True,  # DigitalOcean Spaces always uses SSL
                "verify_ssl": True,
                "signature_version": "s3v4",
                "max_pool_connections": self.s3_max_pool_connections,
            }
        elif self.storage_provider in [
            StorageProviderType.AWS.value,
            StorageProviderType.S3.value,
        ]:
            return {
                "endpoint_url": None,  # Use AWS default endpoint
                "access_key": self.aws_access_key_id or self.s3_access_key,
                "secret_key": self.aws_secret_access_key or self.s3_secret_key,
                "region_name": self.aws_region_name,
                "bucket_name": self.aws_bucket_name,
                "use_ssl": True,  # AWS S3 always uses SSL
                "verify_ssl": True,
                "signature_version": "s3v4",
                "max_pool_connections": self.s3_max_pool_connections,
            }
        else:
            # Default to MinIO configuration
            return {
                "endpoint_url": self.s3_endpoint_url,
                "access_key": self.s3_access_key,
                "secret_key": self.s3_secret_key,
                "region_name": self.s3_region_name,
                "bucket_name": self.s3_bucket_name,
                "use_ssl": self.s3_use_ssl,
                "verify_ssl": self.s3_verify_ssl,
                "signature_version": self.s3_signature_version,
                "max_pool_connections": self.s3_max_pool_connections,
            }

    def get_storage_prefix(self) -> str:
        """
        Get the base storage prefix for object storage.

        Returns:
            Base storage prefix (e.g., "datasets")
        """
        return self.storage_prefix

    def build_storage_path(self, *parts: str) -> str:
        """
        Build a storage path using the configured prefix and separator.

        Args:
            *parts: Path components to join

        Returns:
            Complete storage path
        """
        all_parts = [self.storage_prefix] + list(parts)
        return self.storage_separator.join(filter(None, all_parts))

    def build_dataset_path(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        format_type: str = None,
        date: str = None,
    ) -> str:
        """
        Build a dataset storage path.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Timeframe
            format_type: Data format (optional)
            date: Date (optional)

        Returns:
            Dataset storage path
        """
        parts = [exchange, symbol, timeframe]
        if format_type:
            parts.append(format_type)
        if date:
            parts.append(date)
        return self.build_storage_path(*parts)

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
