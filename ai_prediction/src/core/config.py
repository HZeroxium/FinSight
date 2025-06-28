"""
Configuration utilities for managing API settings, rate limits,
and data collection parameters across different exchanges using Pydantic.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict, BaseSettings

from ..common.logger import LoggerFactory, LoggerType, LogLevel


class ExchangeConfig(BaseSettings):
    """Configuration for exchange API settings"""

    name: str
    rate_limit: bool = True
    timeout: int = 30000  # milliseconds
    retry_count: int = 3
    retry_delay: float = 1.0  # seconds
    sandbox: bool = False
    adjust_for_time_difference: bool = True
    enable_rate_limit: bool = True
    requests_per_minute: Optional[int] = None
    max_concurrent_requests: int = 10

    @field_validator("timeout")
    def validate_timeout(cls, v):
        if v < 1000 or v > 300000:  # 1s to 5min
            raise ValueError("timeout must be between 1000 and 300000 milliseconds")
        return v

    @field_validator("retry_count")
    def validate_retry_count(cls, v):
        if v < 0 or v > 10:
            raise ValueError("retry_count must be between 0 and 10")
        return v


class DataCollectionConfig(BaseSettings):
    """Configuration for data collection parameters"""

    default_symbols: List[str] = Field(
        default_factory=lambda: [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "ADA/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "DOT/USDT",
            "DOGE/USDT",
            "AVAX/USDT",
            "MATIC/USDT",
        ]
    )
    default_timeframes: List[str] = Field(
        default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"]
    )
    max_ohlcv_limit: int = Field(default=1000, ge=1, le=5000)
    max_trades_limit: int = Field(default=1000, ge=1, le=5000)
    max_orderbook_limit: int = Field(default=100, ge=1, le=1000)
    save_raw_data: bool = True
    save_processed_data: bool = True
    data_formats: List[str] = Field(default_factory=lambda: ["json", "csv", "parquet"])
    enable_technical_indicators: bool = True
    enable_data_validation: bool = True

    @field_validator("data_formats")
    def validate_data_formats(cls, v):
        allowed_formats = {"json", "csv", "parquet", "hdf5"}
        invalid_formats = set(v) - allowed_formats
        if invalid_formats:
            raise ValueError(
                f"Invalid data formats: {invalid_formats}. Allowed: {allowed_formats}"
            )
        return v


class StorageConfig(BaseSettings):
    """Configuration for data storage"""

    base_directory: str = "data"
    create_timestamped_files: bool = True
    compress_files: bool = False
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    enable_backup: bool = False
    backup_directory: Optional[str] = None
    retention_days: int = Field(default=30, ge=1)

    @field_validator("base_directory")
    def validate_base_directory(cls, v):
        path = Path(v)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create base directory {v}: {e}")
        return v


class RateLimitConfig(BaseSettings):
    """Configuration for rate limiting"""

    binance: Dict[str, int] = Field(
        default_factory=lambda: {
            "requests_per_minute": 1200,
            "orders_per_second": 10,
            "orders_per_day": 200000,
        }
    )
    ccxt: Dict[str, int] = Field(
        default_factory=lambda: {
            "requests_per_minute": 600,
            "requests_per_second": 10,
        }
    )
    cryptofeed: Dict[str, int] = Field(
        default_factory=lambda: {
            "connections_per_minute": 300,
            "messages_per_second": 1000,
        }
    )


class AIPreedictionSettings(BaseSettings):
    """Main configuration for AI Prediction service"""

    # Service configuration
    app_name: str = "ai-prediction-service"
    debug: bool = False
    environment: str = "development"

    # Exchange configurations
    exchanges: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "binance": {
                "name": "binance",
                "rate_limit": True,
                "timeout": 30000,
                "retry_count": 3,
                "retry_delay": 1.0,
                "enable_rate_limit": True,
                "requests_per_minute": 1200,
            },
            "ccxt_binance": {
                "name": "ccxt_binance",
                "rate_limit": True,
                "timeout": 30000,
                "retry_count": 3,
                "retry_delay": 1.0,
                "enable_rate_limit": True,
                "requests_per_minute": 600,
            },
            "kraken": {
                "name": "kraken",
                "rate_limit": True,
                "timeout": 30000,
                "retry_count": 3,
                "retry_delay": 2.0,
                "enable_rate_limit": True,
                "requests_per_minute": 300,
            },
            "coinbase": {
                "name": "coinbase",
                "rate_limit": True,
                "timeout": 30000,
                "retry_count": 3,
                "retry_delay": 1.5,
                "enable_rate_limit": True,
                "requests_per_minute": 300,
            },
        }
    )

    # Data collection configuration
    data_collection: Dict[str, Any] = Field(
        default_factory=lambda: {
            "default_symbols": [
                "BTC/USDT",
                "ETH/USDT",
                "BNB/USDT",
                "ADA/USDT",
                "SOL/USDT",
                "XRP/USDT",
                "DOT/USDT",
                "DOGE/USDT",
                "AVAX/USDT",
                "MATIC/USDT",
            ],
            "default_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
            "max_ohlcv_limit": 1000,
            "max_trades_limit": 1000,
            "max_orderbook_limit": 100,
            "save_raw_data": True,
            "save_processed_data": True,
            "data_formats": ["json", "csv", "parquet"],
            "enable_technical_indicators": True,
            "enable_data_validation": True,
        }
    )

    # Storage configuration
    storage: Dict[str, Any] = Field(
        default_factory=lambda: {
            "base_directory": "data",
            "create_timestamped_files": True,
            "compress_files": False,
            "max_file_size_mb": 100,
            "enable_backup": False,
            "retention_days": 30,
        }
    )

    # Rate limits configuration
    rate_limits: Dict[str, Dict[str, int]] = Field(
        default_factory=lambda: {
            "binance": {
                "requests_per_minute": 1200,
                "orders_per_second": 10,
                "orders_per_day": 200000,
            },
            "ccxt": {
                "requests_per_minute": 600,
                "requests_per_second": 10,
            },
            "cryptofeed": {
                "connections_per_minute": 300,
                "messages_per_second": 1000,
            },
        }
    )

    # Logging configuration
    log_level: str = "INFO"
    log_file_path: str = "logs/"
    enable_structured_logging: bool = True

    # Cache configuration
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    cache_max_size: int = 1000

    @field_validator("log_level")
    def validate_log_level(cls, v):
        levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in levels:
            raise ValueError(f"log_level must be one of {sorted(levels)}")
        return v.upper()

    @field_validator("environment")
    def validate_environment(cls, v):
        allowed_envs = {"development", "staging", "production", "testing"}
        if v.lower() not in allowed_envs:
            raise ValueError(f"environment must be one of {sorted(allowed_envs)}")
        return v.lower()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="AI_PREDICTION_",
    )


class ConfigManager:
    """Enhanced configuration manager using Pydantic settings"""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize ConfigManager with Pydantic settings

        Args:
            config_file: Optional path to configuration file (for backward compatibility)
        """
        self.logger = LoggerFactory.get_logger(
            name="config_manager",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            use_colors=True,
        )

        try:
            self.settings = AIPreedictionSettings()
            self.logger.info("Loaded configuration using Pydantic settings")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def get_exchange_config(self, exchange_name: str) -> ExchangeConfig:
        """Get configuration for specific exchange"""
        exchange_data = self.settings.exchanges.get(exchange_name, {})

        if not exchange_data:
            # Return default config for unknown exchanges
            self.logger.warning(f"No config found for {exchange_name}, using defaults")
            return ExchangeConfig(name=exchange_name)

        # Remove 'name' from exchange_data if it exists to avoid duplicate parameter
        exchange_data_copy = exchange_data.copy()
        if "name" in exchange_data_copy:
            exchange_data_copy.pop("name")

        return ExchangeConfig(name=exchange_name, **exchange_data_copy)

    def get_data_collection_config(self) -> DataCollectionConfig:
        """Get data collection configuration"""
        return DataCollectionConfig(**self.settings.data_collection)

    def get_storage_config(self) -> StorageConfig:
        """Get storage configuration"""
        return StorageConfig(**self.settings.storage)

    def get_rate_limits(self, exchange_name: str) -> Dict[str, int]:
        """Get rate limits for specific exchange"""
        return self.settings.rate_limits.get(exchange_name, {})

    def get_symbols_for_exchange(self, exchange_name: str) -> List[str]:
        """Get default symbols for specific exchange"""
        data_config = self.get_data_collection_config()

        # Exchange-specific symbol mappings
        symbol_mappings = {
            "binance": [s.replace("/", "") for s in data_config.default_symbols],
            "ccxt": data_config.default_symbols,
            "cryptofeed": [s.replace("/", "") for s in data_config.default_symbols],
        }

        return symbol_mappings.get(exchange_name, data_config.default_symbols)

    def get_timeframes_for_exchange(self, exchange_name: str) -> List[str]:
        """Get supported timeframes for specific exchange"""
        data_config = self.get_data_collection_config()

        # Exchange-specific timeframe mappings
        timeframe_mappings = {
            "binance": data_config.default_timeframes,
            "ccxt": data_config.default_timeframes,
            "cryptofeed": ["1m", "5m", "15m", "1h", "1d"],  # Cryptofeed naming
        }

        return timeframe_mappings.get(exchange_name, data_config.default_timeframes)

    def update_setting(self, key: str, value: Any) -> None:
        """Update a configuration setting"""
        try:
            setattr(self.settings, key, value)
            self.logger.info(f"Updated setting: {key} = {value}")
        except Exception as e:
            self.logger.error(f"Failed to update setting {key}: {e}")
            raise

    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration"""
        return {
            "enable_caching": self.settings.enable_caching,
            "cache_ttl_seconds": self.settings.cache_ttl_seconds,
            "cache_max_size": self.settings.cache_max_size,
        }

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            "log_level": self.settings.log_level,
            "log_file_path": self.settings.log_file_path,
            "enable_structured_logging": self.settings.enable_structured_logging,
        }
