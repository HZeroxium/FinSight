# core/config.py

from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from ..schemas.enums import (
    DataLoaderType,
    ExperimentTrackerType,
    StorageProviderType,
    TimeFrame,
    CryptoSymbol,
    ServingAdapterType,
)


class Settings(BaseSettings):
    """Application configuration settings"""

    # Application info
    app_name: str = Field("FinSight Model Builder", env="APP_NAME")
    app_version: str = Field("1.0.0", env="APP_VERSION")
    debug: bool = Field(False, env="DEBUG")

    # API settings
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")

    # Directory paths
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent
    )
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent / "data",
        env="DATA_DIR",
    )
    models_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent / "models",
        env="MODELS_DIR",
    )
    logs_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent / "logs",
        env="LOGS_DIR",
    )

    # Model management settings
    model_name_pattern: str = Field(
        "{symbol}_{timeframe}_{model_type}", env="MODEL_NAME_PATTERN"
    )
    checkpoint_filename: str = Field("model.pt", env="CHECKPOINT_FILENAME")
    metadata_filename: str = Field("metadata.json", env="METADATA_FILENAME")
    config_filename: str = Field("config.json", env="CONFIG_FILENAME")

    # Model training defaults
    default_context_length: int = Field(64, env="DEFAULT_CONTEXT_LENGTH")
    default_prediction_length: int = Field(1, env="DEFAULT_PREDICTION_LENGTH")
    default_num_epochs: int = Field(10, env="DEFAULT_NUM_EPOCHS")
    default_batch_size: int = Field(32, env="DEFAULT_BATCH_SIZE")
    default_learning_rate: float = Field(1e-3, env="DEFAULT_LEARNING_RATE")

    # Device configuration
    force_cpu: bool = Field(
        True,
        env="FORCE_CPU",
        description="Force CPU usage even when GPU is available. When True, all training and inference will use CPU regardless of GPU availability.",
    )

    # Model limits
    max_context_length: int = Field(512, env="MAX_CONTEXT_LENGTH")
    max_prediction_length: int = Field(24, env="MAX_PREDICTION_LENGTH")
    max_num_epochs: int = Field(100, env="MAX_NUM_EPOCHS")

    # Cache settings
    enable_model_cache: bool = Field(True, env="ENABLE_MODEL_CACHE")
    max_cached_models: int = Field(5, env="MAX_CACHED_MODELS")

    # Serving adapter settings
    serving_adapter_type: str = Field("simple", env="SERVING_ADAPTER_TYPE")

    # Simple adapter settings
    simple_max_models_in_memory: int = Field(5, env="SIMPLE_MAX_MODELS_IN_MEMORY")
    simple_model_timeout_seconds: int = Field(3600, env="SIMPLE_MODEL_TIMEOUT_SECONDS")

    # Triton adapter settings
    triton_server_url: str = Field("localhost:8000", env="TRITON_SERVER_URL")
    triton_server_grpc_url: str = Field("localhost:8001", env="TRITON_SERVER_GRPC_URL")
    triton_use_grpc: bool = Field(False, env="TRITON_USE_GRPC")
    triton_ssl: bool = Field(False, env="TRITON_SSL")
    triton_insecure: bool = Field(True, env="TRITON_INSECURE")
    triton_model_repository: str = Field("/models", env="TRITON_MODEL_REPOSITORY")
    triton_default_model_version: str = Field("1", env="TRITON_DEFAULT_MODEL_VERSION")
    triton_max_batch_size: int = Field(8, env="TRITON_MAX_BATCH_SIZE")
    triton_timeout_seconds: int = Field(30, env="TRITON_TIMEOUT_SECONDS")

    # TorchServe adapter settings
    torchserve_inference_url: str = Field(
        "http://localhost:8080", env="TORCHSERVE_INFERENCE_URL"
    )
    torchserve_management_url: str = Field(
        "http://localhost:8081", env="TORCHSERVE_MANAGEMENT_URL"
    )
    torchserve_metrics_url: str = Field(
        "http://localhost:8082", env="TORCHSERVE_METRICS_URL"
    )
    torchserve_model_store: str = Field("./model_store", env="TORCHSERVE_MODEL_STORE")
    torchserve_batch_size: int = Field(1, env="TORCHSERVE_BATCH_SIZE")
    torchserve_max_batch_delay: int = Field(100, env="TORCHSERVE_MAX_BATCH_DELAY")
    torchserve_timeout_seconds: int = Field(30, env="TORCHSERVE_TIMEOUT_SECONDS")
    torchserve_initial_workers: int = Field(1, env="TORCHSERVE_INITIAL_WORKERS")
    torchserve_max_workers: int = Field(4, env="TORCHSERVE_MAX_WORKERS")

    # Redis settings
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    training_job_repository_type: str = Field(
        "file", env="TRAINING_JOB_REPOSITORY_TYPE"
    )

    # ===== Cloud Storage Configuration (following backtesting patterns) =====

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

    # Storage prefix configuration for object storage
    storage_prefix: str = Field(default="datasets", env="STORAGE_PREFIX")
    storage_separator: str = Field(default="/", env="STORAGE_SEPARATOR")

    # Data loading settings
    data_loader_type: str = Field(
        DataLoaderType.HYBRID.value, env="DATA_LOADER_TYPE"
    )  # local, cloud, hybrid
    cloud_data_cache_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
        / "tmp"
        / "cloud_cache",
        env="CLOUD_DATA_CACHE_DIR",
    )
    cloud_data_cache_ttl_hours: int = Field(24, env="CLOUD_DATA_CACHE_TTL_HOURS")

    # Default symbols and timeframes for data loading
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

    # Experiment Tracker settings
    experiment_tracker_type: str = Field(
        ExperimentTrackerType.MLFLOW.value, env="EXPERIMENT_TRACKER_TYPE"
    )  # simple, mlflow
    experiment_tracker_fallback: str = Field(
        ExperimentTrackerType.SIMPLE.value, env="EXPERIMENT_TRACKER_FALLBACK"
    )

    # MLflow settings
    mlflow_tracking_uri: Optional[str] = Field(
        "http://localhost:5000", env="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field("finsight-ml", env="MLFLOW_EXPERIMENT_NAME")

    # Model saving configuration
    save_multiple_formats: bool = Field(True, env="SAVE_MULTIPLE_FORMATS")
    enabled_adapters: List[str] = Field(
        default_factory=lambda: [
            ServingAdapterType.SIMPLE.value,
            ServingAdapterType.TORCHSCRIPT.value,
            ServingAdapterType.TORCHSERVE.value,
            ServingAdapterType.TRITON.value,
        ],
        env="ENABLED_ADAPTERS",
    )

    # Field validators
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

    @field_validator("data_loader_type")
    @classmethod
    def validate_data_loader_type(cls, v):
        from ..schemas.enums import DataLoaderType

        try:
            return DataLoaderType(v.lower()).value
        except ValueError:
            allowed_types = [loader_type.value for loader_type in DataLoaderType]
            raise ValueError(f"data_loader_type must be one of {sorted(allowed_types)}")

    @field_validator("experiment_tracker_type")
    @classmethod
    def validate_experiment_tracker_type(cls, v):
        from ..schemas.enums import ExperimentTrackerType

        try:
            return ExperimentTrackerType(v.lower()).value
        except ValueError:
            allowed_types = [
                tracker_type.value for tracker_type in ExperimentTrackerType
            ]
            raise ValueError(
                f"experiment_tracker_type must be one of {sorted(allowed_types)}"
            )

    @field_validator("experiment_tracker_fallback")
    @classmethod
    def validate_experiment_tracker_fallback(cls, v):
        from ..schemas.enums import ExperimentTrackerType

        try:
            return ExperimentTrackerType(v.lower()).value
        except ValueError:
            allowed_types = [
                tracker_type.value for tracker_type in ExperimentTrackerType
            ]
            raise ValueError(
                f"experiment_tracker_fallback must be one of {sorted(allowed_types)}"
            )

    @field_validator("serving_adapter_type")
    @classmethod
    def validate_serving_adapter_type(cls, v):
        from ..schemas.enums import ServingAdapterType

        try:
            return ServingAdapterType(v.lower()).value
        except ValueError:
            allowed_types = [adapter_type.value for adapter_type in ServingAdapterType]
            raise ValueError(
                f"serving_adapter_type must be one of {sorted(allowed_types)}"
            )

    @field_validator("storage_provider")
    @classmethod
    def validate_storage_provider(cls, v):
        from ..schemas.enums import StorageProviderType

        try:
            return StorageProviderType(v.lower()).value
        except ValueError:
            allowed_providers = [provider.value for provider in StorageProviderType]
            raise ValueError(
                f"storage_provider must be one of {sorted(allowed_providers)}"
            )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_directories()

    def _setup_directories(self):
        """Create directories if they don't exist"""
        # Create directories if they don't exist
        for directory in [
            self.data_dir,
            self.models_dir,
            self.logs_dir,
            self.cloud_data_cache_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

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

    def get_device_config(self) -> str:
        """
        Get the appropriate device configuration for PyTorch models.

        Returns:
            str: Device string ('cpu' or 'cuda')
        """
        if self.force_cpu:
            return "cpu"

        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def is_gpu_enabled(self) -> bool:
        """
        Check if GPU is enabled and available.

        Returns:
            bool: True if GPU is available and not forced to CPU
        """
        if self.force_cpu:
            return False

        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

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

    @property
    def serving(self) -> dict:
        """Get serving adapter configuration"""
        adapter_type = self.serving_adapter_type.lower()

        if adapter_type == "simple":
            adapter_config = {
                "max_models_in_memory": self.simple_max_models_in_memory,
                "model_timeout_seconds": self.simple_model_timeout_seconds,
            }
        elif adapter_type == "triton":
            adapter_config = {
                "server_url": self.triton_server_url,
                "server_grpc_url": self.triton_server_grpc_url,
                "use_grpc": self.triton_use_grpc,
                "ssl": self.triton_ssl,
                "insecure": self.triton_insecure,
                "model_repository": self.triton_model_repository,
                "default_model_version": self.triton_default_model_version,
                "max_batch_size": self.triton_max_batch_size,
                "timeout_seconds": self.triton_timeout_seconds,
            }
        elif adapter_type == "torchserve":
            adapter_config = {
                "inference_url": self.torchserve_inference_url,
                "management_url": self.torchserve_management_url,
                "metrics_url": self.torchserve_metrics_url,
                "model_store": self.torchserve_model_store,
                "batch_size": self.torchserve_batch_size,
                "max_batch_delay": self.torchserve_max_batch_delay,
                "timeout_seconds": self.torchserve_timeout_seconds,
                "initial_workers": self.torchserve_initial_workers,
                "max_workers": self.torchserve_max_workers,
            }
        else:
            # Default to simple adapter
            adapter_config = {
                "max_models_in_memory": self.simple_max_models_in_memory,
                "model_timeout_seconds": self.simple_model_timeout_seconds,
            }
            adapter_type = "simple"

        return {"adapter_type": adapter_type, "adapter_config": adapter_config}


# Global settings instance
_settings: Optional[Settings] = None

settings = Settings()


def get_settings() -> Settings:
    """Get application settings (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def override_settings(new_settings: Settings) -> None:
    """Override settings (useful for testing)"""
    global _settings
    _settings = new_settings
