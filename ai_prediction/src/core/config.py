# core/config.py

from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


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
    data_dir: Path = Field(None, env="DATA_DIR")
    models_dir: Path = Field(None, env="MODELS_DIR")
    logs_dir: Path = Field(None, env="LOGS_DIR")

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
    
    # Model saving configuration
    save_multiple_formats: bool = Field(True, env="SAVE_MULTIPLE_FORMATS")
    enabled_adapters: List[str] = Field(
        default_factory=lambda: ["simple", "torchscript", "torchserve", "triton"],
        env="ENABLED_ADAPTERS"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_directories()

    def _setup_directories(self):
        """Set up default directories if not specified"""
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"

        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"

        if self.logs_dir is None:
            self.logs_dir = self.base_dir / "logs"

        # Create directories if they don't exist
        for directory in [self.data_dir, self.models_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

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
