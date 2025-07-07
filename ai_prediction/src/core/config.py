# core/config.py

import os
from pathlib import Path
from typing import Optional
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
