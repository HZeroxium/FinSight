# core/config.py

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
from pydantic_core.core_schema import FieldValidationInfo
from enum import Enum


class ModelType(str, Enum):
    """Available model types"""

    TRANSFORMER = "transformer"
    LIGHTWEIGHT_TRANSFORMER = "lightweight_transformer"
    HYBRID_TRANSFORMER = "hybrid_transformer"


class OptimizerType(str, Enum):
    """Available optimizer types"""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class SchedulerType(str, Enum):
    """Available scheduler types"""

    STEP_LR = "step_lr"
    COSINE = "cosine"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    WARM_RESTART = "warm_restart"


class PoolingStrategy(str, Enum):
    """Available pooling strategies"""

    ADAPTIVE = "adaptive"
    ATTENTION = "attention"
    LAST = "last"
    MULTI_SCALE = "multi_scale"


class ModelConfig(BaseSettings):
    """Configuration for transformer model architecture and training"""

    # Model architecture
    model_type: ModelType = ModelType.TRANSFORMER
    d_model: int = Field(default=256, ge=64, le=2048)
    n_heads: int = Field(default=8, ge=1, le=32)
    n_layers: int = Field(default=6, ge=1, le=24)
    d_ff: int = Field(default=1024, ge=64, le=8192)
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    activation: str = Field(default="gelu", pattern="^(relu|gelu|swish)$")

    # Advanced architecture options
    use_relative_position: bool = True
    pre_norm: bool = True
    pooling_strategy: PoolingStrategy = PoolingStrategy.ADAPTIVE
    residual_scaling: float = Field(default=1.0, ge=0.1, le=2.0)
    learnable_pos_encoding: bool = False

    # Input/Output dimensions
    input_dim: int = Field(default=5, ge=1, le=100)  # Changed from 7 to 5
    output_dim: int = Field(default=1, ge=1, le=10)
    sequence_length: int = Field(default=60, ge=10, le=1000)
    prediction_horizon: int = Field(default=1, ge=1, le=30)

    # Training parameters
    batch_size: int = Field(default=32, ge=1, le=1024)
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-1)
    weight_decay: float = Field(default=1e-5, ge=0.0, le=1e-2)
    epochs: int = Field(default=100, ge=1, le=1000)
    patience: int = Field(default=10, ge=1, le=50)

    # Optimizer and scheduler
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    scheduler_type: SchedulerType = SchedulerType.COSINE
    warmup_epochs: int = Field(default=5, ge=0, le=20)
    min_lr: float = Field(default=1e-6, ge=1e-8, le=1e-3)

    # Data split ratios
    train_ratio: float = Field(default=0.7, ge=0.1, le=0.9)
    val_ratio: float = Field(default=0.15, ge=0.05, le=0.3)
    test_ratio: float = Field(default=0.15, ge=0.05, le=0.3)

    # Feature configuration
    features_to_use: List[str] = Field(
        default_factory=lambda: ["Open", "High", "Low", "Close", "Volume"]
    )
    target_column: str = "Close"
    use_all_features: bool = Field(
        default=False,
        description="If True, use all meaningful features from feature engineering. If False, use only features_to_use.",
    )
    feature_selection_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Variance threshold for automatic feature selection when use_all_features=True",
    )
    scale_features: bool = True
    scaler_type: str = Field(default="standard", pattern="^(standard|minmax|robust)$")

    # Model paths and device
    model_save_dir: str = "models"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    use_gpu: bool = True
    gpu_id: Optional[int] = None
    mixed_precision: bool = True

    # Regularization
    gradient_clip_value: float = Field(default=1.0, ge=0.0, le=10.0)
    label_smoothing: float = Field(default=0.0, ge=0.0, le=0.2)

    @field_validator("train_ratio", "val_ratio", "test_ratio")
    def validate_ratios(cls, v: float) -> float:
        """Validate that ratios are positive."""
        if v <= 0:
            raise ValueError("Ratios must be positive")
        return v

    @field_validator("test_ratio")
    def validate_ratio_sum(cls, v: float, info: FieldValidationInfo) -> float:
        """Validate that train + val + test ratios sum to 1.0."""
        train_ratio = info.data.get("train_ratio", 0.7)
        val_ratio = info.data.get("val_ratio", 0.15)
        total = train_ratio + val_ratio + v
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Train, validation, and test ratios must sum to 1.0, got {total}"
            )
        return v

    @field_validator("d_ff")
    def validate_d_ff(cls, v: int, info: FieldValidationInfo) -> int:
        """Ensure d_ff is at least equal to d_model."""
        d_model = info.data.get("d_model", 256)
        if v < d_model:
            raise ValueError("d_ff should be at least equal to d_model")
        return v

    @field_validator("n_heads")
    def validate_n_heads(cls, v: int, info: FieldValidationInfo) -> int:
        """Ensure d_model is divisible by n_heads."""
        d_model = info.data.get("d_model", 256)
        if d_model % v != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return v


class DataConfig(BaseSettings):
    """Configuration for data processing and feature engineering"""

    # File paths
    data_file: str = "data/coin_Bitcoin.csv"
    date_column: str = "Date"

    # Feature engineering flags
    add_technical_indicators: bool = True
    add_cyclical_features: bool = True
    add_price_patterns: bool = True

    # Technical indicators configuration
    technical_indicators: Dict[str, Any] = Field(
        default_factory=lambda: {
            "sma": [5, 10, 20, 50],
            "ema": [5, 10, 20, 50],
            "rsi": [14, 21],
            "bollinger": [20],
            "macd": True,
            "stochastic": True,
            "williams_r": True,
            "atr": True,
        }
    )

    # Data cleaning configuration
    remove_outliers: bool = True
    outlier_threshold: float = Field(default=3.0, ge=1.0, le=5.0)
    fill_missing: bool = True
    missing_method: str = Field(
        default="forward", pattern="^(forward|backward|interpolate)$"
    )

    # Data validation
    min_data_points: int = Field(default=1000, ge=100)
    max_missing_ratio: float = Field(default=0.1, ge=0.0, le=0.5)

    @field_validator("data_file")
    def validate_data_file(cls, v):
        """Validate that data file exists"""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Data file does not exist: {v}")
        return v


class Config(BaseSettings):
    """Main configuration class combining all settings"""

    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    # Global settings
    project_name: str = "ai-prediction"
    version: str = "1.0.0"
    environment: str = Field(
        default="development", pattern="^(development|staging|production)$"
    )
    debug: bool = False

    # Logging configuration
    log_level: str = Field(
        default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    enable_tensorboard: bool = True
    save_model_artifacts: bool = True

    # Reproducibility
    random_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    deterministic: bool = True

    # Performance monitoring
    profile_training: bool = False
    track_memory_usage: bool = True

    # Model configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="AI_PREDICTION_",
        extra="ignore",
    )

    @field_validator("environment")
    def validate_environment(cls, v):
        """Validate environment setting"""
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    def get_model_save_path(self, model_name: str) -> Path:
        """Get full path for saving model"""
        return Path(self.model.model_save_dir) / f"{model_name}.pt"

    def get_checkpoint_path(self, checkpoint_name: str) -> Path:
        """Get full path for saving checkpoint"""
        return Path(self.model.checkpoint_dir) / f"{checkpoint_name}.pt"

    def get_log_path(self, log_name: str) -> Path:
        """Get full path for log files"""
        return Path(self.model.log_dir) / log_name

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            "model": self.model.dict(),
            "data": self.data.dict(),
            "project_name": self.project_name,
            "version": self.version,
            "environment": self.environment,
            "debug": self.debug,
            "log_level": self.log_level,
            "random_seed": self.random_seed,
            "deterministic": self.deterministic,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create config from dictionary"""
        model_config = ModelConfig(**config_dict.get("model", {}))
        data_config = DataConfig(**config_dict.get("data", {}))

        return cls(
            model=model_config,
            data=data_config,
            **{k: v for k, v in config_dict.items() if k not in ["model", "data"]},
        )

    @classmethod
    def load_from_file(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from JSON or YAML file"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if config_path.suffix.lower() == ".json":
            import json

            with open(config_path, "r") as f:
                config_dict = json.load(f)
        elif config_path.suffix.lower() in [".yaml", ".yml"]:
            import yaml

            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        return cls.from_dict(config_dict)

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()

        if config_path.suffix.lower() == ".json":
            import json

            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        elif config_path.suffix.lower() in [".yaml", ".yml"]:
            import yaml

            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


# Factory functions for common configurations
def create_development_config() -> Config:
    """Create configuration optimized for development"""
    return Config(
        environment="development",
        debug=True,
        model=ModelConfig(
            epochs=10,
            batch_size=16,
            sequence_length=30,
            n_layers=2,
            d_model=128,
        ),
        data=DataConfig(
            min_data_points=100,
        ),
    )


def create_production_config() -> Config:
    """Create configuration optimized for production"""
    return Config(
        environment="production",
        debug=False,
        deterministic=True,
        model=ModelConfig(
            epochs=100,
            batch_size=64,
            mixed_precision=True,
            gradient_clip_value=1.0,
        ),
        track_memory_usage=True,
    )


def create_lightweight_config() -> Config:
    """Create configuration for lightweight model"""
    return Config(
        model=ModelConfig(
            model_type=ModelType.LIGHTWEIGHT_TRANSFORMER,
            d_model=128,
            n_heads=4,
            n_layers=3,
            batch_size=64,
        )
    )
