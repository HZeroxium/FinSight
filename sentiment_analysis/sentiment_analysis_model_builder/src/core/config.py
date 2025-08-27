# core/config.py

"""Configuration management for the sentiment analysis model builder."""

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .enums import (
    DataFormat,
    EvaluationStrategy,
    ExportFormat,
    LogLevel,
    MetricType,
    ModelBackbone,
    ModelStage,
    SaveStrategy,
    SentimentLabel,
)


class PreprocessingConfig(BaseSettings):
    """Configuration for text preprocessing."""

    model_config = SettingsConfigDict(env_prefix="PREPROCESSING_")

    # Text cleaning options
    remove_html: bool = Field(default=True, description="Remove HTML tags")
    normalize_unicode: bool = Field(
        default=True, description="Normalize Unicode characters"
    )
    lowercase: bool = Field(default=True, description="Convert text to lowercase")
    remove_urls: bool = Field(default=True, description="Remove URLs from text")
    remove_emails: bool = Field(default=True, description="Remove email addresses")

    # Text length constraints (FinBERT supports max 512 tokens, ~2048 characters)
    max_length: int = Field(
        default=512, description="Maximum sequence length in tokens for model input"
    )
    max_character_length: int = Field(
        default=2048, description="Maximum character length before tokenization"
    )
    min_length: int = Field(
        default=10, description="Minimum sequence length in characters"
    )

    # Label mapping
    label_mapping: Dict[str, int] = Field(
        default={
            SentimentLabel.NEGATIVE.value: 0,
            SentimentLabel.NEUTRAL.value: 1,
            SentimentLabel.POSITIVE.value: 2,
        },
        description="Mapping from label strings to integers",
    )

    @field_validator("max_length")
    @classmethod
    def validate_max_length(cls, v: int) -> int:
        """Validate maximum sequence length."""
        if v < 64:
            raise ValueError("max_length must be at least 64")
        if v > 512:
            raise ValueError("max_length must not exceed 512 for FinBERT")
        return v

    @field_validator("max_character_length")
    @classmethod
    def validate_max_character_length(cls, v: int) -> int:
        """Validate maximum character length."""
        if v < 256:
            raise ValueError("max_character_length must be at least 256")
        if v > 8192:
            raise ValueError("max_character_length must not exceed 8192")
        return v

    @field_validator("min_length")
    @classmethod
    def validate_min_length(cls, v: int, info: Any) -> int:
        """Validate minimum sequence length."""
        max_char_length = info.data.get("max_character_length", 2048)
        if v >= max_char_length:
            raise ValueError("min_length must be less than max_character_length")
        if v < 1:
            raise ValueError("min_length must be at least 1")
        return v


class TrainingConfig(BaseSettings):
    """Configuration for model training."""

    model_config = SettingsConfigDict(env_prefix="TRAINING_")

    # Model configuration
    backbone: ModelBackbone = Field(
        default=ModelBackbone.FINBERT,
        description="Model backbone to use for training",
    )

    # Training hyperparameters
    batch_size: int = Field(default=16, description="Training batch size")
    eval_batch_size: int = Field(default=32, description="Evaluation batch size")
    learning_rate: float = Field(default=2e-5, description="Learning rate")
    num_epochs: int = Field(default=3, description="Number of training epochs")
    warmup_steps: int = Field(default=500, description="Number of warmup steps")
    weight_decay: float = Field(default=0.01, description="Weight decay")
    gradient_clip_val: float = Field(default=1.0, description="Gradient clipping value")

    # Data splitting
    train_split: float = Field(default=0.7, description="Training set proportion")
    val_split: float = Field(default=0.15, description="Validation set proportion")
    test_split: float = Field(default=0.15, description="Test set proportion")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")

    # Early stopping
    early_stopping_patience: int = Field(
        default=3, description="Early stopping patience"
    )
    early_stopping_threshold: float = Field(
        default=0.001, description="Early stopping threshold"
    )

    # Metrics
    primary_metric: MetricType = Field(
        default=MetricType.F1_MACRO, description="Primary metric for evaluation"
    )

    # Evaluation and logging
    evaluation_strategy: EvaluationStrategy = Field(
        default=EvaluationStrategy.STEPS, description="Evaluation strategy"
    )
    eval_steps: int = Field(default=20, description="Evaluation steps")
    logging_steps: int = Field(default=20, description="Logging steps")
    save_steps: int = Field(default=500, description="Save steps")
    save_total_limit: int = Field(default=3, description="Save total limit")
    save_strategy: SaveStrategy = Field(
        default=SaveStrategy.STEPS, description="Save strategy"
    )

    @field_validator("batch_size", "eval_batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size."""
        if v < 1:
            raise ValueError("Batch size must be at least 1")
        if v > 128:
            raise ValueError("Batch size must not exceed 128")
        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Validate learning rate."""
        if v <= 0:
            raise ValueError("Learning rate must be positive")
        if v > 1e-2:
            raise ValueError("Learning rate must not exceed 0.01")
        return v

    @field_validator("train_split", "val_split", "test_split")
    @classmethod
    def validate_splits(cls, v: float) -> float:
        """Validate data split proportions."""
        if v <= 0 or v >= 1:
            raise ValueError("Split proportions must be between 0 and 1")
        return v

    @field_validator("train_split", "val_split", "test_split")
    @classmethod
    def validate_split_sum(cls, v: float, info: Any) -> float:
        """Validate that splits sum to approximately 1.0."""
        splits = [
            info.data.get("train_split", 0.7),
            info.data.get("val_split", 0.15),
            info.data.get("test_split", 0.15),
        ]
        total = sum(splits)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Data splits must sum to 1.0, got {total}")
        return v


class ExportConfig(BaseSettings):
    """Configuration for model export."""

    model_config = SettingsConfigDict(env_prefix="EXPORT_")

    # Export formats
    format: ExportFormat = Field(
        default=ExportFormat.ONNX, description="Export format for the trained model"
    )

    # ONNX specific settings
    onnx_opset_version: int = Field(default=17, description="ONNX opset version")
    onnx_dynamic_axes: bool = Field(
        default=True, description="Use dynamic axes for ONNX"
    )

    # Output paths
    output_dir: Path = Field(
        default=Path("models/exported"), description="Directory to save exported models"
    )

    # Validation
    validate_export: bool = Field(default=True, description="Validate exported model")
    test_batch_size: int = Field(
        default=1, description="Batch size for export validation"
    )

    @field_validator("onnx_opset_version")
    @classmethod
    def validate_onnx_opset(cls, v: int) -> int:
        """Validate ONNX opset version."""
        if v < 11 or v > 18:
            raise ValueError("ONNX opset version must be between 11 and 18")
        return v


class RegistryConfig(BaseSettings):
    """Configuration for model registry (MLflow)."""

    model_config = SettingsConfigDict(env_prefix="REGISTRY_")

    # MLflow settings
    tracking_uri: str = Field(
        default="http://localhost:5000", description="MLflow tracking server URI"
    )
    registry_uri: Optional[str] = Field(
        default=None, description="MLflow model registry URI"
    )
    backend_store_uri: str = Field(
        default="sqlite:///mlflow.db",
        description="MLflow backend store URI (SQLite for local)",
    )

    # Model registry
    model_name: str = Field(
        default="crypto-news-sentiment", description="Name of the model in the registry"
    )
    model_stage: ModelStage = Field(
        default=ModelStage.STAGING, description="Initial stage for the registered model"
    )

    # Artifact storage (MinIO/S3)
    artifact_location: str = Field(
        default="s3://mlflow-artifacts/",
        description="Artifact storage location (S3/MinIO)",
    )

    # MinIO configuration
    aws_access_key_id: str = Field(
        default="minioadmin", description="MinIO access key ID"
    )
    aws_secret_access_key: str = Field(
        default="minioadmin", description="MinIO secret access key"
    )
    aws_region: str = Field(
        default="us-east-1", description="AWS region (required for S3 client)"
    )
    s3_endpoint_url: str = Field(
        default="http://localhost:9000",
        description="MinIO endpoint URL",
    )

    # MLflow server settings
    mlflow_host: str = Field(default="0.0.0.0", description="MLflow server host")
    mlflow_port: int = Field(default=5000, description="MLflow server port")

    @field_validator("tracking_uri")
    @classmethod
    def validate_tracking_uri(cls, v: str) -> str:
        """Validate MLflow tracking URI."""
        if not v:
            raise ValueError("Tracking URI cannot be empty")
        return v

    @field_validator("mlflow_port")
    @classmethod
    def validate_mlflow_port(cls, v: int) -> int:
        """Validate MLflow server port."""
        if v < 1 or v > 65535:
            raise ValueError("MLflow port must be between 1 and 65535")
        return v


class DataConfig(BaseSettings):
    """Configuration for data loading and processing."""

    model_config = SettingsConfigDict(env_prefix="DATA_")

    # Input data - Make input_path optional with default
    input_path: Optional[Path] = Field(
        default=Path("data/transformed_dataset_sample.parquet"),
        description="Path to input data file",
    )
    input_format: DataFormat = Field(
        default=DataFormat.PARQUET, description="Format of input data"
    )

    # Required columns
    text_column: str = Field(default="text", description="Column containing text data")
    label_column: str = Field(default="label", description="Column containing labels")

    # Optional columns
    id_column: Optional[str] = Field(
        default=None, description="Column containing document IDs"
    )
    title_column: Optional[str] = Field(
        default=None, description="Column containing titles"
    )
    published_at_column: Optional[str] = Field(
        default=None, description="Column containing publication dates"
    )
    tickers_column: Optional[str] = Field(
        default=None, description="Column containing ticker symbols"
    )
    split_column: Optional[str] = Field(
        default=None, description="Column containing split information"
    )

    # Data validation
    validate_data: bool = Field(default=True, description="Validate input data")
    max_samples: Optional[int] = Field(
        default=None, description="Maximum number of samples to load"
    )

    @field_validator("input_path")
    @classmethod
    def validate_input_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate input file path."""
        if v is None:
            return v
        if not v.exists():
            # Don't raise error, just log warning
            from loguru import logger

            logger.warning(f"Input file does not exist: {v}")
        return v


class APIConfig(BaseSettings):
    """Configuration for API server."""

    model_config = SettingsConfigDict(env_prefix="API_")

    # Server settings
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, description="API server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    reload: bool = Field(default=False, description="Enable auto-reload in development")

    # API metadata
    title: str = Field(
        default="FinBERT Sentiment Analysis API", description="API title"
    )
    description: str = Field(
        default="RESTful API for financial sentiment analysis using fine-tuned FinBERT",
        description="API description",
    )
    version: str = Field(default="1.0.0", description="API version")

    # Model settings
    model_path: Path = Field(
        default=Path("outputs/model"), description="Path to trained model"
    )
    preprocessing_config_path: Path = Field(
        default=Path("outputs/preprocessing_config.json"),
        description="Path to preprocessing configuration",
    )
    label_mapping_path: Path = Field(
        default=Path("outputs/id2label.json"), description="Path to label mapping file"
    )

    # Inference settings
    max_batch_size: int = Field(
        default=32, description="Maximum batch size for inference"
    )
    max_text_length: int = Field(
        default=512, description="Maximum text length for tokenization"
    )
    device: str = Field(
        default="auto", description="Device for inference (auto, cpu, cuda)"
    )

    # CORS settings
    allow_origins: list[str] = Field(default=["*"], description="Allowed CORS origins")
    allow_credentials: bool = Field(
        default=True, description="Allow credentials in CORS"
    )
    allow_methods: list[str] = Field(default=["*"], description="Allowed HTTP methods")
    allow_headers: list[str] = Field(default=["*"], description="Allowed HTTP headers")

    # Performance settings
    workers: int = Field(default=1, description="Number of worker processes")
    timeout: int = Field(default=30, description="Request timeout in seconds")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port number."""
        if v < 1 or v > 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    @field_validator("max_batch_size")
    @classmethod
    def validate_max_batch_size(cls, v: int) -> int:
        """Validate maximum batch size."""
        if v < 1 or v > 128:
            raise ValueError("max_batch_size must be between 1 and 128")
        return v

    @field_validator("max_text_length")
    @classmethod
    def validate_max_text_length(cls, v: int) -> int:
        """Validate maximum text length."""
        if v < 64 or v > 512:
            raise ValueError("max_text_length must be between 64 and 512")
        return v


class Config(BaseSettings):
    """Main configuration class combining all sub-configurations."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Sub-configurations
    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    # Global settings
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    output_dir: Path = Field(default=Path("outputs"), description="Output directory")
    cache_dir: Path = Field(default=Path(".cache"), description="Cache directory")

    @field_validator("output_dir", "cache_dir")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Create directories if they don't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    def get_label_mapping(self) -> Dict[str, int]:
        """Get the label mapping for the current configuration."""
        return self.preprocessing.label_mapping

    def get_reverse_label_mapping(self) -> Dict[int, str]:
        """Get the reverse label mapping (int -> str)."""
        return {v: k for k, v in self.preprocessing.label_mapping.items()}

    def get_num_labels(self) -> int:
        """Get the number of unique labels."""
        return len(self.preprocessing.label_mapping)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "data": self.data.model_dump(),
            "preprocessing": self.preprocessing.model_dump(),
            "training": self.training.model_dump(),
            "export": self.export.model_dump(),
            "registry": self.registry.model_dump(),
            "log_level": self.log_level.value,
            "output_dir": str(self.output_dir),
            "cache_dir": str(self.cache_dir),
        }
