# core/config.py

"""Core configuration for the sentiment analysis inference engine."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict, BaseSettings

from common.logger.logger_interface import LogLevel


class TritonConfig(BaseSettings):
    """Configuration for Triton Inference Server."""

    model_config = SettingsConfigDict(env_prefix="TRITON_", protected_namespaces=())

    # Server settings
    host: str = Field(default="localhost", description="Triton server host")
    http_port: int = Field(default=8000, description="HTTP port for Triton")
    grpc_port: int = Field(default=8001, description="gRPC port for Triton")
    metrics_port: int = Field(default=8002, description="Metrics port for Triton")

    # Model repository
    model_repository: Path = Field(
        default=Path(
            "../sentiment_analysis_model_builder/models/triton_model_repository"
        ),
        description="Path to Triton model repository",
    )
    model_name: str = Field(
        default="finbert_sentiment", description="Model name in Triton"
    )

    # Docker settings
    docker_image: str = Field(
        default="nvcr.io/nvidia/tritonserver:25.07-py3",
        description="Triton Docker image",
    )
    container_name: str = Field(
        default="triton-inference-server", description="Docker container name"
    )

    # Startup settings
    startup_timeout: int = Field(default=120, description="Startup timeout in seconds")
    health_check_interval: int = Field(
        default=5, description="Health check interval in seconds"
    )

    # GPU settings
    # Default to CPU-only to ensure it runs on Docker Desktop without GPU
    gpu_enabled: bool = Field(default=False, description="Enable GPU acceleration")
    gpu_memory_fraction: float = Field(
        default=0.8, description="GPU memory fraction to use"
    )

    @field_validator("model_repository")
    @classmethod
    def validate_model_repository(cls, v: Path) -> Path:
        """Validate model repository path."""
        resolved_path = v.resolve()
        # Only warn if path doesn't exist, don't fail during config initialization
        # The actual validation will happen during runtime in triton_manager
        if not resolved_path.exists():
            print(f"Warning: Model repository does not exist: {resolved_path}")
        return resolved_path


class SentimentConfig(BaseSettings):
    """Configuration for sentiment analysis service."""

    model_config = SettingsConfigDict(env_prefix="SENTIMENT_", protected_namespaces=())

    # Model settings
    model_name: str = Field(
        default="finbert_sentiment", description="Model name for inference"
    )
    tokenizer_name: str = Field(
        default="ProsusAI/finbert", description="Tokenizer name"
    )
    max_length: int = Field(default=512, description="Maximum sequence length")

    # Batch processing
    max_batch_size: int = Field(
        default=32, description="Maximum batch size for inference"
    )
    batch_timeout_ms: int = Field(
        default=100, description="Batch timeout in milliseconds"
    )

    # Performance
    cache_size: int = Field(default=1000, description="Result cache size")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

    # Labels
    label_mapping: dict[int, str] = Field(
        default={0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"},
        description="Label ID to name mapping",
    )

    @field_validator("max_length")
    @classmethod
    def validate_max_length(cls, v: int) -> int:
        """Validate maximum sequence length."""
        if v <= 0 or v > 1024:
            raise ValueError("max_length must be between 1 and 1024")
        return v


class APIConfig(BaseSettings):
    """Configuration for FastAPI server."""

    model_config = SettingsConfigDict(env_prefix="API_", protected_namespaces=())

    # Server settings
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8080, description="API server port")
    reload: bool = Field(
        default=False, description="Enable auto-reload for development"
    )

    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    access_log: bool = Field(default=True, description="Enable access logging")

    # CORS
    allow_origins: list[str] = Field(default=["*"], description="Allowed CORS origins")
    allow_methods: list[str] = Field(
        default=["GET", "POST"], description="Allowed HTTP methods"
    )
    allow_headers: list[str] = Field(default=["*"], description="Allowed headers")

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, description="Requests per minute")

    # Timeouts
    request_timeout: int = Field(default=30, description="Request timeout in seconds")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port number."""
        if v <= 0 or v > 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class InferenceConfig(BaseSettings):
    """Main configuration for the inference engine."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=(),
        extra="ignore",  # Ignore extra environment variables
    )

    # Sub-configurations - these will be initialized separately
    triton: TritonConfig = Field(default_factory=TritonConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    # Global settings
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Enable debug mode")

    # Paths
    log_dir: Path = Field(default=Path("logs"), description="Log directory")
    data_dir: Path = Field(default=Path("data"), description="Data directory")

    def __init__(self, **kwargs):
        """Initialize configuration and create directories."""
        super().__init__(**kwargs)

        # Create directories if they don't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
