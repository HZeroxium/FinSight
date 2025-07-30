# core/serving_config.py

"""
Configuration for model serving adapters.

This module provides configuration management for different
serving backends including Triton, TorchServe, and simple serving.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field
import os

from .config import get_settings


class SimpleServingConfig(BaseModel):
    """Configuration for simple in-memory serving"""

    max_models_in_memory: int = Field(5, description="Maximum models to keep in memory")
    model_timeout_seconds: int = Field(3600, description="Model timeout in seconds")
    enable_cache: bool = Field(True, description="Enable model caching")
    cache_size_mb: int = Field(1024, description="Cache size limit in MB")


class TritonServingConfig(BaseModel):
    """Configuration for Triton Inference Server"""

    server_url: str = Field("localhost:8000", description="Triton HTTP server URL")
    server_grpc_url: str = Field("localhost:8001", description="Triton gRPC server URL")
    use_grpc: bool = Field(False, description="Use gRPC instead of HTTP")
    ssl: bool = Field(False, description="Use SSL/TLS")
    insecure: bool = Field(True, description="Allow insecure connections")
    model_repository: str = Field("/models", description="Model repository path")
    default_model_version: str = Field("1", description="Default model version")
    max_batch_size: int = Field(8, description="Maximum batch size")
    timeout_seconds: int = Field(30, description="Request timeout in seconds")


class TorchServeConfig(BaseModel):
    """Configuration for TorchServe"""

    management_url: str = Field(
        "http://localhost:8081", description="Management API URL"
    )
    inference_url: str = Field("http://localhost:8080", description="Inference API URL")
    model_store: str = Field("/tmp/model-store", description="Model store directory")
    default_workers: int = Field(2, description="Default number of workers per model")
    max_workers: int = Field(4, description="Maximum workers per model")
    batch_size: int = Field(1, description="Batch size for inference")
    max_batch_delay: int = Field(100, description="Maximum batch delay in ms")
    response_timeout: int = Field(120, description="Response timeout in seconds")


class TorchScriptServingConfig(BaseModel):
    """Configuration for TorchScript serving"""

    device: str = Field("cpu", description="Device for inference (cpu/cuda)")
    optimize_for_inference: bool = Field(
        True, description="Optimize models for inference"
    )
    enable_fusion: bool = Field(True, description="Enable operator fusion")
    compile_mode: str = Field("trace", description="Compilation mode (trace/script)")
    max_models_in_memory: int = Field(5, description="Maximum models to keep in memory")
    model_timeout_seconds: int = Field(3600, description="Model timeout in seconds")


class ServingConfig(BaseModel):
    """Main serving configuration"""

    adapter_type: str = Field("simple", description="Type of serving adapter to use")
    simple: SimpleServingConfig = Field(default_factory=SimpleServingConfig)
    triton: TritonServingConfig = Field(default_factory=TritonServingConfig)
    torchserve: TorchServeConfig = Field(default_factory=TorchServeConfig)
    torchscript: TorchScriptServingConfig = Field(
        default_factory=TorchScriptServingConfig
    )

    # Global settings
    enable_metrics: bool = Field(True, description="Enable serving metrics")
    enable_health_checks: bool = Field(True, description="Enable health checks")
    log_level: str = Field("INFO", description="Logging level")


def get_serving_config() -> ServingConfig:
    """
    Get serving configuration from environment variables and defaults.

    Returns:
        ServingConfig: Configured serving settings
    """
    settings = get_settings()

    # Get adapter type from environment or default
    adapter_type = os.getenv("SERVING_ADAPTER_TYPE", "torchserve").lower()

    # Simple serving configuration
    simple_config = SimpleServingConfig(
        max_models_in_memory=int(os.getenv("SIMPLE_MAX_MODELS", "5")),
        model_timeout_seconds=int(os.getenv("SIMPLE_MODEL_TIMEOUT", "3600")),
        enable_cache=os.getenv("SIMPLE_ENABLE_CACHE", "true").lower() == "true",
        cache_size_mb=int(os.getenv("SIMPLE_CACHE_SIZE_MB", "1024")),
    )

    # Triton serving configuration
    triton_config = TritonServingConfig(
        server_url=os.getenv("TRITON_SERVER_URL", "localhost:8000"),
        server_grpc_url=os.getenv("TRITON_GRPC_URL", "localhost:8001"),
        use_grpc=os.getenv("TRITON_USE_GRPC", "false").lower() == "true",
        ssl=os.getenv("TRITON_SSL", "false").lower() == "true",
        insecure=os.getenv("TRITON_INSECURE", "true").lower() == "true",
        model_repository=os.getenv("TRITON_MODEL_REPOSITORY", "/models"),
        default_model_version=os.getenv("TRITON_DEFAULT_VERSION", "1"),
        max_batch_size=int(os.getenv("TRITON_MAX_BATCH_SIZE", "8")),
        timeout_seconds=int(os.getenv("TRITON_TIMEOUT", "30")),
    )

    # TorchServe configuration
    torchserve_config = TorchServeConfig(
        management_url=os.getenv("TORCHSERVE_MGMT_URL", "http://localhost:8081"),
        inference_url=os.getenv("TORCHSERVE_INFERENCE_URL", "http://localhost:8080"),
        model_store=os.getenv("TORCHSERVE_MODEL_STORE", "/tmp/model-store"),
        default_workers=int(os.getenv("TORCHSERVE_DEFAULT_WORKERS", "2")),
        max_workers=int(os.getenv("TORCHSERVE_MAX_WORKERS", "4")),
        batch_size=int(os.getenv("TORCHSERVE_BATCH_SIZE", "1")),
        max_batch_delay=int(os.getenv("TORCHSERVE_MAX_BATCH_DELAY", "100")),
        response_timeout=int(os.getenv("TORCHSERVE_RESPONSE_TIMEOUT", "120")),
    )

    return ServingConfig(
        adapter_type=adapter_type,
        simple=simple_config,
        triton=triton_config,
        torchserve=torchserve_config,
        enable_metrics=os.getenv("SERVING_ENABLE_METRICS", "true").lower() == "true",
        enable_health_checks=os.getenv("SERVING_ENABLE_HEALTH_CHECKS", "true").lower()
        == "true",
        log_level=os.getenv("SERVING_LOG_LEVEL", "INFO"),
    )


def get_adapter_config(adapter_type: str) -> Dict[str, Any]:
    """
    Get configuration dictionary for a specific adapter type.

    Args:
        adapter_type: Type of adapter ("simple", "triton", "torchserve")

    Returns:
        Dict[str, Any]: Configuration dictionary for the adapter
    """
    serving_config = get_serving_config()

    if adapter_type == "simple":
        return serving_config.simple.model_dump()
    elif adapter_type == "triton":
        return serving_config.triton.model_dump()
    elif adapter_type == "torchserve":
        return serving_config.torchserve.model_dump()
    elif adapter_type == "torchscript":
        return serving_config.torchscript.model_dump()
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}")
