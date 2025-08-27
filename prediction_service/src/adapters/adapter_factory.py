# adapters/adapter_factory.py

"""
Factory for creating model serving adapters.

This factory provides a unified interface for creating different
types of model serving adapters based on configuration.
"""

from enum import Enum
from typing import Any, Dict, Type

from common.logger.logger_factory import LoggerFactory

from ..interfaces.serving_interface import IModelServingAdapter
from .simple_serving import SimpleServingAdapter
from .torchscript_serving import TorchScriptServingAdapter
from .torchserve_serving import TorchServeAdapter
from .triton_serving import TritonServingAdapter


class ServingAdapterType(Enum):
    """Supported serving adapter types"""

    SIMPLE = "simple"
    TRITON = "triton"
    TORCHSERVE = "torchserve"
    TORCHSCRIPT = "torchscript"


class ServingAdapterFactory:
    """
    Factory for creating model serving adapters.

    This factory creates the appropriate serving adapter based on
    configuration and provides a unified interface for model serving.
    """

    _adapter_classes: Dict[ServingAdapterType, Type[IModelServingAdapter]] = {
        ServingAdapterType.SIMPLE: SimpleServingAdapter,
        ServingAdapterType.TRITON: TritonServingAdapter,
        ServingAdapterType.TORCHSERVE: TorchServeAdapter,
        ServingAdapterType.TORCHSCRIPT: TorchScriptServingAdapter,
    }

    def __init__(self):
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)

    @classmethod
    def create_adapter(
        cls, adapter_type: str, config: Dict[str, Any]
    ) -> IModelServingAdapter:
        """
        Create a serving adapter instance.

        Args:
            adapter_type: Type of adapter to create ("simple", "triton", "torchserve")
            config: Configuration dictionary for the adapter

        Returns:
            IModelServingAdapter: Created adapter instance

        Raises:
            ValueError: If adapter type is not supported
            ImportError: If required dependencies are not available
        """
        try:
            # Normalize adapter type
            adapter_type_enum = ServingAdapterType(adapter_type.lower())
        except ValueError:
            supported_types = [t.value for t in ServingAdapterType]
            raise ValueError(
                f"Unsupported adapter type: {adapter_type}. "
                f"Supported types: {supported_types}"
            )

        # Get adapter class
        adapter_class = cls._adapter_classes[adapter_type_enum]

        # Create and return adapter instance
        try:
            adapter = adapter_class(config)
            return adapter
        except ImportError as e:
            raise ImportError(
                f"Failed to create {adapter_type} adapter. "
                f"Missing dependencies: {e}"
            )
        except Exception as e:
            raise Exception(f"Failed to create {adapter_type} adapter: {e}")

    @classmethod
    def get_supported_adapters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about supported adapters.

        Returns:
            Dict[str, Dict[str, Any]]: Information about each supported adapter
        """
        return {
            "simple": {
                "name": "Simple In-Memory Serving",
                "description": "Direct in-memory model serving for development",
                "dependencies": ["numpy", "pandas"],
                "features": [
                    "In-memory model storage",
                    "Direct inference",
                    "Memory management",
                    "Model eviction",
                ],
                "use_cases": [
                    "Development and testing",
                    "Small-scale deployments",
                    "Quick prototyping",
                ],
            },
            "triton": {
                "name": "NVIDIA Triton Inference Server",
                "description": "High-performance serving with GPU acceleration",
                "dependencies": ["tritonclient"],
                "features": [
                    "GPU acceleration",
                    "Dynamic batching",
                    "Model versioning",
                    "Multi-framework support",
                    "Advanced monitoring",
                ],
                "use_cases": [
                    "High-throughput production",
                    "GPU-accelerated inference",
                    "Enterprise deployments",
                ],
            },
            "torchserve": {
                "name": "PyTorch TorchServe",
                "description": "PyTorch native model serving with scaling",
                "dependencies": ["torch", "torchserve"],
                "features": [
                    "Auto-scaling",
                    "Model management",
                    "Metrics and monitoring",
                    "RESTful API",
                    "Custom handlers",
                ],
                "use_cases": [
                    "PyTorch model deployment",
                    "Production serving",
                    "Scalable inference",
                ],
            },
        }

    @classmethod
    def validate_adapter_config(
        cls, adapter_type: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and normalize adapter configuration.

        Args:
            adapter_type: Type of adapter
            config: Configuration to validate

        Returns:
            Dict[str, Any]: Validated and normalized configuration

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            adapter_type_enum = ServingAdapterType(adapter_type.lower())
        except ValueError:
            supported_types = [t.value for t in ServingAdapterType]
            raise ValueError(
                f"Unsupported adapter type: {adapter_type}. "
                f"Supported types: {supported_types}"
            )

        # Default configurations for each adapter type
        default_configs = {
            ServingAdapterType.SIMPLE: {
                "max_models_in_memory": 5,
                "model_timeout_seconds": 3600,
            },
            ServingAdapterType.TRITON: {
                "server_url": "localhost:8000",
                "server_grpc_url": "localhost:8001",
                "use_grpc": False,
                "ssl": False,
                "insecure": True,
                "model_repository": "/models",
                "default_model_version": "1",
                "max_batch_size": 8,
                "timeout_seconds": 30,
            },
            ServingAdapterType.TORCHSERVE: {
                "inference_url": "http://localhost:8080",
                "management_url": "http://localhost:8081",
                "metrics_url": "http://localhost:8082",
                "model_store": "./model_store",
                "batch_size": 1,
                "max_batch_delay": 100,
                "timeout_seconds": 30,
                "initial_workers": 1,
                "max_workers": 4,
            },
        }

        # Merge with defaults
        validated_config = default_configs[adapter_type_enum].copy()
        validated_config.update(config)

        # Adapter-specific validation
        if adapter_type_enum == ServingAdapterType.SIMPLE:
            cls._validate_simple_config(validated_config)
        elif adapter_type_enum == ServingAdapterType.TRITON:
            cls._validate_triton_config(validated_config)
        elif adapter_type_enum == ServingAdapterType.TORCHSERVE:
            cls._validate_torchserve_config(validated_config)

        return validated_config

    @classmethod
    def _validate_simple_config(cls, config: Dict[str, Any]) -> None:
        """Validate simple adapter configuration"""
        required_fields = ["max_models_in_memory", "model_timeout_seconds"]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field for simple adapter: {field}")

        if config["max_models_in_memory"] <= 0:
            raise ValueError("max_models_in_memory must be positive")

        if config["model_timeout_seconds"] <= 0:
            raise ValueError("model_timeout_seconds must be positive")

    @classmethod
    def _validate_triton_config(cls, config: Dict[str, Any]) -> None:
        """Validate Triton adapter configuration"""
        required_fields = ["server_url", "model_repository"]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field for Triton adapter: {field}")

        if config["max_batch_size"] <= 0:
            raise ValueError("max_batch_size must be positive")

        if config["timeout_seconds"] <= 0:
            raise ValueError("timeout_seconds must be positive")

    @classmethod
    def _validate_torchserve_config(cls, config: Dict[str, Any]) -> None:
        """Validate TorchServe adapter configuration"""
        required_fields = ["inference_url", "management_url", "model_store"]

        for field in required_fields:
            if field not in config:
                raise ValueError(
                    f"Missing required field for TorchServe adapter: {field}"
                )

        if config["batch_size"] <= 0:
            raise ValueError("batch_size must be positive")

        if config["initial_workers"] < 0:
            raise ValueError("initial_workers must be non-negative")

        if config["max_workers"] <= 0:
            raise ValueError("max_workers must be positive")

        if config["initial_workers"] > config["max_workers"]:
            raise ValueError("initial_workers cannot exceed max_workers")

    @classmethod
    def create_adapter_from_settings(cls, settings) -> IModelServingAdapter:
        """
        Create adapter from application settings.

        Args:
            settings: Application settings object

        Returns:
            IModelServingAdapter: Created adapter instance
        """
        # Get serving configuration from settings
        serving_config = getattr(settings, "serving", {})

        # Default to simple adapter if not specified
        adapter_type = serving_config.get("adapter_type", "simple")
        adapter_config = serving_config.get("adapter_config", {})

        # Validate configuration
        validated_config = cls.validate_adapter_config(adapter_type, adapter_config)

        # Create adapter
        return cls.create_adapter(adapter_type, validated_config)

    @classmethod
    def get_adapter_requirements(cls, adapter_type: str) -> Dict[str, Any]:
        """
        Get requirements and dependencies for an adapter type.

        Args:
            adapter_type: Type of adapter

        Returns:
            Dict[str, Any]: Requirements information
        """
        supported_adapters = cls.get_supported_adapters()

        if adapter_type.lower() not in supported_adapters:
            supported_types = list(supported_adapters.keys())
            raise ValueError(
                f"Unsupported adapter type: {adapter_type}. "
                f"Supported types: {supported_types}"
            )

        adapter_info = supported_adapters[adapter_type.lower()]

        # Check if dependencies are available
        dependencies_available = {}
        for dep in adapter_info["dependencies"]:
            try:
                __import__(dep)
                dependencies_available[dep] = True
            except ImportError:
                dependencies_available[dep] = False

        return {
            "adapter_type": adapter_type,
            "dependencies": adapter_info["dependencies"],
            "dependencies_available": dependencies_available,
            "all_dependencies_available": all(dependencies_available.values()),
            "features": adapter_info["features"],
            "use_cases": adapter_info["use_cases"],
        }
