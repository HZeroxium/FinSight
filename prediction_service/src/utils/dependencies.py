# utils/dependencies.py

"""
Dependency Injection Container for AI Prediction Module

This module sets up dependency injection using the dependency_injector library
to provide clean separation of concerns and enable easy testing and configuration.
"""

from dependency_injector import containers, providers

from ..interfaces.experiment_tracker_interface import IExperimentTracker
from ..interfaces.data_loader_interface import IDataLoader
from ..adapters.simple_experiment_tracker import SimpleExperimentTracker
from ..data.cloud_data_loader import CloudDataLoader
from ..data.file_data_loader import FileDataLoader
from ..utils.storage_client import StorageClient
from ..utils.model_utils import ModelUtils
from ..services.eureka_client_service import EurekaClientService
from ..core.config import get_settings, Settings
from ..schemas.enums import DataLoaderType, ExperimentTrackerType
from .device_manager import create_device_manager_from_settings, DeviceManager
from common.logger.logger_factory import LoggerFactory

# Initialize logger
logger = LoggerFactory.get_logger("Dependencies")


def _create_data_loader(
    loader_type: str,
    cloud_loader: CloudDataLoader,
    file_loader: FileDataLoader,
) -> IDataLoader:
    """Factory function to create data loader based on type"""

    logger.info(f"Creating data loader with type: {loader_type}")

    loader_type_enum = DataLoaderType(loader_type.lower())

    if loader_type_enum == DataLoaderType.CLOUD:
        logger.info("Selected CloudDataLoader")
        return cloud_loader
    elif loader_type_enum == DataLoaderType.LOCAL:
        logger.info("Selected FileDataLoader")
        return file_loader
    elif loader_type_enum == DataLoaderType.HYBRID:
        # Hybrid mode uses CloudDataLoader which has built-in fallback
        logger.info("Selected CloudDataLoader (hybrid mode)")
        return cloud_loader
    else:
        # Default to hybrid mode
        logger.warning(f"Unknown loader type '{loader_type}', defaulting to hybrid")
        return cloud_loader


def _create_experiment_tracker(
    tracker_type: str,
    fallback_type: str,
    simple_tracker: SimpleExperimentTracker,
    settings: Settings,
) -> IExperimentTracker:
    """Factory function to create experiment tracker with fallback logic"""

    logger.info(f"Creating experiment tracker with type: {tracker_type}")

    # Try primary tracker
    try:
        tracker_type_enum = ExperimentTrackerType(tracker_type.lower())

        if tracker_type_enum == ExperimentTrackerType.MLFLOW:
            from ..adapters.mlflow_experiment_tracker import MLflowExperimentTracker

            tracker = MLflowExperimentTracker(
                tracking_uri=settings.mlflow_tracking_uri,
                experiment_name=settings.mlflow_experiment_name,
            )
            logger.info("Initialized MLflow experiment tracker")
            return tracker

        elif tracker_type_enum == ExperimentTrackerType.SIMPLE:
            logger.info("Initialized Simple experiment tracker")
            return simple_tracker

    except Exception as e:
        logger.warning(f"Failed to initialize primary tracker ({tracker_type}): {e}")

        # Try fallback tracker
        try:
            fallback_enum = ExperimentTrackerType(fallback_type.lower())
            if fallback_enum == ExperimentTrackerType.SIMPLE:
                logger.info("Initialized fallback Simple experiment tracker")
                return simple_tracker
        except Exception as fallback_error:
            logger.error(f"Failed to initialize fallback tracker: {fallback_error}")

    # Default to simple tracker as last resort
    logger.info("Using default Simple experiment tracker as last resort")
    return simple_tracker


def _create_storage_client() -> StorageClient:
    """Factory function to create storage client with configuration"""
    settings = get_settings()
    storage_config = settings.get_storage_config()

    # Filter out unsupported parameters
    supported_params = {
        "endpoint_url",
        "access_key",
        "secret_key",
        "region_name",
        "bucket_name",
        "use_ssl",
        "verify_ssl",
        "signature_version",
        "max_pool_connections",
    }

    filtered_config = {k: v for k, v in storage_config.items() if k in supported_params}
    return StorageClient(**filtered_config)


class Container(containers.DeclarativeContainer):
    """Main dependency injection container for AI Prediction module"""

    # Configuration - using the centralized settings
    config = providers.Object(get_settings())

    # Core utilities
    logger_factory = providers.Singleton(LoggerFactory)

    # Device manager for consistent CPU/GPU handling
    device_manager = providers.Singleton(lambda: create_device_manager_from_settings())

    # Storage client for cloud operations - now initialized with centralized config
    storage_client = providers.Singleton(_create_storage_client)

    # Model utilities, now injected with storage_client
    model_utils = providers.Singleton(ModelUtils, storage_client=storage_client)

    # Eureka Client Service
    eureka_client_service = providers.Singleton(
        EurekaClientService,
    )

    # Data loaders, now injected with storage_client
    cloud_data_loader = providers.Singleton(
        CloudDataLoader,
        data_dir=config.provided.data_dir,
        storage_client=storage_client,
    )

    file_data_loader = providers.Singleton(
        FileDataLoader, data_dir=config.provided.data_dir
    )

    # Experiment trackers
    simple_experiment_tracker = providers.Singleton(SimpleExperimentTracker)

    # Factories
    data_loader = providers.Factory(
        _create_data_loader,
        loader_type=config.provided.data_loader_type,
        cloud_loader=cloud_data_loader,
        file_loader=file_data_loader,
    )

    experiment_tracker = providers.Factory(
        _create_experiment_tracker,
        tracker_type=config.provided.experiment_tracker_type,
        fallback_type=config.provided.experiment_tracker_fallback,
        simple_tracker=simple_experiment_tracker,
        settings=config,
    )


class DependencyManager:
    """
    Dependency Manager for easy access to container services.

    Provides a high-level interface for accessing dependencies
    and managing container configuration.
    """

    def __init__(self):
        self.container = Container()
        # Initialize wire
        self.container.wire(modules=[__name__])

    def get_data_loader(self) -> IDataLoader:
        """Get configured data loader"""
        return self.container.data_loader()

    def get_experiment_tracker(self) -> IExperimentTracker:
        """Get configured experiment tracker"""
        return self.container.experiment_tracker()

    def get_storage_client(self) -> StorageClient:
        """Get configured storage client"""
        return self.container.storage_client()

    def get_cloud_data_loader(self) -> CloudDataLoader:
        """Get cloud data loader"""
        return self.container.cloud_data_loader()

    def get_file_data_loader(self) -> FileDataLoader:
        """Get file data loader"""
        return self.container.file_data_loader()

    def get_dataset_management_service(self):
        """Get dataset management service - lazy import to avoid circular dependency"""
        from ..services.dataset_management_service import DatasetManagementService

        return DatasetManagementService(storage_client=self.get_storage_client())

    def get_device_manager(self) -> DeviceManager:
        """Get device manager"""
        return self.container.device_manager()

    def get_model_utils(self) -> ModelUtils:
        """Get model utilities"""
        return self.container.model_utils()

    def get_eureka_client_service(self) -> EurekaClientService:
        """Get Eureka client service"""
        return self.container.eureka_client_service()

    def reset_configuration(self) -> None:
        """Reset container configuration to defaults"""
        self.container.reset_last_provided()

    def shutdown(self) -> None:
        """Shutdown container and clean up resources"""
        self.container.shutdown_resources()


# Global dependency manager instance
dependency_manager = DependencyManager()


def get_dependency_manager() -> DependencyManager:
    """Get the global dependency manager instance"""
    return dependency_manager


# Convenience functions for backward compatibility
def get_data_loader() -> IDataLoader:
    """Get configured data loader (convenience function)"""
    return dependency_manager.get_data_loader()


def get_experiment_tracker() -> IExperimentTracker:
    """Get configured experiment tracker (convenience function)"""
    return dependency_manager.get_experiment_tracker()


def get_storage_client() -> StorageClient:
    """Get configured storage client (convenience function)"""
    return dependency_manager.get_storage_client()


def get_dataset_management_service():
    """Get dataset management service (convenience function)"""
    return dependency_manager.get_dataset_management_service()


def get_device_manager() -> DeviceManager:
    """Get configured device manager (convenience function)"""
    return dependency_manager.get_device_manager()


def get_model_utils() -> ModelUtils:
    """Get model utilities (convenience function)"""
    return dependency_manager.get_model_utils()


def get_eureka_client_service() -> EurekaClientService:
    """Get Eureka client service instance"""
    try:
        return dependency_manager.container.eureka_client_service()
    except Exception as e:
        logger.error(f"Failed to get Eureka client service: {e}")
        raise


# FastAPI dependency functions for easy injection
async def get_experiment_tracker_dependency() -> IExperimentTracker:
    """FastAPI dependency function for experiment tracker."""
    return get_experiment_tracker()


async def get_data_loader_dependency() -> IDataLoader:
    """FastAPI dependency function for data loader."""
    return get_data_loader()


async def get_storage_client_dependency() -> StorageClient:
    """FastAPI dependency function for storage client."""
    return get_storage_client()


async def get_dataset_management_service_dependency():
    """FastAPI dependency function for dataset management service."""
    return get_dataset_management_service()


async def get_device_manager_dependency() -> DeviceManager:
    """FastAPI dependency function for device manager."""
    return get_device_manager()


async def get_model_utils_dependency() -> ModelUtils:
    """FastAPI dependency function for model utilities."""
    return get_model_utils()


async def get_eureka_client_service_dependency() -> EurekaClientService:
    """FastAPI dependency: Eureka client service"""
    return get_eureka_client_service()


# Health check and info functions
async def health_check_dependencies() -> dict:
    """
    Perform health check on all dependencies.

    Returns:
        dict: Health check results for all dependencies
    """
    results = {
        "experiment_tracker": False,
        "data_loader": False,
        "storage_client": False,
        "dataset_management_service": False,
        "device_manager": False,
        "model_utils": False,
        "timestamp": None,
    }

    try:
        # Check experiment tracker
        tracker = get_experiment_tracker()
        if hasattr(tracker, "health_check"):
            results["experiment_tracker"] = await tracker.health_check()
        else:
            results["experiment_tracker"] = tracker is not None

        # Check data loader (simple existence check)
        loader = get_data_loader()
        results["data_loader"] = loader is not None

        # Check storage client
        storage_client = get_storage_client()
        if hasattr(storage_client, "get_storage_info"):
            try:
                storage_info = await storage_client.get_storage_info()
                results["storage_client"] = storage_info is not None
            except Exception:
                results["storage_client"] = False
        else:
            results["storage_client"] = storage_client is not None

        # Check dataset management service
        try:
            dataset_service = get_dataset_management_service()
            results["dataset_management_service"] = dataset_service is not None
        except Exception:
            results["dataset_management_service"] = False

        # Check device manager
        device_manager = get_device_manager()
        results["device_manager"] = device_manager is not None

        # Check model utils
        model_utils = get_model_utils()
        results["model_utils"] = model_utils is not None

        results["timestamp"] = "Health check completed"
        logger.info("Dependency health check completed")

    except Exception as e:
        logger.error(f"Dependency health check failed: {e}")
        results["error"] = str(e)

    return results


def get_dependency_info() -> dict:
    """
    Get information about current dependency configuration.

    Returns:
        dict: Current dependency configuration and status
    """
    settings = get_settings()

    data_loader = dependency_manager.get_data_loader()
    experiment_tracker = dependency_manager.get_experiment_tracker()
    device_manager = dependency_manager.get_device_manager()
    model_utils = dependency_manager.get_model_utils()

    info = {
        "experiment_tracker": {
            "type": settings.experiment_tracker_type,
            "fallback": settings.experiment_tracker_fallback,
            "instance": type(experiment_tracker).__name__,
        },
        "data_loader": {
            "type": settings.data_loader_type,
            "instance": type(data_loader).__name__,
        },
        "device_manager": {
            "device": device_manager.device,
            "force_cpu": device_manager.force_cpu,
            "gpu_enabled": device_manager.is_gpu_enabled(),
            "torch_available": device_manager.torch_available,
        },
        "cloud_storage": {
            "enabled": settings.enable_cloud_storage,
            "provider": settings.storage_provider,
            "bucket": settings.get_storage_config().get("bucket_name", "unknown"),
            "model_storage_prefix": settings.model_storage_prefix,
            "dataset_storage_prefix": settings.dataset_storage_prefix,
            "endpoint_url": settings.get_storage_config().get(
                "endpoint_url", "unknown"
            ),
            "region": settings.get_storage_config().get("region_name", "unknown"),
            "use_ssl": settings.get_storage_config().get("use_ssl", False),
        },
        "mlflow": {
            "tracking_uri": settings.mlflow_tracking_uri,
            "experiment_name": settings.mlflow_experiment_name,
        },
        "model_utils": {
            "instance": type(model_utils).__name__,
            "storage_client_injected": model_utils.storage_client is not None,
        },
        "eureka": {
            "enabled": settings.enable_eureka_client,
            "server_url": settings.eureka_server_url,
            "app_name": settings.eureka_app_name,
        },
    }

    return info


# Utility functions for testing and development
def reset_experiment_tracker() -> None:
    """Reset the experiment tracker instance (useful for testing)."""
    dependency_manager.reset_configuration()
    logger.info("Reset experiment tracker instance")


def reset_data_loader() -> None:
    """Reset the data loader instance (useful for testing)."""
    dependency_manager.reset_configuration()
    logger.info("Reset data loader instance")


def override_experiment_tracker(tracker: IExperimentTracker) -> None:
    """
    Override the experiment tracker instance (useful for testing).

    Args:
        tracker: IExperimentTracker instance to use
    """
    dependency_manager.container.experiment_tracker.override(providers.Object(tracker))
    logger.info(f"Overridden experiment tracker with {type(tracker).__name__}")


def override_data_loader(loader: IDataLoader) -> None:
    """
    Override the data loader instance (useful for testing).

    Args:
        loader: IDataLoader instance to use
    """
    dependency_manager.container.data_loader.override(providers.Object(loader))
    logger.info(f"Overridden data loader with {type(loader).__name__}")


# === Eureka lifecycle (clone từ service đúng) ===


async def initialize_services() -> None:
    """Initialize all async services (Eureka included)"""
    _logger = logger  # dùng logger đã có sẵn ở đầu file
    _logger.info("Initializing services...")

    try:
        settings = get_settings()
        # Initialize Eureka client service
        eureka_service = dependency_manager.container.eureka_client_service()
        if settings.enable_eureka_client:
            _logger.info("🚀 Initializing Eureka client service...")
            success = await eureka_service.start()
            if success:
                _logger.info("✅ Eureka client service initialized successfully")
            else:
                _logger.warning("⚠️ Eureka client service initialization failed")
        else:
            _logger.info("🔄 Eureka client service is disabled")

        _logger.info("✅ All services initialized successfully")
    except Exception as e:
        _logger.error(f"❌ Failed to initialize services: {e}")
        raise


async def cleanup_services() -> None:
    """Cleanup all async services (stop Eureka)"""
    _logger = logger
    _logger.info("Cleaning up services...")

    try:
        settings = get_settings()
        eureka_service = dependency_manager.container.eureka_client_service()
        # is_registered() giả định đã có trong EurekaClientService
        if (
            settings.enable_eureka_client
            and hasattr(eureka_service, "is_registered")
            and eureka_service.is_registered()
        ):
            _logger.info("🛑 Stopping Eureka client service...")
            await eureka_service.stop()
            _logger.info("✅ Eureka client service stopped")

        _logger.info("✅ All services cleaned up successfully")
    except Exception as e:
        _logger.error(f"❌ Error during service cleanup: {e}")
