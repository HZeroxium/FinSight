# utils/dependencies.py

from typing import Optional
from functools import lru_cache

from ..interfaces.experiment_tracker_interface import IExperimentTracker
from ..interfaces.data_loader_interface import IDataLoader
from ..adapters.simple_experiment_tracker import SimpleExperimentTracker
from ..data.data_loader import CloudDataLoader, FileDataLoader
from ..core.config import get_settings
from common.logger.logger_factory import LoggerFactory

# Initialize logger
logger = LoggerFactory.get_logger("Dependencies")

# Global instances for singleton behavior
_experiment_tracker: Optional[IExperimentTracker] = None
_data_loader: Optional[IDataLoader] = None


def get_experiment_tracker() -> IExperimentTracker:
    """
    Get experiment tracker instance with fallback logic.

    Primary: MLflow (if available and configured)
    Fallback: Simple file-based tracker

    Returns:
        IExperimentTracker: Configured experiment tracker
    """
    global _experiment_tracker

    if _experiment_tracker is not None:
        return _experiment_tracker

    settings = get_settings()
    tracker_type = settings.experiment_tracker_type.lower()
    fallback_type = settings.experiment_tracker_fallback.lower()

    # Try primary tracker
    try:
        if tracker_type == "mlflow":
            from ..adapters.mlflow_experiment_tracker import MLflowExperimentTracker

            _experiment_tracker = MLflowExperimentTracker(
                tracking_uri=settings.mlflow_tracking_uri,
                experiment_name=settings.mlflow_experiment_name,
                artifact_root=settings.mlflow_artifact_root,
            )
            logger.info(f"Initialized MLflow experiment tracker")

        elif tracker_type == "simple":
            _experiment_tracker = SimpleExperimentTracker()
            logger.info(f"Initialized Simple experiment tracker")

        else:
            raise ValueError(f"Unknown experiment tracker type: {tracker_type}")

    except Exception as e:
        logger.warning(f"Failed to initialize primary tracker ({tracker_type}): {e}")

        # Try fallback tracker
        try:
            if fallback_type == "simple":
                _experiment_tracker = SimpleExperimentTracker()
                logger.info(f"Initialized fallback Simple experiment tracker")
            else:
                raise ValueError(f"Unknown fallback tracker type: {fallback_type}")

        except Exception as fallback_error:
            logger.error(f"Failed to initialize fallback tracker: {fallback_error}")
            # Default to simple tracker as last resort
            _experiment_tracker = SimpleExperimentTracker()
            logger.info("Using default Simple experiment tracker as last resort")

    return _experiment_tracker


def get_data_loader() -> IDataLoader:
    """
    Get data loader instance based on configuration.

    Types:
    - hybrid: Cloud-first with local fallback (default)
    - cloud: Cloud-only
    - local: Local files only

    Returns:
        IDataLoader: Configured data loader
    """
    global _data_loader

    if _data_loader is not None:
        return _data_loader

    settings = get_settings()
    loader_type = settings.data_loader_type.lower()

    try:
        if loader_type in ["hybrid", "cloud"]:
            _data_loader = CloudDataLoader()
            logger.info(f"Initialized CloudDataLoader in {loader_type} mode")

        elif loader_type == "local":
            _data_loader = FileDataLoader()
            logger.info("Initialized FileDataLoader for local files only")

        else:
            logger.warning(f"Unknown data loader type: {loader_type}, using hybrid")
            _data_loader = CloudDataLoader()
            logger.info("Initialized CloudDataLoader as default")

    except Exception as e:
        logger.error(f"Failed to initialize data loader: {e}")
        # Fallback to simple file loader
        _data_loader = FileDataLoader()
        logger.info("Using FileDataLoader as fallback")

    return _data_loader


def reset_experiment_tracker() -> None:
    """Reset the experiment tracker instance (useful for testing)."""
    global _experiment_tracker
    _experiment_tracker = None
    logger.info("Reset experiment tracker instance")


def reset_data_loader() -> None:
    """Reset the data loader instance (useful for testing)."""
    global _data_loader
    _data_loader = None
    logger.info("Reset data loader instance")


def override_experiment_tracker(tracker: IExperimentTracker) -> None:
    """
    Override the experiment tracker instance (useful for testing).

    Args:
        tracker: IExperimentTracker instance to use
    """
    global _experiment_tracker
    _experiment_tracker = tracker
    logger.info(f"Overridden experiment tracker with {type(tracker).__name__}")


def override_data_loader(loader: IDataLoader) -> None:
    """
    Override the data loader instance (useful for testing).

    Args:
        loader: IDataLoader instance to use
    """
    global _data_loader
    _data_loader = loader
    logger.info(f"Overridden data loader with {type(loader).__name__}")


async def health_check_dependencies() -> dict:
    """
    Perform health check on all dependencies.

    Returns:
        dict: Health check results for all dependencies
    """
    results = {"experiment_tracker": False, "data_loader": False, "timestamp": None}

    try:
        # Check experiment tracker
        tracker = get_experiment_tracker()
        results["experiment_tracker"] = await tracker.health_check()

        # Check data loader (simple existence check)
        loader = get_data_loader()
        results["data_loader"] = loader is not None

        results["timestamp"] = logger.info("Dependency health check completed")

    except Exception as e:
        logger.error(f"Dependency health check failed: {e}")
        results["error"] = str(e)

    return results


# FastAPI dependency functions for easy injection


async def get_experiment_tracker_dependency() -> IExperimentTracker:
    """FastAPI dependency function for experiment tracker."""
    return get_experiment_tracker()


async def get_data_loader_dependency() -> IDataLoader:
    """FastAPI dependency function for data loader."""
    return get_data_loader()


# Utility functions for dependency information


def get_dependency_info() -> dict:
    """
    Get information about current dependency configuration.

    Returns:
        dict: Current dependency configuration and status
    """
    settings = get_settings()

    info = {
        "experiment_tracker": {
            "type": settings.experiment_tracker_type,
            "fallback": settings.experiment_tracker_fallback,
            "instance": (
                type(_experiment_tracker).__name__ if _experiment_tracker else None
            ),
        },
        "data_loader": {
            "type": settings.data_loader_type,
            "cloud_enabled": settings.enable_cloud_storage,
            "instance": type(_data_loader).__name__ if _data_loader else None,
        },
        "cloud_storage": {
            "enabled": settings.enable_cloud_storage,
            "type": settings.cloud_storage_type,
            "bucket": settings.cloud_storage_bucket,
        },
        "mlflow": {
            "tracking_uri": settings.mlflow_tracking_uri,
            "experiment_name": settings.mlflow_experiment_name,
            "artifact_root": settings.mlflow_artifact_root,
        },
    }

    return info
