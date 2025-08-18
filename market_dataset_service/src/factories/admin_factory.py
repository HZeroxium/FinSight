# factories/admin_factory.py

"""
Admin Factory

Factory functions for creating admin service dependencies.
"""

from ..services.admin_service import AdminService
from ..utils.dependencies import (
    get_market_data_service,
    get_market_data_collector_service,
    get_storage_service,
    get_market_data_job_service,
    get_cross_repository_pipeline,
)
from .market_data_repository_factory import create_repository
from ..core.config import Settings


def get_admin_service() -> AdminService:
    """
    Create and configure admin service with all dependencies.

    Returns:
        Configured AdminService instance
    """
    # Get dependencies
    repository = create_repository("csv")
    market_data_service = get_market_data_service()
    collector_service = get_market_data_collector_service()
    storage_service = get_storage_service()
    market_data_job_service = get_market_data_job_service()
    cross_repository_pipeline = get_cross_repository_pipeline()

    # Create admin service
    admin_service = AdminService(
        market_data_service=market_data_service,
        collector_service=collector_service,
        repository=repository,
        storage_service=storage_service,
        market_data_job_service=market_data_job_service,
        cross_repository_pipeline=cross_repository_pipeline,
    )

    return admin_service


def get_admin_service_with_config(settings: Settings) -> AdminService:
    """
    Create admin service with specific configuration.

    Args:
        settings: Configuration settings

    Returns:
        Configured AdminService instance
    """
    # This could be extended to use settings for specific configurations
    return get_admin_service()
