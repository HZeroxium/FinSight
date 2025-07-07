# factories/admin_factory.py

"""
Admin Factory

Factory functions for creating admin service dependencies.
"""

from ..services.admin_service import AdminService
from .market_data_repository_factory import get_market_data_service, create_repository
from .backtesting_factory import get_market_data_collector_service
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

    # Create admin service
    admin_service = AdminService(
        market_data_service=market_data_service,
        collector_service=collector_service,
        repository=repository,
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
