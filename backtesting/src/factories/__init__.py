"""
Factory package for creating various service instances.

Provides factory patterns for creating repositories, collectors, and services
with different implementations and configurations.
"""

from .market_data_repository_factory import (
    MarketDataRepositoryFactory,
    repository_factory,
    create_repository,
    create_repository_from_config,
)

from .backtesting_factory import (
    BacktestingFactory,
    BacktestingEngineType,
)

from .admin_factory import (
    get_admin_service,
    get_admin_service_with_config,
)

__all__ = [
    "MarketDataRepositoryFactory",
    "repository_factory",
    "create_repository",
    "create_repository_from_config",
    "BacktestingFactory",
    "BacktestingEngineType",
]
