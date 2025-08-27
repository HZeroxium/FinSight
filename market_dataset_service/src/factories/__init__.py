"""
Factory package for creating various service instances.

Provides factory patterns for creating repositories, collectors, and services
with different implementations and configurations.
"""

from .backtesting_factory import BacktestingEngineType, BacktestingFactory

__all__ = [
    "BacktestingFactory",
    "BacktestingEngineType",
]
