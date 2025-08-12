# utils/__init__.py

"""
Utility modules for the FinSight Model Builder system.
"""

from .backtest_strategy_utils import (
    BacktestEngine,
    HyperparameterTuner,
    SignalType,
    Trade,
)
from .device_manager import (
    DeviceManager,
    create_device_manager,
    create_device_manager_from_settings,
)

# Conditional import for visualization utils due to matplotlib dependency
try:
    from .visualization_utils import VisualizationUtils
except ImportError:
    VisualizationUtils = None

__all__ = [
    "BacktestEngine",
    "HyperparameterTuner",
    "SignalType",
    "Trade",
    "DeviceManager",
    "create_device_manager",
    "create_device_manager_from_settings",
    "VisualizationUtils",
]
