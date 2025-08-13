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
from .dataset_utils import (
    validate_timeframe_string,
    parse_datetime_string,
    calculate_file_age_hours,
    calculate_cache_expiry_hours,
    format_file_size,
    validate_dataset_path,
    get_dataset_metadata,
    merge_dataset_lists,
    calculate_dataset_statistics,
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
    # Dataset utilities
    "validate_timeframe_string",
    "parse_datetime_string",
    "calculate_file_age_hours",
    "calculate_cache_expiry_hours",
    "format_file_size",
    "validate_dataset_path",
    "get_dataset_metadata",
    "merge_dataset_lists",
    "calculate_dataset_statistics",
]
