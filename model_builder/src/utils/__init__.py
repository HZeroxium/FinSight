# utils/__init__.py

"""
Utilities package for AI Prediction service.
"""

# Import base utilities first (no dependencies)
from .common_utils import CommonUtils
from .device_utils import DeviceUtils
from .file_utils import FileUtils
from .metric_utils import MetricUtils
from .validation_utils import ValidationUtils

# Import utilities with dependencies second
from .demo_utils import DemoUtils
from .training_utils import TrainingUtils
from .evaluation_utils import EvaluationUtils
from .visualization_utils import VisualizationUtils
from .prediction_utils import PredictionUtils
from .model_utils import ModelUtils

# Convenience functions
get_device = DeviceUtils.get_device
set_seed = CommonUtils.set_seed
ensure_dir = FileUtils.ensure_dir
save_json = FileUtils.save_json
load_json = FileUtils.load_json
calculate_metrics = MetricUtils.calculate_all_metrics
validate_dataframe = ValidationUtils.validate_dataframe
validate_numeric_data = ValidationUtils.validate_numeric_data

# Aliases for backward compatibility
mean_absolute_percentage_error = MetricUtils.calculate_mape

__all__ = [
    # Classes
    "CommonUtils",
    "DeviceUtils",
    "FileUtils",
    "MetricUtils",
    "ValidationUtils",
    "DemoUtils",
    "TrainingUtils",
    "EvaluationUtils",
    "VisualizationUtils",
    "PredictionUtils",
    "ModelUtils",
    # Convenience functions
    "get_device",
    "set_seed",
    "ensure_dir",
    "save_json",
    "load_json",
    "calculate_metrics",
    "validate_dataframe",
    "validate_numeric_data",
    "mean_absolute_percentage_error",
    "load_json",
    "calculate_metrics",
    "validate_dataframe",
    "validate_numeric_data",
    "mean_absolute_percentage_error",
]
