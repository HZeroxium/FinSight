# utils/common_utils.py

import os
import random
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar
import numpy as np
import torch
from pathlib import Path

from ..common.logger.logger_factory import LoggerFactory

F = TypeVar("F", bound=Callable[..., Any])


class CommonUtils:
    """Common utility functions for the AI prediction system"""

    _logger = LoggerFactory.get_logger(__name__)

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """
        Set random seed for reproducibility across all libraries

        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Make CUDA operations deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Set environment variable for additional determinism
        os.environ["PYTHONHASHSEED"] = str(seed)

        CommonUtils._logger.info(f"Random seed set to {seed}")

    @staticmethod
    def get_timestamp() -> str:
        """
        Get current timestamp as string

        Returns:
            str: Timestamp in format YYYYMMDD_HHMMSS
        """
        return time.strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def get_readable_timestamp() -> str:
        """
        Get current timestamp in readable format

        Returns:
            str: Timestamp in format YYYY-MM-DD HH:MM:SS
        """
        return time.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Format duration in seconds to human readable format

        Args:
            seconds: Duration in seconds

        Returns:
            str: Formatted duration string
        """
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{int(minutes)}m {secs:.1f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"

    @staticmethod
    def format_bytes(bytes_val: int) -> str:
        """
        Format bytes to human readable format

        Args:
            bytes_val: Number of bytes

        Returns:
            str: Formatted string (e.g., "1.5 GB")
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f} PB"

    @staticmethod
    def safe_divide(
        numerator: float, denominator: float, default: float = 0.0
    ) -> float:
        """
        Safely divide two numbers, returning default if denominator is zero

        Args:
            numerator: Numerator
            denominator: Denominator
            default: Default value if denominator is zero

        Returns:
            float: Division result or default value
        """
        if abs(denominator) < 1e-10:
            return default
        return numerator / denominator

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """
        Clamp value between minimum and maximum

        Args:
            value: Value to clamp
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            float: Clamped value
        """
        return max(min_val, min(value, max_val))

    @staticmethod
    def get_project_root() -> Path:
        """
        Get project root directory

        Returns:
            Path: Project root path
        """
        current_file = Path(__file__).resolve()
        # Navigate up to find the project root (assuming it contains ai_prediction folder)
        for parent in current_file.parents:
            if (parent / "ai_prediction").exists():
                return parent
        # Fallback to current directory
        return Path.cwd()

    @staticmethod
    def create_run_id() -> str:
        """
        Create unique run ID for experiments

        Returns:
            str: Unique run ID
        """
        timestamp = CommonUtils.get_timestamp()
        random_suffix = random.randint(1000, 9999)
        return f"run_{timestamp}_{random_suffix}"

    @staticmethod
    def timer(func: F) -> F:
        """
        Decorator to time function execution

        Args:
            func: Function to time

        Returns:
            Decorated function
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time

            CommonUtils._logger.info(
                f"{func.__name__} completed in {CommonUtils.format_duration(duration)}"
            )
            return result

        return wrapper

    @staticmethod
    def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
        """
        Decorator to retry function execution on failure

        Args:
            max_attempts: Maximum number of attempts
            delay: Initial delay between attempts
            backoff: Backoff multiplier for delay

        Returns:
            Decorator function
        """

        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args, **kwargs):
                current_delay = delay
                last_exception = None

                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            CommonUtils._logger.warning(
                                f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                                f"Retrying in {current_delay:.1f}s"
                            )
                            time.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            CommonUtils._logger.error(
                                f"All {max_attempts} attempts failed for {func.__name__}"
                            )

                raise last_exception

            return wrapper

        return decorator

    @staticmethod
    def memoize(func: F) -> F:
        """
        Simple memoization decorator

        Args:
            func: Function to memoize

        Returns:
            Memoized function
        """
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = str(args) + str(sorted(kwargs.items()))

            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]

        wrapper.cache = cache
        wrapper.clear_cache = lambda: cache.clear()
        return wrapper

    @staticmethod
    def flatten_dict(
        d: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """
        Flatten nested dictionary

        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator for nested keys

        Returns:
            dict: Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(CommonUtils.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
        """
        Unflatten dictionary (reverse of flatten_dict)

        Args:
            d: Flattened dictionary
            sep: Separator used in keys

        Returns:
            dict: Nested dictionary
        """
        result = {}
        for key, value in d.items():
            parts = key.split(sep)
            d_ref = result
            for part in parts[:-1]:
                if part not in d_ref:
                    d_ref[part] = {}
                d_ref = d_ref[part]
            d_ref[parts[-1]] = value
        return result

    @staticmethod
    def get_environment_info() -> Dict[str, str]:
        """
        Get information about the current environment

        Returns:
            dict: Environment information
        """
        import platform
        import sys

        info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture()[0],
            "hostname": platform.node(),
            "torch_version": torch.__version__,
            "cuda_available": str(torch.cuda.is_available()),
        }

        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = str(torch.cuda.device_count())
            info["gpu_name"] = torch.cuda.get_device_name(0)

        return info

    @staticmethod
    def format_time(seconds: float) -> str:
        """
        Format time duration in human readable format

        Args:
            seconds: Time duration in seconds

        Returns:
            str: Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{int(minutes)}m {remaining_seconds:.1f}s"
        else:
            hours = seconds // 3600
            remaining_minutes = (seconds % 3600) // 60
            return f"{int(hours)}h {int(remaining_minutes)}m"

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        Get current memory usage

        Returns:
            dict: Memory usage information
        """
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
        }

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
        """
        Count model parameters

        Args:
            model: PyTorch model

        Returns:
            dict: Parameter counts
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": total_params - trainable_params,
        }

    @staticmethod
    def create_checkpoint_name(
        model_name: str, epoch: int, metric_value: float, metric_name: str = "loss"
    ) -> str:
        """
        Create standardized checkpoint filename

        Args:
            model_name: Name of the model
            epoch: Training epoch
            metric_value: Metric value for this checkpoint
            metric_name: Name of the metric

        Returns:
            str: Checkpoint filename
        """
        timestamp = CommonUtils.get_timestamp()
        return f"{model_name}_epoch_{epoch}_{metric_name}_{metric_value:.4f}_{timestamp}.pt"

    @staticmethod
    def ensure_reproducibility(seed: int = 42) -> None:
        """
        Ensure reproducibility by setting all random seeds and deterministic flags

        Args:
            seed: Random seed value
        """
        CommonUtils.set_seed(seed)

        # Additional settings for reproducibility
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    @staticmethod
    def pretty_print_dict(
        data: Dict[str, Any], title: Optional[str] = None, indent: int = 2
    ) -> None:
        """
        Pretty print dictionary with optional title

        Args:
            data: Dictionary to print
            title: Optional title
            indent: Indentation level
        """
        if title:
            CommonUtils._logger.info(f"\n{title}")
            CommonUtils._logger.info("=" * len(title))

        for key, value in data.items():
            if isinstance(value, dict):
                CommonUtils._logger.info(f"{' ' * indent}{key}:")
                CommonUtils.pretty_print_dict(value, indent=indent + 2)
            else:
                CommonUtils._logger.info(f"{' ' * indent}{key}: {value}")
