import torch
import psutil
from typing import Optional, Dict, List, Union
from common.logger.logger_factory import LoggerFactory


class DeviceUtils:
    """Utility class for device management (CPU/GPU) with enhanced capabilities"""

    _logger = LoggerFactory.get_logger(__name__)

    @staticmethod
    def get_device(
        prefer_gpu: bool = True, gpu_id: Optional[int] = None
    ) -> torch.device:
        """
        Get the best available device

        Args:
            prefer_gpu: Whether to prefer GPU if available
            gpu_id: Specific GPU ID to use (None for automatic selection)

        Returns:
            torch.device: The selected device
        """
        if prefer_gpu and torch.cuda.is_available():
            if gpu_id is not None:
                if gpu_id >= torch.cuda.device_count():
                    DeviceUtils._logger.warning(
                        f"GPU {gpu_id} not available, using GPU 0"
                    )
                    gpu_id = 0
                device = torch.device(f"cuda:{gpu_id}")
            else:
                # Select GPU with most free memory
                gpu_id = DeviceUtils.get_best_gpu()
                device = torch.device(f"cuda:{gpu_id}")

            DeviceUtils._logger.info(
                f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}"
            )

            # Log GPU memory info
            memory_info = DeviceUtils.get_gpu_memory_info(gpu_id)
            if memory_info:
                DeviceUtils._logger.info(
                    f"GPU Memory: {memory_info['total_gb']:.1f}GB total, "
                    f"{memory_info['free_gb']:.1f}GB free"
                )
        else:
            device = torch.device("cpu")
            DeviceUtils._logger.info("Using CPU")

            # Log CPU info
            cpu_info = DeviceUtils.get_cpu_info()
            DeviceUtils._logger.info(
                f"CPU: {cpu_info['cores']} cores, "
                f"{cpu_info['memory_gb']:.1f}GB RAM available"
            )

        return device

    @staticmethod
    def get_best_gpu() -> int:
        """
        Get GPU with most free memory

        Returns:
            int: GPU ID with most free memory
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        gpu_count = torch.cuda.device_count()
        if gpu_count == 1:
            return 0

        max_free_memory = 0
        best_gpu = 0

        for gpu_id in range(gpu_count):
            memory_info = DeviceUtils.get_gpu_memory_info(gpu_id)
            if memory_info and memory_info["free_gb"] > max_free_memory:
                max_free_memory = memory_info["free_gb"]
                best_gpu = gpu_id

        DeviceUtils._logger.info(
            f"Selected GPU {best_gpu} with {max_free_memory:.1f}GB free memory"
        )
        return best_gpu

    @staticmethod
    def get_gpu_count() -> int:
        """
        Get number of available GPUs

        Returns:
            int: Number of available GPUs
        """
        return torch.cuda.device_count() if torch.cuda.is_available() else 0

    @staticmethod
    def get_available_gpus() -> List[int]:
        """
        Get list of available GPU IDs

        Returns:
            List[int]: List of available GPU IDs
        """
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))

    @staticmethod
    def clear_gpu_cache(device_id: Optional[int] = None) -> None:
        """
        Clear GPU cache for specific device or all devices

        Args:
            device_id: Specific GPU ID (None for all GPUs)
        """
        if not torch.cuda.is_available():
            return

        if device_id is not None:
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
            DeviceUtils._logger.info(f"GPU {device_id} cache cleared")
        else:
            torch.cuda.empty_cache()
            DeviceUtils._logger.info("All GPU caches cleared")

    @staticmethod
    def get_gpu_memory_info(device_id: int = 0) -> Optional[Dict[str, float]]:
        """
        Get GPU memory information for specific device

        Args:
            device_id: GPU device ID

        Returns:
            dict: GPU memory info or None if not available
        """
        if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
            return None

        try:
            with torch.cuda.device(device_id):
                allocated = torch.cuda.memory_allocated(device_id)
                cached = torch.cuda.memory_reserved(device_id)
                total = torch.cuda.get_device_properties(device_id).total_memory

                return {
                    "device_id": device_id,
                    "allocated_gb": allocated / 1e9,
                    "cached_gb": cached / 1e9,
                    "total_gb": total / 1e9,
                    "free_gb": (total - allocated) / 1e9,
                    "utilization_percent": (allocated / total) * 100,
                }
        except Exception as e:
            DeviceUtils._logger.error(f"Failed to get GPU memory info: {str(e)}")
            return None

    @staticmethod
    def get_all_gpu_memory_info() -> List[Dict[str, float]]:
        """
        Get memory information for all available GPUs

        Returns:
            List[dict]: List of GPU memory info dictionaries
        """
        if not torch.cuda.is_available():
            return []

        gpu_info = []
        for gpu_id in range(torch.cuda.device_count()):
            info = DeviceUtils.get_gpu_memory_info(gpu_id)
            if info:
                gpu_info.append(info)

        return gpu_info

    @staticmethod
    def get_cpu_info() -> Dict[str, Union[int, float]]:
        """
        Get CPU information

        Returns:
            dict: CPU information including cores and memory
        """
        return {
            "cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "memory_gb": psutil.virtual_memory().available / 1e9,
            "total_memory_gb": psutil.virtual_memory().total / 1e9,
            "cpu_percent": psutil.cpu_percent(interval=1),
        }

    @staticmethod
    def monitor_memory_usage(device: torch.device) -> Dict[str, float]:
        """
        Monitor current memory usage for a device

        Args:
            device: PyTorch device

        Returns:
            dict: Memory usage information
        """
        if device.type == "cuda":
            return DeviceUtils.get_gpu_memory_info(device.index) or {}
        else:
            return DeviceUtils.get_cpu_info()

    @staticmethod
    def set_memory_fraction(fraction: float, device_id: int = 0) -> None:
        """
        Set memory fraction for GPU to avoid OOM errors

        Args:
            fraction: Fraction of GPU memory to use (0.0 to 1.0)
            device_id: GPU device ID
        """
        if not torch.cuda.is_available():
            DeviceUtils._logger.warning(
                "CUDA not available, cannot set memory fraction"
            )
            return

        if not 0.0 < fraction <= 1.0:
            raise ValueError("Memory fraction must be between 0.0 and 1.0")

        torch.cuda.set_per_process_memory_fraction(fraction, device_id)
        DeviceUtils._logger.info(f"Set GPU {device_id} memory fraction to {fraction}")

    @staticmethod
    def optimize_for_inference(device: torch.device) -> None:
        """
        Optimize device settings for inference

        Args:
            device: PyTorch device
        """
        if device.type == "cuda":
            # Enable optimizations for inference
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            DeviceUtils._logger.info("Optimized CUDA settings for inference")

    @staticmethod
    def optimize_for_training(device: torch.device) -> None:
        """
        Optimize device settings for training

        Args:
            device: PyTorch device
        """
        if device.type == "cuda":
            # Settings for training
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            DeviceUtils._logger.info("Optimized CUDA settings for training")
