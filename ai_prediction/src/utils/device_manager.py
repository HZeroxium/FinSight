# utils/device_manager.py

"""
Device management utilities for CPU/GPU configuration.

This module provides centralized device management for all PyTorch operations
in the AI prediction module, respecting the force_cpu configuration setting.
"""

from typing import Optional, Dict, Any
from common.logger.logger_factory import LoggerFactory


class DeviceManager:
    """
    Centralized device manager for PyTorch operations.

    This class handles device configuration and ensures consistent
    CPU/GPU usage across all model adapters and serving components.
    """

    def __init__(self, force_cpu: bool = False):
        """
        Initialize device manager.

        Args:
            force_cpu: Force CPU usage even when GPU is available
        """
        self.force_cpu = force_cpu
        self.logger = LoggerFactory.get_logger("DeviceManager")
        self._device = None
        self._torch_available = None

    @property
    def torch_available(self) -> bool:
        """Check if PyTorch is available."""
        if self._torch_available is None:
            try:
                import torch

                self._torch_available = True
            except ImportError:
                self._torch_available = False
                self.logger.warning("PyTorch not available, falling back to CPU")
        return self._torch_available

    @property
    def device(self) -> str:
        """
        Get the appropriate device string.

        Returns:
            str: Device string ('cpu' or 'cuda')
        """
        if self._device is None:
            self._device = self._determine_device()
        return self._device

    def _determine_device(self) -> str:
        """
        Determine the appropriate device based on configuration and availability.

        Returns:
            str: Device string
        """
        if self.force_cpu:
            self.logger.info("Force CPU mode enabled - using CPU for all operations")
            return "cpu"

        if not self.torch_available:
            self.logger.warning("PyTorch not available - using CPU")
            return "cpu"

        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
                self.logger.info(
                    f"CUDA available - using GPU: {torch.cuda.get_device_name(0)}"
                )
                self.logger.info(f"CUDA version: {torch.version.cuda}")
                self.logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            else:
                device = "cpu"
                self.logger.info("CUDA not available - using CPU")

            return device

        except Exception as e:
            self.logger.error(f"Error determining device: {e}")
            return "cpu"

    def get_torch_device(self):
        """
        Get PyTorch device object.

        Returns:
            torch.device: PyTorch device object
        """
        if not self.torch_available:
            raise ImportError("PyTorch is required but not available")

        import torch

        return torch.device(self.device)

    def move_to_device(self, tensor_or_model, device: Optional[str] = None):
        """
        Move tensor or model to the configured device.

        Args:
            tensor_or_model: PyTorch tensor or model to move
            device: Optional device override

        Returns:
            Tensor or model on the appropriate device
        """
        if not self.torch_available:
            return tensor_or_model

        target_device = device or self.device

        try:
            import torch

            if hasattr(tensor_or_model, "to"):
                return tensor_or_model.to(torch.device(target_device))
            else:
                self.logger.warning(
                    f"Object {type(tensor_or_model)} doesn't support .to() method"
                )
                return tensor_or_model
        except Exception as e:
            self.logger.error(f"Error moving to device {target_device}: {e}")
            return tensor_or_model

    def is_gpu_enabled(self) -> bool:
        """
        Check if GPU is enabled and available.

        Returns:
            bool: True if GPU is available and not forced to CPU
        """
        return self.device == "cuda"

    def get_device_info(self) -> Dict[str, Any]:
        """
        Get detailed device information.

        Returns:
            Dict with device information
        """
        info = {
            "device": self.device,
            "force_cpu": self.force_cpu,
            "torch_available": self.torch_available,
            "gpu_enabled": self.is_gpu_enabled(),
        }

        if self.torch_available:
            try:
                import torch

                info.update(
                    {
                        "torch_version": torch.__version__,
                        "cuda_available": torch.cuda.is_available(),
                    }
                )

                if torch.cuda.is_available():
                    info.update(
                        {
                            "cuda_version": torch.version.cuda,
                            "cudnn_version": torch.backends.cudnn.version(),
                            "gpu_count": torch.cuda.device_count(),
                            "current_gpu": (
                                torch.cuda.current_device()
                                if torch.cuda.is_available()
                                else None
                            ),
                            "gpu_names": [
                                torch.cuda.get_device_name(i)
                                for i in range(torch.cuda.device_count())
                            ],
                            "gpu_memory": (
                                {
                                    f"gpu_{i}": {
                                        "total": torch.cuda.get_device_properties(
                                            i
                                        ).total_memory,
                                        "allocated": torch.cuda.memory_allocated(i),
                                        "cached": torch.cuda.memory_reserved(i),
                                    }
                                    for i in range(torch.cuda.device_count())
                                }
                                if torch.cuda.is_available()
                                else {}
                            ),
                        }
                    )
            except Exception as e:
                self.logger.error(f"Error getting detailed device info: {e}")

        return info

    def reset_device(self) -> None:
        """Reset device detection (useful for testing or configuration changes)."""
        self._device = None
        self.logger.info("Device configuration reset")

    def set_force_cpu(self, force_cpu: bool) -> None:
        """
        Update force_cpu setting and reset device.

        Args:
            force_cpu: New force_cpu setting
        """
        if self.force_cpu != force_cpu:
            self.force_cpu = force_cpu
            self.reset_device()
            self.logger.info(f"Force CPU setting updated to: {force_cpu}")


def create_device_manager(force_cpu: bool = False) -> DeviceManager:
    """
    Factory function to create a device manager.

    Args:
        force_cpu: Force CPU usage

    Returns:
        DeviceManager instance
    """
    return DeviceManager(force_cpu=force_cpu)


def create_device_manager_from_settings() -> DeviceManager:
    """
    Create device manager from application settings.

    Returns:
        DeviceManager instance configured from settings
    """
    from ..core.config import get_settings

    settings = get_settings()
    return create_device_manager(force_cpu=settings.force_cpu)
