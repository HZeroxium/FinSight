# utils/device_manager.py

"""
Device management utilities for CPU/GPU configuration.

This module provides centralized device management for all PyTorch operations
in the AI prediction module, respecting the force_cpu configuration setting.
"""

import os
from typing import Any, Dict, Optional, Union

import torch
from common.logger.logger_factory import LoggerFactory

from ..schemas.enums import DeviceType


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
        self._device_info = None

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
            str: Normalized device string ('cpu', 'cuda', 'mps')
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
            return DeviceType.CPU.value

        if not self.torch_available:
            self.logger.warning("PyTorch not available - using CPU")
            return DeviceType.CPU.value

        try:
            import torch

            # Check environment variables first
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible == "":
                self.logger.info("CUDA_VISIBLE_DEVICES is empty - using CPU")
                return DeviceType.CPU.value
            elif cuda_visible is not None:
                self.logger.info(f"CUDA_VISIBLE_DEVICES set to: {cuda_visible}")

            if torch.cuda.is_available():
                # Check if CUDA is actually working
                try:
                    # Test CUDA functionality
                    test_tensor = torch.tensor([1.0], device=DeviceType.CUDA.value)
                    del test_tensor
                    torch.cuda.empty_cache()

                    # Set memory fraction if available
                    try:
                        from ..core.config import get_settings

                        settings = get_settings()
                        if hasattr(settings, "cuda_device_memory_fraction"):
                            torch.cuda.set_per_process_memory_fraction(
                                settings.cuda_device_memory_fraction
                            )
                            self.logger.info(
                                f"Set GPU memory fraction to: {settings.cuda_device_memory_fraction}"
                            )
                    except Exception as mem_error:
                        self.logger.debug(
                            f"Could not set GPU memory fraction: {mem_error}"
                        )

                    device = DeviceType.CUDA.value
                    self.logger.info(
                        f"CUDA available and working - using GPU: {torch.cuda.get_device_name(0)}"
                    )
                    self.logger.info(f"CUDA version: {torch.version.cuda}")
                    self.logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
                    self.logger.info(f"Current GPU: {torch.cuda.current_device()}")

                    # Log GPU memory info
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        self.logger.info(
                            f"GPU {i}: {props.name} - {props.total_memory / 1024**3:.1f} GB"
                        )

                except Exception as cuda_error:
                    self.logger.warning(f"CUDA available but not working: {cuda_error}")
                    self.logger.info("Falling back to CPU")
                    device = DeviceType.CPU.value
            else:
                device = DeviceType.CPU.value
                self.logger.info("CUDA not available - using CPU")

            return device

        except Exception as e:
            self.logger.error(f"Error determining device: {e}")
            return DeviceType.CPU.value

    def get_torch_device(self):
        """
        Get PyTorch device object.

        Returns:
            torch.device: PyTorch device object
        """
        if not self.torch_available:
            raise ImportError("PyTorch is required but not available")

        import torch

        # Normalize CUDA device forms (e.g., 'cuda:0' -> 'cuda') when creating torch.device
        dev = self._normalize_device_string(self.device)
        return torch.device(dev)

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

        target_device = self._normalize_device_string(device or self.device)

        try:
            import torch

            if hasattr(tensor_or_model, "to"):
                # Ensure the target device is valid
                if (
                    target_device == DeviceType.CUDA.value
                    and not torch.cuda.is_available()
                ):
                    self.logger.warning("CUDA not available, falling back to CPU")
                    target_device = DeviceType.CPU.value

                result = tensor_or_model.to(torch.device(target_device))

                # Verify the move was successful
                if hasattr(result, "device"):
                    actual_device = self._normalize_device_string(str(result.device))
                    if actual_device != target_device:
                        self.logger.warning(
                            f"Device mismatch: requested {target_device}, got {actual_device}"
                        )

                return result
            else:
                self.logger.warning(
                    f"Object {type(tensor_or_model)} doesn't support .to() method"
                )
                return tensor_or_model
        except Exception as e:
            self.logger.error(f"Error moving to device {target_device}: {e}")
            # Fallback to CPU if GPU fails
            if target_device == DeviceType.CUDA.value:
                self.logger.info("GPU move failed, falling back to CPU")
                try:
                    import torch

                    return tensor_or_model.to(torch.device(DeviceType.CPU.value))
                except Exception as cpu_error:
                    self.logger.error(f"CPU fallback also failed: {cpu_error}")
            return tensor_or_model

    def is_gpu_enabled(self) -> bool:
        """
        Check if GPU is enabled and available.

        Returns:
            bool: True if GPU is available and not forced to CPU
        """
        return self._normalize_device_string(self.device) == DeviceType.CUDA.value

    def get_device_info(self) -> Dict[str, Any]:
        """
        Get detailed device information.

        Returns:
            Dict with device information
        """
        if self._device_info is None:
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

            self._device_info = info

        return self._device_info

    def reset_device(self) -> None:
        """Reset device detection (useful for testing or configuration changes)."""
        self._device = None
        self._device_info = None
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

    def ensure_consistent_device(self, *tensors_or_models) -> None:
        """
        Ensure all tensors/models are on the same device.

        Args:
            *tensors_or_models: Variable number of tensors or models to check
        """
        if not self.torch_available or not tensors_or_models:
            return

        target_device = self._normalize_device_string(self.device)
        inconsistent_items = []

        for item in tensors_or_models:
            if hasattr(item, "device"):
                item_device = self._normalize_device_string(str(item.device))
                if item_device != target_device:
                    inconsistent_items.append((item, item_device))

        if inconsistent_items:
            self.logger.warning(
                f"Found {len(inconsistent_items)} items on inconsistent devices:"
            )
            for item, device in inconsistent_items:
                self.logger.warning(f"  {type(item).__name__} on {device}")

            # Move all items to target device
            for item, _ in inconsistent_items:
                try:
                    item.to(torch.device(target_device))
                except Exception as e:
                    self.logger.error(
                        f"Failed to move {type(item).__name__} to {target_device}: {e}"
                    )

    def get_optimal_batch_size(self, model_size_mb: float = 100.0) -> int:
        """
        Get optimal batch size based on available device memory.

        Args:
            model_size_mb: Estimated model size in MB

        Returns:
            int: Recommended batch size
        """
        if not self.is_gpu_enabled():
            return 32  # Default CPU batch size

        try:
            import torch

            gpu_props = torch.cuda.get_device_properties(0)
            total_memory_gb = gpu_props.total_memory / (1024**3)

            # Conservative memory usage (use 70% of available memory)
            available_memory_gb = total_memory_gb * 0.7
            model_memory_gb = model_size_mb / 1024

            # Estimate memory per sample (rough approximation)
            memory_per_sample_gb = 0.001  # 1MB per sample

            optimal_batch_size = int(
                (available_memory_gb - model_memory_gb) / memory_per_sample_gb
            )

            # Clamp to reasonable bounds
            optimal_batch_size = max(1, min(optimal_batch_size, 128))

            self.logger.info(
                f"GPU memory: {total_memory_gb:.1f}GB, optimal batch size: {optimal_batch_size}"
            )
            return optimal_batch_size

        except Exception as e:
            self.logger.warning(f"Could not determine optimal batch size: {e}")
            return 32

    @staticmethod
    def _normalize_device_string(device: Optional[str]) -> str:
        """Normalize various device string forms to canonical values.

        Examples:
            'cuda:0' -> 'cuda'
            'CUDA' -> 'cuda'
            None -> current default ('cpu')
        """
        if not device:
            return DeviceType.CPU.value
        dev = str(device).strip().lower()
        if dev.startswith(DeviceType.CUDA.value):
            return DeviceType.CUDA.value
        if dev.startswith(DeviceType.MPS.value):
            return DeviceType.MPS.value
        return DeviceType.CPU.value if dev == "cpu" else dev


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
