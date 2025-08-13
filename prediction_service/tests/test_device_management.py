# tests/test_device_management.py

"""
Tests for device management functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.device_manager import DeviceManager, create_device_manager
from src.core.config import Settings


class TestDeviceManager:
    """Test cases for DeviceManager class."""

    def test_force_cpu_true(self):
        """Test device manager with force_cpu=True."""
        device_manager = DeviceManager(force_cpu=True)

        assert device_manager.force_cpu is True
        assert device_manager.device == "cpu"
        assert device_manager.is_gpu_enabled() is False

    def test_force_cpu_false_no_cuda(self):
        """Test device manager with force_cpu=False and no CUDA."""
        with patch("src.utils.device_manager.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            device_manager = DeviceManager(force_cpu=False)

            assert device_manager.force_cpu is False
            assert device_manager.device == "cpu"
            assert device_manager.is_gpu_enabled() is False

    @patch("src.utils.device_manager.torch")
    def test_force_cpu_false_with_cuda(self, mock_torch):
        """Test device manager with force_cpu=False and CUDA available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3080"
        mock_torch.version.cuda = "11.8"
        mock_torch.cuda.device_count.return_value = 1

        device_manager = DeviceManager(force_cpu=False)

        assert device_manager.force_cpu is False
        assert device_manager.device == "cuda"
        assert device_manager.is_gpu_enabled() is True

    def test_torch_not_available(self):
        """Test device manager when PyTorch is not available."""
        device_manager = DeviceManager(force_cpu=False)
        device_manager._torch_available = False

        assert device_manager.torch_available is False
        assert device_manager.device == "cpu"
        assert device_manager.is_gpu_enabled() is False

    @patch("src.utils.device_manager.torch")
    def test_get_torch_device(self, mock_torch):
        """Test getting PyTorch device object."""
        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        device_manager = DeviceManager(force_cpu=True)
        result = device_manager.get_torch_device()

        mock_torch.device.assert_called_once_with("cpu")
        assert result == mock_device

    def test_get_torch_device_no_torch(self):
        """Test getting PyTorch device when torch not available."""
        device_manager = DeviceManager(force_cpu=True)
        device_manager._torch_available = False

        with pytest.raises(ImportError):
            device_manager.get_torch_device()

    @patch("src.utils.device_manager.torch")
    def test_move_to_device(self, mock_torch):
        """Test moving tensor to device."""
        mock_tensor = MagicMock()
        mock_tensor.to.return_value = "tensor_on_device"
        mock_torch.device.return_value = "cpu_device"

        device_manager = DeviceManager(force_cpu=True)
        result = device_manager.move_to_device(mock_tensor)

        mock_tensor.to.assert_called_once_with("cpu_device")
        assert result == "tensor_on_device"

    def test_move_to_device_no_to_method(self):
        """Test moving object without .to() method."""
        device_manager = DeviceManager(force_cpu=True)
        obj = "string_object"

        result = device_manager.move_to_device(obj)

        assert result == obj

    @patch("src.utils.device_manager.torch")
    def test_get_device_info(self, mock_torch):
        """Test getting device information."""
        mock_torch.__version__ = "2.0.1"
        mock_torch.cuda.is_available.return_value = True
        mock_torch.version.cuda = "11.8"
        mock_torch.backends.cudnn.version.return_value = 8700
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3080"
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=10737418240
        )
        mock_torch.cuda.memory_allocated.return_value = 1073741824
        mock_torch.cuda.memory_reserved.return_value = 2147483648

        device_manager = DeviceManager(force_cpu=False)
        info = device_manager.get_device_info()

        assert info["device"] == "cuda"
        assert info["force_cpu"] is False
        assert info["torch_available"] is True
        assert info["gpu_enabled"] is True
        assert info["torch_version"] == "2.0.1"
        assert info["cuda_available"] is True
        assert info["cuda_version"] == "11.8"

    def test_reset_device(self):
        """Test resetting device configuration."""
        device_manager = DeviceManager(force_cpu=True)
        device_manager._device = "some_device"

        device_manager.reset_device()

        assert device_manager._device is None

    def test_set_force_cpu(self):
        """Test updating force_cpu setting."""
        device_manager = DeviceManager(force_cpu=False)
        device_manager._device = "cuda"

        device_manager.set_force_cpu(True)

        assert device_manager.force_cpu is True
        assert device_manager._device is None  # Should be reset


class TestDeviceManagerIntegration:
    """Integration tests for device manager with configuration."""

    @patch("src.utils.device_manager.get_settings")
    def test_create_device_manager_from_settings(self, mock_get_settings):
        """Test creating device manager from settings."""
        mock_settings = MagicMock()
        mock_settings.force_cpu = True
        mock_get_settings.return_value = mock_settings

        from src.utils.device_manager import create_device_manager_from_settings

        device_manager = create_device_manager_from_settings()

        assert device_manager.force_cpu is True

    def test_factory_function(self):
        """Test device manager factory function."""
        device_manager = create_device_manager(force_cpu=True)

        assert isinstance(device_manager, DeviceManager)
        assert device_manager.force_cpu is True


if __name__ == "__main__":
    pytest.main([__file__])
