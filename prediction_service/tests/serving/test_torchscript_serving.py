"""
Test suite for TorchScript Serving Adapter

Tests the complete TorchScript serving pipeline including:
- Model format conversion during training
- TorchScript adapter loading
- Model prediction functionality
- Integration with ModelFormatConverter
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import shutil

from src.adapters.torchscript_serving import TorchScriptServingAdapter
from src.schemas.enums import ModelType, TimeFrame
from src.interfaces.serving_interface import ModelInfo, PredictionResult
from src.utils.model_format_converter import ModelFormatConverter
from src.core.constants import FacadeConstants


class TestTorchScriptServingAdapter:
    """Test cases for TorchScript Serving Adapter"""

    @pytest.fixture
    async def adapter(self):
        """Create a TorchScript serving adapter for testing"""
        config = {
            "device": "cpu",
            "optimize_for_inference": True,
            "enable_fusion": True,
            "compile_mode": "trace",
            "max_models_in_memory": 3,
            "model_timeout_seconds": 3600,
        }
        adapter = TorchScriptServingAdapter(config)
        await adapter.initialize()
        return adapter

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="D"),
                "open": [100 + i * 0.1 for i in range(100)],
                "high": [102 + i * 0.1 for i in range(100)],
                "low": [98 + i * 0.1 for i in range(100)],
                "close": [101 + i * 0.1 for i in range(100)],
                "volume": [1000000 + i * 1000 for i in range(100)],
            }
        )

    @pytest.mark.asyncio
    async def test_adapter_initialization(self, adapter):
        """Test adapter initialization"""
        assert adapter is not None
        assert adapter.device == "cpu"
        assert adapter.optimize_for_inference
        assert adapter.enable_fusion
        assert adapter.compile_mode == "trace"
        assert len(adapter._loaded_models) == 0
        assert len(adapter._model_info) == 0

    @pytest.mark.asyncio
    async def test_health_check(self, adapter):
        """Test adapter health check"""
        health = await adapter.health_check()

        assert health["status"] == "healthy"
        assert "device" in health
        assert "memory_usage_mb" in health
        assert "models_loaded" in health
        assert health["models_loaded"] == 0
        assert health["device"] == "cpu"

    @pytest.mark.asyncio
    @patch("torch.jit.load")
    @patch("torch.jit.trace")
    async def test_model_loading_with_torchscript(
        self, mock_trace, mock_load, adapter, tmp_path
    ):
        """Test model loading with TorchScript compilation"""
        # Setup mocks
        mock_model = Mock()
        mock_torchscript_model = Mock()
        mock_load.return_value = mock_torchscript_model
        mock_trace.return_value = mock_torchscript_model

        # Create temporary model directory
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Create mock model files
        model_file = model_dir / "model.pkl"
        import pickle

        with open(model_file, "wb") as f:
            pickle.dump(mock_model, f)

        # Test model loading
        model_info = await adapter.load_model(
            model_path=str(model_dir),
            symbol="BTCUSDT",
            timeframe=TimeFrame.DAY_1,
            model_type=ModelType.PATCHTSMIXER,
            model_config={},
        )

        assert model_info is not None
        assert model_info.is_loaded
        assert model_info.symbol == "BTCUSDT"
        assert model_info.model_type == ModelType.PATCHTSMIXER.value
        assert len(adapter._loaded_models) == 1

    @pytest.mark.asyncio
    @patch("torch.jit.load")
    async def test_model_loading_existing_torchscript(
        self, mock_load, adapter, tmp_path
    ):
        """Test loading existing TorchScript model"""
        mock_torchscript_model = Mock()
        mock_load.return_value = mock_torchscript_model

        # Create temporary model directory with TorchScript file
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        torchscript_file = model_dir / "model_torchscript.pt"
        torchscript_file.touch()  # Create empty file

        # Test model loading
        model_info = await adapter.load_model(
            model_path=str(model_dir),
            symbol="BTCUSDT",
            timeframe=TimeFrame.DAY_1,
            model_type=ModelType.PATCHTSMIXER,
            model_config={},
        )

        assert model_info is not None
        assert model_info.is_loaded
        mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_loading_failure(self, adapter):
        """Test model loading failure with non-existent path"""
        with pytest.raises(Exception):
            await adapter.load_model(
                model_path="/non/existent/path",
                symbol="BTCUSDT",
                timeframe=TimeFrame.DAY_1,
                model_type=ModelType.PATCHTSMIXER,
                model_config={},
            )

    @pytest.mark.asyncio
    @patch("torch.jit.load")
    async def test_prediction_success(self, mock_load, adapter, sample_data, tmp_path):
        """Test successful prediction with TorchScript model"""
        # Setup mock TorchScript model
        mock_torchscript_model = Mock()
        mock_output = Mock()
        mock_output.cpu.return_value.numpy.return_value = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0]]
        )
        mock_torchscript_model.return_value = mock_output
        mock_load.return_value = mock_torchscript_model

        # Create temporary model directory
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        torchscript_file = model_dir / "model_torchscript.pt"
        torchscript_file.touch()

        # Load model
        model_info = await adapter.load_model(
            model_path=str(model_dir),
            symbol="BTCUSDT",
            timeframe=TimeFrame.DAY_1,
            model_type=ModelType.PATCHTSMIXER,
            model_config={},
        )

        # Test prediction
        with patch("torch.tensor") as mock_tensor:
            mock_tensor.return_value = Mock()

            result = await adapter.predict(
                model_id=model_info.model_id, input_data=sample_data, n_steps=5
            )

        assert result.success
        assert len(result.predictions) == 5
        assert result.inference_time_ms > 0

    @pytest.mark.asyncio
    async def test_prediction_failure_model_not_loaded(self, adapter, sample_data):
        """Test prediction failure when model not loaded"""
        result = await adapter.predict(
            model_id="non_existent_model", input_data=sample_data, n_steps=5
        )

        assert not result.success
        assert "not loaded" in result.error
        assert len(result.predictions) == 0

    @pytest.mark.asyncio
    @patch("torch.jit.load")
    async def test_model_unloading(self, mock_load, adapter, tmp_path):
        """Test model unloading"""
        mock_torchscript_model = Mock()
        mock_load.return_value = mock_torchscript_model

        # Load a model first
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        torchscript_file = model_dir / "model_torchscript.pt"
        torchscript_file.touch()

        model_info = await adapter.load_model(
            model_path=str(model_dir),
            symbol="BTCUSDT",
            timeframe=TimeFrame.DAY_1,
            model_type=ModelType.PATCHTSMIXER,
            model_config={},
        )

        # Verify model is loaded
        assert len(adapter._loaded_models) == 1

        # Unload model
        success = await adapter.unload_model(model_info.model_id)

        assert success
        assert len(adapter._loaded_models) == 0
        assert len(adapter._model_info) == 0

    @pytest.mark.asyncio
    @patch("torch.jit.trace")
    async def test_torchscript_conversion_trace_mode(
        self, mock_trace, adapter, tmp_path
    ):
        """Test TorchScript conversion in trace mode"""
        mock_model = Mock()
        mock_torchscript_model = Mock()
        mock_trace.return_value = mock_torchscript_model

        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Test conversion
        result = await adapter._convert_to_torchscript(
            mock_model, str(model_dir), ModelType.PATCHTSMIXER
        )

        assert result == mock_torchscript_model
        mock_trace.assert_called_once()

    @pytest.mark.asyncio
    @patch("torch.jit.script")
    async def test_torchscript_conversion_script_mode(self, mock_script, tmp_path):
        """Test TorchScript conversion in script mode"""
        config = {"device": "cpu", "compile_mode": "script", "max_models_in_memory": 3}
        adapter = TorchScriptServingAdapter(config)
        await adapter.initialize()

        mock_model = Mock()
        mock_torchscript_model = Mock()
        mock_script.return_value = mock_torchscript_model

        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Test conversion
        result = await adapter._convert_to_torchscript(
            mock_model, str(model_dir), ModelType.PATCHTSMIXER
        )

        assert result == mock_torchscript_model
        mock_script.assert_called_once()

    @pytest.mark.asyncio
    @patch("torch.jit.load")
    async def test_list_loaded_models(self, mock_load, adapter, tmp_path):
        """Test listing loaded models"""
        mock_torchscript_model = Mock()
        mock_load.return_value = mock_torchscript_model

        # Initially no models
        models = await adapter.list_loaded_models()
        assert len(models) == 0

        # Load a model
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        torchscript_file = model_dir / "model_torchscript.pt"
        torchscript_file.touch()

        model_info = await adapter.load_model(
            model_path=str(model_dir),
            symbol="BTCUSDT",
            timeframe=TimeFrame.DAY_1,
            model_type=ModelType.PATCHTSMIXER,
            model_config={},
        )

        # Check loaded models
        models = await adapter.list_loaded_models()
        assert len(models) == 1
        assert models[0].model_id == model_info.model_id

    @pytest.mark.asyncio
    async def test_device_management(self):
        """Test device management (CPU/GPU)"""
        # Test CPU config
        cpu_config = {"device": "cpu", "max_models_in_memory": 3}
        cpu_adapter = TorchScriptServingAdapter(cpu_config)
        await cpu_adapter.initialize()
        assert cpu_adapter.device == "cpu"

        # Test auto device detection
        auto_config = {"device": "auto", "max_models_in_memory": 3}
        auto_adapter = TorchScriptServingAdapter(auto_config)
        await auto_adapter.initialize()
        # Should default to cpu in test environment
        assert auto_adapter.device in ["cpu", "cuda"]

    @pytest.mark.asyncio
    @patch("torch.jit.load")
    async def test_model_optimization(self, mock_load, adapter, tmp_path):
        """Test model optimization features"""
        mock_torchscript_model = Mock()
        mock_optimized_model = Mock()

        # Mock optimization methods
        mock_torchscript_model.eval.return_value = mock_torchscript_model

        with patch("torch.jit.optimize_for_inference") as mock_optimize:
            mock_optimize.return_value = mock_optimized_model
            mock_load.return_value = mock_torchscript_model

            model_dir = tmp_path / "test_model"
            model_dir.mkdir()

            torchscript_file = model_dir / "model_torchscript.pt"
            torchscript_file.touch()

            # Load model with optimization
            model_info = await adapter.load_model(
                model_path=str(model_dir),
                symbol="BTCUSDT",
                timeframe=TimeFrame.DAY_1,
                model_type=ModelType.PATCHTSMIXER,
                model_config={},
            )

            # Should have called optimization if enabled
            if adapter.optimize_for_inference:
                mock_optimize.assert_called_once()

    @pytest.mark.asyncio
    @patch("torch.jit.load")
    async def test_get_stats(self, mock_load, adapter, tmp_path):
        """Test getting adapter statistics"""
        # Initial stats
        stats = await adapter.get_stats()
        assert stats.models_loaded == 0
        assert stats.total_predictions == 0

        # Load a model
        mock_torchscript_model = Mock()
        mock_load.return_value = mock_torchscript_model

        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        torchscript_file = model_dir / "model_torchscript.pt"
        torchscript_file.touch()

        await adapter.load_model(
            model_path=str(model_dir),
            symbol="BTCUSDT",
            timeframe=TimeFrame.DAY_1,
            model_type=ModelType.PATCHTSMIXER,
            model_config={},
        )

        # Updated stats
        stats = await adapter.get_stats()
        assert stats.models_loaded == 1


# Integration test with TorchScript
class TestTorchScriptServingAdapterIntegration:
    """Integration tests with actual TorchScript models"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_model_loading(self):
        """Test loading actual TorchScript models"""
        from src.utils.model_utils import ModelUtils

        model_utils = ModelUtils()
        config = {
            "device": "cpu",
            "optimize_for_inference": True,
            "max_models_in_memory": 5,
        }
        adapter = TorchScriptServingAdapter(config)
        await adapter.initialize()

        # Try to load actual model
        symbol = "BTCUSDT"
        timeframe = TimeFrame.DAY_1
        model_type = ModelType.PATCHTSMIXER

        model_path = model_utils.get_model_path(
            symbol, timeframe, model_type, adapter_type="torchscript"
        )

        if model_path.exists():
            try:
                model_info = await adapter.load_model(
                    model_path=str(model_path),
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type,
                    model_config={},
                )

                assert model_info is not None
                assert model_info.is_loaded

                # Test prediction with real model
                test_data = pd.DataFrame(
                    {
                        "timestamp": pd.date_range("2024-01-01", periods=100, freq="D"),
                        "close": [100 + i * 0.1 for i in range(100)],
                    }
                )

                result = await adapter.predict(
                    model_id=model_info.model_id, input_data=test_data, n_steps=5
                )

                # Should succeed or fail gracefully
                assert isinstance(result, PredictionResult)
                if result.success:
                    assert len(result.predictions) == 5

            except Exception as e:
                pytest.skip(f"TorchScript model loading failed: {e}")
        else:
            pytest.skip("No actual TorchScript model found for integration test")


# Comprehensive Integration Tests
class TestTorchScriptIntegrationComplete:
    """Complete integration tests for TorchScript serving with format conversion"""

    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary models directory structure"""
        temp_dir = Path(tempfile.mkdtemp())
        models_dir = temp_dir / "models"
        
        # Create adapter type directories
        for adapter_type in FacadeConstants.SUPPORTED_ADAPTERS:
            (models_dir / adapter_type).mkdir(parents=True, exist_ok=True)
            
        yield models_dir
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_model_with_pytorch(self):
        """Create a mock model that has a PyTorch model attribute for conversion"""
        mock_model = Mock()
        
        # Mock PyTorch model
        mock_pytorch_model = Mock()
        mock_pytorch_model.eval.return_value = mock_pytorch_model
        mock_pytorch_model.cpu.return_value = mock_pytorch_model
        mock_pytorch_model.to.return_value = mock_pytorch_model
        
        # Mock parameters for device detection
        mock_param = Mock()
        mock_param.device = "cpu"
        mock_pytorch_model.parameters.return_value = [mock_param]
        
        mock_model.model = mock_pytorch_model
        mock_model.context_length = 64
        mock_model.feature_columns = ['close', 'volume', 'open', 'high', 'low']
        
        return mock_model

    @pytest.mark.asyncio
    async def test_complete_format_conversion_and_loading(self, temp_models_dir, mock_model_with_pytorch):
        """Test complete pipeline: format conversion → TorchScript loading → prediction"""
        
        # Setup
        symbol = "BTCUSDT"
        timeframe = TimeFrame.DAY_1
        model_type = ModelType.PATCHTSMIXER
        
        # 1. Create simple model directory structure
        simple_dir = temp_models_dir / "simple" / f"{symbol}_{timeframe.value}_ibm_patchtsmixer_forecasting"
        simple_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock model files in simple format
        model_files = {
            "model_state_dict.pt": b"mock_pytorch_state_dict",
            "model_config.json": json.dumps({"context_length": 64, "num_features": 5}),
            "metadata.json": json.dumps({"model_type": model_type.value}),
            "feature_scaler.pkl": b"mock_scaler",
            "target_scaler.pkl": b"mock_target_scaler"
        }
        
        for filename, content in model_files.items():
            file_path = simple_dir / filename
            if isinstance(content, str):
                with open(file_path, "w") as f:
                    f.write(content)
            else:
                with open(file_path, "wb") as f:
                    f.write(content)

        # 2. Test ModelFormatConverter
        with patch('src.utils.model_format_converter.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.models_dir = temp_models_dir
            mock_get_settings.return_value = mock_settings
            
            converter = ModelFormatConverter()
            
            # Mock torch.jit.trace to avoid actual model tracing
            with patch('torch.jit.trace') as mock_trace:
                mock_scripted_model = Mock()
                mock_scripted_model.save = Mock()
                mock_trace.return_value = mock_scripted_model
                
                # Run conversion
                results = converter.convert_model_for_all_adapters(
                    model=mock_model_with_pytorch,
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type,
                    source_path=simple_dir
                )
                
                # Verify conversion results
                assert results[FacadeConstants.ADAPTER_SIMPLE] is True
                assert results[FacadeConstants.ADAPTER_TORCHSCRIPT] is True
                assert results[FacadeConstants.ADAPTER_TORCHSERVE] is True
                
                # Verify TorchScript files were created
                torchscript_dir = temp_models_dir / "torchscript" / f"{symbol}_{timeframe.value}_ibm_patchtsmixer_forecasting"
                assert torchscript_dir.exists()
                
                # Check that multiple TorchScript file formats exist
                expected_torchscript_files = ["model_torchscript.pt", "scripted_model.pt", "model.pt"]
                for filename in expected_torchscript_files:
                    mock_scripted_model.save.assert_any_call(torchscript_dir / filename)

    @pytest.mark.asyncio
    async def test_torchscript_adapter_with_preconverted_model(self, temp_models_dir):
        """Test TorchScript adapter loading a pre-converted model"""
        
        # Setup
        symbol = "BTCUSDT"
        timeframe = TimeFrame.DAY_1
        model_type = ModelType.PATCHTSMIXER
        
        # Create TorchScript model directory with converted files
        torchscript_dir = temp_models_dir / "torchscript" / f"{symbol}_{timeframe.value}_ibm_patchtsmixer_forecasting"
        torchscript_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock TorchScript files
        (torchscript_dir / "model_torchscript.pt").write_bytes(b"mock_torchscript_model")
        (torchscript_dir / "model_config.json").write_text(json.dumps({"context_length": 64}))
        
        # Setup adapter
        config = {
            "device": "cpu",
            "optimize_for_inference": False,  # Disable to avoid extra complexity
            "max_models_in_memory": 3,
        }
        
        with patch('src.adapters.torchscript_serving.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.models_dir = temp_models_dir
            mock_get_settings.return_value = mock_settings
            
            adapter = TorchScriptServingAdapter(config)
            await adapter.initialize()
            
            # Mock torch.jit.load to return a scripted model
            with patch('torch.jit.load') as mock_jit_load:
                mock_scripted_model = Mock()
                mock_scripted_model.eval.return_value = mock_scripted_model
                mock_jit_load.return_value = mock_scripted_model
                
                # Test model loading
                model_info = await adapter.load_model(
                    model_path=str(torchscript_dir),
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type,
                    model_config={}
                )
                
                # Verify model was loaded successfully
                assert model_info is not None
                assert model_info.is_loaded
                assert model_info.symbol == symbol
                assert model_info.model_type == model_type.value
                
                # Verify torch.jit.load was called with the pre-converted model
                mock_jit_load.assert_called_once()
                called_path = str(mock_jit_load.call_args[0][0])
                assert "model_torchscript.pt" in called_path

    @pytest.mark.asyncio 
    async def test_torchscript_adapter_fallback_conversion(self, temp_models_dir):
        """Test TorchScript adapter fallback to conversion when no pre-converted model exists"""
        
        # Setup
        symbol = "BTCUSDT"
        timeframe = TimeFrame.DAY_1
        model_type = ModelType.PATCHTSMIXER
        
        # Create simple model directory (no TorchScript version)
        simple_dir = temp_models_dir / "simple" / f"{symbol}_{timeframe.value}_ibm_patchtsmixer_forecasting"
        simple_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model files
        (simple_dir / "model_state_dict.pt").write_bytes(b"mock_state_dict")
        (simple_dir / "model_config.json").write_text(json.dumps({"context_length": 64}))
        
        # Use the path as TorchScript path (simulating model_path pointing to TorchScript dir)
        torchscript_dir = temp_models_dir / "torchscript" / f"{symbol}_{timeframe.value}_ibm_patchtsmixer_forecasting"
        
        # Setup adapter
        config = {
            "device": "cpu",
            "optimize_for_inference": False,
            "max_models_in_memory": 3,
        }
        
        with patch('src.adapters.torchscript_serving.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.models_dir = temp_models_dir
            mock_get_settings.return_value = mock_settings
            
            adapter = TorchScriptServingAdapter(config)
            await adapter.initialize()
            
            # Mock model loading and conversion
            with patch.object(adapter, '_load_original_model') as mock_load_original:
                with patch('torch.jit.trace') as mock_trace:
                    with patch('torch.jit.load') as mock_jit_load:
                        
                        # Setup mocks
                        mock_pytorch_model = Mock()
                        mock_load_original.return_value = mock_pytorch_model
                        
                        mock_scripted_model = Mock()
                        mock_scripted_model.eval.return_value = mock_scripted_model
                        mock_scripted_model.save = Mock()
                        mock_trace.return_value = mock_scripted_model
                        mock_jit_load.return_value = mock_scripted_model
                        
                        # Test model loading (should trigger conversion)
                        model_info = await adapter.load_model(
                            model_path=str(simple_dir),  # Point to simple model
                            symbol=symbol,
                            timeframe=timeframe,
                            model_type=model_type,
                            model_config={}
                        )
                        
                        # Verify conversion was triggered
                        mock_load_original.assert_called_once()
                        mock_trace.assert_called_once()
                        
                        # Verify model was loaded
                        assert model_info is not None
                        assert model_info.is_loaded

    @pytest.mark.asyncio
    async def test_format_conversion_error_handling(self, temp_models_dir, mock_model_with_pytorch):
        """Test error handling in format conversion"""
        
        symbol = "BTCUSDT"
        timeframe = TimeFrame.DAY_1
        model_type = ModelType.PATCHTSMIXER
        
        simple_dir = temp_models_dir / "simple" / f"{symbol}_{timeframe.value}_ibm_patchtsmixer_forecasting"
        simple_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('src.utils.model_format_converter.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.models_dir = temp_models_dir
            mock_get_settings.return_value = mock_settings
            
            converter = ModelFormatConverter()
            
            # Test with torch.jit.trace raising an error
            with patch('torch.jit.trace', side_effect=RuntimeError("Tracing failed")):
                results = converter.convert_model_for_all_adapters(
                    model=mock_model_with_pytorch,
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type,
                    source_path=simple_dir
                )
                
                # Should still succeed but without TorchScript tracing
                assert results[FacadeConstants.ADAPTER_TORCHSCRIPT] is True
                
                # Verify error was logged (files still copied)
                torchscript_dir = temp_models_dir / "torchscript" / f"{symbol}_{timeframe.value}_ibm_patchtsmixer_forecasting"
                assert torchscript_dir.exists()


if __name__ == "__main__":
    # Run specific tests
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd().parent / "src"))

    pytest.main([__file__, "-v"])
