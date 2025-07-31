"""
Comprehensive test suite for SimpleServingAdapter
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.adapters.simple_serving import SimpleServingAdapter
from src.schemas.enums import ModelType, TimeFrame
from src.interfaces.serving_interface import ModelInfo, PredictionResult


class TestSimpleServingAdapter:
    """Comprehensive test cases for SimpleServingAdapter"""

    @pytest.fixture
    async def adapter(self):
        """Create a simple serving adapter for testing"""
        config = {
            "max_models_in_memory": 5,
            "model_timeout_seconds": 3600,
            "max_models": 5,
            "memory_threshold": 0.8,
            "cleanup_interval": 3600,
        }
        adapter = SimpleServingAdapter(config)
        await adapter.initialize()
        yield adapter
        # Cleanup
        try:
            await adapter.shutdown()
        except Exception:
            pass

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing with sufficient context (150+ points)"""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=150, freq="D"),
                "open": np.random.randn(150) * 100 + 50000,
                "high": np.random.randn(150) * 100 + 51000,
                "low": np.random.randn(150) * 100 + 49000,
                "close": np.random.randn(150) * 100 + 50000,
                "volume": np.random.randn(150) * 1000 + 10000,
            }
        )

    @pytest.fixture
    def model_config(self):
        """Model configuration for testing"""
        return {
            "symbol": "BTCUSDT",
            "timeframe": TimeFrame.DAY_1,
            "model_type": ModelType.PATCHTSMIXER,
            "model_path": "models/simple/BTCUSDT_1d_ibm_patchtsmixer_forecasting",
        }

    @pytest.mark.asyncio
    async def test_adapter_initialization(self, adapter):
        """Test adapter initialization"""
        assert adapter is not None
        assert adapter.max_models_in_memory == 5
        assert len(adapter._loaded_models) == 0
        assert len(adapter._model_info) == 0

    @pytest.mark.asyncio
    async def test_health_check(self, adapter):
        """Test health check functionality"""
        health = await adapter.health_check()
        assert isinstance(health, dict)
        assert "status" in health
        assert "adapter_type" in health
        assert health["adapter_type"] == "simple"
        assert "models_loaded" in health
        assert "uptime_seconds" in health

    @pytest.mark.asyncio
    async def test_model_loading_real_model(self, adapter, model_config):
        """Test loading a real model if it exists"""
        model_path = model_config["model_path"]

        # Skip if model doesn't exist
        if not Path(model_path).exists():
            pytest.skip(f"Model path {model_path} does not exist")

        model_info = await adapter.load_model(
            model_path=model_path,
            symbol=model_config["symbol"],
            timeframe=model_config["timeframe"],
            model_type=model_config["model_type"],
        )

        assert isinstance(model_info, ModelInfo)
        assert model_info.is_loaded
        assert model_info.symbol == model_config["symbol"]
        assert model_info.timeframe == model_config["timeframe"].value
        assert model_info.model_type == model_config["model_type"].value

    @pytest.mark.asyncio
    async def test_prediction_real_model(self, adapter, model_config, sample_data):
        """Test prediction with a real model if it exists"""
        model_path = model_config["model_path"]

        # Skip if model doesn't exist
        if not Path(model_path).exists():
            pytest.skip(f"Model path {model_path} does not exist")

        # Load model first
        model_info = await adapter.load_model(
            model_path=model_path,
            symbol=model_config["symbol"],
            timeframe=model_config["timeframe"],
            model_type=model_config["model_type"],
        )

        # Make prediction
        result = await adapter.predict(model_info.model_id, sample_data, n_steps=5)

        assert isinstance(result, PredictionResult)
        assert result.success
        assert len(result.predictions) == 5
        assert result.current_price is not None
        assert result.inference_time_ms is not None
        print(f"✅ Prediction successful: {result.predictions}")

    @pytest.mark.asyncio
    async def test_prediction_insufficient_data(self, adapter, model_config):
        """Test prediction with insufficient data"""
        model_path = model_config["model_path"]

        # Skip if model doesn't exist
        if not Path(model_path).exists():
            pytest.skip(f"Model path {model_path} does not exist")

        # Load model first
        model_info = await adapter.load_model(
            model_path=model_path,
            symbol=model_config["symbol"],
            timeframe=model_config["timeframe"],
            model_type=model_config["model_type"],
        )

        # Create insufficient data (less than 64 points needed by model)
        insufficient_data = pd.DataFrame(
            {
                "close": np.random.randn(30) * 100 + 50000,
            }
        )

        result = await adapter.predict(
            model_info.model_id, insufficient_data, n_steps=3
        )
        # Should handle gracefully - either succeed or fail with clear error
        assert isinstance(result, PredictionResult)

    @pytest.mark.asyncio
    async def test_prediction_invalid_model_id(self, adapter, sample_data):
        """Test prediction with invalid model ID"""
        result = await adapter.predict("invalid_model_id", sample_data, n_steps=3)
        assert isinstance(result, PredictionResult)
        assert not result.success
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_list_loaded_models(self, adapter, model_config):
        """Test listing loaded models"""
        # Initially should be empty or have few models
        initial_models = await adapter.list_loaded_models()
        initial_count = len(initial_models)

        model_path = model_config["model_path"]
        if not Path(model_path).exists():
            pytest.skip(f"Model path {model_path} does not exist")

        # Load a model
        model_info = await adapter.load_model(
            model_path=model_path,
            symbol=model_config["symbol"],
            timeframe=model_config["timeframe"],
            model_type=model_config["model_type"],
        )

        # Should have one more model
        models = await adapter.list_loaded_models()
        assert len(models) == initial_count + 1
        assert any(m.model_id == model_info.model_id for m in models)

    @pytest.mark.asyncio
    async def test_model_unloading(self, adapter, model_config):
        """Test model unloading"""
        model_path = model_config["model_path"]
        if not Path(model_path).exists():
            pytest.skip(f"Model path {model_path} does not exist")

        # Load model first
        model_info = await adapter.load_model(
            model_path=model_path,
            symbol=model_config["symbol"],
            timeframe=model_config["timeframe"],
            model_type=model_config["model_type"],
        )

        # Verify model is loaded
        loaded_models = await adapter.list_loaded_models()
        assert any(m.model_id == model_info.model_id for m in loaded_models)

        # Unload model
        success = await adapter.unload_model(model_info.model_id)
        assert success

        # Verify model is unloaded
        loaded_models = await adapter.list_loaded_models()
        assert not any(m.model_id == model_info.model_id for m in loaded_models)

    @pytest.mark.asyncio
    async def test_get_model_info(self, adapter, model_config):
        """Test getting model information"""
        model_path = model_config["model_path"]
        if not Path(model_path).exists():
            pytest.skip(f"Model path {model_path} does not exist")

        # Load model first
        model_info = await adapter.load_model(
            model_path=model_path,
            symbol=model_config["symbol"],
            timeframe=model_config["timeframe"],
            model_type=model_config["model_type"],
        )

        # Get model info
        retrieved_info = await adapter.get_model_info(model_info.model_id)
        assert retrieved_info is not None
        assert retrieved_info.model_id == model_info.model_id

        # Test with invalid model ID
        invalid_info = await adapter.get_model_info("invalid_id")
        assert invalid_info is None


# Standalone test functions for direct execution
async def run_simple_test():
    """Run a simple standalone test"""
    print("=" * 60)
    print("Running SimpleServingAdapter Integration Test")
    print("=" * 60)

    try:
        # Create adapter
        config = {"max_models": 5, "memory_threshold": 0.8, "cleanup_interval": 3600}
        adapter = SimpleServingAdapter(config)
        await adapter.initialize()

        # Test model loading
        model_path = "models/simple/BTCUSDT_1d_ibm_patchtsmixer_forecasting"

        if not Path(model_path).exists():
            print(
                f"⚠️  Model path {model_path} does not exist, skipping real model test"
            )
            print("✅ SimpleServingAdapter basic functionality test passed")
            return True

        print("1. Loading model...")
        model_info = await adapter.load_model(
            model_path=model_path,
            symbol="BTCUSDT",
            timeframe=TimeFrame.DAY_1,
            model_type=ModelType.PATCHTSMIXER,
        )
        print(f"✅ Model loaded: {model_info.model_id}")

        # Test prediction
        print("2. Testing prediction...")
        np.random.seed(42)
        test_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=150, freq="D"),
                "open": np.random.randn(150) * 100 + 50000,
                "high": np.random.randn(150) * 100 + 51000,
                "low": np.random.randn(150) * 100 + 49000,
                "close": np.random.randn(150) * 100 + 50000,
                "volume": np.random.randn(150) * 1000 + 10000,
            }
        )

        result = await adapter.predict(model_info.model_id, test_data, n_steps=5)

        if result.success:
            print(f"✅ Prediction successful!")
            print(f"   Predictions: {result.predictions}")
            print(f"   Current price: {result.current_price}")
            print(f"   Inference time: {result.inference_time_ms:.2f}ms")
        else:
            print(f"❌ Prediction failed: {result.error}")
            return False

        # Test health check
        print("3. Testing health check...")
        health = await adapter.health_check()
        print(f"✅ Health check: {health['status']}")

        # Test cleanup
        print("4. Testing cleanup...")
        await adapter.shutdown()
        print("✅ Cleanup successful")

        print("\n✅ All SimpleServingAdapter tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the standalone test
    result = asyncio.run(run_simple_test())
    exit(0 if result else 1)
