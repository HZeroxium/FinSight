# routers/serving.py

"""
FastAPI router for model serving management.

This router provides endpoints for managing the model serving
adapter and monitoring serving performance.
"""

import asyncio
from typing import Any, Dict, List, Optional

from common.logger.logger_factory import LoggerFactory
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ..adapters import ServingAdapterFactory
from ..core.config import get_settings
from ..facades import EnhancedModelFacade
from ..schemas.enums import ModelType, TimeFrame

# Create router
router = APIRouter(prefix="/serving", tags=["Model Serving"])

# Logger
logger = LoggerFactory.get_logger("ServingRouter")

# Global facade instance
_facade: Optional[EnhancedModelFacade] = None


async def get_facade() -> EnhancedModelFacade:
    """Get or create the enhanced model facade instance"""
    global _facade
    if _facade is None:
        _facade = EnhancedModelFacade()
        await _facade.initialize()
    elif not _facade._adapter_initialized:
        await _facade.initialize()
    return _facade


@router.get("/health", summary="Get serving adapter health status")
async def get_serving_health(facade: EnhancedModelFacade = Depends(get_facade)):
    """
    Get the health status of the model serving adapter.

    Returns:
        Dict containing health status and adapter information
    """
    try:
        health_status = await facade.get_serving_health()
        return {"success": True, "health": health_status}
    except Exception as e:
        logger.error(f"Failed to get serving health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", summary="Get serving adapter statistics")
async def get_serving_stats(facade: EnhancedModelFacade = Depends(get_facade)):
    """
    Get statistics from the model serving adapter.

    Returns:
        Dict containing serving statistics
    """
    try:
        stats = await facade.get_serving_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.error(f"Failed to get serving stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", summary="List loaded models in serving adapter")
async def list_serving_models(facade: EnhancedModelFacade = Depends(get_facade)):
    """
    List all models currently loaded in the serving adapter.

    Returns:
        List of loaded model information
    """
    try:
        models = await facade.list_serving_models()
        return {
            "success": True,
            "models": [
                {
                    "model_id": model.model_id,
                    "symbol": model.symbol,
                    "timeframe": model.timeframe,
                    "model_type": model.model_type,
                    "is_loaded": model.is_loaded,
                    "loaded_at": (
                        model.loaded_at.isoformat() if model.loaded_at else None
                    ),
                    "memory_usage_mb": model.memory_usage_mb,
                    "version": model.version,
                }
                for model in models
            ],
            "total_models": len(models),
        }
    except Exception as e:
        logger.error(f"Failed to list serving models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/load", summary="Load a model into serving adapter")
async def load_model_to_serving(
    symbol: str,
    timeframe: TimeFrame,
    model_type: ModelType,
    background_tasks: BackgroundTasks,
    facade: EnhancedModelFacade = Depends(get_facade),
):
    """
    Load a specific model into the serving adapter.

    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        model_type: Model type

    Returns:
        Success status and model information
    """
    try:
        # Check if model exists on disk
        if not facade.model_exists(symbol, timeframe, model_type):
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {symbol} {timeframe.value} {model_type.value}",
            )

        # Load model in background
        success = await facade.load_model_to_serving(symbol, timeframe, model_type)

        if success:
            # Get model info from serving adapter
            model_id = facade.serving_adapter.generate_model_id(
                symbol, timeframe, model_type
            )
            model_info = await facade.serving_adapter.get_model_info(model_id)

            return {
                "success": True,
                "message": f"Model loaded successfully: {symbol} {timeframe.value} {model_type.value}",
                "model_info": (
                    {
                        "model_id": model_info.model_id,
                        "symbol": model_info.symbol,
                        "timeframe": model_info.timeframe,
                        "model_type": model_info.model_type,
                        "memory_usage_mb": model_info.memory_usage_mb,
                        "loaded_at": (
                            model_info.loaded_at.isoformat()
                            if model_info.loaded_at
                            else None
                        ),
                    }
                    if model_info
                    else None
                ),
            }
        else:
            raise HTTPException(
                status_code=500, detail="Failed to load model into serving adapter"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load model to serving: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/unload", summary="Unload a model from serving adapter")
async def unload_model_from_serving(
    symbol: str,
    timeframe: TimeFrame,
    model_type: ModelType,
    facade: EnhancedModelFacade = Depends(get_facade),
):
    """
    Unload a specific model from the serving adapter.

    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        model_type: Model type

    Returns:
        Success status
    """
    try:
        success = await facade.unload_model_from_serving(symbol, timeframe, model_type)

        if success:
            return {
                "success": True,
                "message": f"Model unloaded successfully: {symbol} {timeframe.value} {model_type.value}",
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found in serving adapter: {symbol} {timeframe.value} {model_type.value}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unload model from serving: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/predict", summary="Make prediction using serving adapter")
async def predict_with_serving(
    symbol: str,
    timeframe: TimeFrame,
    model_type: ModelType,
    input_data: Dict[str, Any],
    n_steps: int = 1,
    facade: EnhancedModelFacade = Depends(get_facade),
):
    """
    Make predictions using the serving adapter.

    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        model_type: Model type
        input_data: Input data for prediction
        n_steps: Number of prediction steps

    Returns:
        Prediction results
    """
    try:
        # Validate input data
        if not input_data:
            raise HTTPException(status_code=400, detail="Input data is required")

        # Make prediction using serving adapter
        result = await facade.predict(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            recent_data=input_data,
            n_steps=n_steps,
            use_serving_adapter=True,
        )

        if result.get("success", False):
            return {"success": True, "result": result}
        else:
            raise HTTPException(
                status_code=500, detail=result.get("error", "Prediction failed")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to make prediction with serving: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/adapters", summary="Get information about available serving adapters")
async def get_available_adapters():
    """
    Get information about all available serving adapters.

    Returns:
        Information about supported adapters and their requirements
    """
    try:
        adapters = ServingAdapterFactory.get_supported_adapters()

        # Check requirements for each adapter
        adapter_info = {}
        for adapter_type, info in adapters.items():
            requirements = ServingAdapterFactory.get_adapter_requirements(adapter_type)
            adapter_info[adapter_type] = {**info, "requirements": requirements}

        return {"success": True, "adapters": adapter_info}

    except Exception as e:
        logger.error(f"Failed to get adapter information: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config", summary="Get current serving configuration")
async def get_serving_config():
    """
    Get the current serving adapter configuration.

    Returns:
        Current serving configuration
    """
    try:
        settings = get_settings()
        serving_config = settings.serving

        return {"success": True, "config": serving_config}

    except Exception as e:
        logger.error(f"Failed to get serving config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/validate", summary="Validate serving adapter configuration")
async def validate_serving_config(adapter_type: str, config: Dict[str, Any]):
    """
    Validate a serving adapter configuration.

    Args:
        adapter_type: Type of adapter to validate
        config: Configuration to validate

    Returns:
        Validation results
    """
    try:
        validated_config = ServingAdapterFactory.validate_adapter_config(
            adapter_type, config
        )

        return {
            "success": True,
            "message": "Configuration is valid",
            "validated_config": validated_config,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to validate config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/warmup", summary="Warm-up serving adapter with common models")
async def warmup_serving(
    models: Optional[List[Dict[str, str]]] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    facade: EnhancedModelFacade = Depends(get_facade),
):
    """
    Warm-up the serving adapter by loading commonly used models.

    Args:
        models: Optional list of models to load. Each model should have
                'symbol', 'timeframe', and 'model_type' keys.
                If not provided, will load the most recently trained models.

    Returns:
        Warm-up status
    """
    try:
        if models is None:
            # Get recently trained models
            available_models = facade.list_available_models()
            # Sort by creation date and take the 3 most recent
            available_models.sort(key=lambda x: x.created_at, reverse=True)
            models = [
                {
                    "symbol": model.symbol,
                    "timeframe": model.timeframe.value,
                    "model_type": model.model_type.value,
                }
                for model in available_models[:3]
            ]

        async def warmup_task():
            """Background warmup task"""
            loaded_count = 0
            failed_count = 0

            for model_spec in models:
                try:
                    symbol = model_spec["symbol"]
                    timeframe = TimeFrame(model_spec["timeframe"])
                    model_type = ModelType(model_spec["model_type"])

                    success = await facade.load_model_to_serving(
                        symbol, timeframe, model_type
                    )
                    if success:
                        loaded_count += 1
                        logger.info(
                            f"Warmed up model: {symbol} {timeframe.value} {model_type.value}"
                        )
                    else:
                        failed_count += 1
                        logger.warning(
                            f"Failed to warm up model: {symbol} {timeframe.value} {model_type.value}"
                        )

                except Exception as e:
                    failed_count += 1
                    logger.error(f"Error warming up model {model_spec}: {e}")

            logger.info(
                f"Warmup completed: {loaded_count} loaded, {failed_count} failed"
            )

        # Start warmup in background
        background_tasks.add_task(warmup_task)

        return {
            "success": True,
            "message": f"Started warmup process for {len(models)} models",
            "models_to_load": models,
        }

    except Exception as e:
        logger.error(f"Failed to start warmup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmark", summary="Benchmark serving adapter performance")
async def benchmark_serving(
    symbol: str,
    timeframe: TimeFrame,
    model_type: ModelType,
    num_requests: int = 10,
    concurrent_requests: int = 1,
    facade: EnhancedModelFacade = Depends(get_facade),
):
    """
    Benchmark the serving adapter with a specific model.

    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        model_type: Model type
        num_requests: Total number of requests to make
        concurrent_requests: Number of concurrent requests

    Returns:
        Benchmark results
    """
    try:
        if num_requests > 100:
            raise HTTPException(
                status_code=400, detail="Maximum 100 requests allowed for benchmarking"
            )

        if concurrent_requests > 10:
            raise HTTPException(
                status_code=400, detail="Maximum 10 concurrent requests allowed"
            )

        # Sample input data (simplified)
        sample_input = {
            "close": [100.0, 101.0, 102.0, 101.5, 103.0] * 13  # 65 data points
        }

        # Ensure model is loaded
        model_id = facade.serving_adapter.generate_model_id(
            symbol, timeframe, model_type
        )
        model_info = await facade.serving_adapter.get_model_info(model_id)
        if model_info is None:
            success = await facade.load_model_to_serving(symbol, timeframe, model_type)
            if not success:
                raise HTTPException(
                    status_code=404, detail="Failed to load model for benchmarking"
                )

        # Run benchmark
        import time

        async def make_request():
            """Single benchmark request"""
            start_time = time.time()
            result = await facade.predict(
                symbol=symbol,
                timeframe=timeframe,
                model_type=model_type,
                recent_data=sample_input,
                n_steps=1,
                use_serving_adapter=True,
            )
            end_time = time.time()

            return {
                "success": result.get("success", False),
                "inference_time_ms": (end_time - start_time) * 1000,
                "error": result.get("error"),
            }

        # Execute benchmark
        start_time = time.time()

        results = []
        for batch_start in range(0, num_requests, concurrent_requests):
            batch_size = min(concurrent_requests, num_requests - batch_start)
            batch_tasks = [make_request() for _ in range(batch_size)]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    results.append(
                        {"success": False, "inference_time_ms": 0, "error": str(result)}
                    )
                else:
                    results.append(result)

        end_time = time.time()

        # Calculate statistics
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]

        if successful_requests:
            inference_times = [r["inference_time_ms"] for r in successful_requests]
            avg_inference_time = sum(inference_times) / len(inference_times)
            min_inference_time = min(inference_times)
            max_inference_time = max(inference_times)
        else:
            avg_inference_time = min_inference_time = max_inference_time = 0

        total_time = (end_time - start_time) * 1000  # Convert to milliseconds
        requests_per_second = num_requests / ((end_time - start_time) or 1)

        return {
            "success": True,
            "benchmark_results": {
                "total_requests": num_requests,
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "success_rate": len(successful_requests) / num_requests * 100,
                "total_time_ms": total_time,
                "requests_per_second": requests_per_second,
                "average_inference_time_ms": avg_inference_time,
                "min_inference_time_ms": min_inference_time,
                "max_inference_time_ms": max_inference_time,
                "concurrent_requests": concurrent_requests,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to run benchmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))
