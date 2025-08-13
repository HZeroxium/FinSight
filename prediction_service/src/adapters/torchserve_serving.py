# adapters/torchserve_serving.py

"""
TorchServe adapter for PyTorch model serving.

This adapter integrates with PyTorch TorchServe to provide scalable
PyTorch model serving with features like dynamic batching,
multi-model management, and monitoring.
"""

import asyncio
import time
import json
import aiohttp
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from ..interfaces.serving_interface import (
    IModelServingAdapter,
    ModelInfo,
    PredictionResult,
    ServingStats,
)
from ..schemas.enums import ModelType, TimeFrame
from ..core.config import get_settings
from common.logger.logger_factory import LoggerFactory


class TorchServeAdapter(IModelServingAdapter):
    """
    TorchServe adapter for PyTorch model serving.

    This adapter provides enterprise-grade PyTorch model serving using
    TorchServe with features like:
    - Dynamic batching
    - Model versioning
    - Auto-scaling
    - Metrics and logging
    - RESTful API
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TorchServe adapter

        Args:
            config: Configuration dictionary with TorchServe server details
        """
        super().__init__(config)
        self.settings = get_settings()

        # TorchServe server configuration
        self.inference_url = config.get("inference_url", "http://localhost:8080")
        self.management_url = config.get("management_url", "http://localhost:8081")
        self.metrics_url = config.get("metrics_url", "http://localhost:8082")

        # Model configuration
        self.model_store = config.get("model_store", "./model_store")
        self.batch_size = config.get("batch_size", 1)
        self.max_batch_delay = config.get("max_batch_delay", 100)
        self.timeout_seconds = config.get("timeout_seconds", 30)
        self.initial_workers = config.get("initial_workers", 1)
        self.max_workers = config.get("max_workers", 4)

        # HTTP session for API calls
        self.session: Optional[aiohttp.ClientSession] = None
        self._model_configs: Dict[str, Dict[str, Any]] = {}

        self.logger.info(
            f"TorchServeAdapter initialized for server: {self.inference_url}"
        )

    async def initialize(self) -> bool:
        """
        Initialize connection to TorchServe server

        Returns:
            bool: True if initialization successful
        """
        try:
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            self.session = aiohttp.ClientSession(timeout=timeout)

            # Test connection to TorchServe
            async with self.session.get(f"{self.management_url}/ping") as response:
                if response.status != 200:
                    raise ConnectionError(f"TorchServe ping failed: {response.status}")

            # Create model store directory
            Path(self.model_store).mkdir(exist_ok=True, parents=True)

            self.logger.info("Successfully connected to TorchServe")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize TorchServe client: {e}")
            if self.session:
                await self.session.close()
            return False

    async def load_model(
        self,
        model_path: str,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> ModelInfo:
        """
        Load a model into TorchServe

        Args:
            model_path: Path to the model files
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Type of model
            model_config: Optional model configuration

        Returns:
            ModelInfo: Information about the loaded model
        """
        model_id = self.generate_model_id(symbol, timeframe, model_type)

        try:
            self.logger.info(f"Loading model {model_id} into TorchServe")

            # Create TorchServe model archive (.mar file)
            mar_file_path = await self._create_model_archive(
                model_path, model_id, model_type, model_config
            )

            # Register model with TorchServe
            torchserve_model_name = await self._register_model(
                mar_file_path, model_id, model_config
            )

            # Start model workers
            await self._start_model_workers(torchserve_model_name, self.initial_workers)

            # Get model status and metadata
            model_status = await self._get_model_status(torchserve_model_name)
            memory_usage = await self._estimate_model_memory(torchserve_model_name)

            model_info = ModelInfo(
                model_id=model_id,
                symbol=symbol,
                timeframe=timeframe.value,
                model_type=model_type.value,
                model_path=model_path,
                is_loaded=True,
                loaded_at=datetime.now(),
                memory_usage_mb=memory_usage,
                metadata={
                    "torchserve_model_name": torchserve_model_name,
                    "mar_file_path": mar_file_path,
                    "model_status": model_status,
                    "workers": self.initial_workers,
                    **(model_config or {}),
                },
            )

            # Store model configuration
            self._model_configs[model_id] = {
                "torchserve_model_name": torchserve_model_name,
                "model_info": model_info,
                "mar_file_path": mar_file_path,
            }

            # Update stats
            self._stats.models_loaded += 1
            self._stats.total_memory_usage_mb += memory_usage

            self.logger.info(f"Model {model_id} loaded successfully in TorchServe")
            return model_info

        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            raise

    async def predict(
        self,
        model_id: str,
        input_data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
        n_steps: int = 1,
        **kwargs,
    ) -> PredictionResult:
        """
        Make predictions using TorchServe

        Args:
            model_id: Identifier of the loaded model
            input_data: Input data for prediction
            n_steps: Number of prediction steps
            **kwargs: Additional prediction parameters

        Returns:
            PredictionResult: Prediction results
        """
        start_time = time.time()

        try:
            self.logger.debug(f"Making prediction with model {model_id}")

            # Check if model is loaded
            if model_id not in self._model_configs:
                error_msg = f"Model {model_id} not loaded in TorchServe"
                self.logger.error(error_msg)
                self._update_stats(success=False)
                return PredictionResult(success=False, predictions=[], error=error_msg)

            model_config = self._model_configs[model_id]
            torchserve_model_name = model_config["torchserve_model_name"]

            # Prepare input data for TorchServe
            payload = await self._prepare_torchserve_input(
                input_data, n_steps, **kwargs
            )

            # Make inference request
            prediction_url = f"{self.inference_url}/predictions/{torchserve_model_name}"

            async with self.session.post(
                prediction_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:

                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"TorchServe inference failed: {response.status} - {error_text}"
                    )

                response_data = await response.json()

            # Process response
            result = await self._process_torchserve_response(
                response_data, input_data, model_config
            )

            inference_time_ms = (time.time() - start_time) * 1000

            # Update stats
            self._update_stats(success=True, inference_time_ms=inference_time_ms)

            return PredictionResult(
                success=True,
                predictions=result["predictions"],
                current_price=result.get("current_price"),
                predicted_change_pct=result.get("predicted_change_pct"),
                confidence_score=result.get("confidence_score", 0.85),
                model_info={
                    "model_id": model_id,
                    "torchserve_model_name": torchserve_model_name,
                    "model_type": model_config["model_info"].model_type,
                    "version": "1.0",
                },
                inference_time_ms=inference_time_ms,
            )

        except Exception as e:
            inference_time_ms = (time.time() - start_time) * 1000
            self.logger.error(f"TorchServe prediction failed for model {model_id}: {e}")
            self._update_stats(success=False, inference_time_ms=inference_time_ms)

            return PredictionResult(
                success=False,
                predictions=[],
                error=str(e),
                inference_time_ms=inference_time_ms,
            )

    async def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from TorchServe

        Args:
            model_id: Identifier of the model to unload

        Returns:
            bool: True if unload successful
        """
        try:
            if model_id in self._model_configs:
                torchserve_model_name = self._model_configs[model_id][
                    "torchserve_model_name"
                ]

                # Stop model workers
                await self._stop_model_workers(torchserve_model_name)

                # Unregister model
                await self._unregister_model(torchserve_model_name)

                # Get memory usage before removal
                model_info = self._model_configs[model_id]["model_info"]
                memory_usage = model_info.memory_usage_mb or 0.0

                # Remove from local cache
                del self._model_configs[model_id]

                # Update stats
                self._stats.models_loaded -= 1
                self._stats.total_memory_usage_mb -= memory_usage

                self.logger.info(f"Model {model_id} unloaded from TorchServe")
                return True
            else:
                self.logger.warning(f"Model {model_id} not found for unloading")
                return False

        except Exception as e:
            self.logger.error(f"Failed to unload model {model_id}: {e}")
            return False

    async def list_loaded_models(self) -> List[ModelInfo]:
        """
        List all models loaded in TorchServe

        Returns:
            List[ModelInfo]: List of loaded model information
        """
        try:
            # Get models from TorchServe management API
            async with self.session.get(f"{self.management_url}/models") as response:
                if response.status == 200:
                    torchserve_models = await response.json()
                else:
                    torchserve_models = {}

            loaded_models = []

            for model_config in self._model_configs.values():
                model_info = model_config["model_info"]
                torchserve_name = model_config["torchserve_model_name"]

                # Check if model is still loaded in TorchServe
                if torchserve_name in torchserve_models.get("models", []):
                    # Update model status
                    status = await self._get_model_status(torchserve_name)
                    if status.get("workers", []):
                        loaded_models.append(model_info)

            return loaded_models

        except Exception as e:
            self.logger.error(f"Failed to list loaded models: {e}")
            return []

    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model

        Args:
            model_id: Model identifier

        Returns:
            Optional[ModelInfo]: Model information if found
        """
        if model_id in self._model_configs:
            return self._model_configs[model_id]["model_info"]
        return None

    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of TorchServe

        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            # Check server status
            server_healthy = False
            server_version = "unknown"

            try:
                async with self.session.get(f"{self.management_url}/ping") as response:
                    server_healthy = response.status == 200

                # Get server status
                async with self.session.get(
                    f"{self.management_url}/models"
                ) as response:
                    if response.status == 200:
                        server_status = await response.json()
                    else:
                        server_status = {}

            except Exception as e:
                self.logger.warning(f"TorchServe health check failed: {e}")
                server_status = {}

            # Get metrics if available
            total_requests = 0
            try:
                async with self.session.get(f"{self.metrics_url}/metrics") as response:
                    if response.status == 200:
                        metrics_text = await response.text()
                        # Parse Prometheus metrics (simplified)
                        for line in metrics_text.split("\n"):
                            if line.startswith("ts_inference_requests_total"):
                                total_requests += float(line.split()[-1])
            except Exception:
                pass

            return {
                "status": "healthy" if server_healthy else "unhealthy",
                "adapter_type": "torchserve",
                "server_healthy": server_healthy,
                "server_version": server_version,
                "models_loaded": len(self._model_configs),
                "torchserve_models": len(server_status.get("models", [])),
                "total_memory_usage_mb": self._stats.total_memory_usage_mb,
                "total_server_requests": total_requests,
                "uptime_seconds": self.get_stats().uptime_seconds,
                "total_predictions": self._stats.total_predictions,
                "success_rate": self._stats.get_success_rate(),
            }

        except Exception as e:
            self.logger.error(f"TorchServe health check failed: {e}")
            return {
                "status": "unhealthy",
                "adapter_type": "torchserve",
                "error": str(e),
            }

    async def shutdown(self) -> bool:
        """
        Shutdown the TorchServe serving backend gracefully

        Returns:
            bool: True if shutdown successful
        """
        try:
            self.logger.info("Shutting down TorchServeAdapter")

            # Unload all models
            model_ids = list(self._model_configs.keys())
            for model_id in model_ids:
                await self.unload_model(model_id)

            # Close HTTP session
            if self.session:
                await self.session.close()

            self.logger.info("TorchServeAdapter shutdown completed")
            return True

        except Exception as e:
            self.logger.error(f"Error during TorchServe shutdown: {e}")
            return False

    # Private helper methods

    async def _create_model_archive(
        self,
        model_path: str,
        model_id: str,
        model_type: ModelType,
        model_config: Optional[Dict[str, Any]],
    ) -> str:
        """
        Create TorchServe model archive (.mar file)

        This method packages the model and creates a .mar file for TorchServe
        """
        import tempfile
        import subprocess
        import shutil

        try:
            # Create temporary directory for building
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Copy model files
                model_files_dir = temp_path / "model_files"
                shutil.copytree(model_path, model_files_dir)

                # Create handler file
                handler_path = await self._create_torchserve_handler(
                    temp_path, model_type, model_config
                )

                # Create model archive name
                mar_name = f"finsight_{model_id}".replace(":", "_").replace("/", "_")
                mar_file = f"{mar_name}.mar"
                mar_path = Path(self.model_store) / mar_file

                # Build .mar file using torch-model-archiver
                cmd = [
                    "torch-model-archiver",
                    "--model-name",
                    mar_name,
                    "--version",
                    "1.0",
                    "--handler",
                    str(handler_path),
                    "--extra-files",
                    str(model_files_dir),
                    "--export-path",
                    str(self.model_store),
                    "--force",
                ]

                # Add requirements if needed
                requirements_file = temp_path / "requirements.txt"
                with open(requirements_file, "w") as f:
                    f.write("torch\n")
                    f.write("numpy\n")
                    f.write("pandas\n")
                    if model_type in [ModelType.PATCHTST, ModelType.PATCHTSMIXER]:
                        f.write("transformers\n")

                cmd.extend(["--requirements-file", str(requirements_file)])

                # Execute command
                result = subprocess.run(
                    cmd, capture_output=True, text=True, cwd=temp_dir
                )

                if result.returncode != 0:
                    raise Exception(f"torch-model-archiver failed: {result.stderr}")

                self.logger.info(f"Created model archive: {mar_path}")
                return str(mar_path)

        except Exception as e:
            self.logger.error(f"Failed to create model archive: {e}")
            raise

    async def _create_torchserve_handler(
        self,
        temp_path: Path,
        model_type: ModelType,
        model_config: Optional[Dict[str, Any]],
    ) -> Path:
        """Create TorchServe custom handler"""

        handler_code = '''
import json
import logging
import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class FinSightModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, context):
        """Initialize the model"""
        try:
            properties = context.system_properties
            model_dir = properties.get("model_dir")
            
            # Load your model here based on model type
            # This is a simplified example
            self.model = self._load_model(model_dir)
            self.initialized = True
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _load_model(self, model_dir):
        """Load the specific model"""
        import pickle
        import os
        
        # Try different model file patterns
        model_files = [
            os.path.join(model_dir, "model_files", "model.pkl"),
            os.path.join(model_dir, "model_files", "model.pth"),
            os.path.join(model_dir, "model_files", "pytorch_model.bin")
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                if model_file.endswith('.pkl'):
                    with open(model_file, 'rb') as f:
                        return pickle.load(f)
                elif model_file.endswith(('.pth', '.bin')):
                    return torch.load(model_file, map_location='cpu')
        
        raise FileNotFoundError("No valid model file found")
    
    def preprocess(self, data):
        """Preprocess input data"""
        try:
            # Handle different input formats
            if isinstance(data, list):
                # Multiple inputs
                processed_data = []
                for item in data:
                    if isinstance(item, dict):
                        # JSON input
                        if 'data' in item:
                            input_data = np.array(item['data'], dtype=np.float32)
                        else:
                            input_data = np.array(list(item.values()), dtype=np.float32)
                    else:
                        # Direct array input
                        input_data = np.array(item, dtype=np.float32)
                    
                    processed_data.append(torch.tensor(input_data))
                
                return processed_data
            else:
                # Single input
                if isinstance(data, dict):
                    if 'data' in data:
                        input_data = np.array(data['data'], dtype=np.float32)
                    else:
                        input_data = np.array(list(data.values()), dtype=np.float32)
                else:
                    input_data = np.array(data, dtype=np.float32)
                
                return [torch.tensor(input_data)]
                
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
    
    def inference(self, data):
        """Run inference"""
        try:
            predictions = []
            
            for input_tensor in data:
                if hasattr(self.model, 'predict'):
                    # Scikit-learn style
                    pred = self.model.predict(input_tensor.numpy().reshape(1, -1))
                elif hasattr(self.model, 'forward'):
                    # PyTorch model
                    with torch.no_grad():
                        pred = self.model.forward(input_tensor.unsqueeze(0))
                        pred = pred.cpu().numpy()
                else:
                    # Fallback
                    pred = input_tensor.numpy()[-1:] * 1.01  # Simple prediction
                
                predictions.append(pred.flatten().tolist())
            
            return predictions
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def postprocess(self, data):
        """Postprocess output data"""
        try:
            # Format output for JSON response
            result = []
            for prediction in data:
                result.append({
                    "predictions": prediction,
                    "model_type": "finsight_model",
                    "version": "1.0"
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Postprocessing failed: {e}")
            raise
'''

        handler_path = temp_path / "handler.py"
        with open(handler_path, "w") as f:
            f.write(handler_code)

        return handler_path

    async def _register_model(
        self, mar_file_path: str, model_id: str, model_config: Optional[Dict[str, Any]]
    ) -> str:
        """Register model with TorchServe"""
        model_name = f"finsight_{model_id}".replace(":", "_").replace("/", "_")

        try:
            # Register model
            register_url = f"{self.management_url}/models"
            params = {
                "url": mar_file_path,
                "model_name": model_name,
                "initial_workers": 0,  # Start with 0 workers, will scale up later
                "synchronous": "true",
                "batch_size": self.batch_size,
                "max_batch_delay": self.max_batch_delay,
            }

            async with self.session.post(register_url, params=params) as response:
                if response.status not in [200, 201]:
                    error_text = await response.text()
                    raise Exception(
                        f"Model registration failed: {response.status} - {error_text}"
                    )

            self.logger.info(f"Model {model_name} registered successfully")
            return model_name

        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            raise

    async def _start_model_workers(self, model_name: str, num_workers: int) -> bool:
        """Start model workers"""
        try:
            # Scale up workers
            scale_url = f"{self.management_url}/models/{model_name}"
            params = {
                "min_worker": num_workers,
                "max_worker": self.max_workers,
                "synchronous": "true",
            }

            async with self.session.put(scale_url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"Worker scaling failed: {response.status} - {error_text}"
                    )

            # Wait for workers to be ready
            for _ in range(30):  # 30 second timeout
                status = await self._get_model_status(model_name)
                if len(status.get("workers", [])) >= num_workers:
                    break
                await asyncio.sleep(1)

            self.logger.info(f"Started {num_workers} workers for model {model_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start workers: {e}")
            raise

    async def _stop_model_workers(self, model_name: str) -> bool:
        """Stop model workers"""
        try:
            # Scale down to 0 workers
            scale_url = f"{self.management_url}/models/{model_name}"
            params = {"min_worker": 0, "max_worker": 0, "synchronous": "true"}

            async with self.session.put(scale_url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.warning(
                        f"Worker shutdown failed: {response.status} - {error_text}"
                    )

            return True

        except Exception as e:
            self.logger.error(f"Failed to stop workers: {e}")
            return False

    async def _unregister_model(self, model_name: str) -> bool:
        """Unregister model from TorchServe"""
        try:
            unregister_url = f"{self.management_url}/models/{model_name}"

            async with self.session.delete(unregister_url) as response:
                if response.status not in [200, 404]:  # 404 is OK if model not found
                    error_text = await response.text()
                    self.logger.warning(
                        f"Model unregistration failed: {response.status} - {error_text}"
                    )

            self.logger.info(f"Model {model_name} unregistered")
            return True

        except Exception as e:
            self.logger.error(f"Failed to unregister model: {e}")
            return False

    async def _get_model_status(self, model_name: str) -> Dict[str, Any]:
        """Get model status from TorchServe"""
        try:
            status_url = f"{self.management_url}/models/{model_name}"

            async with self.session.get(status_url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}

        except Exception as e:
            self.logger.error(f"Failed to get model status: {e}")
            return {}

    async def _estimate_model_memory(self, model_name: str) -> float:
        """Estimate model memory usage"""
        try:
            # Get model metrics if available
            # This is approximate as TorchServe doesn't directly expose memory per model
            return 150.0  # MB - placeholder
        except Exception:
            return 150.0  # Default estimate

    async def _prepare_torchserve_input(
        self,
        input_data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
        n_steps: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare input data for TorchServe request"""
        try:
            # Convert input data to appropriate format
            if isinstance(input_data, pd.DataFrame):
                data_array = input_data.values.astype(float).tolist()
            elif isinstance(input_data, np.ndarray):
                data_array = input_data.astype(float).tolist()
            elif isinstance(input_data, dict):
                if "close" in input_data:
                    data_array = input_data["close"]
                else:
                    data_array = list(input_data.values())
            else:
                raise ValueError(f"Unsupported input data type: {type(input_data)}")

            # Prepare payload
            payload = {"data": data_array, "n_steps": n_steps, **kwargs}

            return payload

        except Exception as e:
            self.logger.error(f"Failed to prepare TorchServe input: {e}")
            raise

    async def _process_torchserve_response(
        self,
        response_data: Dict[str, Any],
        input_data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
        model_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process TorchServe response"""
        try:
            # Extract predictions from response
            if isinstance(response_data, list) and len(response_data) > 0:
                predictions = response_data[0].get("predictions", [])
            elif isinstance(response_data, dict):
                predictions = response_data.get("predictions", [])
            else:
                predictions = []

            result = {"predictions": predictions}

            # Calculate additional metrics
            if isinstance(input_data, (pd.DataFrame, np.ndarray)):
                if isinstance(input_data, pd.DataFrame):
                    input_array = input_data.values
                else:
                    input_array = input_data

                if len(input_array) > 0:
                    current_price = float(input_array[-1])
                    result["current_price"] = current_price

                    if len(predictions) > 0:
                        predicted_price = predictions[0]
                        change_pct = (
                            (predicted_price - current_price) / current_price
                        ) * 100
                        result["predicted_change_pct"] = change_pct

            return result

        except Exception as e:
            self.logger.error(f"Failed to process TorchServe response: {e}")
            raise
