# adapters/triton_serving.py

"""
Triton Inference Server adapter for model serving.

This adapter integrates with NVIDIA Triton Inference Server to provide
high-performance, scalable model serving with advanced features like
dynamic batching, model versioning, and GPU acceleration.
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import tritonclient.grpc as grpcclient
    import tritonclient.http as httpclient
    from tritonclient.utils import InferenceServerException

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    httpclient = grpcclient = InferenceServerException = None

from common.logger.logger_factory import LoggerFactory

from ..core.config import get_settings
from ..interfaces.serving_interface import (IModelServingAdapter, ModelInfo,
                                            PredictionResult, ServingStats)
from ..schemas.enums import ModelType, TimeFrame


class TritonServingAdapter(IModelServingAdapter):
    """
    Triton Inference Server adapter for high-performance model serving.

    This adapter provides enterprise-grade model serving using NVIDIA Triton
    Inference Server with features like:
    - Dynamic batching
    - Model versioning
    - GPU acceleration
    - Multi-model serving
    - Health monitoring
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Triton serving adapter

        Args:
            config: Configuration dictionary with Triton server details
        """
        super().__init__(config)

        if not TRITON_AVAILABLE:
            raise ImportError(
                "tritonclient is required for TritonServingAdapter. "
                "Install with: pip install tritonclient[all]"
            )

        self.settings = get_settings()

        # Triton server configuration
        self.server_url = config.get("server_url", "localhost:8000")
        self.server_grpc_url = config.get("server_grpc_url", "localhost:8001")
        self.use_grpc = config.get("use_grpc", False)
        self.ssl = config.get("ssl", False)
        self.insecure = config.get("insecure", True)

        # Model configuration
        self.model_repository = config.get("model_repository", "/models")
        self.default_model_version = config.get("default_model_version", "1")
        self.max_batch_size = config.get("max_batch_size", 8)
        self.timeout_seconds = config.get("timeout_seconds", 30)

        # Client initialization
        self.client = None
        self._model_configs: Dict[str, Dict[str, Any]] = {}

        self.logger.info(
            f"TritonServingAdapter initialized for server: {self.server_url}"
        )

    async def initialize(self) -> bool:
        """
        Initialize connection to Triton Inference Server

        Returns:
            bool: True if initialization successful
        """
        try:
            # Create Triton client
            if self.use_grpc:
                self.client = grpcclient.InferenceServerClient(
                    url=self.server_grpc_url, ssl=self.ssl, insecure=self.insecure
                )
            else:
                self.client = httpclient.InferenceServerClient(
                    url=self.server_url, ssl=self.ssl, insecure=self.insecure
                )

            # Test connection
            if not self.client.is_server_live():
                raise ConnectionError("Triton server is not live")

            if not self.client.is_server_ready():
                raise ConnectionError("Triton server is not ready")

            self.logger.info("Successfully connected to Triton Inference Server")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Triton client: {e}")
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
        Load a model into Triton Inference Server

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
            self.logger.info(f"Loading model {model_id} into Triton server")

            # Convert model to Triton format if needed
            triton_model_name = await self._prepare_triton_model(
                model_path, model_id, model_type, model_config
            )

            # Load model in Triton
            await self._load_model_in_triton(triton_model_name)

            # Get model metadata
            metadata = await self._get_model_metadata(triton_model_name)

            # Calculate memory usage (approximate)
            memory_usage = await self._estimate_triton_model_memory(triton_model_name)

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
                    "triton_model_name": triton_model_name,
                    "triton_metadata": metadata,
                    **(model_config or {}),
                },
            )

            # Store model configuration
            self._model_configs[model_id] = {
                "triton_model_name": triton_model_name,
                "model_info": model_info,
                "inputs": metadata.get("inputs", []),
                "outputs": metadata.get("outputs", []),
            }

            # Update stats
            self._stats.models_loaded += 1
            self._stats.total_memory_usage_mb += memory_usage

            self.logger.info(f"Model {model_id} loaded successfully in Triton")
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
        Make predictions using Triton Inference Server

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
                error_msg = f"Model {model_id} not loaded in Triton"
                self.logger.error(error_msg)
                self._update_stats(success=False)
                return PredictionResult(success=False, predictions=[], error=error_msg)

            model_config = self._model_configs[model_id]
            triton_model_name = model_config["triton_model_name"]

            # Prepare input for Triton
            triton_inputs = await self._prepare_triton_inputs(
                input_data, model_config, n_steps, **kwargs
            )

            # Prepare outputs
            triton_outputs = await self._prepare_triton_outputs(model_config)

            # Make inference request
            if self.use_grpc:
                response = self.client.infer(
                    model_name=triton_model_name,
                    inputs=triton_inputs,
                    outputs=triton_outputs,
                    model_version=self.default_model_version,
                )
            else:
                response = self.client.infer(
                    model_name=triton_model_name,
                    inputs=triton_inputs,
                    outputs=triton_outputs,
                    model_version=self.default_model_version,
                )

            # Process response
            result = await self._process_triton_response(
                response, input_data, model_config
            )

            inference_time_ms = (time.time() - start_time) * 1000

            # Update stats
            self._update_stats(success=True, inference_time_ms=inference_time_ms)

            return PredictionResult(
                success=True,
                predictions=result["predictions"],
                current_price=result.get("current_price"),
                predicted_change_pct=result.get("predicted_change_pct"),
                confidence_score=result.get("confidence_score", 0.9),
                model_info={
                    "model_id": model_id,
                    "triton_model_name": triton_model_name,
                    "model_type": model_config["model_info"].model_type,
                    "version": self.default_model_version,
                },
                inference_time_ms=inference_time_ms,
            )

        except Exception as e:
            inference_time_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Triton prediction failed for model {model_id}: {e}")
            self._update_stats(success=False, inference_time_ms=inference_time_ms)

            return PredictionResult(
                success=False,
                predictions=[],
                error=str(e),
                inference_time_ms=inference_time_ms,
            )

    async def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from Triton server

        Args:
            model_id: Identifier of the model to unload

        Returns:
            bool: True if unload successful
        """
        try:
            if model_id in self._model_configs:
                triton_model_name = self._model_configs[model_id]["triton_model_name"]

                # Unload from Triton
                self.client.unload_model(triton_model_name)

                # Get memory usage before removal
                model_info = self._model_configs[model_id]["model_info"]
                memory_usage = model_info.memory_usage_mb or 0.0

                # Remove from local cache
                del self._model_configs[model_id]

                # Update stats
                self._stats.models_loaded -= 1
                self._stats.total_memory_usage_mb -= memory_usage

                self.logger.info(f"Model {model_id} unloaded from Triton")
                return True
            else:
                self.logger.warning(f"Model {model_id} not found for unloading")
                return False

        except Exception as e:
            self.logger.error(f"Failed to unload model {model_id}: {e}")
            return False

    async def list_loaded_models(self) -> List[ModelInfo]:
        """
        List all models loaded in Triton server

        Returns:
            List[ModelInfo]: List of loaded model information
        """
        try:
            # Get models from Triton server
            triton_models = self.client.get_model_repository_index()
            loaded_models = []

            for model_config in self._model_configs.values():
                model_info = model_config["model_info"]
                triton_name = model_config["triton_model_name"]

                # Check if model is still loaded in Triton
                model_status = None
                for tm in triton_models:
                    if tm["name"] == triton_name:
                        model_status = tm
                        break

                if model_status and model_status.get("state") == "READY":
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
        Check the health status of Triton server

        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            # Check server status
            server_live = self.client.is_server_live()
            server_ready = self.client.is_server_ready()

            # Get server metadata
            server_metadata = self.client.get_server_metadata()

            # Get model repository status
            model_repository = self.client.get_model_repository_index()
            ready_models = [m for m in model_repository if m.get("state") == "READY"]

            return {
                "status": "healthy" if server_live and server_ready else "unhealthy",
                "adapter_type": "triton",
                "server_live": server_live,
                "server_ready": server_ready,
                "server_version": server_metadata.get("version", "unknown"),
                "models_loaded": len(self._model_configs),
                "triton_ready_models": len(ready_models),
                "total_memory_usage_mb": self._stats.total_memory_usage_mb,
                "uptime_seconds": self.get_stats().uptime_seconds,
                "total_predictions": self._stats.total_predictions,
                "success_rate": self._stats.get_success_rate(),
            }

        except Exception as e:
            self.logger.error(f"Triton health check failed: {e}")
            return {"status": "unhealthy", "adapter_type": "triton", "error": str(e)}

    async def shutdown(self) -> bool:
        """
        Shutdown the Triton serving backend gracefully

        Returns:
            bool: True if shutdown successful
        """
        try:
            self.logger.info("Shutting down TritonServingAdapter")

            # Unload all models
            model_ids = list(self._model_configs.keys())
            for model_id in model_ids:
                await self.unload_model(model_id)

            # Close client connection
            if self.client:
                self.client.close()

            self.logger.info("TritonServingAdapter shutdown completed")
            return True

        except Exception as e:
            self.logger.error(f"Error during Triton shutdown: {e}")
            return False

    # Private helper methods

    async def _prepare_triton_model(
        self,
        model_path: str,
        model_id: str,
        model_type: ModelType,
        model_config: Optional[Dict[str, Any]],
    ) -> str:
        """
        Prepare model for Triton Inference Server

        This method converts the model to a Triton-compatible format
        and creates the necessary configuration files.
        """
        triton_model_name = f"finsight_{model_id}".replace(":", "_").replace("/", "_")

        try:
            # Create model directory in Triton model repository
            model_dir = (
                Path(self.model_repository)
                / triton_model_name
                / self.default_model_version
            )
            model_dir.mkdir(parents=True, exist_ok=True)

            # Convert model based on type
            if model_type in [ModelType.PATCHTST, ModelType.PATCHTSMIXER]:
                await self._convert_transformer_to_triton(model_path, model_dir)
            elif model_type == ModelType.PYTORCH_TRANSFORMER:
                await self._convert_pytorch_to_triton(model_path, model_dir)
            else:
                await self._convert_generic_to_triton(model_path, model_dir)

            # Create Triton model configuration
            await self._create_triton_config(
                triton_model_name, model_type, model_config
            )

            return triton_model_name

        except Exception as e:
            self.logger.error(f"Failed to prepare Triton model: {e}")
            raise

    async def _convert_transformer_to_triton(self, model_path: str, output_dir: Path):
        """Convert HuggingFace transformer model to Triton format"""
        try:
            # For now, copy the model files directly
            # In production, you would convert to ONNX or TensorRT
            import shutil

            shutil.copytree(model_path, output_dir, dirs_exist_ok=True)

            # Create a simple Python backend script
            python_script = """
import json
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        # Load your model here
        pass
    
    def execute(self, requests):
        responses = []
        for request in requests:
            # Process request and generate response
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            input_data = input_tensor.as_numpy()
            
            # Your prediction logic here
            output_data = input_data  # Placeholder
            
            output_tensor = pb_utils.Tensor("OUTPUT", output_data)
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)
        
        return responses
"""

            with open(output_dir / "model.py", "w") as f:
                f.write(python_script)

        except Exception as e:
            self.logger.error(f"Failed to convert transformer model: {e}")
            raise

    async def _convert_pytorch_to_triton(self, model_path: str, output_dir: Path):
        """Convert PyTorch model to Triton format"""
        # Similar implementation for PyTorch models
        await self._convert_transformer_to_triton(model_path, output_dir)

    async def _convert_generic_to_triton(self, model_path: str, output_dir: Path):
        """Convert generic model to Triton format"""
        # Generic conversion implementation
        await self._convert_transformer_to_triton(model_path, output_dir)

    async def _create_triton_config(
        self,
        model_name: str,
        model_type: ModelType,
        model_config: Optional[Dict[str, Any]],
    ):
        """Create Triton model configuration file"""
        config = {
            "name": model_name,
            "platform": "python",
            "max_batch_size": self.max_batch_size,
            "input": [
                {
                    "name": "INPUT",
                    "data_type": "TYPE_FP32",
                    "dims": [-1, 64],  # Sequence length, features
                }
            ],
            "output": [
                {
                    "name": "OUTPUT",
                    "data_type": "TYPE_FP32",
                    "dims": [-1, 1],  # Predictions
                }
            ],
            "dynamic_batching": {
                "preferred_batch_size": [1, 2, 4],
                "max_queue_delay_microseconds": 100,
            },
        }

        config_path = Path(self.model_repository) / model_name / "config.pbtxt"

        # Convert to protobuf text format
        config_text = self._dict_to_pbtxt(config)

        with open(config_path, "w") as f:
            f.write(config_text)

    def _dict_to_pbtxt(self, config: Dict[str, Any]) -> str:
        """Convert configuration dictionary to protobuf text format"""

        def format_value(key: str, value: Any, indent: int = 0) -> str:
            spaces = "  " * indent

            if isinstance(value, dict):
                result = f"{spaces}{key} {{\n"
                for k, v in value.items():
                    result += format_value(k, v, indent + 1)
                result += f"{spaces}}}\n"
                return result
            elif isinstance(value, list):
                result = ""
                for item in value:
                    if isinstance(item, dict):
                        result += f"{spaces}{key} {{\n"
                        for k, v in item.items():
                            result += format_value(k, v, indent + 1)
                        result += f"{spaces}}}\n"
                    else:
                        result += f"{spaces}{key}: {item}\n"
                return result
            elif isinstance(value, str):
                return f'{spaces}{key}: "{value}"\n'
            else:
                return f"{spaces}{key}: {value}\n"

        result = ""
        for key, value in config.items():
            result += format_value(key, value)

        return result

    async def _load_model_in_triton(self, model_name: str):
        """Load model in Triton server"""
        try:
            self.client.load_model(model_name)

            # Wait for model to be ready
            max_wait = 30  # seconds
            wait_time = 0
            while wait_time < max_wait:
                if self.client.is_model_ready(model_name):
                    break
                await asyncio.sleep(1)
                wait_time += 1

            if not self.client.is_model_ready(model_name):
                raise TimeoutError(f"Model {model_name} not ready after {max_wait}s")

        except Exception as e:
            self.logger.error(f"Failed to load model in Triton: {e}")
            raise

    async def _get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """Get model metadata from Triton server"""
        try:
            metadata = self.client.get_model_metadata(model_name)
            return {
                "name": metadata.name,
                "versions": metadata.versions,
                "platform": metadata.platform,
                "inputs": [
                    {"name": inp.name, "datatype": inp.datatype, "shape": inp.shape}
                    for inp in metadata.inputs
                ],
                "outputs": [
                    {"name": out.name, "datatype": out.datatype, "shape": out.shape}
                    for out in metadata.outputs
                ],
            }
        except Exception as e:
            self.logger.error(f"Failed to get model metadata: {e}")
            return {}

    async def _estimate_triton_model_memory(self, model_name: str) -> float:
        """Estimate model memory usage in Triton server"""
        try:
            # Get model statistics if available
            stats = self.client.get_model_statistics(model_name)
            # This is approximate - Triton doesn't directly expose memory usage
            return 100.0  # MB - placeholder
        except Exception:
            return 100.0  # Default estimate

    async def _prepare_triton_inputs(
        self,
        input_data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
        model_config: Dict[str, Any],
        n_steps: int,
        **kwargs,
    ) -> List:
        """Prepare input tensors for Triton inference"""
        try:
            # Convert input data to numpy array
            if isinstance(input_data, pd.DataFrame):
                input_array = input_data.values.astype(np.float32)
            elif isinstance(input_data, np.ndarray):
                input_array = input_data.astype(np.float32)
            elif isinstance(input_data, dict):
                if "close" in input_data:
                    input_array = np.array(input_data["close"]).astype(np.float32)
                else:
                    raise ValueError("Invalid input data format")
            else:
                raise ValueError(f"Unsupported input data type: {type(input_data)}")

            # Reshape for batch processing
            if len(input_array.shape) == 1:
                input_array = input_array.reshape(1, -1)

            # Create Triton input tensor
            if self.use_grpc:
                inputs = [grpcclient.InferInput("INPUT", input_array.shape, "FP32")]
                inputs[0].set_data_from_numpy(input_array)
            else:
                inputs = [httpclient.InferInput("INPUT", input_array.shape, "FP32")]
                inputs[0].set_data_from_numpy(input_array)

            return inputs

        except Exception as e:
            self.logger.error(f"Failed to prepare Triton inputs: {e}")
            raise

    async def _prepare_triton_outputs(self, model_config: Dict[str, Any]) -> List:
        """Prepare output tensors for Triton inference"""
        try:
            if self.use_grpc:
                outputs = [grpcclient.InferRequestedOutput("OUTPUT")]
            else:
                outputs = [httpclient.InferRequestedOutput("OUTPUT")]

            return outputs

        except Exception as e:
            self.logger.error(f"Failed to prepare Triton outputs: {e}")
            raise

    async def _process_triton_response(
        self,
        response,
        input_data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
        model_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process Triton inference response"""
        try:
            # Get output data
            output_data = response.as_numpy("OUTPUT")
            predictions_list = output_data.flatten().tolist()

            result = {"predictions": predictions_list}

            # Calculate additional metrics
            if isinstance(input_data, (pd.DataFrame, np.ndarray)):
                if isinstance(input_data, pd.DataFrame):
                    input_array = input_data.values
                else:
                    input_array = input_data

                if len(input_array) > 0:
                    current_price = float(input_array[-1])
                    result["current_price"] = current_price

                    if len(predictions_list) > 0:
                        predicted_price = predictions_list[0]
                        change_pct = (
                            (predicted_price - current_price) / current_price
                        ) * 100
                        result["predicted_change_pct"] = change_pct

            return result

        except Exception as e:
            self.logger.error(f"Failed to process Triton response: {e}")
            raise
