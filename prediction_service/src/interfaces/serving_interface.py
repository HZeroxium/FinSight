# interfaces/serving_interface.py

"""
Abstract interface for model serving implementations.

This interface defines the contract for different model serving backends
including Triton Inference Server, TorchServe, and simple in-memory serving.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel
import numpy as np
import pandas as pd

from ..schemas.enums import ModelType, TimeFrame
from common.logger.logger_factory import LoggerFactory


class ModelInfo(BaseModel):
    """Model information schema for serving"""

    model_id: str
    symbol: str
    timeframe: str
    model_type: str
    model_path: str
    is_loaded: bool = False
    loaded_at: Optional[datetime] = None
    memory_usage_mb: Optional[float] = None
    version: str = "1.0"
    metadata: Dict[str, Any] = {}


class PredictionResult(BaseModel):
    """Prediction result schema"""

    success: bool
    predictions: List[float]
    current_price: Optional[float] = None
    predicted_change_pct: Optional[float] = None
    confidence_score: Optional[float] = None
    model_info: Dict[str, Any] = {}
    inference_time_ms: Optional[float] = None
    error: Optional[str] = None


class ServingStats(BaseModel):
    """Serving statistics schema"""

    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    average_inference_time_ms: float = 0.0
    models_loaded: int = 0
    total_memory_usage_mb: float = 0.0
    uptime_seconds: float = 0.0

    def get_success_rate(self) -> float:
        """Calculate prediction success rate"""
        if self.total_predictions == 0:
            return 0.0
        return self.successful_predictions / self.total_predictions


class IModelServingAdapter(ABC):
    """
    Abstract interface for model serving adapters.

    This interface defines the contract that all model serving implementations
    must follow, ensuring consistency across different backends.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the serving adapter

        Args:
            config: Configuration dictionary for the adapter
        """
        self.config = config
        self.logger = LoggerFactory.get_logger(f"{self.__class__.__name__}")
        self._stats = ServingStats()
        self._start_time = datetime.now()

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the serving backend

        Returns:
            bool: True if initialization successful
        """
        pass

    @abstractmethod
    async def load_model(
        self,
        model_path: str,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> ModelInfo:
        """
        Load a model for serving

        Args:
            model_path: Path to the model files
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Type of model
            model_config: Optional model configuration

        Returns:
            ModelInfo: Information about the loaded model
        """
        pass

    @abstractmethod
    async def predict(
        self,
        model_id: str,
        input_data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
        n_steps: int = 1,
        **kwargs,
    ) -> PredictionResult:
        """
        Make predictions using a loaded model

        Args:
            model_id: Identifier of the loaded model
            input_data: Input data for prediction
            n_steps: Number of prediction steps
            **kwargs: Additional prediction parameters

        Returns:
            PredictionResult: Prediction results
        """
        pass

    @abstractmethod
    async def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from memory

        Args:
            model_id: Identifier of the model to unload

        Returns:
            bool: True if unload successful
        """
        pass

    @abstractmethod
    async def list_loaded_models(self) -> List[ModelInfo]:
        """
        List all currently loaded models

        Returns:
            List[ModelInfo]: List of loaded model information
        """
        pass

    @abstractmethod
    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model

        Args:
            model_id: Model identifier

        Returns:
            Optional[ModelInfo]: Model information if found
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the serving backend

        Returns:
            Dict[str, Any]: Health status information
        """
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Shutdown the serving backend gracefully

        Returns:
            bool: True if shutdown successful
        """
        pass

    # Common utility methods

    def generate_model_id(
        self, symbol: str, timeframe: TimeFrame, model_type: ModelType
    ) -> str:
        """
        Generate a unique model identifier

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type

        Returns:
            str: Unique model identifier
        """
        return f"{symbol}_{timeframe.value}_{model_type.value}"

    def get_stats(self) -> ServingStats:
        """Get serving statistics"""
        current_time = datetime.now()
        self._stats.uptime_seconds = (current_time - self._start_time).total_seconds()
        return self._stats

    def reset_stats(self) -> None:
        """Reset serving statistics"""
        self._stats = ServingStats()
        self._start_time = datetime.now()

    def _update_stats(self, success: bool, inference_time_ms: float = 0.0) -> None:
        """Update serving statistics"""
        self._stats.total_predictions += 1
        if success:
            self._stats.successful_predictions += 1
        else:
            self._stats.failed_predictions += 1

        # Update average inference time
        if self._stats.total_predictions == 1:
            self._stats.average_inference_time_ms = inference_time_ms
        else:
            # Running average
            total_time = (
                self._stats.average_inference_time_ms
                * (self._stats.total_predictions - 1)
                + inference_time_ms
            )
            self._stats.average_inference_time_ms = (
                total_time / self._stats.total_predictions
            )
