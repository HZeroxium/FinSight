# utils/model/path_manager.py

"""
Model Path Management Utilities

This module handles all model path generation and management operations,
providing consistent path structures for different adapter types.
"""

from pathlib import Path
from typing import Optional

from common.logger.logger_factory import LoggerFactory

from ...core.config import get_settings
from ...schemas.enums import ModelType, TimeFrame


class ModelPathManager:
    """Handles model path generation and management operations"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger("ModelPathManager")
        self.settings = get_settings()

    def generate_model_identifier(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> str:
        """Generate standardized model identifier using configuration pattern"""
        # Ensure symbol is a string (handle enum case)
        if hasattr(symbol, "value"):
            symbol = symbol.value

        # Clean model type for filesystem compatibility
        clean_model_type = model_type.value.replace("/", "_").replace("-", "_")

        return self.settings.model_name_pattern.format(
            symbol=symbol,
            timeframe=timeframe.value,
            model_type=clean_model_type,
        )

    def generate_cloud_object_key(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        file_name: Optional[str] = None,
    ) -> str:
        """
        Generate cloud storage object key for model artifacts.

        Structure: {model_storage_prefix}/{adapter_type}/{symbol}_{timeframe}_{model_type}/{file_name}

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            adapter_type: Adapter type for storage location
            file_name: Optional specific file name

        Returns:
            Cloud storage object key
        """
        model_id = self.generate_model_identifier(symbol, timeframe, model_type)
        base_key = f"{self.settings.model_storage_prefix}/{adapter_type}/{model_id}"

        if file_name:
            return f"{base_key}/{file_name}"
        return base_key

    def generate_cloud_serving_key(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        file_name: Optional[str] = None,
    ) -> str:
        """
        Generate cloud storage object key for model serving artifacts.

        This method is specifically for serving operations and uses the same structure
        as generate_cloud_object_key but is separated for clarity.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            adapter_type: Adapter type for storage location
            file_name: Optional specific file name

        Returns:
            Cloud storage object key for serving
        """
        return self.generate_cloud_object_key(
            symbol, timeframe, model_type, adapter_type, file_name
        )

    def generate_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Path:
        """
        Generate standardized model directory path with adapter-specific structure.

        The model storage is organized as:
        /models/{adapter_type}/{model_identifier}/

        Where adapter_type can be: simple, torchscript, torchserve, triton
        """
        model_id = self.generate_model_identifier(symbol, timeframe, model_type)
        return self.settings.models_dir / adapter_type / model_id

    def get_checkpoint_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Path:
        """Get model checkpoint file path - check for multiple possible file names"""
        model_dir = self.generate_model_path(
            symbol, timeframe, model_type, adapter_type
        )

        # Check for different possible model file names in order of preference
        possible_files = [
            "model_state_dict.pt",  # Used by HuggingFace adapters
            "model.pt",  # Default checkpoint filename
            "pytorch_model.bin",  # Alternative HuggingFace format
        ]

        for filename in possible_files:
            checkpoint_path = model_dir / filename
            if checkpoint_path.exists():
                return checkpoint_path

        # Return default path even if it doesn't exist (for creation)
        return model_dir / self.settings.checkpoint_filename

    def get_metadata_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Path:
        """Get model metadata file path"""
        model_dir = self.generate_model_path(
            symbol, timeframe, model_type, adapter_type
        )
        return model_dir / self.settings.metadata_filename

    def get_config_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Path:
        """Get model config file path"""
        model_dir = self.generate_model_path(
            symbol, timeframe, model_type, adapter_type
        )
        return model_dir / self.settings.config_filename

    def ensure_model_directory(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Path:
        """Ensure model directory exists and return its path"""
        model_dir = self.generate_model_path(
            symbol, timeframe, model_type, adapter_type
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def get_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Path:
        """Alias for generate_model_path for backward compatibility"""
        return self.generate_model_path(symbol, timeframe, model_type, adapter_type)

    def get_simple_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Path:
        """Get path for simple adapter model storage."""
        return self.generate_model_path(symbol, timeframe, model_type, "simple")

    def get_torchscript_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Path:
        """Get path for TorchScript adapter model storage."""
        return self.generate_model_path(symbol, timeframe, model_type, "torchscript")

    def get_torchserve_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Path:
        """Get path for TorchServe adapter model storage."""
        return self.generate_model_path(symbol, timeframe, model_type, "torchserve")

    def get_triton_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Path:
        """Get path for Triton adapter model storage."""
        return self.generate_model_path(symbol, timeframe, model_type, "triton")
