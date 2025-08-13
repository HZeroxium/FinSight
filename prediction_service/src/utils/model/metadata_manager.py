# utils/model/metadata_manager.py

"""
Model Metadata Management Utilities

This module handles model metadata operations including saving, loading,
and JSON operations for model information.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from ...schemas.enums import ModelType, TimeFrame
from ...core.config import get_settings
from .path_manager import ModelPathManager
from common.logger.logger_factory import LoggerFactory


class ModelMetadataManager:
    """Handles model metadata operations"""

    def __init__(self, path_manager: Optional[ModelPathManager] = None):
        self.logger = LoggerFactory.get_logger("ModelMetadataManager")
        self.settings = get_settings()
        self.path_manager = path_manager or ModelPathManager()

    def save_json(self, data: Dict[str, Any], file_path: str) -> None:
        """
        Save data as JSON file

        Args:
            data: Data to save
            file_path: Path to save the JSON file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load data from JSON file

        Args:
            file_path: Path to the JSON file

        Returns:
            Loaded data or None if file doesn't exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON from {file_path}: {e}")
            return None

    def save_model_metadata(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        metadata: Dict[str, Any],
        adapter_type: str = "simple",
    ) -> Path:
        """
        Save model metadata

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            metadata: Metadata dictionary to save
            adapter_type: Adapter type for storage location

        Returns:
            Path to saved metadata file
        """
        model_dir = self.path_manager.generate_model_path(
            symbol, timeframe, model_type, adapter_type
        )
        model_dir.mkdir(parents=True, exist_ok=True)

        # Ensure symbol is string
        if hasattr(symbol, "value"):
            symbol = symbol.value

        # Add standardized metadata
        metadata.update(
            {
                "symbol": symbol,
                "timeframe": timeframe.value,
                "model_type": model_type.value,
                "adapter_type": adapter_type,
                "saved_at": datetime.now().isoformat(),
                "model_identifier": self.path_manager.generate_model_identifier(
                    symbol, timeframe, model_type
                ),
            }
        )

        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return metadata_path

    def load_model_metadata(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Optional[Dict[str, Any]]:
        """
        Load model metadata

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            adapter_type: Adapter type for storage location

        Returns:
            Metadata dictionary or None if not found
        """
        model_dir = self.path_manager.generate_model_path(
            symbol, timeframe, model_type, adapter_type
        )
        metadata_path = model_dir / "metadata.json"

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def update_model_metadata(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        updates: Dict[str, Any],
        adapter_type: str = "simple",
    ) -> bool:
        """
        Update existing model metadata

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            updates: Dictionary of updates to apply
            adapter_type: Adapter type for storage location

        Returns:
            True if update successful, False otherwise
        """
        try:
            # Load existing metadata
            existing_metadata = self.load_model_metadata(
                symbol, timeframe, model_type, adapter_type
            )

            if existing_metadata is None:
                self.logger.warning(
                    f"No existing metadata found for {symbol}_{timeframe.value}_{model_type.value}"
                )
                return False

            # Update metadata
            existing_metadata.update(updates)
            existing_metadata["updated_at"] = datetime.now().isoformat()

            # Save updated metadata
            self.save_model_metadata(
                symbol, timeframe, model_type, existing_metadata, adapter_type
            )

            return True
        except Exception as e:
            self.logger.error(f"Failed to update model metadata: {e}")
            return False
