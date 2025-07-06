# utils/model_utils.py

"""
Model utilities for centralized model path management and operations.

This module provides utilities for model path generation, model metadata handling,
and consistent model saving/loading operations across the system.
"""
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..schemas.enums import ModelType, TimeFrame
from ..core.config import get_settings
from ..logger.logger_factory import LoggerFactory


class ModelUtils:
    """Utilities for model path management and operations"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger("ModelUtils")
        self.settings = get_settings()

    def generate_model_identifier(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> str:
        """Generate standardized model identifier using configuration pattern"""
        # Clean model type for filesystem compatibility
        clean_model_type = model_type.value.replace("/", "_").replace("-", "_")

        return self.settings.model_name_pattern.format(
            symbol=symbol,
            timeframe=timeframe.value,
            model_type=clean_model_type,
        )

    def generate_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Path:
        """Generate standardized model directory path"""
        model_id = self.generate_model_identifier(symbol, timeframe, model_type)
        return self.settings.models_dir / model_id

    def get_checkpoint_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Path:
        """Get model checkpoint file path"""
        model_dir = self.generate_model_path(symbol, timeframe, model_type)
        return model_dir / self.settings.checkpoint_filename

    def get_metadata_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Path:
        """Get model metadata file path"""
        model_dir = self.generate_model_path(symbol, timeframe, model_type)
        return model_dir / self.settings.metadata_filename

    def get_config_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Path:
        """Get model config file path"""
        model_dir = self.generate_model_path(symbol, timeframe, model_type)
        return model_dir / self.settings.config_filename

    def ensure_model_directory(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Path:
        """Ensure model directory exists and return its path"""
        model_dir = self.generate_model_path(symbol, timeframe, model_type)
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    @staticmethod
    def save_model_metadata(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        metadata: Dict[str, Any],
    ) -> Path:
        """
        Save model metadata

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            metadata: Metadata dictionary to save

        Returns:
            Path to saved metadata file
        """
        model_dir = ModelUtils.generate_model_path(symbol, timeframe, model_type)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Add standardized metadata
        metadata.update(
            {
                "symbol": symbol,
                "timeframe": timeframe.value,
                "model_type": model_type.value,
                "saved_at": datetime.now().isoformat(),
                "model_identifier": ModelUtils.generate_model_identifier(
                    symbol, timeframe, model_type
                ),
            }
        )

        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return metadata_path

    @staticmethod
    def load_model_metadata(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Optional[Dict[str, Any]]:
        """
        Load model metadata

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type

        Returns:
            Metadata dictionary or None if not found
        """
        model_dir = ModelUtils.generate_model_path(symbol, timeframe, model_type)
        metadata_path = model_dir / "metadata.json"

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def model_exists(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> bool:
        """
        Check if model exists

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type

        Returns:
            True if model exists, False otherwise
        """
        model_path = ModelUtils.generate_checkpoint_path(symbol, timeframe, model_type)
        return model_path.exists()

    @staticmethod
    def list_available_models(base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        List all available models

        Args:
            base_dir: Base directory to search (if None, uses settings)

        Returns:
            List of model information dictionaries
        """
        if base_dir is None:
            settings = get_settings()
            base_dir = settings.models_dir

        if not base_dir.exists():
            return []

        models = []
        for model_dir in base_dir.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        models.append(metadata)
                    except Exception:
                        continue

        return models

    @staticmethod
    def delete_model(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> bool:
        """
        Delete model and all associated files

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type

        Returns:
            True if deletion successful, False otherwise
        """
        import shutil

        model_dir = ModelUtils.generate_model_path(symbol, timeframe, model_type)

        if not model_dir.exists():
            return False

        try:
            shutil.rmtree(model_dir)
            return True
        except Exception:
            return False

    @staticmethod
    def get_model_size(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Optional[int]:
        """
        Get model size in bytes

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type

        Returns:
            Model size in bytes or None if model doesn't exist
        """
        checkpoint_path = ModelUtils.generate_checkpoint_path(
            symbol, timeframe, model_type
        )

        if not checkpoint_path.exists():
            return None

        return checkpoint_path.stat().st_size

    @staticmethod
    def backup_model(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        backup_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Create backup of model

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            backup_dir: Backup directory (if None, creates in models directory)

        Returns:
            Path to backup or None if failed
        """
        import shutil

        model_dir = ModelUtils.generate_model_path(symbol, timeframe, model_type)

        if not model_dir.exists():
            return None

        if backup_dir is None:
            settings = get_settings()
            backup_dir = settings.models_dir / "backups"

        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = ModelUtils.generate_model_identifier(symbol, timeframe, model_type)
        backup_path = backup_dir / f"{model_id}_{timestamp}"

        try:
            shutil.copytree(model_dir, backup_path)
            return backup_path
        except Exception:
            return None
