# utils/model_utils.py

"""
Model utilities for centralized model path management and operations.

This module provides utilities for model path generation, model metadata handling,
and consistent model saving/loading operations across the system.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..schemas.enums import ModelType, TimeFrame
from ..core.config import get_settings
from common.logger.logger_factory import LoggerFactory


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

    def save_json(self, data: Dict[str, Any], file_path: str) -> None:
        """
        Save data as JSON file

        Args:
            data: Data to save
            file_path: Path to save the JSON file
        """
        import json
        from pathlib import Path

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
        import json
        from pathlib import Path

        file_path = Path(file_path)
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON from {file_path}: {e}")
            return None

    def model_exists(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> bool:
        """Check if a model exists (instance method)"""
        checkpoint_path = self.get_checkpoint_path(
            symbol, timeframe, model_type, adapter_type
        )
        return checkpoint_path.exists()

    def copy_model_for_adapter(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        source_adapter: str = "simple",
        target_adapter: str = "torchscript",
    ) -> bool:
        """
        Copy a model from one adapter directory to another for compatibility.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            source_adapter: Source adapter type
            target_adapter: Target adapter type

        Returns:
            True if copy successful, False otherwise
        """
        try:
            import shutil

            source_dir = self.generate_model_path(
                symbol, timeframe, model_type, source_adapter
            )
            target_dir = self.generate_model_path(
                symbol, timeframe, model_type, target_adapter
            )

            if not source_dir.exists():
                self.logger.warning(
                    f"Source model directory does not exist: {source_dir}"
                )
                return False

            # Create target directory
            target_dir.parent.mkdir(parents=True, exist_ok=True)

            # Copy the entire model directory
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(source_dir, target_dir)

            self.logger.info(
                f"Copied model from {source_adapter} to {target_adapter}: {target_dir}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to copy model for adapter: {e}")
            return False

    def ensure_adapter_compatibility(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        target_adapters: List[str] = None,
    ) -> None:
        """
        Ensure model is available for multiple adapters by copying from simple adapter.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            target_adapters: List of target adapter types
        """
        if target_adapters is None:
            target_adapters = ["simple", "torchscript", "torchserve", "triton"]

        # Check if model exists in simple adapter (training default)
        simple_exists = self.model_exists(symbol, timeframe, model_type, "simple")
        if not simple_exists:
            self.logger.warning(
                f"No model found in simple adapter for {symbol}_{timeframe}_{model_type}"
            )
            return

        # Copy to other adapters if they don't exist
        for adapter in target_adapters:
            if adapter != "simple":
                if not self.model_exists(symbol, timeframe, model_type, adapter):
                    self.copy_model_for_adapter(
                        symbol, timeframe, model_type, "simple", adapter
                    )

    @staticmethod
    def save_model_metadata(
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
        utils = ModelUtils()  # Create instance to access instance methods
        model_dir = utils.generate_model_path(
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
                "model_identifier": utils.generate_model_identifier(
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
        utils = ModelUtils()  # Create instance to access instance methods
        model_dir = utils.generate_model_path(
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

    @staticmethod
    def model_exists(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> bool:
        """
        Check if model exists

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            adapter_type: Adapter type for storage location

        Returns:
            True if model exists, False otherwise
        """
        utils = ModelUtils()  # Create instance to access instance methods
        checkpoint_path = utils.get_checkpoint_path(
            symbol, timeframe, model_type, adapter_type
        )
        return checkpoint_path.exists()

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
