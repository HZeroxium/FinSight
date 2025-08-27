# utils/model/local_operations.py

"""
Local Model Operations

This module handles local model operations including existence checks,
listing, deletion, backup, and size calculations.
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.logger.logger_factory import LoggerFactory

from ...core.config import get_settings
from ...schemas.enums import ModelType, TimeFrame
from .metadata_manager import ModelMetadataManager
from .path_manager import ModelPathManager


class LocalModelOperations:
    """Handles local model operations"""

    def __init__(
        self,
        path_manager: Optional[ModelPathManager] = None,
        metadata_manager: Optional[ModelMetadataManager] = None,
    ):
        self.logger = LoggerFactory.get_logger("LocalModelOperations")
        self.settings = get_settings()
        self.path_manager = path_manager or ModelPathManager()
        self.metadata_manager = metadata_manager or ModelMetadataManager(
            self.path_manager
        )

    def model_exists(
        self,
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
        checkpoint_path = self.path_manager.get_checkpoint_path(
            symbol, timeframe, model_type, adapter_type
        )
        return checkpoint_path.exists()

    def list_available_models(
        self, base_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        List all available models

        Args:
            base_dir: Base directory to search (if None, uses settings)

        Returns:
            List of model information dictionaries
        """
        if base_dir is None:
            base_dir = self.settings.models_dir

        if not base_dir.exists():
            return []

        models = []

        # Check both old flat structure and new adapter-based structure
        for adapter_dir in base_dir.iterdir():
            if adapter_dir.is_dir():
                # Check if this is an adapter directory (contains model subdirectories)
                for model_dir in adapter_dir.iterdir():
                    if model_dir.is_dir():
                        metadata_path = model_dir / "metadata.json"
                        if metadata_path.exists():
                            try:
                                metadata = self.metadata_manager.load_json(
                                    str(metadata_path)
                                )
                                if metadata:
                                    models.append(metadata)
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to load metadata from {metadata_path}: {e}"
                                )
                                continue
                        else:
                            # Try to infer model info from directory structure if no metadata
                            try:
                                parts = model_dir.name.split("_")
                                if len(parts) >= 3:
                                    symbol = parts[0]
                                    timeframe_str = parts[1]
                                    model_type_str = "_".join(parts[2:])

                                    # Basic model info without full metadata
                                    model_info = {
                                        "symbol": symbol,
                                        "timeframe": timeframe_str,
                                        "model_type": model_type_str,
                                        "adapter_type": adapter_dir.name,
                                        "model_path": str(model_dir),
                                        "created_at": datetime.fromtimestamp(
                                            model_dir.stat().st_ctime
                                        ).isoformat(),
                                        "metadata_available": False,
                                    }
                                    models.append(model_info)
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to parse model directory {model_dir}: {e}"
                                )
                                continue

        return models

    def delete_model(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> bool:
        """
        Delete model and all associated files from local storage

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            adapter_type: Adapter type for storage location

        Returns:
            True if deletion successful, False otherwise
        """
        model_dir = self.path_manager.generate_model_path(
            symbol, timeframe, model_type, adapter_type
        )

        if not model_dir.exists():
            return False

        try:
            shutil.rmtree(model_dir)
            self.logger.info(f"Deleted model: {model_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_dir}: {e}")
            return False

    def get_model_size(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Optional[int]:
        """
        Get local model size in bytes

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            adapter_type: Adapter type

        Returns:
            Model size in bytes or None if model doesn't exist
        """
        checkpoint_path = self.path_manager.get_checkpoint_path(
            symbol, timeframe, model_type, adapter_type
        )

        if not checkpoint_path.exists():
            return None

        return checkpoint_path.stat().st_size

    def backup_model(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        backup_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Create backup of model in local storage

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            adapter_type: Adapter type
            backup_dir: Backup directory (if None, creates in models directory)

        Returns:
            Path to backup or None if failed
        """
        model_dir = self.path_manager.generate_model_path(
            symbol, timeframe, model_type, adapter_type
        )

        if not model_dir.exists():
            return None

        if backup_dir is None:
            backup_dir = self.settings.models_dir / "backups"

        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = self.path_manager.generate_model_identifier(
            symbol, timeframe, model_type
        )
        backup_path = backup_dir / f"{model_id}_{adapter_type}_{timestamp}"

        try:
            shutil.copytree(model_dir, backup_path)
            self.logger.info(f"Created model backup: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Failed to create model backup: {e}")
            return None

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
            source_dir = self.path_manager.generate_model_path(
                symbol, timeframe, model_type, source_adapter
            )
            target_dir = self.path_manager.generate_model_path(
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
        Ensure model is available for multiple adapters.

        This method is deprecated and kept for backward compatibility.
        New training should use ModelFormatConverter for proper format conversion.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            target_adapters: List of target adapter types
        """
        from ...core.constants import FacadeConstants

        if target_adapters is None:
            target_adapters = FacadeConstants.SUPPORTED_ADAPTERS

        # Check if model exists in simple adapter (training default)
        simple_exists = self.model_exists(
            symbol, timeframe, model_type, FacadeConstants.ADAPTER_SIMPLE
        )
        if not simple_exists:
            self.logger.warning(
                f"No model found in simple adapter for {symbol}_{timeframe}_{model_type}"
            )
            return

        # Copy to other adapters if they don't exist (fallback behavior)
        for adapter in target_adapters:
            if adapter != FacadeConstants.ADAPTER_SIMPLE:
                if not self.model_exists(symbol, timeframe, model_type, adapter):
                    self.copy_model_for_adapter(
                        symbol,
                        timeframe,
                        model_type,
                        FacadeConstants.ADAPTER_SIMPLE,
                        adapter,
                    )

        self.logger.warning(
            "ensure_adapter_compatibility uses basic file copying. "
            "For proper format conversion, use ModelFormatConverter."
        )

    def get_model_directory_size(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Optional[int]:
        """
        Get total size of model directory in bytes

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            adapter_type: Adapter type

        Returns:
            Total directory size in bytes or None if directory doesn't exist
        """
        model_dir = self.path_manager.generate_model_path(
            symbol, timeframe, model_type, adapter_type
        )

        if not model_dir.exists():
            return None

        try:
            total_size = 0
            for file_path in model_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception as e:
            self.logger.error(f"Failed to calculate directory size: {e}")
            return None
