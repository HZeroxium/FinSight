# utils/model_utils.py

"""
Model utilities for centralized model path management and operations.

This module provides utilities for model path generation, model metadata handling,
and consistent model saving/loading operations across both local and cloud storage.
Integrates with experiment tracker for comprehensive artifact management.
"""
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from ..schemas.enums import ModelType, TimeFrame
from ..core.config import get_settings
from common.logger.logger_factory import LoggerFactory


class ModelUtils:
    """Utilities for model path management and operations with cloud storage support"""

    def __init__(self, storage_client: Optional[Any] = None):
        self.logger = LoggerFactory.get_logger("ModelUtils")
        self.settings = get_settings()
        self._storage_client = storage_client
        self._experiment_tracker = None

    @property
    def storage_client(self):
        """Storage client instance - now injected via dependency injection"""
        if self._storage_client is None and self.settings.enable_cloud_storage:
            # Fallback to getting from dependencies if not explicitly injected (shouldn't happen in DI setup)
            from .dependencies import get_storage_client

            self._storage_client = get_storage_client()
        return self._storage_client

    @property
    def experiment_tracker(self):
        """Lazy-loaded experiment tracker instance"""
        if self._experiment_tracker is None:
            from .dependencies import get_experiment_tracker

            self._experiment_tracker = get_experiment_tracker()
        return self._experiment_tracker

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

        Structure: models/{adapter_type}/{symbol}_{timeframe}_{model_type}/{file_name}

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
        base_key = f"models/{adapter_type}/{model_id}"

        if file_name:
            return f"{base_key}/{file_name}"
        return base_key

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
        Ensure model is available for multiple adapters.

        This method is deprecated and kept for backward compatibility.
        New training should use ModelFormatConverter for proper format conversion.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            target_adapters: List of target adapter types
        """
        from ..core.constants import FacadeConstants

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
        import shutil

        utils = ModelUtils()
        model_dir = utils.generate_model_path(
            symbol, timeframe, model_type, adapter_type
        )

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
        utils = ModelUtils()
        checkpoint_path = utils.get_checkpoint_path(
            symbol, timeframe, model_type, adapter_type
        )

        if not checkpoint_path.exists():
            return None

        return checkpoint_path.stat().st_size

    @staticmethod
    def backup_model(
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
        import shutil

        utils = ModelUtils()
        model_dir = utils.generate_model_path(
            symbol, timeframe, model_type, adapter_type
        )

        if not model_dir.exists():
            return None

        if backup_dir is None:
            settings = get_settings()
            backup_dir = settings.models_dir / "backups"

        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = utils.generate_model_identifier(symbol, timeframe, model_type)
        backup_path = backup_dir / f"{model_id}_{timestamp}"

        try:
            shutil.copytree(model_dir, backup_path)
            return backup_path
        except Exception:
            return None

    async def sync_model_to_cloud(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        run_id: Optional[str] = None,
        force_upload: bool = False,
    ) -> Dict[str, Any]:
        """
        Synchronize local model to cloud storage with experiment tracking.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            adapter_type: Adapter type
            run_id: Optional experiment run ID for tracking
            force_upload: Whether to force upload even if cloud version exists

        Returns:
            Synchronization result dictionary
        """
        try:
            # Check if local model exists
            if not self.model_exists(symbol, timeframe, model_type, adapter_type):
                return {"success": False, "error": "Local model does not exist"}

            # Check if cloud model exists (unless force upload)
            if not force_upload:
                cloud_exists = await self.model_exists_in_cloud(
                    symbol, timeframe, model_type, adapter_type
                )
                if cloud_exists:
                    return {
                        "success": True,
                        "message": "Model already exists in cloud",
                        "action": "skipped",
                    }

            # Upload model to cloud
            upload_result = await self.upload_model_to_cloud(
                symbol, timeframe, model_type, adapter_type, run_id
            )

            if upload_result["success"]:
                # Upload metadata to cloud
                local_metadata = self.load_model_metadata(
                    symbol, timeframe, model_type, adapter_type
                )

                if local_metadata:
                    metadata_result = await self.save_model_metadata_to_cloud(
                        symbol,
                        timeframe,
                        model_type,
                        local_metadata,
                        adapter_type,
                        run_id,
                    )
                    upload_result["metadata_uploaded"] = metadata_result["success"]

                upload_result["action"] = "uploaded"

            return upload_result

        except Exception as e:
            self.logger.error(f"Failed to sync model to cloud: {e}")
            return {"success": False, "error": str(e)}

    async def sync_model_from_cloud(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """
        Synchronize cloud model to local storage.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            adapter_type: Adapter type
            force_download: Whether to force download even if local version exists

        Returns:
            Synchronization result dictionary
        """
        try:
            # Check if cloud model exists
            cloud_exists = await self.model_exists_in_cloud(
                symbol, timeframe, model_type, adapter_type
            )

            if not cloud_exists:
                return {"success": False, "error": "Model does not exist in cloud"}

            # Check if local model exists (unless force download)
            if not force_download:
                local_exists = self.model_exists(
                    symbol, timeframe, model_type, adapter_type
                )
                if local_exists:
                    return {
                        "success": True,
                        "message": "Model already exists locally",
                        "action": "skipped",
                    }

            # Download model from cloud
            download_result = await self.download_model_from_cloud(
                symbol, timeframe, model_type, adapter_type, force_download
            )

            if download_result["success"]:
                download_result["action"] = "downloaded"

            return download_result

        except Exception as e:
            self.logger.error(f"Failed to sync model from cloud: {e}")
            return {"success": False, "error": str(e)}

    async def upload_model_to_cloud(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload model to cloud storage.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            adapter_type: Adapter type
            run_id: Optional experiment run ID for tracking

        Returns:
            Upload result dictionary
        """
        try:
            if not self.storage_client:
                return {"success": False, "error": "Cloud storage not available"}

            local_model_path = self.generate_model_path(
                symbol, timeframe, model_type, adapter_type
            )

            if not local_model_path.exists():
                return {"success": False, "error": "Local model path does not exist"}

            # Generate cloud object key
            model_id = self.generate_model_identifier(symbol, timeframe, model_type)
            cloud_base_key = self.generate_cloud_object_key(
                symbol, timeframe, model_type, adapter_type
            )

            # Upload all files in the model directory
            uploaded_files = []
            failed_files = []

            for file_path in local_model_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_model_path)
                    cloud_key = f"{cloud_base_key}/{relative_path}"

                    try:
                        # Add metadata for tracking
                        metadata = {
                            "model_id": model_id,
                            "symbol": symbol,
                            "timeframe": timeframe.value,
                            "model_type": model_type.value,
                            "adapter_type": adapter_type,
                            "upload_timestamp": datetime.now().isoformat(),
                        }
                        if run_id:
                            metadata["run_id"] = run_id

                        success = await self.storage_client.upload_file(
                            local_file_path=str(file_path),
                            object_key=cloud_key,
                            metadata=metadata,
                        )

                        if success:
                            uploaded_files.append(str(relative_path))
                        else:
                            failed_files.append(str(relative_path))

                    except Exception as e:
                        self.logger.error(f"Failed to upload {file_path}: {e}")
                        failed_files.append(str(relative_path))

            if failed_files:
                return {
                    "success": False,
                    "error": f"Failed to upload files: {failed_files}",
                    "uploaded_files": uploaded_files,
                    "failed_files": failed_files,
                }

            self.logger.info(f"Successfully uploaded model {model_id} to cloud storage")
            return {
                "success": True,
                "uploaded_files": uploaded_files,
                "cloud_key": cloud_base_key,
            }

        except Exception as e:
            self.logger.error(f"Failed to upload model to cloud: {e}")
            return {"success": False, "error": str(e)}

    async def download_model_from_cloud(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """
        Download model from cloud storage.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            adapter_type: Adapter type
            force_download: Whether to force download

        Returns:
            Download result dictionary
        """
        try:
            if not self.storage_client:
                return {"success": False, "error": "Cloud storage not available"}

            local_model_path = self.generate_model_path(
                symbol, timeframe, model_type, adapter_type
            )

            # Check if local model exists (unless force download)
            if not force_download and local_model_path.exists():
                return {
                    "success": True,
                    "message": "Local model already exists",
                    "action": "skipped",
                }

            # Generate cloud object key
            cloud_base_key = self.generate_cloud_object_key(
                symbol, timeframe, model_type, adapter_type
            )

            # List objects in cloud with the model prefix
            try:
                objects = await self.storage_client.list_objects(prefix=cloud_base_key)
            except Exception as e:
                self.logger.error(f"Failed to list cloud objects: {e}")
                return {"success": False, "error": f"Failed to list cloud objects: {e}"}

            if not objects:
                return {"success": False, "error": "No model files found in cloud"}

            # Create local directory
            local_model_path.mkdir(parents=True, exist_ok=True)

            # Download all files
            downloaded_files = []
            failed_files = []

            for obj in objects:
                try:
                    # Extract relative path from cloud key
                    relative_path = obj["key"].replace(f"{cloud_base_key}/", "")
                    if not relative_path:  # Skip the base directory object
                        continue

                    local_file_path = local_model_path / relative_path
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)

                    success = await self.storage_client.download_file(
                        object_key=obj["key"],
                        local_file_path=str(local_file_path),
                    )

                    if success:
                        downloaded_files.append(relative_path)
                    else:
                        failed_files.append(relative_path)

                except Exception as e:
                    self.logger.error(f"Failed to download {obj['key']}: {e}")
                    failed_files.append(obj["key"])

            if failed_files:
                return {
                    "success": False,
                    "error": f"Failed to download files: {failed_files}",
                    "downloaded_files": downloaded_files,
                    "failed_files": failed_files,
                }

            self.logger.info(
                f"Successfully downloaded model from cloud storage: {len(downloaded_files)} files"
            )
            return {
                "success": True,
                "downloaded_files": downloaded_files,
                "local_path": str(local_model_path),
            }

        except Exception as e:
            self.logger.error(f"Failed to download model from cloud: {e}")
            return {"success": False, "error": str(e)}

    async def model_exists_in_cloud(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> bool:
        """
        Check if model exists in cloud storage.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            adapter_type: Adapter type

        Returns:
            True if model exists in cloud
        """
        try:
            if not self.storage_client:
                return False

            cloud_base_key = self.generate_cloud_object_key(
                symbol, timeframe, model_type, adapter_type
            )

            # Check for model config file existence
            config_key = f"{cloud_base_key}/config.json"
            return await self.storage_client.object_exists(config_key)

        except Exception as e:
            self.logger.error(f"Failed to check cloud model existence: {e}")
            return False

    async def save_model_metadata_to_cloud(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        metadata: Dict[str, Any],
        adapter_type: str = "simple",
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save model metadata to cloud storage.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            metadata: Model metadata
            adapter_type: Adapter type
            run_id: Optional experiment run ID

        Returns:
            Save result dictionary
        """
        try:
            if not self.storage_client:
                return {"success": False, "error": "Cloud storage not available"}

            # Add metadata
            metadata["cloud_upload_timestamp"] = datetime.now().isoformat()
            if run_id:
                metadata["run_id"] = run_id

            # Generate cloud object key
            cloud_base_key = self.generate_cloud_object_key(
                symbol, timeframe, model_type, adapter_type
            )
            metadata_key = f"{cloud_base_key}/cloud_metadata.json"

            # Convert metadata to JSON
            import json

            metadata_json = json.dumps(metadata, indent=2, default=str)

            success = await self.storage_client.upload_bytes(
                data=metadata_json.encode("utf-8"),
                object_key=metadata_key,
                content_type="application/json",
            )

            if success:
                self.logger.info(f"Saved model metadata to cloud: {metadata_key}")
                return {"success": True, "metadata_key": metadata_key}
            else:
                return {"success": False, "error": "Failed to upload metadata"}

        except Exception as e:
            self.logger.error(f"Failed to save model metadata to cloud: {e}")
            return {"success": False, "error": str(e)}

    async def load_model_with_cloud_fallback(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        force_cloud_download: bool = False,
    ) -> Dict[str, Any]:
        """
        Load model with cloud-first strategy and local fallback.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            adapter_type: Adapter type
            force_cloud_download: Whether to force download from cloud

        Returns:
            Load result dictionary
        """
        try:
            settings = get_settings()

            # Check if cloud storage is enabled
            if not settings.enable_cloud_storage or not self.storage_client:
                # Fallback to local only
                local_exists = self.model_exists(
                    symbol, timeframe, model_type, adapter_type
                )
                if local_exists:
                    return {
                        "success": True,
                        "source": "local",
                        "path": str(
                            self.generate_model_path(
                                symbol, timeframe, model_type, adapter_type
                            )
                        ),
                    }
                else:
                    return {"success": False, "error": "Model not found locally"}

            # Cloud-first strategy
            local_exists = self.model_exists(
                symbol, timeframe, model_type, adapter_type
            )
            cloud_exists = await self.model_exists_in_cloud(
                symbol, timeframe, model_type, adapter_type
            )

            if not local_exists and not cloud_exists:
                return {
                    "success": False,
                    "error": "Model not found locally or in cloud",
                }

            # If local doesn't exist but cloud does, download from cloud
            if not local_exists and cloud_exists:
                self.logger.info(
                    f"Model not found locally, downloading from cloud: {symbol}_{timeframe.value}_{model_type.value}"
                )
                download_result = await self.download_model_from_cloud(
                    symbol, timeframe, model_type, adapter_type, force_download=True
                )

                if download_result["success"]:
                    return {
                        "success": True,
                        "source": "cloud",
                        "path": download_result["local_path"],
                        "downloaded_files": download_result.get("downloaded_files", []),
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to download from cloud: {download_result.get('error')}",
                    }

            # If local exists but cloud doesn't, upload to cloud (if enabled)
            if local_exists and not cloud_exists and settings.enable_model_cloud_sync:
                self.logger.info(
                    f"Model exists locally but not in cloud, uploading: {symbol}_{timeframe.value}_{model_type.value}"
                )
                upload_result = await self.sync_model_to_cloud(
                    symbol, timeframe, model_type, adapter_type
                )

                if not upload_result["success"]:
                    self.logger.warning(
                        f"Failed to sync model to cloud: {upload_result.get('error')}"
                    )

            # Return local model path
            return {
                "success": True,
                "source": "local",
                "path": str(
                    self.generate_model_path(
                        symbol, timeframe, model_type, adapter_type
                    )
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to load model with cloud fallback: {e}")
            return {"success": False, "error": str(e)}

    async def health_check_cloud_storage(self) -> Dict[str, Any]:
        """
        Perform health check on cloud storage.

        Returns:
            Health check result dictionary
        """
        try:
            if not self.storage_client:
                return {
                    "status": "disabled",
                    "message": "Cloud storage not configured",
                    "details": {},
                }

            # Test basic operations
            test_key = "health_check/test.txt"
            test_data = b"health_check_data"
            test_metadata = {
                "health_check": "true",
                "timestamp": datetime.now().isoformat(),
            }

            # Test upload
            upload_success = await self.storage_client.upload_bytes(
                data=test_data,
                object_key=test_key,
                metadata=test_metadata,
            )

            if not upload_success:
                return {
                    "status": "failed",
                    "message": "Failed to upload test file",
                    "details": {"operation": "upload"},
                }

            # Test download
            download_success = await self.storage_client.download_bytes(test_key)
            if not download_success or download_success != test_data:
                return {
                    "status": "failed",
                    "message": "Failed to download test file or data mismatch",
                    "details": {"operation": "download"},
                }

            # Test delete
            delete_success = await self.storage_client.delete_object(test_key)
            if not delete_success:
                return {
                    "status": "failed",
                    "message": "Failed to delete test file",
                    "details": {"operation": "delete"},
                }

            return {
                "status": "healthy",
                "message": "Cloud storage is working correctly",
                "details": {
                    "operations_tested": ["upload", "download", "delete"],
                    "timestamp": datetime.now().isoformat(),
                },
            }

        except Exception as e:
            self.logger.error(f"Cloud storage health check failed: {e}")
            return {
                "status": "failed",
                "message": f"Health check failed: {str(e)}",
                "details": {"error": str(e)},
            }
