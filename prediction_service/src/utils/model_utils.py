# utils/model_utils.py

"""
Model Utils Orchestrator - Coordinated interface for model operations

This module provides a unified interface that orchestrates different model utility
components including path management, metadata operations, local operations,
and cloud storage operations. Maintains backward compatibility while delegating
to specialized components.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from ..schemas.enums import ModelType, TimeFrame
from ..core.config import get_settings
from .model.path_manager import ModelPathManager
from .model.metadata_manager import ModelMetadataManager
from .model.local_operations import LocalModelOperations
from .model.cloud_operations import CloudModelOperations
from common.logger.logger_factory import LoggerFactory


class ModelUtils:
    """
    Model utilities orchestrator providing unified interface for all model operations.

    This class coordinates between different specialized components:
    - PathManager: Model path generation and management
    - MetadataManager: Model metadata operations
    - LocalOperations: Local model operations
    - CloudOperations: Cloud storage operations
    """

    def __init__(self, storage_client: Optional[Any] = None):
        self.logger = LoggerFactory.get_logger("ModelUtils")
        self.settings = get_settings()

        # Initialize specialized components
        self.path_manager = ModelPathManager()
        self.metadata_manager = ModelMetadataManager(self.path_manager)
        self.local_ops = LocalModelOperations(self.path_manager, self.metadata_manager)
        self.cloud_ops = CloudModelOperations(
            storage_client, self.path_manager, self.metadata_manager, self.local_ops
        )

        # Backward compatibility properties
        self._storage_client = storage_client
        self._experiment_tracker = None

    @property
    def storage_client(self):
        """Storage client instance - delegated to cloud operations"""
        return self.cloud_ops.storage_client

    @property
    def experiment_tracker(self):
        """Lazy-loaded experiment tracker instance"""
        if self._experiment_tracker is None:
            from .dependencies import get_experiment_tracker

            self._experiment_tracker = get_experiment_tracker()
        return self._experiment_tracker

    # ===== Path Management Operations (delegated to PathManager) =====

    def generate_model_identifier(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> str:
        """Generate standardized model identifier - delegated to path manager"""
        return self.path_manager.generate_model_identifier(
            symbol, timeframe, model_type
        )

    def generate_cloud_object_key(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        file_name: Optional[str] = None,
    ) -> str:
        """Generate cloud storage object key - delegated to path manager"""
        return self.path_manager.generate_cloud_object_key(
            symbol, timeframe, model_type, adapter_type, file_name
        )

    def generate_cloud_serving_key(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        file_name: Optional[str] = None,
    ) -> str:
        """Generate cloud serving key - delegated to path manager"""
        return self.path_manager.generate_cloud_serving_key(
            symbol, timeframe, model_type, adapter_type, file_name
        )

    def generate_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Path:
        """Generate model path - delegated to path manager"""
        return self.path_manager.generate_model_path(
            symbol, timeframe, model_type, adapter_type
        )

    def get_checkpoint_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Path:
        """Get checkpoint path - delegated to path manager"""
        return self.path_manager.get_checkpoint_path(
            symbol, timeframe, model_type, adapter_type
        )

    def get_metadata_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Path:
        """Get metadata path - delegated to path manager"""
        return self.path_manager.get_metadata_path(
            symbol, timeframe, model_type, adapter_type
        )

    def get_config_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Path:
        """Get config path - delegated to path manager"""
        return self.path_manager.get_config_path(
            symbol, timeframe, model_type, adapter_type
        )

    def ensure_model_directory(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Path:
        """Ensure model directory - delegated to path manager"""
        return self.path_manager.ensure_model_directory(
            symbol, timeframe, model_type, adapter_type
        )

    def get_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Path:
        """Get model path - delegated to path manager"""
        return self.path_manager.get_model_path(
            symbol, timeframe, model_type, adapter_type
        )

    def get_simple_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Path:
        """Get simple model path - delegated to path manager"""
        return self.path_manager.get_simple_model_path(symbol, timeframe, model_type)

    def get_torchscript_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Path:
        """Get torchscript model path - delegated to path manager"""
        return self.path_manager.get_torchscript_model_path(
            symbol, timeframe, model_type
        )

    def get_torchserve_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Path:
        """Get torchserve model path - delegated to path manager"""
        return self.path_manager.get_torchserve_model_path(
            symbol, timeframe, model_type
        )

    def get_triton_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Path:
        """Get triton model path - delegated to path manager"""
        return self.path_manager.get_triton_model_path(symbol, timeframe, model_type)

    # ===== Metadata Operations (delegated to MetadataManager) =====

    def save_json(self, data: Dict[str, Any], file_path: str) -> None:
        """Save JSON data - delegated to metadata manager"""
        return self.metadata_manager.save_json(data, file_path)

    def load_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load JSON data - delegated to metadata manager"""
        return self.metadata_manager.load_json(file_path)

    # ===== Local Operations (delegated to LocalOperations) =====

    def model_exists(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> bool:
        """Check if model exists - delegated to local operations"""
        return self.local_ops.model_exists(symbol, timeframe, model_type, adapter_type)

    def copy_model_for_adapter(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        source_adapter: str = "simple",
        target_adapter: str = "torchscript",
    ) -> bool:
        """Copy model for adapter - delegated to local operations"""
        return self.local_ops.copy_model_for_adapter(
            symbol, timeframe, model_type, source_adapter, target_adapter
        )

    def ensure_adapter_compatibility(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        target_adapters: List[str] = None,
    ) -> None:
        """Ensure adapter compatibility - delegated to local operations"""
        return self.local_ops.ensure_adapter_compatibility(
            symbol, timeframe, model_type, target_adapters
        )

    # ===== Cloud Operations (delegated to CloudOperations) =====

    async def sync_model_to_cloud(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        run_id: Optional[str] = None,
        force_upload: bool = False,
        enable_upsert: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Sync model to cloud - delegated to cloud operations"""
        return await self.cloud_ops.sync_model_to_cloud(
            symbol,
            timeframe,
            model_type,
            adapter_type,
            run_id,
            force_upload,
            enable_upsert,
        )

    async def sync_model_from_cloud(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """Sync model from cloud - delegated to cloud operations"""
        return await self.cloud_ops.sync_model_from_cloud(
            symbol, timeframe, model_type, adapter_type, force_download
        )

    async def upload_model_to_cloud(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload model to cloud - delegated to cloud operations"""
        return await self.cloud_ops.upload_model_to_cloud(
            symbol, timeframe, model_type, adapter_type, run_id
        )

    async def download_model_from_cloud(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """Download model from cloud - delegated to cloud operations"""
        return await self.cloud_ops.download_model_from_cloud(
            symbol, timeframe, model_type, adapter_type, force_download
        )

    async def model_exists_in_cloud(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> bool:
        """Check if model exists in cloud - delegated to cloud operations"""
        return await self.cloud_ops.model_exists_in_cloud(
            symbol, timeframe, model_type, adapter_type
        )

    async def save_model_metadata_to_cloud(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        metadata: Dict[str, Any],
        adapter_type: str = "simple",
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save model metadata to cloud - delegated to cloud operations"""
        return await self.cloud_ops.save_model_metadata_to_cloud(
            symbol, timeframe, model_type, metadata, adapter_type, run_id
        )

    async def load_model_with_cloud_fallback(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        force_cloud_download: bool = False,
    ) -> Dict[str, Any]:
        """Load model with cloud fallback - delegated to cloud operations"""
        return await self.cloud_ops.load_model_with_cloud_fallback(
            symbol, timeframe, model_type, adapter_type, force_cloud_download
        )

    async def health_check_cloud_storage(self) -> Dict[str, Any]:
        """Health check cloud storage - delegated to cloud operations"""
        return await self.cloud_ops.health_check_cloud_storage()

    async def load_model_for_serving(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        force_cloud_download: bool = False,
    ) -> Dict[str, Any]:
        """
        Load model for serving with cloud-first strategy - delegated to cloud operations.

        This method implements cloud-first loading for serving operations:
        1. Try to load from cloud storage first
        2. Fall back to local storage if cloud fails
        3. Cache the model locally for future use
        """
        return await self.cloud_ops.load_model_with_cloud_fallback(
            symbol, timeframe, model_type, adapter_type, force_cloud_download
        )

    # ===== Static Methods for Backward Compatibility =====

    @staticmethod
    def save_model_metadata(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        metadata: Dict[str, Any],
        adapter_type: str = "simple",
    ) -> Path:
        """Save model metadata - static method for backward compatibility"""
        utils = ModelUtils()
        return utils.metadata_manager.save_model_metadata(
            symbol, timeframe, model_type, metadata, adapter_type
        )

    @staticmethod
    def load_model_metadata(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Optional[Dict[str, Any]]:
        """Load model metadata - static method for backward compatibility"""
        utils = ModelUtils()
        return utils.metadata_manager.load_model_metadata(
            symbol, timeframe, model_type, adapter_type
        )

    @staticmethod
    def model_exists(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> bool:
        """Check if model exists - static method for backward compatibility"""
        utils = ModelUtils()
        return utils.local_ops.model_exists(symbol, timeframe, model_type, adapter_type)

    @staticmethod
    def list_available_models(base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """List available models - static method for backward compatibility"""
        utils = ModelUtils()
        return utils.local_ops.list_available_models(base_dir)

    @staticmethod
    def delete_model(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> bool:
        """Delete model - static method for backward compatibility"""
        utils = ModelUtils()
        return utils.local_ops.delete_model(symbol, timeframe, model_type, adapter_type)

    @staticmethod
    def get_model_size(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Optional[int]:
        """Get model size - static method for backward compatibility"""
        utils = ModelUtils()
        return utils.local_ops.get_model_size(
            symbol, timeframe, model_type, adapter_type
        )

    @staticmethod
    def backup_model(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        backup_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Backup model - static method for backward compatibility"""
        utils = ModelUtils()
        return utils.local_ops.backup_model(
            symbol, timeframe, model_type, adapter_type, backup_dir
        )
