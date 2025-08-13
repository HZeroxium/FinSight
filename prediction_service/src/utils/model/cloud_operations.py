# utils/model/cloud_operations.py

"""
Cloud Model Operations

This module handles cloud storage operations for models including upload, download,
sync operations, and cloud-first loading strategies.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from ...schemas.enums import ModelType, TimeFrame
from ...core.config import get_settings
from .path_manager import ModelPathManager
from .metadata_manager import ModelMetadataManager
from .local_operations import LocalModelOperations
from common.logger.logger_factory import LoggerFactory


class CloudModelOperations:
    """Handles cloud storage operations for models"""

    def __init__(
        self,
        storage_client: Optional[Any] = None,
        path_manager: Optional[ModelPathManager] = None,
        metadata_manager: Optional[ModelMetadataManager] = None,
        local_ops: Optional[LocalModelOperations] = None,
    ):
        self.logger = LoggerFactory.get_logger("CloudModelOperations")
        self.settings = get_settings()
        self._storage_client = storage_client
        self.path_manager = path_manager or ModelPathManager()
        self.metadata_manager = metadata_manager or ModelMetadataManager(
            self.path_manager
        )
        self.local_ops = local_ops or LocalModelOperations(
            self.path_manager, self.metadata_manager
        )

    @property
    def storage_client(self):
        """Storage client instance - injected via dependency injection"""
        if self._storage_client is None and self.settings.enable_cloud_storage:
            # Fallback to getting from dependencies if not explicitly injected
            from ..dependencies import get_storage_client

            self._storage_client = get_storage_client()
        return self._storage_client

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
        """
        Synchronize local model to cloud storage with experiment tracking and upsert support.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            adapter_type: Adapter type
            run_id: Optional experiment run ID for tracking
            force_upload: Whether to force upload even if cloud version exists
            enable_upsert: Whether to enable overwriting existing cloud files (overrides global setting)

        Returns:
            Synchronization result dictionary
        """
        model_identifier = (
            f"{symbol}_{timeframe.value}_{model_type.value}_{adapter_type}"
        )

        # Determine upsert setting (parameter overrides global setting)
        should_upsert = (
            enable_upsert
            if enable_upsert is not None
            else self.settings.enable_cloud_upsert
        )

        try:
            # Check if local model exists
            if not self.local_ops.model_exists(
                symbol, timeframe, model_type, adapter_type
            ):
                self.logger.error(
                    f"❌ Local model does not exist for cloud sync: {model_identifier}"
                )
                return {"success": False, "error": "Local model does not exist"}

            # Check cloud model existence
            cloud_exists = await self.model_exists_in_cloud(
                symbol, timeframe, model_type, adapter_type
            )

            # Determine action based on existence and settings
            if cloud_exists and not force_upload and not should_upsert:
                self.logger.info(
                    f"⏭️ Cloud model exists and upsert disabled, skipping: {model_identifier}"
                )
                return {
                    "success": True,
                    "message": "Model already exists in cloud and upsert is disabled",
                    "action": "skipped",
                }
            elif cloud_exists and (force_upload or should_upsert):
                self.logger.info(
                    f"🔄 Cloud model exists but overwriting (upsert={should_upsert}, force={force_upload}): {model_identifier}"
                )
                action = "overwritten"
            else:
                self.logger.info(f"🆕 Uploading new model to cloud: {model_identifier}")
                action = "uploaded"

            # Upload model to cloud
            upload_result = await self.upload_model_to_cloud(
                symbol, timeframe, model_type, adapter_type, run_id
            )

            if upload_result["success"]:
                # Upload metadata to cloud
                local_metadata = self.metadata_manager.load_model_metadata(
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

                upload_result["action"] = action
                self.logger.info(
                    f"✅ Successfully {action} model to cloud: {model_identifier}"
                )

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
                local_exists = self.local_ops.model_exists(
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

            local_model_path = self.path_manager.generate_model_path(
                symbol, timeframe, model_type, adapter_type
            )

            if not local_model_path.exists():
                return {"success": False, "error": "Local model path does not exist"}

            # Generate cloud object key
            model_id = self.path_manager.generate_model_identifier(
                symbol, timeframe, model_type
            )
            cloud_base_key = self.path_manager.generate_cloud_object_key(
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

            local_model_path = self.path_manager.generate_model_path(
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
            cloud_base_key = self.path_manager.generate_cloud_object_key(
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

            cloud_base_key = self.path_manager.generate_cloud_object_key(
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
            cloud_base_key = self.path_manager.generate_cloud_object_key(
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
        model_identifier = (
            f"{symbol}_{timeframe.value}_{model_type.value}_{adapter_type}"
        )
        self.logger.info(
            f"🔍 Loading model with cloud-first strategy: {model_identifier}"
        )

        try:
            # Check if cloud storage is enabled
            if not self.settings.enable_cloud_storage or not self.storage_client:
                self.logger.warning(
                    f"☁️ Cloud storage disabled - using local-only strategy for {model_identifier}"
                )
                # Fallback to local only
                local_exists = self.local_ops.model_exists(
                    symbol, timeframe, model_type, adapter_type
                )
                if local_exists:
                    self.logger.info(f"✅ Model found locally: {model_identifier}")
                    return {
                        "success": True,
                        "source": "local",
                        "path": str(
                            self.path_manager.generate_model_path(
                                symbol, timeframe, model_type, adapter_type
                            )
                        ),
                    }
                else:
                    self.logger.error(
                        f"❌ Model not found locally and cloud storage is disabled: {model_identifier}"
                    )
                    return {
                        "success": False,
                        "error": "Model not found locally and cloud storage is disabled",
                    }

            # Cloud-first strategy implementation
            self.logger.info(
                f"☁️ Checking model availability - Cloud storage enabled for {model_identifier}"
            )

            local_exists = self.local_ops.model_exists(
                symbol, timeframe, model_type, adapter_type
            )
            self.logger.info(
                f"📁 Local model exists: {local_exists} for {model_identifier}"
            )

            cloud_exists = await self.model_exists_in_cloud(
                symbol, timeframe, model_type, adapter_type
            )
            self.logger.info(
                f"☁️ Cloud model exists: {cloud_exists} for {model_identifier}"
            )

            # Implement true cloud-first strategy
            if force_cloud_download or not local_exists:
                if cloud_exists:
                    self.logger.info(
                        f"🚀 CLOUD-FIRST: Downloading model from cloud: {model_identifier}"
                    )
                    download_result = await self.download_model_from_cloud(
                        symbol, timeframe, model_type, adapter_type, force_download=True
                    )

                    if download_result["success"]:
                        self.logger.info(
                            f"✅ Successfully downloaded model from cloud: {model_identifier}"
                        )
                        return {
                            "success": True,
                            "source": "cloud",
                            "path": download_result["local_path"],
                            "downloaded_files": download_result.get(
                                "downloaded_files", []
                            ),
                        }
                    else:
                        self.logger.error(
                            f"❌ Failed to download from cloud: {download_result.get('error')} for {model_identifier}"
                        )
                        # Continue to check local fallback

                # If cloud download failed or cloud doesn't exist, check local
                if local_exists:
                    self.logger.info(
                        f"📁 FALLBACK: Using local model as cloud download failed: {model_identifier}"
                    )
                    return {
                        "success": True,
                        "source": "local",
                        "path": str(
                            self.path_manager.generate_model_path(
                                symbol, timeframe, model_type, adapter_type
                            )
                        ),
                    }
                else:
                    self.logger.error(
                        f"❌ Model not found in cloud or locally: {model_identifier}"
                    )
                    return {
                        "success": False,
                        "error": "Model not found in cloud or locally",
                    }

            # If local exists and we're not forcing cloud download
            if local_exists:
                self.logger.info(
                    f"📁 Using local model (cloud-first not forced): {model_identifier}"
                )

                # Check if we should sync to cloud (if cloud doesn't exist)
                if not cloud_exists and self.settings.enable_model_cloud_sync:
                    self.logger.info(
                        f"📤 Background sync: Uploading local model to cloud: {model_identifier}"
                    )
                    upload_result = await self.sync_model_to_cloud(
                        symbol, timeframe, model_type, adapter_type
                    )

                    if upload_result["success"]:
                        self.logger.info(
                            f"✅ Successfully synced local model to cloud: {model_identifier}"
                        )
                    else:
                        self.logger.warning(
                            f"⚠️ Failed to sync model to cloud: {upload_result.get('error')} for {model_identifier}"
                        )

                return {
                    "success": True,
                    "source": "local",
                    "path": str(
                        self.path_manager.generate_model_path(
                            symbol, timeframe, model_type, adapter_type
                        )
                    ),
                }

            # Neither local nor cloud exists
            self.logger.error(f"❌ Model not found anywhere: {model_identifier}")
            return {
                "success": False,
                "error": "Model not found locally or in cloud",
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
