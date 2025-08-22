# registry/mlflow_registry.py

"""MLflow model registry integration for sentiment analysis models."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

from ..core.config import RegistryConfig, ModelStage


class MLflowRegistry:
    """Handles model registration and versioning with MLflow."""

    def __init__(self, config: RegistryConfig):
        """Initialize the MLflow registry.

        Args:
            config: Registry configuration
        """
        self.config = config
        self.client = MlflowClient()

        # Setup MLflow tracking
        mlflow.set_tracking_uri(config.tracking_uri)
        if config.registry_uri:
            mlflow.set_registry_uri(config.registry_uri)

        # Setup S3/MinIO client if configured
        self.s3_client = None
        if self._is_s3_configured():
            self.s3_client = self._create_s3_client()

        logger.info(
            f"MLflow registry initialized with tracking URI: {config.tracking_uri}"
        )

    def _is_s3_configured(self) -> bool:
        """Check if S3/MinIO is configured.

        Returns:
            True if S3/MinIO is configured
        """
        return all(
            [
                self.config.aws_access_key_id,
                self.config.aws_secret_access_key,
                self.config.artifact_location,
            ]
        )

    def _create_s3_client(self) -> boto3.client:
        """Create S3 client for artifact storage.

        Returns:
            Configured S3 client
        """
        session = boto3.Session(
            aws_access_key_id=self.config.aws_access_key_id,
            aws_secret_access_key=self.config.aws_secret_access_key,
            region_name=self.config.aws_region or "us-east-1",
        )

        return session.client("s3", endpoint_url=self.config.s3_endpoint_url)

    def register_model(
        self,
        model_path: Path,
        run_id: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register a trained model in the MLflow model registry.

        Args:
            model_path: Path to the trained model
            run_id: MLflow run ID that produced the model
            description: Model description
            tags: Additional tags for the model

        Returns:
            Model version URI

        Raises:
            ValueError: If registration fails
        """
        logger.info(f"Registering model from run {run_id}")

        try:
            # Create or get the model
            model_name = self.config.model_name

            try:
                model = self.client.get_registered_model(model_name)
                logger.info(f"Found existing model: {model_name}")
            except Exception:
                logger.info(f"Creating new model: {model_name}")
                model = self.client.create_registered_model(
                    name=model_name, description=description, tags=tags or {}
                )

            # Create a new model version
            model_version = self.client.create_model_version(
                name=model_name,
                source=str(model_path),
                run_id=run_id,
                description=description,
                tags=tags or {},
            )

            # Transition to the configured stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=self.config.model_stage.value,
            )

            # Upload artifacts to S3/MinIO if configured
            if self.s3_client and self.config.artifact_location:
                self._upload_artifacts_to_s3(model_path, model_version)

            logger.info(
                f"Model registered successfully: {model_name} v{model_version.version}"
            )
            return f"models:/{model_name}/{model_version.version}"

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise ValueError(f"Model registration failed: {e}")

    def _upload_artifacts_to_s3(self, model_path: Path, model_version: Any) -> None:
        """Upload model artifacts to S3/MinIO.

        Args:
            model_path: Path to the model artifacts
            model_version: MLflow model version object
        """
        try:
            # Parse S3 URI
            s3_uri = self.config.artifact_location
            if s3_uri.startswith("s3://"):
                bucket_name = s3_uri.split("/")[2]
                prefix = "/".join(s3_uri.split("/")[3:])
            else:
                logger.warning(f"Invalid S3 URI format: {s3_uri}")
                return

            # Upload model files
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    # Calculate S3 key
                    relative_path = file_path.relative_to(model_path)
                    s3_key = f"{prefix}/{model_version.name}/v{model_version.version}/{relative_path}"

                    # Upload file
                    self.s3_client.upload_file(str(file_path), bucket_name, s3_key)

                    logger.debug(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")

            logger.info("Model artifacts uploaded to S3/MinIO successfully")

        except Exception as e:
            logger.warning(f"Failed to upload artifacts to S3/MinIO: {e}")

    def list_model_versions(
        self, model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all versions of a model.

        Args:
            model_name: Name of the model (uses configured name if None)

        Returns:
            List of model version information
        """
        model_name = model_name or self.config.model_name

        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")

            version_info = []
            for version in versions:
                version_info.append(
                    {
                        "version": version.version,
                        "stage": version.current_stage,
                        "status": version.status,
                        "run_id": version.run_id,
                        "created_at": version.creation_timestamp,
                        "last_updated": version.last_updated_timestamp,
                        "description": version.description,
                    }
                )

            return version_info

        except Exception as e:
            logger.error(f"Failed to list model versions: {e}")
            return []

    def get_model_version(
        self, version: str, model_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get information about a specific model version.

        Args:
            version: Model version number
            model_name: Name of the model (uses configured name if None)

        Returns:
            Model version information or None if not found
        """
        model_name = model_name or self.config.model_name

        try:
            model_version = self.client.get_model_version(model_name, version)

            return {
                "version": model_version.version,
                "stage": model_version.current_stage,
                "status": model_version.status,
                "run_id": model_version.run_id,
                "source": model_version.source,
                "created_at": model_version.creation_timestamp,
                "last_updated": model_version.last_updated_timestamp,
                "description": model_version.description,
            }

        except Exception as e:
            logger.error(f"Failed to get model version {version}: {e}")
            return None

    def transition_model_stage(
        self, version: str, stage: ModelStage, model_name: Optional[str] = None
    ) -> bool:
        """Transition a model version to a new stage.

        Args:
            version: Model version number
            stage: New stage for the model
            model_name: Name of the model (uses configured name if None)

        Returns:
            True if transition successful, False otherwise
        """
        model_name = model_name or self.config.model_name

        try:
            self.client.transition_model_version_stage(
                name=model_name, version=version, stage=stage.value
            )

            logger.info(f"Model {model_name} v{version} transitioned to {stage.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            return False

    def delete_model_version(
        self, version: str, model_name: Optional[str] = None
    ) -> bool:
        """Delete a model version.

        Args:
            version: Model version number
            model_name: Name of the model (uses configured name if None)

        Returns:
            True if deletion successful, False otherwise
        """
        model_name = model_name or self.config.model_name

        try:
            self.client.delete_model_version(model_name, version)

            logger.info(f"Model version {model_name} v{version} deleted")
            return True

        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            return False

    def download_model(
        self, version: str, local_path: Path, model_name: Optional[str] = None
    ) -> bool:
        """Download a model version to local storage.

        Args:
            version: Model version number
            local_path: Local path to save the model
            model_name: Name of the model (uses configured name if None)

        Returns:
            True if download successful, False otherwise
        """
        model_name = model_name or self.config.model_name

        try:
            # Create local directory
            local_path.mkdir(parents=True, exist_ok=True)

            # Download model
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{model_name}/{version}", dst_path=str(local_path)
            )

            logger.info(f"Model {model_name} v{version} downloaded to {local_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False

    def get_latest_model_version(
        self, stage: Optional[ModelStage] = None, model_name: Optional[str] = None
    ) -> Optional[str]:
        """Get the latest model version for a given stage.

        Args:
            stage: Stage to filter by (None for any stage)
            model_name: Name of the model (uses configured name if None)

        Returns:
            Latest model version number or None if not found
        """
        model_name = model_name or self.config.model_name

        try:
            if stage:
                filter_string = f"name='{model_name}' and current_stage='{stage.value}'"
            else:
                filter_string = f"name='{model_name}'"

            versions = self.client.search_model_versions(filter_string)

            if not versions:
                return None

            # Sort by version number and return the latest
            latest_version = max(versions, key=lambda v: v.version)
            return str(latest_version.version)

        except Exception as e:
            logger.error(f"Failed to get latest model version: {e}")
            return None

    def get_model_metrics(self, run_id: str) -> Dict[str, Any]:
        """Get metrics for a specific MLflow run.

        Args:
            run_id: MLflow run ID

        Returns:
            Dictionary of metrics
        """
        try:
            run = self.client.get_run(run_id)
            return run.data.metrics

        except Exception as e:
            logger.error(f"Failed to get metrics for run {run_id}: {e}")
            return {}

    def get_model_params(self, run_id: str) -> Dict[str, Any]:
        """Get parameters for a specific MLflow run.

        Args:
            run_id: MLflow run ID

        Returns:
            Dictionary of parameters
        """
        try:
            run = self.client.get_run(run_id)
            return run.data.params

        except Exception as e:
            logger.error(f"Failed to get parameters for run {run_id}: {e}")
            return {}

    def search_models(self, filter_string: str = "") -> List[Dict[str, Any]]:
        """Search for models in the registry.

        Args:
            filter_string: Filter string for the search

        Returns:
            List of matching models
        """
        try:
            models = self.client.search_registered_models(filter_string=filter_string)

            model_info = []
            for model in models:
                model_info.append(
                    {
                        "name": model.name,
                        "description": model.description,
                        "latest_versions": [
                            {
                                "version": v.version,
                                "stage": v.current_stage,
                                "status": v.status,
                            }
                            for v in model.latest_versions
                        ],
                        "created_at": model.creation_timestamp,
                        "last_updated": model.last_updated_timestamp,
                    }
                )

            return model_info

        except Exception as e:
            logger.error(f"Failed to search models: {e}")
            return []

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get a summary of the model registry.

        Returns:
            Registry summary information
        """
        try:
            # Get model information
            model_name = self.config.model_name
            versions = self.list_model_versions(model_name)

            # Count versions by stage
            stage_counts = {}
            for version in versions:
                stage = version["stage"]
                stage_counts[stage] = stage_counts.get(stage, 0) + 1

            # Get latest version for each stage
            latest_versions = {}
            for stage in ModelStage:
                latest_version = self.get_latest_model_version(stage, model_name)
                if latest_version:
                    latest_versions[stage.value] = latest_version

            return {
                "model_name": model_name,
                "total_versions": len(versions),
                "stage_counts": stage_counts,
                "latest_versions": latest_versions,
                "tracking_uri": self.config.tracking_uri,
                "artifact_storage": "S3/MinIO" if self.s3_client else "Local",
            }

        except Exception as e:
            logger.error(f"Failed to get registry summary: {e}")
            return {}
