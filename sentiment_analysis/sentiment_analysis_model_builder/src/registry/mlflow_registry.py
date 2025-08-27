# registry/mlflow_registry.py

"""MLflow model registry integration for sentiment analysis models."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from loguru import logger
from mlflow.tracking import MlflowClient

import mlflow

from ..core.config import ModelStage, RegistryConfig


class MLflowRegistry:
    """Handles model registration and versioning with MLflow."""

    def __init__(self, config: RegistryConfig):
        """Initialize the MLflow registry.

        Args:
            config: Registry configuration
        """
        self.config = config

        # Setup environment variables for MLflow
        self._setup_environment()

        # Initialize MLflow client
        self.client = MlflowClient(tracking_uri=config.tracking_uri)

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

    def _setup_environment(self) -> None:
        """Setup environment variables for MLflow and S3."""
        # Set environment variables for MLflow S3 integration
        if self.config.aws_access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = self.config.aws_access_key_id
        if self.config.aws_secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.config.aws_secret_access_key
        if self.config.aws_region:
            os.environ["AWS_DEFAULT_REGION"] = self.config.aws_region
        if self.config.s3_endpoint_url:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.config.s3_endpoint_url

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

    def _create_s3_client(self) -> Optional[boto3.client]:
        """Create S3 client for artifact storage.

        Returns:
            Configured S3 client or None if creation fails
        """
        try:
            return boto3.client(
                "s3",
                endpoint_url=self.config.s3_endpoint_url,
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key,
                region_name=self.config.aws_region,
            )
        except Exception as e:
            logger.error(f"Failed to create S3 client: {e}")
            return None

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
        try:
            logger.info(f"Registering model from run {run_id}")

            # Register model
            model_uri = f"runs:/{run_id}/model"
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=self.config.model_name,
                description=description,
                tags=tags,
            )

            # Upload additional artifacts if S3 is configured
            if self.s3_client:
                self._upload_artifacts_to_s3(model_path, model_version)

            logger.info(
                f"Model registered successfully: {self.config.model_name} v{model_version.version}"
            )
            return model_version.source

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise ValueError(f"Model registration failed: {e}")

    def _upload_artifacts_to_s3(self, model_path: Path, model_version: Any) -> None:
        """Upload model artifacts to S3/MinIO.

        Args:
            model_path: Local path to model artifacts
            model_version: MLflow model version object
        """
        try:
            bucket_name = self.config.artifact_location.split("//")[1].split("/")[0]
            key_prefix = f"models/{self.config.model_name}/{model_version.version}/"

            # Upload model files
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(model_path)
                    s3_key = f"{key_prefix}{relative_path}"

                    self.s3_client.upload_file(str(file_path), bucket_name, s3_key)
                    logger.debug(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")

        except Exception as e:
            logger.warning(f"Failed to upload artifacts to S3: {e}")

    def list_model_versions(
        self, model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all versions of a registered model.

        Args:
            model_name: Name of the model (defaults to config model name)

        Returns:
            List of model version information
        """
        model_name = model_name or self.config.model_name

        try:
            versions = self.client.search_model_versions(
                filter_string=f"name='{model_name}'"
            )

            return [
                {
                    "version": version.version,
                    "stage": version.current_stage,
                    "status": version.status,
                    "creation_timestamp": version.creation_timestamp,
                    "last_updated_timestamp": version.last_updated_timestamp,
                    "description": version.description,
                    "source": version.source,
                    "run_id": version.run_id,
                }
                for version in versions
            ]
        except Exception as e:
            logger.error(f"Failed to list model versions: {e}")
            return []

    def get_model_version(
        self, version: str, model_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get specific model version information.

        Args:
            version: Model version
            model_name: Name of the model (defaults to config model name)

        Returns:
            Model version information or None if not found
        """
        model_name = model_name or self.config.model_name

        try:
            version_info = self.client.get_model_version(
                name=model_name, version=version
            )

            return {
                "version": version_info.version,
                "stage": version_info.current_stage,
                "status": version_info.status,
                "creation_timestamp": version_info.creation_timestamp,
                "last_updated_timestamp": version_info.last_updated_timestamp,
                "description": version_info.description,
                "source": version_info.source,
                "run_id": version_info.run_id,
                "tags": version_info.tags,
            }
        except Exception as e:
            logger.error(f"Failed to get model version {version}: {e}")
            return None

    def transition_model_stage(
        self, version: str, stage: ModelStage, model_name: Optional[str] = None
    ) -> bool:
        """Transition model version to a new stage.

        Args:
            version: Model version
            stage: Target stage
            model_name: Name of the model (defaults to config model name)

        Returns:
            True if successful, False otherwise
        """
        model_name = model_name or self.config.model_name

        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage.value,
                archive_existing_versions=True,
            )
            logger.info(f"Transitioned model {model_name} v{version} to {stage.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            return False

    def delete_model_version(
        self, version: str, model_name: Optional[str] = None
    ) -> bool:
        """Delete a specific model version.

        Args:
            version: Model version to delete
            model_name: Name of the model (defaults to config model name)

        Returns:
            True if successful, False otherwise
        """
        model_name = model_name or self.config.model_name

        try:
            self.client.delete_model_version(name=model_name, version=version)
            logger.info(f"Deleted model {model_name} v{version}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            return False

    def download_model(
        self, version: str, local_path: Path, model_name: Optional[str] = None
    ) -> bool:
        """Download a model version to local path.

        Args:
            version: Model version to download
            local_path: Local path to save the model
            model_name: Name of the model (defaults to config model name)

        Returns:
            True if successful, False otherwise
        """
        model_name = model_name or self.config.model_name

        try:
            model_uri = f"models:/{model_name}/{version}"
            mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri, dst_path=str(local_path)
            )
            logger.info(f"Downloaded model {model_name} v{version} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False

    def get_latest_model_version(
        self, stage: Optional[ModelStage] = None, model_name: Optional[str] = None
    ) -> Optional[str]:
        """Get the latest model version for a given stage.

        Args:
            stage: Model stage to filter by (optional)
            model_name: Name of the model (defaults to config model name)

        Returns:
            Latest model version or None if not found
        """
        model_name = model_name or self.config.model_name

        try:
            if stage:
                versions = self.client.get_latest_versions(
                    name=model_name, stages=[stage.value]
                )
            else:
                versions = self.client.search_model_versions(
                    filter_string=f"name='{model_name}'"
                )

            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                return latest.version
            return None
        except Exception as e:
            logger.error(f"Failed to get latest model version: {e}")
            return None

    def get_model_metrics(self, run_id: str) -> Dict[str, Any]:
        """Get metrics for a specific run.

        Args:
            run_id: MLflow run ID

        Returns:
            Dictionary of metrics
        """
        try:
            run = self.client.get_run(run_id)
            return run.data.metrics
        except Exception as e:
            logger.error(f"Failed to get model metrics: {e}")
            return {}

    def get_model_params(self, run_id: str) -> Dict[str, Any]:
        """Get parameters for a specific run.

        Args:
            run_id: MLflow run ID

        Returns:
            Dictionary of parameters
        """
        try:
            run = self.client.get_run(run_id)
            return run.data.params
        except Exception as e:
            logger.error(f"Failed to get model parameters: {e}")
            return {}

    def search_models(self, filter_string: str = "") -> List[Dict[str, Any]]:
        """Search for models in the registry.

        Args:
            filter_string: MLflow search filter string

        Returns:
            List of matching models
        """
        try:
            models = self.client.search_registered_models(filter_string=filter_string)
            return [
                {
                    "name": model.name,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "description": model.description,
                    "latest_versions": [
                        {
                            "version": version.version,
                            "stage": version.current_stage,
                            "status": version.status,
                        }
                        for version in model.latest_versions
                    ],
                }
                for model in models
            ]
        except Exception as e:
            logger.error(f"Failed to search models: {e}")
            return []

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get a summary of the model registry.

        Returns:
            Registry summary information
        """
        try:
            models = self.search_models()
            total_models = len(models)
            total_versions = sum(len(model["latest_versions"]) for model in models)

            return {
                "total_models": total_models,
                "total_versions": total_versions,
                "models": models,
                "tracking_uri": self.config.tracking_uri,
                "artifact_location": self.config.artifact_location,
                "s3_configured": self._is_s3_configured(),
            }
        except Exception as e:
            logger.error(f"Failed to get registry summary: {e}")
            return {
                "total_models": 0,
                "total_versions": 0,
                "models": [],
                "tracking_uri": self.config.tracking_uri,
                "artifact_location": self.config.artifact_location,
                "s3_configured": self._is_s3_configured(),
            }
