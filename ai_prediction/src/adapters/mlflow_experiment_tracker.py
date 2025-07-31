# adapters/mlflow_experiment_tracker.py

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import asyncio

try:
    import mlflow
    import mlflow.tracking
    from mlflow.entities import Experiment, Run, ViewType
    from mlflow.exceptions import RestException, MlflowException

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

from ..interfaces.experiment_tracker_interface import (
    IExperimentTracker,
    ModelStage,
    RunStatus,
    ModelRegistryInfo,
    ExperimentInfo,
    RunInfo,
    ModelArtifact,
)
from ..schemas.enums import TimeFrame, ModelType
from ..core.config import get_settings
from common.logger.logger_factory import LoggerFactory


class MLflowExperimentTracker(IExperimentTracker):
    """
    MLflow-based experiment tracker for cloud storage and advanced experiment management.

    This implementation provides full MLflow capabilities including:
    - Remote tracking server support
    - Object storage for artifacts
    - Advanced model registry features
    - Enterprise-grade experiment tracking
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        artifact_root: Optional[str] = None,
    ):
        """
        Initialize MLflow experiment tracker.

        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Default experiment name
            artifact_root: Root location for artifacts
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "MLflow is not installed. Please install with: pip install mlflow"
            )

        self.settings = get_settings()
        self.logger = LoggerFactory.get_logger("MLflowExperimentTracker")

        # Configure MLflow
        self.tracking_uri = (
            tracking_uri or self.settings.mlflow_tracking_uri or "sqlite:///mlflow.db"
        )
        self.experiment_name = experiment_name or self.settings.mlflow_experiment_name
        self.artifact_root = artifact_root or self.settings.mlflow_artifact_root

        mlflow.set_tracking_uri(self.tracking_uri)

        if self.artifact_root:
            # Set default artifact root if specified
            self.settings.mlflow_default_artifact_root = self.artifact_root

        self.logger.info(
            f"MLflow experiment tracker initialized with URI: {self.tracking_uri}"
        )

        # Ensure default experiment exists
        asyncio.create_task(self._ensure_default_experiment())

    async def _ensure_default_experiment(self) -> None:
        """Ensure default experiment exists."""
        try:
            await self.get_experiment_by_name(self.experiment_name)
        except Exception:
            await self.create_experiment(
                name=self.experiment_name,
                artifact_location=self.artifact_root,
                tags={"type": "default", "domain": "finance"},
            )

    def _convert_mlflow_experiment(self, experiment: Experiment) -> ExperimentInfo:
        """Convert MLflow experiment to ExperimentInfo."""
        return ExperimentInfo(
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            artifact_location=experiment.artifact_location,
            lifecycle_stage=experiment.lifecycle_stage,
            creation_time=(
                datetime.fromtimestamp(experiment.creation_time / 1000, tz=timezone.utc)
                if experiment.creation_time
                else None
            ),
            last_update_time=(
                datetime.fromtimestamp(
                    experiment.last_update_time / 1000, tz=timezone.utc
                )
                if experiment.last_update_time
                else None
            ),
            tags=experiment.tags,
        )

    def _convert_mlflow_run(self, run: Run) -> RunInfo:
        """Convert MLflow run to RunInfo."""
        return RunInfo(
            run_id=run.info.run_id,
            experiment_id=run.info.experiment_id,
            run_name=run.data.tags.get(mlflow.utils.mlflow_tags.MLFLOW_RUN_NAME),
            status=RunStatus(run.info.status),
            start_time=(
                datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc)
                if run.info.start_time
                else None
            ),
            end_time=(
                datetime.fromtimestamp(run.info.end_time / 1000, tz=timezone.utc)
                if run.info.end_time
                else None
            ),
            artifact_uri=run.info.artifact_uri,
            lifecycle_stage=run.info.lifecycle_stage,
            user_id=run.info.user_id,
            tags=run.data.tags,
        )

    def _run_in_executor(self, func, *args, **kwargs):
        """Run synchronous MLflow operations in executor."""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, func, *args, **kwargs)

    # ===== Experiment Management =====

    async def create_experiment(
        self,
        name: str,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create a new experiment."""

        def _create():
            return mlflow.create_experiment(
                name=name, artifact_location=artifact_location, tags=tags
            )

        experiment_id = await self._run_in_executor(_create)
        self.logger.info(f"Created MLflow experiment '{name}' with ID: {experiment_id}")
        return experiment_id

    async def get_experiment(self, experiment_id: str) -> Optional[ExperimentInfo]:
        """Get experiment information."""

        def _get():
            try:
                experiment = mlflow.get_experiment(experiment_id)
                return experiment if experiment else None
            except RestException:
                return None

        experiment = await self._run_in_executor(_get)
        return self._convert_mlflow_experiment(experiment) if experiment else None

    async def get_experiment_by_name(self, name: str) -> Optional[ExperimentInfo]:
        """Get experiment by name."""

        def _get():
            try:
                experiment = mlflow.get_experiment_by_name(name)
                return experiment if experiment else None
            except RestException:
                return None

        experiment = await self._run_in_executor(_get)
        return self._convert_mlflow_experiment(experiment) if experiment else None

    # ===== Run Management =====

    async def start_run(
        self,
        experiment_id: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> str:
        """Start a new experiment run."""

        def _start():
            # If no experiment_id provided, use default experiment
            if experiment_id is None:
                mlflow.set_experiment(self.experiment_name)
            else:
                mlflow.set_experiment(experiment_id=experiment_id)

            run_tags = tags or {}
            if run_name:
                run_tags[mlflow.utils.mlflow_tags.MLFLOW_RUN_NAME] = run_name

            run = mlflow.start_run(run_name=run_name, tags=run_tags, nested=nested)
            return run.info.run_id

        run_id = await self._run_in_executor(_start)
        self.logger.info(f"Started MLflow run {run_id}")
        return run_id

    async def end_run(
        self, run_id: str, status: RunStatus = RunStatus.FINISHED
    ) -> None:
        """End an experiment run."""

        def _end():
            # MLflow uses different status names
            mlflow_status = {
                RunStatus.FINISHED: "FINISHED",
                RunStatus.FAILED: "FAILED",
                RunStatus.KILLED: "KILLED",
                RunStatus.RUNNING: "RUNNING",
            }[status]

            with mlflow.start_run(run_id=run_id):
                mlflow.end_run(status=mlflow_status)

        await self._run_in_executor(_end)
        self.logger.info(f"Ended MLflow run {run_id} with status {status.value}")

    async def get_run(self, run_id: str) -> Optional[RunInfo]:
        """Get run information."""

        def _get():
            try:
                run = mlflow.get_run(run_id)
                return run
            except RestException:
                return None

        run = await self._run_in_executor(_get)
        return self._convert_mlflow_run(run) if run else None

    # ===== Parameter and Metric Logging =====

    async def log_param(self, run_id: str, key: str, value: Any) -> None:
        """Log a parameter for the run."""

        def _log():
            with mlflow.start_run(run_id=run_id):
                mlflow.log_param(key, value)

        await self._run_in_executor(_log)

    async def log_params(self, run_id: str, params: Dict[str, Any]) -> None:
        """Log multiple parameters for the run."""

        def _log():
            with mlflow.start_run(run_id=run_id):
                mlflow.log_params(params)

        await self._run_in_executor(_log)

    async def log_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        step: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Log a metric for the run."""

        def _log():
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric(
                    key,
                    value,
                    step=step,
                    timestamp=int(timestamp.timestamp() * 1000) if timestamp else None,
                )

        await self._run_in_executor(_log)

    async def log_metrics(
        self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics for the run."""

        def _log():
            with mlflow.start_run(run_id=run_id):
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step)

        await self._run_in_executor(_log)

    # ===== Tag Management =====

    async def set_tag(self, run_id: str, key: str, value: str) -> None:
        """Set a tag for the run."""

        def _set():
            with mlflow.start_run(run_id=run_id):
                mlflow.set_tag(key, value)

        await self._run_in_executor(_set)

    async def set_tags(self, run_id: str, tags: Dict[str, str]) -> None:
        """Set multiple tags for the run."""

        def _set():
            with mlflow.start_run(run_id=run_id):
                mlflow.set_tags(tags)

        await self._run_in_executor(_set)

    # ===== Artifact Management =====

    async def log_artifact(
        self,
        run_id: str,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log an artifact for the run."""

        def _log():
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(str(local_path), artifact_path)

        await self._run_in_executor(_log)

    async def log_artifacts(
        self,
        run_id: str,
        local_dir: Union[str, Path],
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log multiple artifacts from a directory."""

        def _log():
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifacts(str(local_dir), artifact_path)

        await self._run_in_executor(_log)

    async def download_artifacts(
        self, run_id: str, path: str = "", dst_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Download artifacts from the run."""

        def _download():
            return mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=path,
                dst_path=str(dst_path) if dst_path else None,
            )

        downloaded_path = await self._run_in_executor(_download)
        return Path(downloaded_path)

    async def list_artifacts(self, run_id: str, path: str = "") -> List[ModelArtifact]:
        """List artifacts for the run."""

        def _list():
            client = mlflow.tracking.MlflowClient()
            return client.list_artifacts(run_id, path)

        mlflow_artifacts = await self._run_in_executor(_list)

        artifacts = []
        for artifact in mlflow_artifacts:
            artifacts.append(
                ModelArtifact(
                    path=artifact.path,
                    is_dir=artifact.is_dir,
                    file_size=artifact.file_size,
                )
            )

        return artifacts

    # ===== Model Registry =====

    async def register_model(
        self,
        name: str,
        model_uri: str,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> str:
        """Register a model in the model registry."""

        def _register():
            if run_id:
                # Use model from specific run
                result = mlflow.register_model(
                    model_uri=f"runs:/{run_id}/{model_uri}", name=name, tags=tags
                )
            else:
                # Use direct model URI
                result = mlflow.register_model(
                    model_uri=model_uri, name=name, tags=tags
                )

            # Update description if provided
            if description and result:
                client = mlflow.tracking.MlflowClient()
                client.update_model_version(
                    name=name, version=result.version, description=description
                )

            return result.version

        version = await self._run_in_executor(_register)
        self.logger.info(f"Registered MLflow model {name} version {version}")
        return version

    async def get_model_version(
        self, name: str, version: str
    ) -> Optional[ModelRegistryInfo]:
        """Get model version information."""

        def _get():
            try:
                client = mlflow.tracking.MlflowClient()
                model_version = client.get_model_version(name, version)
                return model_version
            except RestException:
                return None

        model_version = await self._run_in_executor(_get)

        if not model_version:
            return None

        # Convert MLflow stage to our ModelStage
        stage_mapping = {
            "None": ModelStage.NONE,
            "Staging": ModelStage.STAGING,
            "Production": ModelStage.PRODUCTION,
            "Archived": ModelStage.ARCHIVED,
        }

        return ModelRegistryInfo(
            name=model_version.name,
            version=model_version.version,
            stage=stage_mapping.get(model_version.current_stage, ModelStage.NONE),
            description=model_version.description,
            tags=model_version.tags,
            source=model_version.source,
            run_id=model_version.run_id,
            model_uri=model_version.source,
            creation_timestamp=(
                datetime.fromtimestamp(
                    model_version.creation_timestamp / 1000, tz=timezone.utc
                )
                if model_version.creation_timestamp
                else None
            ),
            last_updated_timestamp=(
                datetime.fromtimestamp(
                    model_version.last_updated_timestamp / 1000, tz=timezone.utc
                )
                if model_version.last_updated_timestamp
                else None
            ),
        )

    async def get_latest_versions(
        self, name: str, stages: Optional[List[ModelStage]] = None
    ) -> List[ModelRegistryInfo]:
        """Get latest model versions for given stages."""

        def _get():
            client = mlflow.tracking.MlflowClient()

            if stages:
                # Convert our stages to MLflow stages
                mlflow_stages = []
                stage_mapping = {
                    ModelStage.NONE: "None",
                    ModelStage.STAGING: "Staging",
                    ModelStage.PRODUCTION: "Production",
                    ModelStage.ARCHIVED: "Archived",
                }
                for stage in stages:
                    mlflow_stages.append(stage_mapping[stage])
            else:
                mlflow_stages = None

            return client.get_latest_versions(name, stages=mlflow_stages)

        model_versions = await self._run_in_executor(_get)

        results = []
        for model_version in model_versions:
            stage_mapping = {
                "None": ModelStage.NONE,
                "Staging": ModelStage.STAGING,
                "Production": ModelStage.PRODUCTION,
                "Archived": ModelStage.ARCHIVED,
            }

            results.append(
                ModelRegistryInfo(
                    name=model_version.name,
                    version=model_version.version,
                    stage=stage_mapping.get(
                        model_version.current_stage, ModelStage.NONE
                    ),
                    description=model_version.description,
                    tags=model_version.tags,
                    source=model_version.source,
                    run_id=model_version.run_id,
                    model_uri=model_version.source,
                    creation_timestamp=(
                        datetime.fromtimestamp(
                            model_version.creation_timestamp / 1000, tz=timezone.utc
                        )
                        if model_version.creation_timestamp
                        else None
                    ),
                    last_updated_timestamp=(
                        datetime.fromtimestamp(
                            model_version.last_updated_timestamp / 1000, tz=timezone.utc
                        )
                        if model_version.last_updated_timestamp
                        else None
                    ),
                )
            )

        return results

    async def transition_model_version_stage(
        self,
        name: str,
        version: str,
        stage: ModelStage,
        archive_existing_versions: bool = False,
    ) -> ModelRegistryInfo:
        """Transition model version to a new stage."""

        def _transition():
            client = mlflow.tracking.MlflowClient()

            # Convert our stage to MLflow stage
            stage_mapping = {
                ModelStage.NONE: "None",
                ModelStage.STAGING: "Staging",
                ModelStage.PRODUCTION: "Production",
                ModelStage.ARCHIVED: "Archived",
            }
            mlflow_stage = stage_mapping[stage]

            return client.transition_model_version_stage(
                name=name,
                version=version,
                stage=mlflow_stage,
                archive_existing_versions=archive_existing_versions,
            )

        await self._run_in_executor(_transition)
        return await self.get_model_version(name, version)

    async def update_model_version(
        self,
        name: str,
        version: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ModelRegistryInfo:
        """Update model version metadata."""

        def _update():
            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=name, version=version, description=description
            )

            if tags:
                for key, value in tags.items():
                    client.set_model_version_tag(name, version, key, value)

        await self._run_in_executor(_update)
        return await self.get_model_version(name, version)

    async def delete_model_version(self, name: str, version: str) -> None:
        """Delete a model version."""

        def _delete():
            client = mlflow.tracking.MlflowClient()
            client.delete_model_version(name, version)

        await self._run_in_executor(_delete)
        self.logger.info(f"Deleted MLflow model {name} version {version}")

    # ===== Health and Utility Methods =====

    async def health_check(self) -> bool:
        """Check if the experiment tracker is healthy."""

        def _check():
            try:
                # Try to list experiments to test connectivity
                client = mlflow.tracking.MlflowClient()
                client.list_experiments()
                return True
            except Exception:
                return False

        try:
            return await self._run_in_executor(_check)
        except Exception as e:
            self.logger.error(f"MLflow health check failed: {e}")
            return False

    async def get_tracking_uri(self) -> str:
        """Get the tracking URI."""
        return self.tracking_uri
