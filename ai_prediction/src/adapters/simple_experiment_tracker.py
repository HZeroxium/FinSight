# adapters/simple_experiment_tracker.py

import json
import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import asyncio

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


class SimpleExperimentTracker(IExperimentTracker):
    """
    Simple file-based experiment tracker that wraps current local file system logic.

    This implementation provides experiment tracking capabilities using local file storage,
    maintaining compatibility with existing model management patterns.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize simple experiment tracker.

        Args:
            base_dir: Base directory for storing experiments (uses settings if None)
        """
        self.settings = get_settings()
        self.base_dir = base_dir or (self.settings.models_dir / "experiments")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.experiments_dir = self.base_dir / "experiments"
        self.runs_dir = self.base_dir / "runs"
        self.models_dir = self.base_dir / "models"
        self.artifacts_dir = self.base_dir / "artifacts"

        for directory in [
            self.experiments_dir,
            self.runs_dir,
            self.models_dir,
            self.artifacts_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        self.logger = LoggerFactory.get_logger("SimpleExperimentTracker")
        self.logger.info(f"Simple experiment tracker initialized at {self.base_dir}")

        # Create default experiment if it doesn't exist
        asyncio.create_task(self._ensure_default_experiment())

    async def _ensure_default_experiment(self) -> None:
        """Ensure default experiment exists."""
        default_exp = await self.get_experiment_by_name(
            self.settings.mlflow_experiment_name
        )
        if not default_exp:
            await self.create_experiment(
                name=self.settings.mlflow_experiment_name,
                tags={"type": "default", "domain": "finance"},
            )

    # ===== Experiment Management =====

    async def create_experiment(
        self,
        name: str,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create a new experiment."""
        experiment_id = str(uuid.uuid4())

        if artifact_location is None:
            artifact_location = str(self.artifacts_dir / experiment_id)

        experiment_info = {
            "experiment_id": experiment_id,
            "name": name,
            "artifact_location": artifact_location,
            "lifecycle_stage": "active",
            "creation_time": datetime.now(timezone.utc).isoformat(),
            "last_update_time": datetime.now(timezone.utc).isoformat(),
            "tags": tags or {},
        }

        # Save experiment info
        exp_file = self.experiments_dir / f"{experiment_id}.json"
        await self._save_json(exp_file, experiment_info)

        # Create artifact directory
        Path(artifact_location).mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Created experiment '{name}' with ID: {experiment_id}")
        return experiment_id

    async def get_experiment(self, experiment_id: str) -> Optional[ExperimentInfo]:
        """Get experiment information."""
        exp_file = self.experiments_dir / f"{experiment_id}.json"
        if not exp_file.exists():
            return None

        data = await self._load_json(exp_file)
        return ExperimentInfo(
            experiment_id=data["experiment_id"],
            name=data["name"],
            artifact_location=data["artifact_location"],
            lifecycle_stage=data["lifecycle_stage"],
            creation_time=(
                datetime.fromisoformat(data["creation_time"])
                if data.get("creation_time")
                else None
            ),
            last_update_time=(
                datetime.fromisoformat(data["last_update_time"])
                if data.get("last_update_time")
                else None
            ),
            tags=data.get("tags"),
        )

    async def get_experiment_by_name(self, name: str) -> Optional[ExperimentInfo]:
        """Get experiment by name."""
        for exp_file in self.experiments_dir.glob("*.json"):
            data = await self._load_json(exp_file)
            if data.get("name") == name:
                return await self.get_experiment(data["experiment_id"])
        return None

    # ===== Run Management =====

    async def start_run(
        self,
        experiment_id: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> str:
        """Start a new experiment run."""
        if experiment_id is None:
            default_exp = await self.get_experiment_by_name(
                self.settings.mlflow_experiment_name
            )
            experiment_id = (
                default_exp.experiment_id
                if default_exp
                else await self.create_experiment(self.settings.mlflow_experiment_name)
            )

        run_id = str(uuid.uuid4())

        run_info = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "run_name": run_name,
            "status": RunStatus.RUNNING.value,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "artifact_uri": str(self.artifacts_dir / run_id),
            "lifecycle_stage": "active",
            "user_id": "system",
            "tags": tags or {},
            "params": {},
            "metrics": {},
        }

        # Save run info
        run_file = self.runs_dir / f"{run_id}.json"
        await self._save_json(run_file, run_info)

        # Create artifact directory for run
        run_artifact_dir = Path(run_info["artifact_uri"])
        run_artifact_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Started run {run_id} in experiment {experiment_id}")
        return run_id

    async def end_run(
        self, run_id: str, status: RunStatus = RunStatus.FINISHED
    ) -> None:
        """End an experiment run."""
        run_file = self.runs_dir / f"{run_id}.json"
        if not run_file.exists():
            raise ValueError(f"Run {run_id} not found")

        data = await self._load_json(run_file)
        data["status"] = status.value
        data["end_time"] = datetime.now(timezone.utc).isoformat()

        await self._save_json(run_file, data)
        self.logger.info(f"Ended run {run_id} with status {status.value}")

    async def get_run(self, run_id: str) -> Optional[RunInfo]:
        """Get run information."""
        run_file = self.runs_dir / f"{run_id}.json"
        if not run_file.exists():
            return None

        data = await self._load_json(run_file)
        return RunInfo(
            run_id=data["run_id"],
            experiment_id=data["experiment_id"],
            run_name=data.get("run_name"),
            status=RunStatus(data["status"]),
            start_time=(
                datetime.fromisoformat(data["start_time"])
                if data.get("start_time")
                else None
            ),
            end_time=(
                datetime.fromisoformat(data["end_time"])
                if data.get("end_time")
                else None
            ),
            artifact_uri=data.get("artifact_uri"),
            lifecycle_stage=data.get("lifecycle_stage", "active"),
            user_id=data.get("user_id"),
            tags=data.get("tags"),
        )

    # ===== Parameter and Metric Logging =====

    async def log_param(self, run_id: str, key: str, value: Any) -> None:
        """Log a parameter for the run."""
        await self.log_params(run_id, {key: value})

    async def log_params(self, run_id: str, params: Dict[str, Any]) -> None:
        """Log multiple parameters for the run."""
        run_file = self.runs_dir / f"{run_id}.json"
        if not run_file.exists():
            raise ValueError(f"Run {run_id} not found")

        data = await self._load_json(run_file)
        if "params" not in data:
            data["params"] = {}

        # Convert all values to strings for consistency
        for key, value in params.items():
            data["params"][key] = str(value)

        await self._save_json(run_file, data)

    async def log_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        step: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Log a metric for the run."""
        await self.log_metrics(run_id, {key: value}, step)

    async def log_metrics(
        self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics for the run."""
        run_file = self.runs_dir / f"{run_id}.json"
        if not run_file.exists():
            raise ValueError(f"Run {run_id} not found")

        data = await self._load_json(run_file)
        if "metrics" not in data:
            data["metrics"] = {}

        timestamp = datetime.now(timezone.utc).isoformat()

        for key, value in metrics.items():
            if key not in data["metrics"]:
                data["metrics"][key] = []

            metric_entry = {"value": float(value), "timestamp": timestamp, "step": step}
            data["metrics"][key].append(metric_entry)

        await self._save_json(run_file, data)

    # ===== Tag Management =====

    async def set_tag(self, run_id: str, key: str, value: str) -> None:
        """Set a tag for the run."""
        await self.set_tags(run_id, {key: value})

    async def set_tags(self, run_id: str, tags: Dict[str, str]) -> None:
        """Set multiple tags for the run."""
        run_file = self.runs_dir / f"{run_id}.json"
        if not run_file.exists():
            raise ValueError(f"Run {run_id} not found")

        data = await self._load_json(run_file)
        if "tags" not in data:
            data["tags"] = {}

        data["tags"].update(tags)
        await self._save_json(run_file, data)

    # ===== Artifact Management =====

    async def log_artifact(
        self,
        run_id: str,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log an artifact for the run."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local path not found: {local_path}")

        run_artifact_dir = self.artifacts_dir / run_id
        run_artifact_dir.mkdir(parents=True, exist_ok=True)

        if artifact_path:
            dest_path = run_artifact_dir / artifact_path / local_path.name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            dest_path = run_artifact_dir / local_path.name

        if local_path.is_file():
            shutil.copy2(local_path, dest_path)
        else:
            shutil.copytree(local_path, dest_path, dirs_exist_ok=True)

        self.logger.debug(f"Logged artifact {local_path} to {dest_path}")

    async def log_artifacts(
        self,
        run_id: str,
        local_dir: Union[str, Path],
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log multiple artifacts from a directory."""
        local_dir = Path(local_dir)
        if not local_dir.exists():
            raise FileNotFoundError(f"Local directory not found: {local_dir}")

        for item in local_dir.iterdir():
            await self.log_artifact(run_id, item, artifact_path)

    async def download_artifacts(
        self, run_id: str, path: str = "", dst_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Download artifacts from the run."""
        run_artifact_dir = self.artifacts_dir / run_id

        if not run_artifact_dir.exists():
            raise FileNotFoundError(f"Artifacts not found for run {run_id}")

        source_path = run_artifact_dir / path if path else run_artifact_dir

        if dst_path is None:
            dst_path = Path.cwd() / "downloaded_artifacts" / run_id
        else:
            dst_path = Path(dst_path)

        dst_path.mkdir(parents=True, exist_ok=True)

        if source_path.is_file():
            shutil.copy2(source_path, dst_path / source_path.name)
        else:
            shutil.copytree(source_path, dst_path, dirs_exist_ok=True)

        return dst_path

    async def list_artifacts(self, run_id: str, path: str = "") -> List[ModelArtifact]:
        """List artifacts for the run."""
        run_artifact_dir = self.artifacts_dir / run_id
        artifacts = []

        if not run_artifact_dir.exists():
            return artifacts

        search_path = run_artifact_dir / path if path else run_artifact_dir

        if search_path.exists():
            for item in search_path.rglob("*"):
                relative_path = item.relative_to(run_artifact_dir)
                artifacts.append(
                    ModelArtifact(
                        path=str(relative_path),
                        is_dir=item.is_dir(),
                        file_size=item.stat().st_size if item.is_file() else None,
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
        # Generate new version
        version = await self._get_next_model_version(name)

        model_info = {
            "name": name,
            "version": version,
            "stage": ModelStage.NONE.value,
            "description": description,
            "tags": tags or {},
            "source": model_uri,
            "run_id": run_id,
            "model_uri": model_uri,
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
            "last_updated_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Save model info
        model_dir = self.models_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        model_file = model_dir / f"version_{version}.json"
        await self._save_json(model_file, model_info)

        self.logger.info(f"Registered model {name} version {version}")
        return version

    async def get_model_version(
        self, name: str, version: str
    ) -> Optional[ModelRegistryInfo]:
        """Get model version information."""
        model_file = self.models_dir / name / f"version_{version}.json"
        if not model_file.exists():
            return None

        data = await self._load_json(model_file)
        return ModelRegistryInfo(
            name=data["name"],
            version=data["version"],
            stage=ModelStage(data["stage"]),
            description=data.get("description"),
            tags=data.get("tags"),
            source=data.get("source"),
            run_id=data.get("run_id"),
            model_uri=data.get("model_uri"),
            creation_timestamp=(
                datetime.fromisoformat(data["creation_timestamp"])
                if data.get("creation_timestamp")
                else None
            ),
            last_updated_timestamp=(
                datetime.fromisoformat(data["last_updated_timestamp"])
                if data.get("last_updated_timestamp")
                else None
            ),
        )

    async def get_latest_versions(
        self, name: str, stages: Optional[List[ModelStage]] = None
    ) -> List[ModelRegistryInfo]:
        """Get latest model versions for given stages."""
        model_dir = self.models_dir / name
        if not model_dir.exists():
            return []

        versions = {}

        for model_file in model_dir.glob("version_*.json"):
            data = await self._load_json(model_file)
            stage = ModelStage(data["stage"])

            if stages is None or stage in stages:
                if (
                    stage not in versions
                    or data["version"] > versions[stage]["version"]
                ):
                    versions[stage] = data

        results = []
        for data in versions.values():
            results.append(
                ModelRegistryInfo(
                    name=data["name"],
                    version=data["version"],
                    stage=ModelStage(data["stage"]),
                    description=data.get("description"),
                    tags=data.get("tags"),
                    source=data.get("source"),
                    run_id=data.get("run_id"),
                    model_uri=data.get("model_uri"),
                    creation_timestamp=(
                        datetime.fromisoformat(data["creation_timestamp"])
                        if data.get("creation_timestamp")
                        else None
                    ),
                    last_updated_timestamp=(
                        datetime.fromisoformat(data["last_updated_timestamp"])
                        if data.get("last_updated_timestamp")
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
        model_file = self.models_dir / name / f"version_{version}.json"
        if not model_file.exists():
            raise ValueError(f"Model {name} version {version} not found")

        # Archive existing versions in the target stage if requested
        if archive_existing_versions and stage != ModelStage.NONE:
            existing_versions = await self.get_latest_versions(name, [stage])
            for existing in existing_versions:
                await self.transition_model_version_stage(
                    name, existing.version, ModelStage.ARCHIVED, False
                )

        # Update the stage
        data = await self._load_json(model_file)
        data["stage"] = stage.value
        data["last_updated_timestamp"] = datetime.now(timezone.utc).isoformat()

        await self._save_json(model_file, data)

        return await self.get_model_version(name, version)

    async def update_model_version(
        self,
        name: str,
        version: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ModelRegistryInfo:
        """Update model version metadata."""
        model_file = self.models_dir / name / f"version_{version}.json"
        if not model_file.exists():
            raise ValueError(f"Model {name} version {version} not found")

        data = await self._load_json(model_file)

        if description is not None:
            data["description"] = description
        if tags is not None:
            data["tags"].update(tags)

        data["last_updated_timestamp"] = datetime.now(timezone.utc).isoformat()

        await self._save_json(model_file, data)

        return await self.get_model_version(name, version)

    async def delete_model_version(self, name: str, version: str) -> None:
        """Delete a model version."""
        model_file = self.models_dir / name / f"version_{version}.json"
        if model_file.exists():
            model_file.unlink()
            self.logger.info(f"Deleted model {name} version {version}")

    # ===== Health and Utility Methods =====

    async def health_check(self) -> bool:
        """Check if the experiment tracker is healthy."""
        try:
            # Check if all required directories exist and are writable
            for directory in [
                self.experiments_dir,
                self.runs_dir,
                self.models_dir,
                self.artifacts_dir,
            ]:
                if not directory.exists():
                    return False
                # Test write access
                test_file = directory / ".health_check"
                test_file.touch()
                test_file.unlink()
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    async def get_tracking_uri(self) -> str:
        """Get the tracking URI."""
        return f"file://{self.base_dir.absolute()}"

    # ===== Private Utility Methods =====

    async def _save_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Save data to JSON file."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    async def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load data from JSON file."""
        with open(file_path, "r") as f:
            return json.load(f)

    async def _get_next_model_version(self, name: str) -> str:
        """Get the next version number for a model."""
        model_dir = self.models_dir / name
        if not model_dir.exists():
            return "1"

        versions = []
        for model_file in model_dir.glob("version_*.json"):
            try:
                version_str = model_file.stem.replace("version_", "")
                versions.append(int(version_str))
            except ValueError:
                continue

        return str(max(versions) + 1) if versions else "1"
