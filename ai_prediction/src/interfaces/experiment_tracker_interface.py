# interfaces/experiment_tracker_interface.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

from ..schemas.enums import TimeFrame, ModelType


class ModelStage(Enum):
    """Model lifecycle stages"""

    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class RunStatus(Enum):
    """Experiment run status"""

    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"


class ModelRegistryInfo(BaseModel):
    """Model registry information"""

    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    stage: ModelStage = Field(..., description="Model stage")
    description: Optional[str] = Field(None, description="Model description")
    tags: Optional[Dict[str, str]] = Field(None, description="Model tags")
    source: Optional[str] = Field(None, description="Model source")
    run_id: Optional[str] = Field(None, description="Run ID")
    model_uri: Optional[str] = Field(None, description="Model URI")
    creation_timestamp: Optional[datetime] = Field(
        None, description="Creation timestamp"
    )
    last_updated_timestamp: Optional[datetime] = Field(
        None, description="Last updated timestamp"
    )


class ExperimentInfo(BaseModel):
    """Experiment information"""

    experiment_id: str = Field(..., description="Experiment ID")
    name: str = Field(..., description="Experiment name")
    artifact_location: str = Field(..., description="Artifact location")
    lifecycle_stage: str = Field(..., description="Lifecycle stage")
    creation_time: Optional[datetime] = Field(None, description="Creation time")
    last_update_time: Optional[datetime] = Field(None, description="Last update time")
    tags: Optional[Dict[str, str]] = Field(None, description="Tags")


class RunInfo(BaseModel):
    """Experiment run information"""

    run_id: str = Field(..., description="Run ID")
    experiment_id: str = Field(..., description="Experiment ID")
    run_name: Optional[str] = Field(None, description="Run name")
    status: RunStatus = Field(..., description="Run status")
    start_time: Optional[datetime] = Field(None, description="Start time")
    end_time: Optional[datetime] = Field(None, description="End time")
    artifact_uri: Optional[str] = Field(None, description="Artifact URI")
    lifecycle_stage: str = Field(..., description="Lifecycle stage")
    user_id: Optional[str] = Field(None, description="User ID")
    tags: Optional[Dict[str, str]] = Field(None, description="Tags")


class ModelArtifact(BaseModel):
    """Model artifact information"""

    path: str = Field(..., description="Path")
    is_dir: bool = Field(..., description="Is directory")
    file_size: Optional[int] = Field(None, description="File size")
    tags: Optional[Dict[str, str]] = Field(None, description="Tags")


class IExperimentTracker(ABC):
    """
    Interface for experiment tracking, model registry, and artifact management.

    This interface provides a unified API for:
    - Model Registry: version, stage, and tag management
    - Experiment Tracking: logging parameters, metrics, and tags
    - Artifact Store: storing and retrieving model files, datasets, charts
    - Run Management: managing experiment runs with IDs, descriptions, tags
    """

    # ===== Experiment Management =====

    @abstractmethod
    async def create_experiment(
        self,
        name: str,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create a new experiment.

        Args:
            name: Experiment name
            artifact_location: Location for storing artifacts
            tags: Optional tags for the experiment

        Returns:
            Experiment ID
        """
        pass

    @abstractmethod
    async def get_experiment(self, experiment_id: str) -> Optional[ExperimentInfo]:
        """Get experiment information."""
        pass

    @abstractmethod
    async def get_experiment_by_name(self, name: str) -> Optional[ExperimentInfo]:
        """Get experiment by name."""
        pass

    # ===== Run Management =====

    @abstractmethod
    async def start_run(
        self,
        experiment_id: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> str:
        """
        Start a new experiment run.

        Args:
            experiment_id: Experiment ID (uses default if None)
            run_name: Optional name for the run
            tags: Optional tags for the run
            nested: Whether this is a nested run

        Returns:
            Run ID
        """
        pass

    @abstractmethod
    async def end_run(
        self, run_id: str, status: RunStatus = RunStatus.FINISHED
    ) -> None:
        """End an experiment run."""
        pass

    @abstractmethod
    async def get_run(self, run_id: str) -> Optional[RunInfo]:
        """Get run information."""
        pass

    # ===== Parameter and Metric Logging =====

    @abstractmethod
    async def log_param(self, run_id: str, key: str, value: Any) -> None:
        """Log a parameter for the run."""
        pass

    @abstractmethod
    async def log_params(self, run_id: str, params: Dict[str, Any]) -> None:
        """Log multiple parameters for the run."""
        pass

    @abstractmethod
    async def log_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        step: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Log a metric for the run."""
        pass

    @abstractmethod
    async def log_metrics(
        self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics for the run."""
        pass

    # ===== Tag Management =====

    @abstractmethod
    async def set_tag(self, run_id: str, key: str, value: str) -> None:
        """Set a tag for the run."""
        pass

    @abstractmethod
    async def set_tags(self, run_id: str, tags: Dict[str, str]) -> None:
        """Set multiple tags for the run."""
        pass

    # ===== Artifact Management =====

    @abstractmethod
    async def log_artifact(
        self,
        run_id: str,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
    ) -> None:
        """
        Log an artifact for the run.

        Args:
            run_id: Run ID
            local_path: Local path to the artifact
            artifact_path: Optional path within the artifact store
        """
        pass

    @abstractmethod
    async def log_artifacts(
        self,
        run_id: str,
        local_dir: Union[str, Path],
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log multiple artifacts from a directory."""
        pass

    @abstractmethod
    async def download_artifacts(
        self, run_id: str, path: str = "", dst_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Download artifacts from the run.

        Args:
            run_id: Run ID
            path: Relative path of the artifact to download
            dst_path: Local destination path

        Returns:
            Path to downloaded artifacts
        """
        pass

    @abstractmethod
    async def list_artifacts(self, run_id: str, path: str = "") -> List[ModelArtifact]:
        """List artifacts for the run."""
        pass

    # ===== Model Registry =====

    @abstractmethod
    async def register_model(
        self,
        name: str,
        model_uri: str,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> str:
        """
        Register a model in the model registry.

        Args:
            name: Model name
            model_uri: URI pointing to the model artifacts
            run_id: Associated run ID
            tags: Optional tags
            description: Optional description

        Returns:
            Model version
        """
        pass

    @abstractmethod
    async def get_model_version(
        self, name: str, version: str
    ) -> Optional[ModelRegistryInfo]:
        """Get model version information."""
        pass

    @abstractmethod
    async def get_latest_versions(
        self, name: str, stages: Optional[List[ModelStage]] = None
    ) -> List[ModelRegistryInfo]:
        """Get latest model versions for given stages."""
        pass

    @abstractmethod
    async def transition_model_version_stage(
        self,
        name: str,
        version: str,
        stage: ModelStage,
        archive_existing_versions: bool = False,
    ) -> ModelRegistryInfo:
        """Transition model version to a new stage."""
        pass

    @abstractmethod
    async def update_model_version(
        self,
        name: str,
        version: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ModelRegistryInfo:
        """Update model version metadata."""
        pass

    @abstractmethod
    async def delete_model_version(self, name: str, version: str) -> None:
        """Delete a model version."""
        pass

    # ===== Health and Utility Methods =====

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the experiment tracker is healthy."""
        pass

    @abstractmethod
    async def get_tracking_uri(self) -> str:
        """Get the tracking URI."""
        pass

    # ===== Financial Domain Specific Methods =====

    async def log_training_config(
        self,
        run_id: str,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        config: Dict[str, Any],
    ) -> None:
        """
        Log training configuration for financial models.

        Args:
            run_id: Run ID
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            config: Training configuration
        """
        params = {
            "symbol": symbol,
            "timeframe": timeframe.value,
            "model_type": model_type.value,
            **config,
        }
        await self.log_params(run_id, params)

        # Set financial domain tags
        tags = {
            "domain": "finance",
            "asset_class": "crypto",
            "symbol": symbol,
            "timeframe": timeframe.value,
            "model_type": model_type.value,
        }
        await self.set_tags(run_id, tags)

    async def log_model_performance(
        self,
        run_id: str,
        training_metrics: Dict[str, float],
        validation_metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log model performance metrics.

        Args:
            run_id: Run ID
            training_metrics: Training metrics
            validation_metrics: Validation metrics
            step: Optional step number
        """
        # Add prefixes to distinguish training vs validation metrics
        prefixed_metrics = {}
        for key, value in training_metrics.items():
            prefixed_metrics[f"train_{key}"] = value
        for key, value in validation_metrics.items():
            prefixed_metrics[f"val_{key}"] = value

        await self.log_metrics(run_id, prefixed_metrics, step)
