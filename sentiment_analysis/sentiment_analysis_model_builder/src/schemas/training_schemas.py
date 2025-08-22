# schemas/training_schemas.py

"""Training schemas for sentiment analysis."""

from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator


class ClassificationReport(BaseModel):
    """Classification report for model evaluation."""

    accuracy: float = Field(..., description="Overall accuracy")
    macro_avg: Dict[str, float] = Field(..., description="Macro-averaged metrics")
    weighted_avg: Dict[str, float] = Field(..., description="Weighted-averaged metrics")
    per_class_metrics: Dict[str, Dict[str, float]] = Field(
        ..., description="Per-class metrics"
    )

    @field_validator("accuracy")
    @classmethod
    def validate_accuracy(cls, v: float) -> float:
        """Validate accuracy value."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Accuracy must be between 0.0 and 1.0")
        return round(v, 4)

    @field_validator("macro_avg", "weighted_avg")
    @classmethod
    def validate_avg_metrics(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate average metrics."""
        required_keys = {"precision", "recall", "f1-score", "support"}
        if not all(key in v for key in required_keys):
            raise ValueError(f"Average metrics must contain all keys: {required_keys}")

        # Round values to 4 decimal places
        return {k: round(v, 4) for k, v in v.items()}


class EvaluationResult(BaseModel):
    """Evaluation result for a dataset split."""

    split_name: str = Field(..., description="Dataset split name")
    eval_loss: float = Field(..., description="Evaluation loss")
    eval_accuracy: float = Field(..., description="Evaluation accuracy")
    eval_f1_macro: float = Field(..., description="Macro F1 score")
    eval_f1_weighted: float = Field(..., description="Weighted F1 score")
    eval_runtime: float = Field(..., description="Evaluation runtime")
    eval_samples_per_second: float = Field(..., description="Samples per second")
    eval_steps_per_second: float = Field(..., description="Steps per second")

    @field_validator("eval_loss")
    @classmethod
    def validate_eval_loss(cls, v: float) -> float:
        """Validate evaluation loss."""
        if v < 0:
            raise ValueError("Evaluation loss must be non-negative")
        return round(v, 4)

    @field_validator("eval_accuracy", "eval_f1_macro", "eval_f1_weighted")
    @classmethod
    def validate_metrics(cls, v: float) -> float:
        """Validate metric values."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Metric values must be between 0.0 and 1.0")
        return round(v, 4)


class TrainingMetrics(BaseModel):
    """Training metrics and results."""

    train_loss: float = Field(..., description="Training loss")
    eval_results: Dict[str, EvaluationResult] = Field(
        ..., description="Evaluation results by split"
    )
    classification_report: Optional[ClassificationReport] = Field(
        default=None, description="Detailed classification report"
    )
    run_id: str = Field(..., description="MLflow run ID")
    experiment_id: str = Field(..., description="MLflow experiment ID")
    training_duration: Optional[float] = Field(
        default=None, description="Total training duration in seconds"
    )
    best_checkpoint: Optional[str] = Field(
        default=None, description="Path to best checkpoint"
    )

    @field_validator("train_loss")
    @classmethod
    def validate_train_loss(cls, v: float) -> float:
        """Validate training loss."""
        if v < 0:
            raise ValueError("Training loss must be non-negative")
        return round(v, 4)

    @field_validator("eval_results")
    @classmethod
    def validate_eval_results(
        cls, v: Dict[str, EvaluationResult]
    ) -> Dict[str, EvaluationResult]:
        """Validate evaluation results."""
        if not v:
            raise ValueError("Evaluation results cannot be empty")
        return v


class ModelArtifacts(BaseModel):
    """Model artifacts and metadata."""

    model_path: Path = Field(..., description="Path to saved model")
    tokenizer_path: Path = Field(..., description="Path to saved tokenizer")
    preprocessing_config_path: Path = Field(
        ..., description="Path to preprocessing config"
    )
    label_mapping_path: Path = Field(..., description="Path to label mapping")
    training_summary_path: Path = Field(..., description="Path to training summary")
    model_size_mb: Optional[float] = Field(default=None, description="Model size in MB")
    created_at: str = Field(..., description="Creation timestamp")
    version: str = Field(default="1.0.0", description="Model version")

    @field_validator(
        "model_path",
        "tokenizer_path",
        "preprocessing_config_path",
        "label_mapping_path",
        "training_summary_path",
    )
    @classmethod
    def validate_paths(cls, v: Path) -> Path:
        """Validate file paths."""
        if not v.parent.exists():
            v.parent.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("model_size_mb")
    @classmethod
    def validate_model_size(cls, v: Optional[float]) -> Optional[float]:
        """Validate model size."""
        if v is not None and v <= 0:
            raise ValueError("Model size must be positive")
        return round(v, 2) if v is not None else None


class TrainingConfig(BaseModel):
    """Training configuration parameters."""

    backbone: str = Field(..., description="Model backbone")
    batch_size: int = Field(..., description="Training batch size")
    learning_rate: float = Field(..., description="Learning rate")
    num_epochs: int = Field(..., description="Number of epochs")
    warmup_steps: int = Field(..., description="Warmup steps")
    weight_decay: float = Field(..., description="Weight decay")
    gradient_clip_val: float = Field(..., description="Gradient clipping value")
    random_seed: int = Field(..., description="Random seed")
    early_stopping_patience: int = Field(..., description="Early stopping patience")
    primary_metric: str = Field(..., description="Primary evaluation metric")

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size."""
        if v < 1:
            raise ValueError("Batch size must be at least 1")
        if v > 128:
            raise ValueError("Batch size must not exceed 128")
        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Validate learning rate."""
        if v <= 0:
            raise ValueError("Learning rate must be positive")
        if v > 1e-2:
            raise ValueError("Learning rate must not exceed 0.01")
        return v
