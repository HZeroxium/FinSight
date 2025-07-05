# finetune/config.py

"""
Configuration for fine-tuning pipeline using modern Pydantic v2 BaseModel.
"""

from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class ModelType(str, Enum):
    """Supported time series model types for fine-tuning"""

    AUTOFORMER = "huggingface/autoformer-tourism-monthly"
    INFORMER = "huggingface/informer-tourism-monthly"
    PATCH_TST = "ibm/patchtst-forecasting"
    PATCH_TSMIXER = "ibm/patchtsmixer-forecasting"
    TIME_SERIES_TRANSFORMER = "huggingface/time-series-transformer"
    TIMESFM = "google/timesfm-1.0-200m"


class PeftMethod(str, Enum):
    """PEFT methods for efficient fine-tuning"""

    LORA = "lora"
    ADALORA = "adalora"
    PROMPT_TUNING = "prompt_tuning"
    PREFIX_TUNING = "prefix_tuning"


class TaskType(str, Enum):
    """Task types for different prediction scenarios"""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    FORECASTING = "forecasting"


class WandBConfig(BaseModel):
    """Weights & Biases configuration"""

    enabled: bool = Field(default=False, description="Enable W&B logging")
    project: str = Field(default="finsight-finetune", description="W&B project name")
    run_name: Optional[str] = Field(default=None, description="W&B run name")
    tags: List[str] = Field(default_factory=list, description="W&B tags")
    notes: Optional[str] = Field(default=None, description="W&B run notes")


class FineTuneConfig(BaseModel):
    """Modern configuration for fine-tuning pipeline"""

    model_config = ConfigDict(
        use_enum_values=True, validate_assignment=True, extra="forbid"
    )

    # Model configuration
    model_name: str = Field(
        default=ModelType.PATCH_TSMIXER, description="Pre-trained model to fine-tune"
    )
    task_type: TaskType = Field(
        default=TaskType.FORECASTING, description="Type of prediction task"
    )

    # PEFT configuration - Disabled by default for time series models
    use_peft: bool = Field(
        default=False,
        description="Enable PEFT for efficient fine-tuning (disabled for time series models)",
    )
    peft_method: PeftMethod = Field(
        default=PeftMethod.LORA, description="PEFT method to use"
    )
    lora_rank: int = Field(default=8, description="LoRA rank parameter")
    lora_alpha: int = Field(default=16, description="LoRA alpha parameter")
    lora_dropout: float = Field(default=0.1, description="LoRA dropout rate")
    target_modules: Optional[List[str]] = Field(
        default=None, description="Target modules for PEFT (auto-detected if None)"
    )

    # Training configuration
    learning_rate: float = Field(default=5e-5, description="Learning rate")
    batch_size: int = Field(default=4, description="Training batch size")
    num_epochs: int = Field(default=3, description="Number of training epochs")
    warmup_steps: int = Field(default=100, description="Warmup steps")
    max_grad_norm: float = Field(default=1.0, description="Gradient clipping norm")

    # Data configuration
    sequence_length: int = Field(
        default=60, description="Input sequence length for time series"
    )
    prediction_horizon: int = Field(default=1, description="Number of steps to predict")
    features: List[str] = Field(
        default=["open", "high", "low", "close", "volume"],
        description="Features to use for training",
    )
    target_column: str = Field(
        default="close", description="Target column for prediction"
    )
    train_split: float = Field(default=0.7, description="Training data split ratio")
    val_split: float = Field(default=0.15, description="Validation data split ratio")

    # Optimization configuration - Made more conservative for time series models
    use_fp16: bool = Field(default=False, description="Use mixed precision training")
    gradient_checkpointing: bool = Field(
        default=False, description="Enable gradient checkpointing (auto-detected)"
    )
    dataloader_num_workers: int = Field(
        default=0,
        description="Number of dataloader workers (0 for Windows compatibility)",
    )

    # Paths
    output_dir: Path = Field(
        default=Path("./finetune_outputs"),
        description="Output directory for fine-tuned models",
    )
    cache_dir: Path = Field(
        default=Path("./model_cache"),
        description="Cache directory for pre-trained models",
    )

    # Evaluation configuration
    eval_steps: int = Field(default=100, description="Evaluation frequency")
    save_steps: int = Field(default=500, description="Model saving frequency")
    logging_steps: int = Field(default=50, description="Logging frequency")

    # W&B configuration
    wandb: WandBConfig = Field(
        default_factory=WandBConfig, description="W&B configuration"
    )

    # Advanced configuration
    use_flash_attention: bool = Field(
        default=False, description="Use Flash Attention if available"
    )
    compile_model: bool = Field(
        default=False, description="Use torch.compile for optimization"
    )

    def get_model_output_dir(self) -> Path:
        """Get model-specific output directory"""
        model_name = self.model_name.split("/")[-1]
        suffix = self.peft_method if self.use_peft else "full"
        return self.output_dir / f"{model_name}_{suffix}"
