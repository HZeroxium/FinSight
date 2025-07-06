# schemas/training_schemas.py

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from .enums import ModelType, TimeFrame
from .base_schemas import BaseResponse


class TrainingRequest(BaseModel):
    """Request schema for model training"""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    symbol: str = Field(..., min_length=1, description="Trading symbol (e.g., BTCUSDT)")
    timeframe: TimeFrame = Field(..., description="Data timeframe")
    model_type: ModelType = Field(..., description="Type of model to train")

    # Target and features
    target_column: str = Field("close", description="Target column to predict")
    feature_columns: Optional[List[str]] = Field(
        None, description="Specific feature columns to use (if None, will use default)"
    )

    # Training hyperparameters
    context_length: int = Field(64, gt=0, le=512, description="Input sequence length")
    prediction_length: int = Field(1, gt=0, le=24, description="Prediction horizon")
    num_epochs: int = Field(10, gt=0, le=100, description="Number of training epochs")
    batch_size: int = Field(32, gt=0, le=256, description="Training batch size")
    learning_rate: float = Field(1e-3, gt=0, le=1, description="Learning rate")

    # Feature engineering parameters
    use_technical_indicators: bool = Field(
        True, description="Whether to add technical indicators"
    )
    normalize_features: bool = Field(True, description="Whether to normalize features")

    # Training configuration
    train_ratio: float = Field(0.8, gt=0, lt=1, description="Training data ratio")
    val_ratio: float = Field(0.1, gt=0, lt=1, description="Validation data ratio")

    @field_validator("val_ratio")
    @classmethod
    def validate_val_ratio(cls, v, info):
        # Access other field values through info.data during validation
        values = info.data if hasattr(info, "data") else {}
        train_ratio = values.get("train_ratio", 0.8)
        if train_ratio + v >= 1.0:
            raise ValueError("train_ratio + val_ratio must be less than 1.0")
        return v


class TrainingResponse(BaseResponse):
    """Response schema for model training"""

    model_config = ConfigDict(validate_assignment=True)

    training_id: Optional[str] = Field(None, description="Training job identifier")
    model_path: Optional[str] = Field(None, description="Path to saved model")
    training_metrics: Optional[Dict[str, float]] = Field(
        None, description="Training performance metrics"
    )
    validation_metrics: Optional[Dict[str, float]] = Field(
        None, description="Validation performance metrics"
    )
    training_time: Optional[float] = Field(
        None, description="Training duration in seconds"
    )
    model_configuration: Optional[Dict[str, Any]] = Field(
        None, description="Model configuration used"
    )


class TrainingStatus(BaseModel):
    """Training status schema"""

    model_config = ConfigDict(validate_assignment=True)

    training_id: str = Field(..., description="Training job identifier")
    status: str = Field(..., description="Training status")
    progress: float = Field(0.0, ge=0, le=1, description="Training progress")
    current_epoch: int = Field(0, ge=0, description="Current epoch")
    total_epochs: int = Field(0, ge=0, description="Total epochs")
    metrics: Optional[Dict[str, float]] = Field(None, description="Current metrics")
    error_message: Optional[str] = Field(None, description="Error message if failed")
