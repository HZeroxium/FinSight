from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .base_schemas import BaseResponse
from .enums import CryptoSymbol, ModelSelectionPriority, ModelType, TimeFrame


class ModelConfig(BaseModel):
    """Model configuration schema"""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    # Core model parameters
    context_length: int = Field(64, gt=0, le=512, description="Input sequence length")
    prediction_length: int = Field(1, gt=0, le=24, description="Prediction horizon")
    target_column: str = Field("close", description="Target column to predict")
    feature_columns: Optional[List[str]] = Field(
        None, description="Specific feature columns to use (if None, will use default)"
    )

    # Training hyperparameters
    num_epochs: int = Field(10, gt=0, le=100, description="Number of training epochs")
    batch_size: int = Field(32, gt=0, le=256, description="Training batch size")
    learning_rate: float = Field(1e-3, gt=0, le=1, description="Learning rate")

    # Feature engineering parameters
    use_technical_indicators: bool = Field(
        True, description="Whether to add technical indicators"
    )
    add_datetime_features: bool = Field(
        False, description="Whether to add datetime features"
    )
    normalize_features: bool = Field(True, description="Whether to normalize features")

    # Data splitting
    train_ratio: float = Field(0.8, gt=0, lt=1, description="Training data ratio")
    val_ratio: float = Field(0.1, gt=0, lt=1, description="Validation data ratio")

    # Model-specific parameters (flexible for different models)
    model_specific_params: Dict[str, Any] = Field(
        default_factory=dict, description="Model-specific parameters"
    )

    @field_validator("val_ratio")
    @classmethod
    def validate_val_ratio(cls, v, info):
        values = info.data if hasattr(info, "data") else {}
        train_ratio = values.get("train_ratio", 0.8)
        if train_ratio + v >= 1.0:
            raise ValueError("train_ratio + val_ratio must be less than 1.0")
        return v


class TrainingRequest(BaseModel):
    """Request schema for model training"""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    symbol: str = Field(
        CryptoSymbol.BTCUSDT, min_length=1, description="Trading symbol (e.g., BTCUSDT)"
    )
    timeframe: TimeFrame = Field(TimeFrame.DAY_1, description="Data timeframe")
    model_type: ModelType = Field(
        ModelType.PATCHTST, description="Type of model to train"
    )
    config: ModelConfig = Field(
        default_factory=ModelConfig, description="Model configuration"
    )


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
    training_duration: Optional[float] = Field(
        None, description="Training duration in seconds"
    )


class PredictionRequest(BaseModel):
    """Request schema for model prediction"""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    symbol: CryptoSymbol = Field(
        CryptoSymbol.BTCUSDT, min_length=1, description="Trading symbol"
    )
    timeframe: TimeFrame = Field(TimeFrame.DAY_1, description="Data timeframe")
    model_type: Optional[ModelType] = Field(
        None,
        description="Model type (if None, will auto-select best available)",
    )
    n_steps: int = Field(1, gt=0, le=100, description="Number of prediction steps")
    enable_fallback: bool = Field(
        True, description="Whether to enable model fallback strategies"
    )


class ModelSelectionInfo(BaseModel):
    """Information about the selected model for prediction"""

    model_config = ConfigDict(validate_assignment=True)

    symbol: str = Field(..., description="Symbol used for prediction")
    timeframe: str = Field(..., description="Timeframe used for prediction")
    model_type: str = Field(..., description="Model type used for prediction")
    model_path: Optional[str] = Field(None, description="Path to the selected model")
    selection_priority: str = Field(
        ..., description="Priority level of model selection"
    )
    fallback_applied: bool = Field(..., description="Whether fallback was applied")
    fallback_reason: Optional[str] = Field(
        None, description="Reason for fallback if applied"
    )
    confidence_score: float = Field(
        ..., description="Confidence score of the selection"
    )


class FallbackInfo(BaseModel):
    """Information about fallback operations"""

    model_config = ConfigDict(validate_assignment=True)

    fallback_applied: bool = Field(..., description="Whether fallback was applied")
    original_request: Dict[str, Any] = Field(
        ..., description="Original request parameters"
    )
    selected_model: ModelSelectionInfo = Field(
        ..., description="Actually selected model"
    )
    fallback_reason: Optional[str] = Field(None, description="Reason for fallback")
    confidence_score: float = Field(
        ..., description="Confidence score of the selection"
    )


class PredictionResponse(BaseResponse):
    """Response schema for model prediction"""

    model_config = ConfigDict(validate_assignment=True)

    predictions: Optional[List[float]] = Field(
        None, description="Raw prediction values"
    )
    prediction_percentages: Optional[List[float]] = Field(
        None,
        description="Prediction values as percentage changes (positive = increase, negative = decrease)",
    )
    prediction_timestamps: Optional[List[str]] = Field(
        None, description="Prediction timestamps"
    )
    current_price: Optional[float] = Field(None, description="Current price")
    predicted_change_pct: Optional[float] = Field(
        None, description="Predicted change percentage"
    )
    confidence_score: Optional[float] = Field(None, description="Confidence score")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model information")

    # New fallback-related fields
    fallback_info: Optional[FallbackInfo] = Field(
        None, description="Information about fallback operations"
    )
    model_selection: Optional[ModelSelectionInfo] = Field(
        None, description="Information about the selected model"
    )
    prediction_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional prediction metadata"
    )


class ModelInfo(BaseModel):
    """Model information schema"""

    model_config = ConfigDict(validate_assignment=True)

    symbol: str = Field(..., description="Trading symbol")
    timeframe: TimeFrame = Field(..., description="Data timeframe")
    model_type: ModelType = Field(..., description="Model type")
    model_path: str = Field(..., description="Path to model")
    created_at: datetime = Field(..., description="Model creation timestamp")
    config: ModelConfig = Field(..., description="Model configuration")
    file_size_mb: Optional[float] = Field(None, description="Model file size in MB")
    is_available: bool = Field(True, description="Whether model is available")
