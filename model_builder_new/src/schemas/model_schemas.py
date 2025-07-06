from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from .enums import ModelType, TimeFrame
from .base_schemas import BaseResponse


class ModelConfig(BaseModel):
    """Configuration for model training and prediction"""

    model_config = ConfigDict(validate_assignment=True)

    context_length: int = Field(64, ge=8, le=512)
    prediction_length: int = Field(1, ge=1, le=24)
    target_column: str = Field("close")
    feature_columns: Optional[List[str]] = Field(None)

    # Training hyperparameters
    num_epochs: int = Field(10, ge=1, le=100)
    batch_size: int = Field(32, ge=1, le=256)
    learning_rate: float = Field(1e-3, gt=0, le=1)

    # Feature engineering
    use_technical_indicators: bool = Field(True)
    add_datetime_features: bool = Field(False)
    normalize_features: bool = Field(True)

    # Data split
    train_ratio: float = Field(0.8, gt=0, lt=1)
    val_ratio: float = Field(0.1, gt=0, lt=1)


class TrainingRequest(BaseModel):
    """Request for model training"""

    model_config = ConfigDict(validate_assignment=True)

    symbol: str = Field(..., min_length=1)
    timeframe: TimeFrame
    model_type: ModelType
    config: ModelConfig = Field(default_factory=ModelConfig)


class TrainingResponse(BaseResponse):
    """Response for model training"""

    model_config = ConfigDict(validate_assignment=True)

    training_id: Optional[str] = None
    model_path: Optional[str] = None
    training_metrics: Optional[Dict[str, float]] = None
    validation_metrics: Optional[Dict[str, float]] = None
    training_duration: Optional[float] = None


class PredictionRequest(BaseModel):
    """Request for model prediction"""

    model_config = ConfigDict(validate_assignment=True)

    symbol: str = Field(..., min_length=1)
    timeframe: TimeFrame
    model_type: Optional[ModelType] = None
    n_steps: int = Field(1, ge=1, le=24)


class PredictionResponse(BaseResponse):
    """Response for model prediction"""

    model_config = ConfigDict(validate_assignment=True)

    predictions: Optional[List[float]] = None
    prediction_timestamps: Optional[List[str]] = None
    current_price: Optional[float] = None
    predicted_change_pct: Optional[float] = None
    confidence_score: Optional[float] = None
    model_info: Optional[Dict[str, Any]] = None


class ModelInfo(BaseModel):
    """Model information and metadata"""

    model_config = ConfigDict(validate_assignment=True)

    symbol: str
    timeframe: TimeFrame
    model_type: ModelType
    model_path: str
    created_at: datetime
    config: ModelConfig
    file_size_mb: Optional[float] = None
    is_available: bool = True


class ModelListResponse(BaseResponse):
    """Response for listing available models"""

    model_config = ConfigDict(validate_assignment=True)

    models: List[ModelInfo] = Field(default_factory=list)
    total_count: int = 0
