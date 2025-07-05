from typing import List, Dict, Optional
from pydantic import BaseModel
from ..finetune.config import ModelType


class TrainingRequest(BaseModel):
    """Request schema for training service"""

    data_path: str
    model_name: str = ModelType.PATCH_TSMIXER
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    sequence_length: int = 60
    prediction_horizon: int = 1
    features: List[str] = ["open", "high", "low", "close", "volume"]
    target_column: str = "close"
    use_peft: bool = False
    output_dir: Optional[str] = None


class TrainingResponse(BaseModel):
    """Response schema for training service"""

    success: bool
    model_path: Optional[str] = None
    training_loss: Optional[float] = None
    validation_metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    training_duration: Optional[float] = None


class PredictionRequest(BaseModel):
    """Request schema for prediction service"""

    model_path: str
    data_path: Optional[str] = None
    data: Optional[List[Dict[str, float]]] = None
    prediction_timeframe: str = "1d"  # 1h, 4h, 12h, 1d, 1w
    n_steps: int = 1


class PredictionResponse(BaseModel):
    """Response schema for prediction service"""

    success: bool
    predictions: Optional[List[float]] = None
    prediction_dates: Optional[List[str]] = None
    current_price: Optional[float] = None
    predicted_change_pct: Optional[float] = None
    confidence: Optional[float] = None
    error_message: Optional[str] = None
