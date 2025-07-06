# schemas/prediction_schemas.py

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from .enums import TimeFrame, ModelType
from .base_schemas import BaseResponse


class PredictionRequest(BaseModel):
    """Request schema for prediction"""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    symbol: str = Field(..., min_length=1, description="Trading symbol (e.g., BTCUSDT)")
    timeframe: TimeFrame = Field(..., description="Data timeframe")
    n_steps: int = Field(1, ge=1, le=24, description="Number of prediction steps")
    model_type: Optional[ModelType] = Field(
        None,
        description="Specific model type to use (if None, will use best available)",
    )
    use_latest_data: bool = Field(
        True, description="Whether to use the latest available data"
    )
    context_length: Optional[int] = Field(
        None,
        gt=0,
        le=512,
        description="Override context length (if None, use model default)",
    )


class PredictionResponse(BaseResponse):
    """Response schema for prediction"""

    model_config = ConfigDict(validate_assignment=True)

    predictions: Optional[List[float]] = Field(None, description="Predicted values")
    prediction_timestamps: Optional[List[str]] = Field(
        None, description="Timestamps for predictions"
    )
    current_price: Optional[float] = Field(None, description="Current market price")
    predicted_change_pct: Optional[float] = Field(
        None, description="Predicted percentage change"
    )
    confidence_score: Optional[float] = Field(
        None, ge=0, le=1, description="Prediction confidence score"
    )
    model_info: Optional[Dict[str, Any]] = Field(
        None, description="Information about the model used"
    )
    data_info: Optional[Dict[str, Any]] = Field(
        None, description="Information about the input data"
    )
    execution_time: Optional[float] = Field(
        None, description="Prediction execution time in seconds"
    )


class BacktestRequest(BaseModel):
    """Request schema for backtesting"""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    symbol: str = Field(..., min_length=1, description="Trading symbol (e.g., BTCUSDT)")
    timeframe: TimeFrame = Field(..., description="Data timeframe")
    model_type: ModelType = Field(..., description="Model type to backtest")
    start_date: Optional[str] = Field(
        None, description="Start date for backtesting (YYYY-MM-DD)"
    )
    end_date: Optional[str] = Field(
        None, description="End date for backtesting (YYYY-MM-DD)"
    )
    n_steps: int = Field(1, ge=1, le=24, description="Number of prediction steps")


class BacktestResponse(BaseResponse):
    """Response schema for backtesting"""

    model_config = ConfigDict(validate_assignment=True)

    backtest_results: Optional[Dict[str, Any]] = Field(
        None, description="Detailed backtesting results"
    )
    performance_metrics: Optional[Dict[str, float]] = Field(
        None, description="Performance metrics (MSE, MAE, etc.)"
    )
    predictions_vs_actual: Optional[Dict[str, List[float]]] = Field(
        None, description="Predictions vs actual values"
    )
    visualization_data: Optional[Dict[str, Any]] = Field(
        None, description="Data for creating visualizations"
    )
