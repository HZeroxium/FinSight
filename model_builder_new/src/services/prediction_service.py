# services/prediction_service.py

import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from ..models.model_facade import ModelFacade
from ..services.data_service import DataService
from ..schemas.model_schemas import PredictionRequest, PredictionResponse
from ..schemas.enums import ModelType, TimeFrame
from ..logger.logger_factory import LoggerFactory
from ..core.config import get_settings


class PredictionService:
    """Service for handling model predictions"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger("PredictionService")
        self.settings = get_settings()
        self.model_facade = ModelFacade()
        self.data_service = DataService()

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Make predictions using trained model

        Args:
            request: Prediction request parameters

        Returns:
            Prediction response with results or error
        """
        start_time = time.time()

        try:
            self.logger.info(
                f"Making prediction for {request.symbol} {request.timeframe} ({request.n_steps} steps)"
            )

            # Auto-select model type if not specified
            if request.model_type is None:
                request.model_type = self._select_best_model(
                    request.symbol, request.timeframe
                )
                if request.model_type is None:
                    return PredictionResponse(
                        success=False,
                        message="No trained model found",
                        error=f"No trained model available for {request.symbol} {request.timeframe}",
                    )

            # Check if model exists
            if not self.model_facade.model_exists(
                request.symbol, request.timeframe, request.model_type
            ):
                return PredictionResponse(
                    success=False,
                    message="Model not found",
                    error=f"No trained {request.model_type.value} model found for {request.symbol} {request.timeframe}",
                )

            # Load recent data for prediction
            recent_data = self.data_service.data_loader.load_data(
                request.symbol, request.timeframe
            )

            if recent_data.empty:
                return PredictionResponse(
                    success=False,
                    message="No data available",
                    error=f"No data found for {request.symbol} {request.timeframe}",
                )

            # Make prediction using model facade
            prediction_result = self.model_facade.predict(
                symbol=request.symbol,
                timeframe=request.timeframe,
                model_type=request.model_type,
                recent_data=recent_data,
                n_steps=request.n_steps,
            )

            execution_time = time.time() - start_time

            if prediction_result.get("success", False):
                self.logger.info(f"Prediction completed in {execution_time:.3f}s")

                # Generate prediction timestamps
                prediction_timestamps = self._generate_prediction_timestamps(
                    request.timeframe, request.n_steps
                )

                return PredictionResponse(
                    success=True,
                    message="Prediction completed successfully",
                    predictions=prediction_result.get("predictions", []),
                    prediction_timestamps=prediction_timestamps,
                    current_price=prediction_result.get("current_price"),
                    predicted_change_pct=prediction_result.get("predicted_change_pct"),
                    confidence_score=0.8,  # Default confidence
                    model_info=prediction_result.get("model_info", {}),
                )
            else:
                return PredictionResponse(
                    success=False,
                    message="Prediction failed",
                    error=prediction_result.get("error", "Unknown error"),
                )

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return PredictionResponse(
                success=False, message="Prediction failed", error=str(e)
            )

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get information about all available trained models"""
        try:
            models = self.model_facade.list_available_models()
            return [
                {
                    "symbol": model.symbol,
                    "timeframe": model.timeframe.value,
                    "model_type": model.model_type.value,
                    "model_path": model.model_path,
                    "created_at": model.created_at.isoformat(),
                    "file_size_mb": model.file_size_mb,
                    "is_available": model.is_available,
                }
                for model in models
            ]
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            return []

    def _select_best_model(
        self, symbol: str, timeframe: TimeFrame
    ) -> Optional[ModelType]:
        """Select the best available model for given symbol and timeframe"""
        # Priority order for model selection
        model_priority = [
            ModelType.PATCHTSMIXER,
            ModelType.PATCHTST,
            ModelType.PYTORCH_TRANSFORMER,
        ]

        for model_type in model_priority:
            if self.model_facade.model_exists(symbol, timeframe, model_type):
                return model_type

        return None

    def _generate_prediction_timestamps(
        self, timeframe: TimeFrame, n_steps: int
    ) -> List[str]:
        """Generate timestamps for predictions"""
        try:
            # Use current time as base
            base_time = datetime.now()

            # Calculate time delta based on timeframe
            time_deltas = {
                TimeFrame.MINUTE_1: timedelta(minutes=1),
                TimeFrame.MINUTE_5: timedelta(minutes=5),
                TimeFrame.MINUTE_15: timedelta(minutes=15),
                TimeFrame.HOUR_1: timedelta(hours=1),
                TimeFrame.HOUR_4: timedelta(hours=4),
                TimeFrame.HOUR_12: timedelta(hours=12),
                TimeFrame.DAY_1: timedelta(days=1),
                TimeFrame.WEEK_1: timedelta(weeks=1),
            }

            delta = time_deltas.get(timeframe, timedelta(hours=1))

            # Generate future timestamps
            timestamps = []
            for i in range(1, n_steps + 1):
                future_time = base_time + (delta * i)
                timestamps.append(future_time.isoformat())

            return timestamps

        except Exception as e:
            self.logger.error(f"Failed to generate timestamps: {e}")
            return [f"Step_{i+1}" for i in range(n_steps)]
