# services/prediction_service.py

import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from ..facades import get_serving_facade, get_unified_facade
from ..services.data_service import DataService
from ..schemas.model_schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelSelectionInfo,
    FallbackInfo,
)
from ..schemas.enums import ModelType, TimeFrame, FallbackStrategy
from ..utils.model_fallback_utils import ModelFallbackUtils, ModelSelectionResult
from common.logger.logger_factory import LoggerFactory
from ..core.config import get_settings
from ..utils.model_utils import ModelUtils


class PredictionService:
    """Service for handling model predictions with intelligent fallback strategies"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger("PredictionService")
        self.settings = get_settings()
        self.model_facade = get_serving_facade()
        self.unified_facade = get_unified_facade()  # For cloud-first model checking
        self.data_service = DataService()
        self.fallback_utils = ModelFallbackUtils()
        self.model_utils = ModelUtils()

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Make predictions using trained model with intelligent fallback strategies

        Args:
            request: Prediction request parameters

        Returns:
            Prediction response with results, fallback information, and model selection details
        """
        start_time = time.time()

        try:
            # Extract string values from enums
            symbol = (
                request.symbol.value
                if hasattr(request.symbol, "value")
                else str(request.symbol)
            )
            timeframe = request.timeframe
            model_type = request.model_type

            self.logger.info(
                f"Making prediction for {symbol} {timeframe} ({request.n_steps} steps) "
                f"with fallback enabled: {request.enable_fallback}"
            )

            # Use intelligent model selection with fallback
            model_selection = await self._select_model_with_fallback(
                symbol=symbol,
                timeframe=timeframe,
                model_type=model_type,
                enable_fallback=request.enable_fallback,
            )

            if not model_selection or not model_selection.model_path:
                return PredictionResponse(
                    success=False,
                    message="No suitable model found",
                    error=f"No trained model available for {symbol} {timeframe} after fallback attempts",
                    fallback_info=(
                        self._create_fallback_info(model_selection)
                        if model_selection
                        else None
                    ),
                    model_selection=(
                        self._create_model_selection_info(model_selection)
                        if model_selection
                        else None
                    ),
                )

            # Load recent data for prediction
            # FIX: Use the symbol and timeframe from the selected model (which might be a fallback)
            recent_data = await self.data_service.data_loader.load_data(
                model_selection.symbol, model_selection.timeframe
            )

            if recent_data.empty:
                return PredictionResponse(
                    success=False,
                    message="No data available",
                    error=f"No data found for {model_selection.symbol} {model_selection.timeframe.value}",
                    fallback_info=self._create_fallback_info(model_selection),
                    model_selection=self._create_model_selection_info(model_selection),
                )

            # Make prediction using the selected model
            self.logger.info(
                f"🚀 Making prediction with selected model: {model_selection.symbol} "
                f"{model_selection.timeframe.value} {model_selection.model_type.value}"
            )

            prediction_result = await self.model_facade.predict_async(
                symbol=model_selection.symbol,
                timeframe=model_selection.timeframe,
                model_type=model_selection.model_type,
                recent_data=recent_data,
                n_steps=request.n_steps,
                use_serving_adapter=True,  # Use serving adapter for better performance
            )

            execution_time = time.time() - start_time

            if prediction_result.get("success", False):
                self.logger.info(f"Prediction completed in {execution_time:.3f}s")

                # Get raw predictions and calculate percentages
                raw_predictions = prediction_result.get("predictions", [])
                current_price = prediction_result.get("current_price")

                # Calculate percentage changes using model_utils
                prediction_percentages = (
                    self.model_utils.calculate_prediction_percentages(
                        raw_predictions=raw_predictions, current_price=current_price
                    )
                )

                # Generate prediction timestamps
                # FIX: Use the timeframe from the selected model (which might be a fallback)
                prediction_timestamps = self._generate_prediction_timestamps(
                    model_selection.timeframe, request.n_steps
                )

                # Create comprehensive response with fallback information
                response = PredictionResponse(
                    success=True,
                    message=self._generate_success_message(model_selection, request),
                    predictions=raw_predictions,
                    prediction_percentages=prediction_percentages,
                    prediction_timestamps=prediction_timestamps,
                    current_price=current_price,
                    predicted_change_pct=prediction_result.get("predicted_change_pct"),
                    confidence_score=model_selection.confidence_score,
                    model_info=prediction_result.get("model_info", {}),
                    fallback_info=self._create_fallback_info(model_selection),
                    model_selection=self._create_model_selection_info(model_selection),
                    prediction_metadata={
                        "execution_time_ms": round(execution_time * 1000, 2),
                        "data_points_used": len(recent_data),
                        "fallback_applied": model_selection.fallback_applied,
                        "selection_priority": model_selection.selection_priority.value,
                        "percentage_calculations": self.model_utils.calculate_prediction_metadata(
                            raw_predictions, prediction_percentages, current_price
                        ),
                    },
                )

                # Log fallback information if applied
                if model_selection.fallback_applied:
                    self.logger.info(
                        f"Fallback applied: {request.symbol.value} {request.timeframe.value} -> "
                        f"{model_selection.symbol} {model_selection.timeframe.value} "
                        f"({model_selection.fallback_reason})"
                    )

                return response
            else:
                return PredictionResponse(
                    success=False,
                    message="Prediction failed",
                    error=prediction_result.get("error", "Unknown error"),
                    fallback_info=self._create_fallback_info(model_selection),
                    model_selection=self._create_model_selection_info(model_selection),
                )

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return PredictionResponse(
                success=False, message="Prediction failed", error=str(e)
            )

    async def _select_model_with_fallback(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: Optional[ModelType],
        enable_fallback: bool = True,
    ) -> Optional[ModelSelectionResult]:
        """
        Select the best available model using intelligent fallback strategies.

        Args:
            symbol: Requested trading symbol
            timeframe: Requested timeframe
            model_type: Requested model type (optional)
            enable_fallback: Whether to enable fallback strategies

        Returns:
            ModelSelectionResult with the selected model and fallback information
        """
        try:
            # Use the fallback utilities to find the best available model
            model_selection = await self.fallback_utils.find_best_available_model(
                requested_symbol=symbol,
                requested_timeframe=timeframe,
                requested_model_type=model_type,
                enable_fallback=enable_fallback,
                fallback_strategy=FallbackStrategy.TIMEFRAME_AND_SYMBOL,
            )

            if model_selection:
                self.logger.info(
                    f"Model selection result: {model_selection.symbol} "
                    f"{model_selection.timeframe.value} {model_selection.model_type.value} "
                    f"(priority: {model_selection.selection_priority.value}, "
                    f"fallback: {model_selection.fallback_applied})"
                )

            return model_selection

        except Exception as e:
            self.logger.error(f"Error in model selection with fallback: {e}")
            return None

    def _create_fallback_info(
        self, model_selection: Optional[ModelSelectionResult]
    ) -> Optional[FallbackInfo]:
        """Create fallback information for the response."""
        if not model_selection:
            return None

        return FallbackInfo(
            fallback_applied=model_selection.fallback_applied,
            original_request=model_selection.original_request,
            selected_model=ModelSelectionInfo(
                symbol=model_selection.symbol,
                timeframe=model_selection.timeframe.value,
                model_type=model_selection.model_type.value,
                model_path=(
                    str(model_selection.model_path)
                    if model_selection.model_path
                    else None
                ),
                selection_priority=model_selection.selection_priority.value,
                fallback_applied=model_selection.fallback_applied,
                fallback_reason=model_selection.fallback_reason,
                confidence_score=model_selection.confidence_score,
            ),
            fallback_reason=model_selection.fallback_reason,
            confidence_score=model_selection.confidence_score,
        )

    def _create_model_selection_info(
        self, model_selection: Optional[ModelSelectionResult]
    ) -> Optional[ModelSelectionInfo]:
        """Create model selection information for the response."""
        if not model_selection:
            return None

        return ModelSelectionInfo(
            symbol=model_selection.symbol,
            timeframe=model_selection.timeframe.value,
            model_type=model_selection.model_type.value,
            model_path=(
                str(model_selection.model_path) if model_selection.model_path else None
            ),
            selection_priority=model_selection.selection_priority.value,
            fallback_applied=model_selection.fallback_applied,
            fallback_reason=model_selection.fallback_reason,
            confidence_score=model_selection.confidence_score,
        )

    def _generate_success_message(
        self, model_selection: ModelSelectionResult, request: PredictionRequest
    ) -> str:
        """Generate appropriate success message based on fallback usage."""
        if not model_selection.fallback_applied:
            return "Prediction completed successfully using requested model"

        # Generate fallback-aware message
        original_symbol = request.symbol.value
        original_timeframe = request.timeframe.value
        selected_symbol = model_selection.symbol
        selected_timeframe = model_selection.timeframe.value

        if (
            original_symbol != selected_symbol
            and original_timeframe != selected_timeframe
        ):
            return f"Prediction completed successfully using fallback model ({original_symbol} {original_timeframe} -> {selected_symbol} {selected_timeframe})"
        elif original_symbol != selected_symbol:
            return f"Prediction completed successfully using fallback symbol ({original_symbol} -> {selected_symbol})"
        elif original_timeframe != selected_timeframe:
            return f"Prediction completed successfully using fallback timeframe ({original_timeframe} -> {selected_timeframe})"
        else:
            return "Prediction completed successfully using fallback model type"

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get information about all available trained models"""
        try:
            models = self.model_facade.list_available_models()
            self.logger.info(f"Found {len(models)} available models for prediction")
            for model in models:
                self.logger.info(
                    f"Model: {model.symbol} {model.timeframe} {model.model_type} at {model.model_path}"
                )
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

    async def _select_best_model(
        self, symbol: str, timeframe: TimeFrame
    ) -> Optional[ModelType]:
        """Select the best available model for given symbol and timeframe using cloud-first strategy"""
        # Ensure symbol is string
        if hasattr(symbol, "value"):
            symbol = symbol.value

        # Priority order for model selection
        model_priority = [
            ModelType.PATCHTSMIXER,
            ModelType.PATCHTST,
            ModelType.PYTORCH_TRANSFORMER,
        ]

        self.logger.info(
            f"🔍 Searching for best model with cloud-first strategy: {symbol} {timeframe}"
        )

        for model_type in model_priority:
            # Use cloud-first model checking
            model_exists = await self.unified_facade.model_exists(
                symbol, timeframe, model_type
            )
            if model_exists:
                self.logger.info(
                    f"✅ Selected model: {symbol} {timeframe} {model_type}"
                )
                return model_type

        # Debug: List what models actually exist locally
        available_models = self.model_facade.list_available_models()
        if available_models:
            self.logger.warning(f"📁 Available local models but none match criteria:")
            for model in available_models:
                self.logger.warning(
                    f"  - {model.symbol} {model.timeframe} {model.model_type}"
                )

        self.logger.warning(
            f"❌ No available model found for {symbol} {timeframe} (checked both cloud and local)"
        )
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
