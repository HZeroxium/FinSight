# services/prediction_service.py

import time
import json
from typing import Dict, Any, List

from datetime import datetime, timedelta

from ..models.model_facade import ModelFacade
from ..schemas.prediction_schemas import PredictionRequest, PredictionResponse
from ..schemas.enums import ModelType, TimeFrame
from ..logger.logger_factory import LoggerFactory
from ..core.config import get_settings


class PredictionService:
    """Service for handling model predictions"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger("PredictionService")
        self.settings = get_settings()
        self.model_facade = ModelFacade()

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
                f"Making prediction for {request.symbol} {request.timeframe}"
            )

            # Use ModelFacade for prediction
            prediction_result = self.model_facade.predict(
                symbol=request.symbol,
                timeframe=request.timeframe.value,
                model_type=request.model_type.value if request.model_type else None,
                n_steps=request.n_steps,
                config={
                    "context_length": request.context_length,
                    "use_latest_data": request.use_latest_data,
                },
            )

            execution_time = time.time() - start_time

            if prediction_result["success"]:
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
                    confidence_score=prediction_result.get("confidence_score", 0.8),
                    model_info=prediction_result.get("model_info", {}),
                    data_info=prediction_result.get("data_info", {}),
                    execution_time=execution_time,
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

    def get_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get information about all available trained models"""

        models_info = {}

        try:
            if not self.settings.models_dir.exists():
                return models_info

            for model_dir in self.settings.models_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                metadata_file = model_dir / "metadata.json"
                if not metadata_file.exists():
                    continue

                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)

                    symbol_timeframe = f"{metadata['symbol']}_{metadata['timeframe']}"

                    if symbol_timeframe not in models_info:
                        models_info[symbol_timeframe] = []

                    model_files = list(model_dir.glob("model_*.pkl"))
                    if model_files:
                        models_info[symbol_timeframe].append(
                            {
                                "model_type": metadata["model_type"],
                                "training_id": metadata["training_id"],
                                "created_at": metadata["created_at"],
                                "config": metadata.get("config", {}),
                                "model_path": str(model_files[0]),
                                "status": "available",
                            }
                        )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to read metadata from {metadata_file}: {e}"
                    )
                    continue

        except Exception as e:
            self.logger.error(f"Failed to scan models: {e}")

        return models_info
