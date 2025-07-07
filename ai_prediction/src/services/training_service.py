# services/training_service.py

import time
import uuid
from typing import Dict, Any, Optional

from ..models.model_facade import ModelFacade
from ..services.data_service import DataService
from ..schemas.model_schemas import TrainingRequest, TrainingResponse
from ..logger.logger_factory import LoggerFactory
from ..core.config import get_settings


class TrainingService:
    """Service for handling model training operations"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger("TrainingService")
        self.settings = get_settings()
        self.model_facade = ModelFacade()
        self.data_service = DataService()
        self.active_trainings: Dict[str, Dict[str, Any]] = {}

    def start_training(self, request: TrainingRequest) -> TrainingResponse:
        """
        Start model training process

        Args:
            request: Training request parameters

        Returns:
            Training response with results or error
        """
        training_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            self.logger.info(
                f"Starting training {training_id} for {request.symbol} {request.timeframe} {request.model_type}"
            )

            # Track training
            self.active_trainings[training_id] = {
                "status": "starting",
                "start_time": start_time,
                "request": request,
                "progress": 0.0,
            }

            # Check data availability
            data_availability = self.data_service.check_data_availability(
                request.symbol, request.timeframe
            )

            if not data_availability.get("exists", False):
                return TrainingResponse(
                    success=False,
                    message=f"Dataset not found for {request.symbol} {request.timeframe}",
                    error=f"No dataset available for the specified symbol and timeframe",
                )

            # Update progress
            self.active_trainings[training_id]["status"] = "loading_data"
            self.active_trainings[training_id]["progress"] = 0.2

            # Load and prepare data
            data_result = self.data_service.load_and_prepare_data(
                symbol=request.symbol,
                timeframe=request.timeframe,
                config=request.config,
            )

            # Update progress
            self.active_trainings[training_id]["status"] = "training"
            self.active_trainings[training_id]["progress"] = 0.4

            # Train model
            train_result = self.model_facade.train_model(
                symbol=request.symbol,
                timeframe=request.timeframe,
                model_type=request.model_type,
                train_data=data_result["train_data"],
                val_data=data_result["val_data"],
                feature_engineering=data_result["feature_engineering"],
                config=request.config,
            )

            # Update progress
            self.active_trainings[training_id]["progress"] = 1.0
            self.active_trainings[training_id]["status"] = "completed"

            training_time = time.time() - start_time

            if train_result.get("success", False):
                self.logger.info(
                    f"Training {training_id} completed in {training_time:.2f}s"
                )

                return TrainingResponse(
                    success=True,
                    message="Model training completed successfully",
                    training_id=training_id,
                    model_path=train_result.get("model_path"),
                    training_metrics=train_result.get("training_metrics", {}),
                    validation_metrics=train_result.get("validation_metrics", {}),
                    training_duration=training_time,
                )
            else:
                # Training failed
                self.active_trainings[training_id]["status"] = "failed"
                self.active_trainings[training_id]["error"] = train_result.get(
                    "error", "Unknown error"
                )

                return TrainingResponse(
                    success=False,
                    message="Training failed",
                    error=train_result.get("error", "Training failed"),
                    training_duration=training_time,
                )

        except Exception as e:
            self.logger.error(f"Training {training_id} failed: {str(e)}")

            if training_id in self.active_trainings:
                self.active_trainings[training_id]["status"] = "failed"
                self.active_trainings[training_id]["error"] = str(e)

            training_time = time.time() - start_time
            return TrainingResponse(
                success=False,
                message="Training failed",
                error=str(e),
                training_duration=training_time,
            )

    def get_training_status(self, training_id: str) -> Optional[Dict[str, Any]]:
        """Get status of active training"""
        return self.active_trainings.get(training_id)
