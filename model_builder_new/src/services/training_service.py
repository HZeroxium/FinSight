# services/training_service.py

import time
import uuid
from typing import Dict, Any, Optional


from ..models.model_facade import ModelFacade
from ..schemas.training_schemas import TrainingRequest, TrainingResponse
from ..logger.logger_factory import LoggerFactory
from ..core.config import get_settings


class TrainingService:
    """Service for handling model training operations"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger("TrainingService")
        self.settings = get_settings()
        self.model_facade = ModelFacade()
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
                f"Starting training {training_id} for {request.symbol} {request.timeframe}"
            )

            # Track training
            self.active_trainings[training_id] = {
                "status": "starting",
                "start_time": start_time,
                "request": request,
                "progress": 0.0,
            }

            # Check if dataset exists
            data_file = (
                self.settings.data_dir / f"{request.symbol}_{request.timeframe}.csv"
            )
            if not data_file.exists():
                return TrainingResponse(
                    success=False,
                    message=f"Dataset not found for {request.symbol} {request.timeframe}",
                    error=f"No dataset file found at {data_file}",
                )

            # Update progress
            self.active_trainings[training_id]["status"] = "training"
            self.active_trainings[training_id]["progress"] = 0.2

            # Use ModelFacade for training
            train_result = self.model_facade.train_model(
                symbol=request.symbol,
                timeframe=request.timeframe.value,
                model_type=request.model_type.value,
                config={
                    "context_length": request.context_length,
                    "prediction_length": request.prediction_length,
                    "target_column": request.target_column,
                    "num_epochs": request.num_epochs,
                    "batch_size": request.batch_size,
                    "learning_rate": request.learning_rate,
                    "use_technical_indicators": request.use_technical_indicators,
                    "normalize_features": request.normalize_features,
                    "train_ratio": request.train_ratio,
                    "val_ratio": request.val_ratio,
                    "feature_columns": request.feature_columns,
                },
            )

            # Update progress
            self.active_trainings[training_id]["progress"] = 1.0
            self.active_trainings[training_id]["status"] = "completed"

            training_time = time.time() - start_time

            if train_result["success"]:
                self.logger.info(
                    f"Training {training_id} completed in {training_time:.2f}s"
                )

                return TrainingResponse(
                    success=True,
                    message="Model training completed successfully",
                    training_id=training_id,
                    model_path=train_result.get("model_path", ""),
                    training_metrics=train_result.get("training_metrics", {}),
                    validation_metrics=train_result.get("validation_metrics", {}),
                    training_time=training_time,
                    model_configuration={
                        "model_type": request.model_type,
                        "context_length": request.context_length,
                        "prediction_length": request.prediction_length,
                        "target_column": request.target_column,
                    },
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
                )

        except Exception as e:
            self.logger.error(f"Training {training_id} failed: {str(e)}")

            if training_id in self.active_trainings:
                self.active_trainings[training_id]["status"] = "failed"
                self.active_trainings[training_id]["error"] = str(e)

            return TrainingResponse(
                success=False, message="Training failed", error=str(e)
            )

    def get_training_status(self, training_id: str) -> Optional[Dict[str, Any]]:
        """Get status of active training"""
        return self.active_trainings.get(training_id)
