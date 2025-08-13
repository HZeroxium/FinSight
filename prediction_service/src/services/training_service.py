# services/training_service.py

"""
Consolidated training service for handling both synchronous and asynchronous model training
"""

import time
import uuid
import asyncio
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from ..facades import get_training_facade, get_unified_facade
from ..services.data_service import DataService
from ..services.background_task_manager import BackgroundTaskManager
from ..repositories.training_job_facade import (
    TrainingJobFacade,
    get_training_job_facade,
)
from ..schemas.model_schemas import TrainingRequest, TrainingResponse
from ..schemas.training_schemas import (
    AsyncTrainingRequest,
    AsyncTrainingResponse,
    TrainingJobInfo,
    TrainingJobStatusResponse,
    TrainingJobListResponse,
    TrainingJobCancelRequest,
    TrainingJobCancelResponse,
    TrainingQueueResponse,
    TrainingProgressUpdate,
    TrainingJobFilter,
    BackgroundTaskHealthResponse,
    TrainingJobPriority,
)
from ..core.constants import TrainingJobStatus, TrainingConstants, ResponseMessages
from common.logger.logger_factory import LoggerFactory
from ..core.config import get_settings
from ..utils.dependencies import get_experiment_tracker
from ..interfaces.experiment_tracker_interface import RunStatus


class TrainingService:
    """Consolidated service for handling both synchronous and asynchronous model training operations"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger("TrainingService")
        self.settings = get_settings()
        self.model_facade = get_training_facade()
        self.data_service = DataService()

        # Initialize async components (but don't start them yet)
        self.job_facade: Optional[TrainingJobFacade] = None
        self.background_manager: Optional[BackgroundTaskManager] = None

        # Legacy sync support
        self.active_trainings: Dict[str, Dict[str, Any]] = {}
        self._is_initialized = False

    async def initialize(self) -> None:
        """
        Initialize async components (call this when event loop is running)
        """
        if self._is_initialized:
            return

        try:
            # Initialize job facade
            self.job_facade = await get_training_job_facade()

            # Initialize background task manager with the facade
            self.background_manager = BackgroundTaskManager(self.job_facade)
            await self.background_manager.initialize()

            self._is_initialized = True
            self.logger.info("TrainingService initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize TrainingService: {e}")
            raise

    async def _ensure_initialized(self) -> None:
        """Ensure service is initialized before use"""
        if not self._is_initialized:
            await self.initialize()

    async def start_async_training(
        self, request: AsyncTrainingRequest
    ) -> AsyncTrainingResponse:
        """
        Start asynchronous model training

        Args:
            request: Async training request parameters

        Returns:
            AsyncTrainingResponse: Response with job ID and status
        """
        await self._ensure_initialized()

        try:
            self.logger.info(
                f"Starting async training for {request.symbol} {request.timeframe} {request.model_type}"
            )

            # Check if similar training is already running (optional duplicate check)
            if not request.force_retrain:
                existing_job = await self._check_existing_training(request)
                if existing_job:
                    return AsyncTrainingResponse(
                        success=False,
                        message=ResponseMessages.TRAINING_ALREADY_EXISTS,
                        error=f"Training already in progress: {existing_job.job_id}",
                        job_id=existing_job.job_id,
                        status=existing_job.status,
                    )

            # Submit job to background manager
            job_id = await self.background_manager.submit_job(
                request=request,
                training_function=self._execute_training_task,
                progress_callback=self._training_progress_callback,
            )

            # Get queue position estimate
            queue_info = await self.background_manager.get_queue_info()
            queue_position = queue_info.get("queued_jobs", 0) + queue_info.get(
                "running_jobs", 0
            )

            # Estimate duration
            estimated_duration = self._estimate_training_duration(request)

            self.logger.info(f"Submitted async training job {job_id}")

            return AsyncTrainingResponse(
                success=True,
                message=ResponseMessages.TRAINING_STARTED,
                job_id=job_id,
                status=TrainingJobStatus.QUEUED,
                estimated_duration_seconds=estimated_duration,
                queue_position=queue_position,
                status_endpoint=f"/training/status/{job_id}",
            )

        except ValueError as e:
            # Handle queue full or system issues
            self.logger.warning(f"Cannot start training: {e}")
            return AsyncTrainingResponse(success=False, message=str(e), error=str(e))

        except Exception as e:
            self.logger.error(f"Failed to start async training: {e}")
            return AsyncTrainingResponse(
                success=False, message=ResponseMessages.INTERNAL_ERROR, error=str(e)
            )

    async def get_training_status(self, job_id: str) -> TrainingJobStatusResponse:
        """
        Get status of a training job

        Args:
            job_id: Training job identifier

        Returns:
            TrainingJobStatusResponse: Job status information
        """
        try:
            await self._ensure_initialized()

            job_info = await self.job_facade.get_job(job_id)

            if not job_info:
                return TrainingJobStatusResponse(
                    success=False,
                    message=ResponseMessages.TRAINING_NOT_FOUND,
                    error=f"Training job {job_id} not found",
                )

            return TrainingJobStatusResponse(
                success=True,
                message=ResponseMessages.JOB_STATUS_RETRIEVED,
                job=job_info,
            )

        except Exception as e:
            self.logger.error(f"Failed to get training status for {job_id}: {e}")
            return TrainingJobStatusResponse(
                success=False, message=ResponseMessages.INTERNAL_ERROR, error=str(e)
            )

    async def list_training_jobs(
        self, filter_criteria: Optional[TrainingJobFilter] = None
    ) -> TrainingJobListResponse:
        """
        List training jobs with optional filtering

        Args:
            filter_criteria: Optional filter criteria

        Returns:
            TrainingJobListResponse: List of training jobs
        """
        try:
            await self._ensure_initialized()

            jobs = await self.job_facade.list_jobs(filter_criteria)

            # Get statistics
            stats = await self.job_facade.get_job_statistics()

            return TrainingJobListResponse(
                success=True,
                message=ResponseMessages.JOB_HISTORY_RETRIEVED,
                jobs=jobs,
                total_count=stats.get("total_jobs", 0),
                active_count=stats.get("active_jobs", 0),
                completed_count=stats.get("completed_jobs", 0),
                failed_count=stats.get("failed_jobs", 0),
            )

        except Exception as e:
            self.logger.error(f"Failed to list training jobs: {e}")
            return TrainingJobListResponse(
                success=False, message=ResponseMessages.INTERNAL_ERROR, error=str(e)
            )

    async def cancel_training_job(
        self, job_id: str, cancel_request: TrainingJobCancelRequest
    ) -> TrainingJobCancelResponse:
        """
        Cancel a training job

        Args:
            job_id: Job identifier
            cancel_request: Cancellation request parameters

        Returns:
            TrainingJobCancelResponse: Cancellation result
        """
        try:
            await self._ensure_initialized()

            # Get job info to check if it was running
            job_info = await self.job_facade.get_job(job_id)
            if not job_info:
                return TrainingJobCancelResponse(
                    success=False,
                    message=ResponseMessages.TRAINING_NOT_FOUND,
                    error=f"Training job {job_id} not found",
                    job_id=job_id,
                    was_running=False,
                    cleanup_completed=False,
                )

            was_running = TrainingJobStatus.is_active(
                job_info.status.value
                if hasattr(job_info.status, "value")
                else job_info.status
            )

            # Cancel the job
            success = await self.background_manager.cancel_job(
                job_id, cancel_request.force
            )

            if success:
                return TrainingJobCancelResponse(
                    success=True,
                    message=ResponseMessages.TRAINING_CANCELLED,
                    job_id=job_id,
                    was_running=was_running,
                    cleanup_completed=True,
                )
            else:
                return TrainingJobCancelResponse(
                    success=False,
                    message="Failed to cancel training job",
                    error="Job may be in a critical stage or already completed",
                    job_id=job_id,
                    was_running=was_running,
                    cleanup_completed=False,
                )

        except Exception as e:
            self.logger.error(f"Failed to cancel training job {job_id}: {e}")
            return TrainingJobCancelResponse(
                success=False,
                message=ResponseMessages.INTERNAL_ERROR,
                error=str(e),
                job_id=job_id,
                was_running=False,
                cleanup_completed=False,
            )

    async def get_queue_info(self) -> TrainingQueueResponse:
        """Get information about the training queue"""
        try:
            await self._ensure_initialized()

            queue_info = await self.background_manager.get_queue_info()

            from ..schemas.training_schemas import TrainingQueueInfo

            queue_info_obj = TrainingQueueInfo(**queue_info)

            return TrainingQueueResponse(
                success=True,
                message="Training queue information retrieved",
                queue_info=queue_info_obj,
            )

        except Exception as e:
            self.logger.error(f"Failed to get queue info: {e}")
            return TrainingQueueResponse(
                success=False, message=ResponseMessages.INTERNAL_ERROR, error=str(e)
            )

    async def get_background_health(self) -> BackgroundTaskHealthResponse:
        """Get background task system health"""
        try:
            await self._ensure_initialized()

            health_status = await self.background_manager.get_health_status()

            return BackgroundTaskHealthResponse(
                success=True,
                message="Background task health retrieved",
                health=health_status,
            )

        except Exception as e:
            self.logger.error(f"Failed to get background health: {e}")
            return BackgroundTaskHealthResponse(
                success=False, message=ResponseMessages.INTERNAL_ERROR, error=str(e)
            )

    # Legacy synchronous methods for backward compatibility

    async def start_training(self, request: TrainingRequest) -> TrainingResponse:
        """
        Legacy synchronous training method (deprecated - use start_async_training)

        Args:
            request: Training request parameters

        Returns:
            TrainingResponse: Training response with results or error
        """
        self.logger.warning(
            "Using deprecated synchronous training method. Consider using start_async_training."
        )

        training_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            self.logger.info(
                f"Starting legacy training {training_id} for {request.symbol} {request.timeframe} {request.model_type}"
            )

            # Track training
            self.active_trainings[training_id] = {
                "status": "starting",
                "start_time": start_time,
                "request": request,
                "progress": 0.0,
            }

            # Check data availability
            data_availability = await self.data_service.check_data_availability(
                request.symbol, request.timeframe
            )

            if not data_availability.get("exists", False):
                return TrainingResponse(
                    success=False,
                    message=ResponseMessages.DATA_NOT_AVAILABLE,
                    error=f"No dataset available for the specified symbol and timeframe",
                )

            # Update progress
            self.active_trainings[training_id]["status"] = "loading_data"
            self.active_trainings[training_id]["progress"] = 0.2

            # Load and prepare data
            data_result = await self.data_service.load_and_prepare_data(
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
                    message=ResponseMessages.TRAINING_COMPLETED,
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

    # Private async methods for background task execution

    async def _execute_training_task(
        self,
        request: AsyncTrainingRequest,
        progress_callback: Callable[[TrainingJobStatus, float, str], None],
    ) -> Dict[str, Any]:
        """
        Execute training task in background with experiment tracking

        Args:
            request: Training request
            progress_callback: Progress update callback

        Returns:
            Dict[str, Any]: Training results
        """
        experiment_tracker = get_experiment_tracker()
        run_id = None

        try:
            self.logger.info(f"Executing background training for {request.symbol}")

            # Stage 0: Initialize experiment tracking
            await progress_callback(
                TrainingJobStatus.INITIALIZING,
                0.05,
                "Starting experiment tracking (5.0%)",
            )

            # Start experiment run with error handling
            run_name = f"{request.symbol}_{request.timeframe.value}_{request.model_type.value}_{int(time.time())}"
            try:
                run_id = await experiment_tracker.start_run(run_name=run_name)
                self.logger.info(f"Started experiment run: {run_id}")
            except Exception as e:
                self.logger.error(f"Failed to start experiment run: {e}")
                # Continue without experiment tracking if it fails
                run_id = None

            # Log training configuration if run was started successfully
            if run_id:
                try:
                    config_dict = {}
                    if hasattr(request.config, "model_dump"):
                        config_dict = request.config.model_dump()
                    elif hasattr(request.config, "_asdict"):
                        config_dict = request.config._asdict()
                    elif hasattr(request.config, "__dict__"):
                        config_dict = vars(request.config)

                    await experiment_tracker.log_training_config(
                        run_id=run_id,
                        symbol=request.symbol,
                        timeframe=request.timeframe,
                        model_type=request.model_type,
                        config=config_dict,
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to log training config: {e}")

            # Stage 1: Initialize
            await progress_callback(
                TrainingJobStatus.INITIALIZING, 0.1, "Initializing training environment"
            )

            # Stage 2: Check data availability
            await progress_callback(
                TrainingJobStatus.LOADING_DATA, 0.2, "Checking data availability"
            )

            data_availability = await self.data_service.check_data_availability(
                request.symbol, request.timeframe
            )

            if not data_availability.get("exists", False):
                if run_id:
                    try:
                        await experiment_tracker.end_run(run_id, RunStatus.FAILED)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to end run on data availability error: {e}"
                        )
                return {
                    "success": False,
                    "error": "No dataset available for the specified symbol and timeframe",
                    "error_code": "E001",
                    "run_id": run_id,
                }

            # Stage 3: Load and prepare data
            await progress_callback(
                TrainingJobStatus.LOADING_DATA,
                0.3,
                "Loading and preparing training data",
            )

            data_result = await self.data_service.load_and_prepare_data(
                symbol=request.symbol,
                timeframe=request.timeframe,
                config=request.config,
            )

            # Log data statistics if run tracking is enabled
            if run_id and "train_data" in data_result and "val_data" in data_result:
                try:
                    train_data = data_result["train_data"]
                    val_data = data_result["val_data"]

                    await experiment_tracker.log_params(
                        run_id,
                        {
                            "train_samples": len(train_data),
                            "val_samples": len(val_data),
                            "total_samples": len(train_data) + len(val_data),
                            "train_ratio": len(train_data)
                            / (len(train_data) + len(val_data)),
                        },
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to log data statistics: {e}")

            # Stage 4: Feature preparation
            await progress_callback(
                TrainingJobStatus.PREPARING_FEATURES, 0.4, "Preparing features"
            )

            # Stage 5: Training
            await progress_callback(
                TrainingJobStatus.TRAINING, 0.5, "Starting model training"
            )

            # Execute training
            train_result = await self.model_facade.train_model(
                symbol=request.symbol,
                timeframe=request.timeframe,
                model_type=request.model_type,
                train_data=data_result["train_data"],
                val_data=data_result["val_data"],
                feature_engineering=data_result["feature_engineering"],
                config=request.config,
            )

            if not train_result.get("success", False):
                if run_id:
                    try:
                        await experiment_tracker.end_run(run_id, RunStatus.FAILED)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to end run on training failure: {e}"
                        )
                return {
                    "success": False,
                    "error": train_result.get("error", "Training failed"),
                    "error_code": "E003",
                    "run_id": run_id,
                }

            # Log training and validation metrics if run tracking is enabled
            training_metrics = train_result.get("training_metrics", {})
            validation_metrics = train_result.get("validation_metrics", {})

            if run_id and (training_metrics or validation_metrics):
                try:
                    await experiment_tracker.log_model_performance(
                        run_id=run_id,
                        training_metrics=training_metrics,
                        validation_metrics=validation_metrics,
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to log model performance metrics: {e}")

            # Stage 6: Validation
            await progress_callback(
                TrainingJobStatus.VALIDATING, 0.85, "Validating trained model"
            )

            # Stage 7: Saving model
            await progress_callback(
                TrainingJobStatus.SAVING_MODEL, 0.95, "Saving trained model"
            )

            # Log model artifacts and register model if run tracking is enabled
            model_size_bytes = None
            model_version = None
            model_path = train_result.get("model_path")

            if run_id and model_path:
                try:
                    from pathlib import Path

                    model_file = Path(model_path)
                    if model_file.exists():
                        # Get model file size
                        model_size_bytes = model_file.stat().st_size
                        try:
                            await experiment_tracker.log_param(
                                run_id, "model_size_bytes", model_size_bytes
                            )
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to log model size parameter: {e}"
                            )

                        # Log model artifacts
                        artifacts_logged = False
                        try:
                            if model_file.is_file():
                                # Log single model file
                                await experiment_tracker.log_artifact(
                                    run_id=run_id,
                                    local_path=model_file,
                                    artifact_path="model",
                                )
                                artifacts_logged = True
                            else:
                                # Log entire model directory
                                await experiment_tracker.log_artifacts(
                                    run_id=run_id,
                                    local_dir=model_file,
                                    artifact_path="model",
                                )
                                artifacts_logged = True

                            self.logger.info(
                                f"Successfully logged model artifacts for run {run_id}"
                            )
                        except Exception as e:
                            self.logger.warning(f"Failed to log model artifacts: {e}")

                        # Register model in model registry only if artifacts were logged
                        if artifacts_logged:
                            model_name = f"{request.symbol}_{request.timeframe.value}_{request.model_type.value}"
                            try:
                                # Use the full run-based URI for model registration
                                model_uri = f"runs:/{run_id}/model"
                                model_version = await experiment_tracker.register_model(
                                    name=model_name,
                                    model_uri=model_uri,
                                    run_id=run_id,
                                    description=f"Model trained for {request.symbol} {request.timeframe.value} using {request.model_type.value}",
                                    tags={
                                        "symbol": request.symbol,
                                        "timeframe": request.timeframe.value,
                                        "model_type": request.model_type.value,
                                        "framework": "pytorch",
                                    },
                                )
                                self.logger.info(
                                    f"Registered model {model_name} version {model_version}"
                                )
                            except Exception as e:
                                self.logger.warning(f"Could not register model: {e}")
                                # Try without model registry if registration fails
                                self.logger.info(
                                    "Model artifacts are still logged and accessible via MLflow run"
                                )
                        else:
                            self.logger.warning(
                                "Skipping model registration due to artifact logging failure"
                            )

                except Exception as e:
                    self.logger.warning(f"Could not process model artifacts: {e}")

            # Stage 8: Completed
            await progress_callback(
                TrainingJobStatus.COMPLETED, 1.0, "Training completed successfully"
            )

            # End experiment run successfully
            if run_id:
                try:
                    await experiment_tracker.end_run(run_id, RunStatus.FINISHED)
                except Exception as e:
                    self.logger.warning(f"Failed to end run successfully: {e}")

            return {
                "success": True,
                "model_path": model_path,
                "model_size_bytes": model_size_bytes,
                "training_metrics": training_metrics,
                "validation_metrics": validation_metrics,
                "run_id": run_id,
                "model_version": model_version,
            }

        except Exception as e:
            self.logger.error(f"Training task failed: {e}")
            if run_id:
                try:
                    await experiment_tracker.end_run(run_id, RunStatus.FAILED)
                except Exception as end_run_error:
                    self.logger.warning(
                        f"Failed to end run on exception: {end_run_error}"
                    )
            return {
                "success": False,
                "error": str(e),
                "error_code": "E999",
                "run_id": run_id,
            }

    async def _training_progress_callback(
        self, progress_update: TrainingProgressUpdate
    ) -> None:
        """Handle training progress updates"""
        try:
            # Log progress updates
            self.logger.info(
                f"Job {progress_update.job_id}: {progress_update.current_stage} "
                f"({progress_update.progress:.1%})"
            )

            # Here you could add webhook notifications if needed
            # await self._send_webhook_notification(progress_update)

        except Exception as e:
            self.logger.warning(f"Error in progress callback: {e}")

    async def _check_existing_training(
        self, request: AsyncTrainingRequest
    ) -> Optional[TrainingJobInfo]:
        """Check if similar training is already running"""
        try:
            # Get active jobs
            active_jobs = await self.job_facade.get_active_jobs()

            # Look for matching jobs
            for job in active_jobs:
                if (
                    job.symbol == request.symbol.upper()
                    and job.timeframe == request.timeframe.value
                    and job.model_type == request.model_type.value
                ):
                    return job

            return None

        except Exception as e:
            self.logger.error(f"Error checking existing training: {e}")
            return None

    def _estimate_training_duration(self, request: AsyncTrainingRequest) -> int:
        """Estimate training duration based on configuration"""
        base_duration = 300  # 5 minutes base

        # Adjust based on epochs
        epoch_factor = request.config.num_epochs / 10

        # Adjust based on model type
        model_factors = {
            "patchtst": 1.0,
            "patchtsmixer": 1.2,
            "pytorch_lightning_transformer": 1.5,
        }
        model_factor = model_factors.get(request.model_type.value, 1.0)

        # Adjust based on batch size (smaller = longer)
        batch_factor = 64 / request.config.batch_size

        estimated = int(base_duration * epoch_factor * model_factor * batch_factor)
        return max(60, min(estimated, 3600))  # Between 1 minute and 1 hour

    async def shutdown(self) -> None:
        """Shutdown the training service and cleanup resources"""
        try:
            self.logger.info("Shutting down training service...")

            # Shutdown background manager
            if self.background_manager:
                await self.background_manager.shutdown()

            # Shutdown job facade
            if self.job_facade:
                await self.job_facade.shutdown()

            self.logger.info("Training service shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during training service shutdown: {e}")


# No alias needed - TrainingService is the main class
