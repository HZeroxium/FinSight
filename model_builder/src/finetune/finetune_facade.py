# finetune/finetune_facade.py

"""
Enhanced main facade for the finetune module providing a comprehensive API for fine-tuning models.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import time
from datetime import datetime

from ..common.logger.logger_factory import LoggerFactory, LoggerType, LogLevel
from ..schemas.model import TrainingRequest, TrainingResponse
from .config import FineTuneConfig, ModelType, TaskType
from .data_processor import FinancialDataProcessor
from .model_factory import ModelFactory
from .trainer import FineTuneTrainer
from .evaluator import FineTuneEvaluator
from .predictor import FineTunePredictor


class FineTuneFacade:
    """
    Enhanced facade class providing a comprehensive interface for fine-tuning financial models.
    Orchestrates all components and provides both simple and advanced APIs.
    """

    def __init__(self, config: Optional[FineTuneConfig] = None):
        """
        Initialize the fine-tuning facade

        Args:
            config: Fine-tuning configuration. If None, creates default config.
        """
        self.config = config or FineTuneConfig()
        self.logger = LoggerFactory.get_logger(
            name="finetune_facade",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
        )

        # Initialize components
        self.data_processor = FinancialDataProcessor(self.config)
        self.model_factory = ModelFactory(self.config)
        self.evaluator = FineTuneEvaluator(self.config)
        self.trainer = None
        self.predictor = None

        # State tracking
        self._last_trained_model_path = None
        self._training_history = []

        self.logger.info("✓ Enhanced FineTune Facade initialized")

    def prepare_data(self, data_path: str) -> Dict[str, Any]:
        """
        Prepare data for fine-tuning

        Args:
            data_path: Path to the dataset file

        Returns:
            Prepared datasets with metadata
        """
        self.logger.info(f"Preparing data from {data_path}")
        start_time = time.time()

        # Load and process data
        dataset_dict = self.data_processor.load_and_prepare_data(Path(data_path))

        # Create torch datasets
        torch_datasets = self.data_processor.create_torch_datasets(dataset_dict)

        preparation_time = time.time() - start_time

        # Add metadata
        metadata = {
            "preparation_time": preparation_time,
            "total_samples": sum(len(ds) for ds in torch_datasets.values()),
            "features_used": self.config.features,
            "sequence_length": self.config.sequence_length,
            "data_splits": {split: len(ds) for split, ds in torch_datasets.items()},
        }

        self.logger.info(f"✓ Data preparation completed in {preparation_time:.2f}s")
        self.logger.info(f"  Total samples: {metadata['total_samples']}")
        self.logger.info(f"  Train/Val/Test: {metadata['data_splits']}")

        return {
            "datasets": torch_datasets,
            "metadata": metadata,
            "config_snapshot": self.config.model_dump(),
        }

    def finetune(
        self,
        data_path: str,
        output_dir: Optional[str] = None,
        validate_on_test: bool = True,
    ) -> Dict[str, Any]:
        """
        Complete fine-tuning pipeline with enhanced monitoring

        Args:
            data_path: Path to the dataset file
            output_dir: Output directory for saving models
            validate_on_test: Whether to run validation on test set

        Returns:
            Comprehensive training and evaluation results
        """
        self.logger.info("🚀 Starting enhanced fine-tuning pipeline")
        pipeline_start = time.time()

        # Update output directory if provided
        if output_dir:
            self.config.output_dir = Path(output_dir)

        try:
            # Prepare data
            data_results = self.prepare_data(data_path)
            datasets = data_results["datasets"]
            data_metadata = data_results["metadata"]

            # Initialize trainer
            self.trainer = FineTuneTrainer(
                config=self.config,
                model_factory=self.model_factory,
                data_processor=self.data_processor,
                evaluator=self.evaluator,
            )

            # Setup model
            model_setup_start = time.time()
            self.trainer.setup_model()
            model_setup_time = time.time() - model_setup_start

            # Train model
            training_start = time.time()
            train_results = self.trainer.train(
                datasets["train"], datasets.get("validation")
            )
            training_time = time.time() - training_start

            # Evaluate on test set if requested
            test_results = {}
            if validate_on_test and "test" in datasets:
                eval_start = time.time()
                test_results = self.trainer.evaluate_model(datasets["test"])
                eval_time = time.time() - eval_start
                test_results["evaluation_time"] = eval_time

            # Save model
            save_start = time.time()
            model_path = self.trainer.save_model()
            save_time = time.time() - save_start
            self._last_trained_model_path = model_path

            # Calculate total pipeline time
            total_pipeline_time = time.time() - pipeline_start

            # Compile comprehensive results
            results = {
                "status": "success",
                "model_path": model_path,
                "training": {
                    **train_results,
                    "training_time": training_time,
                },
                "evaluation": test_results,
                "data_metadata": data_metadata,
                "timing": {
                    "total_pipeline_time": total_pipeline_time,
                    "data_preparation_time": data_metadata["preparation_time"],
                    "model_setup_time": model_setup_time,
                    "training_time": training_time,
                    "save_time": save_time,
                },
                "config": self.config.model_dump(),
                "pipeline_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model_name": self.config.model_name,
                    "task_type": self.config.task_type,
                    "use_peft": self.config.use_peft,
                },
            }

            # Store training history
            self._training_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "model_path": model_path,
                    "training_loss": train_results.get("training_loss"),
                    "config": self.config.model_dump(),
                }
            )

            self.logger.info("✅ Fine-tuning pipeline completed successfully")
            self.logger.info(f"  Total time: {total_pipeline_time:.2f}s")
            self.logger.info(f"  Model saved to: {model_path}")

            return results

        except Exception as e:
            error_msg = f"Fine-tuning failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {
                "status": "failed",
                "error": error_msg,
                "model_path": None,
                "training": {},
                "evaluation": {},
                "timestamp": datetime.now().isoformat(),
                "config": self.config.model_dump(),
            }
        finally:
            if self.trainer:
                self.trainer.finish()

    def quick_finetune(
        self,
        data_path: str,
        model_name: str = "google/flan-t5-small",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
    ) -> Dict[str, Any]:
        """
        Quick fine-tuning with simplified parameters

        Args:
            data_path: Path to the dataset file
            model_name: Hugging Face model name
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate

        Returns:
            Training and evaluation results
        """
        # Update config with quick parameters
        self.config.model_name = model_name
        self.config.num_epochs = num_epochs
        self.config.batch_size = batch_size
        self.config.learning_rate = learning_rate

        return self.finetune(data_path)

    def create_predictor(self, model_path: Optional[str] = None) -> FineTunePredictor:
        """
        Create and configure a predictor instance

        Args:
            model_path: Path to trained model. If None, uses last trained model.

        Returns:
            Configured predictor instance
        """
        if model_path is None:
            model_path = self._last_trained_model_path

        if model_path is None:
            raise ValueError("No model path provided and no model has been trained")

        self.predictor = FineTunePredictor(self.config)
        self.predictor.load_model(model_path)

        self.logger.info(f"✓ Predictor created with model: {model_path}")
        return self.predictor

    def predict_price(
        self,
        data: pd.DataFrame,
        model_path: Optional[str] = None,
        timeframe: str = "1d",
    ) -> Dict[str, Any]:
        """
        High-level price prediction interface

        Args:
            data: Historical price data
            model_path: Path to trained model
            timeframe: Prediction timeframe (1h, 4h, 12h, 1d, 1w)

        Returns:
            Prediction results with metadata
        """
        if self.predictor is None:
            self.create_predictor(model_path)

        # Map timeframe to steps
        timeframe_steps = {"1h": 1, "4h": 4, "12h": 12, "1d": 24, "1w": 168}

        n_steps = timeframe_steps.get(timeframe, 1)

        if n_steps == 1:
            result = self.predictor.predict_single(data)
        else:
            result = self.predictor.predict_sequence(data, n_steps=n_steps)

        result["timeframe"] = timeframe
        result["model_path"] = self._last_trained_model_path

        return result

    def backtest_model(
        self, data: pd.DataFrame, model_path: Optional[str] = None, test_size: int = 100
    ) -> Dict[str, Any]:
        """
        Backtest model on historical data

        Args:
            data: Historical data for backtesting
            model_path: Path to trained model
            test_size: Number of periods to test

        Returns:
            Backtest results with metrics
        """
        if self.predictor is None:
            self.create_predictor(model_path)

        self.logger.info(f"Starting backtest on {test_size} periods...")

        predictions = []
        actual_values = []
        test_start = max(0, len(data) - test_size - self.config.sequence_length)

        for i in range(test_start, len(data) - self.config.sequence_length):
            try:
                historical_data = data.iloc[: i + self.config.sequence_length]
                result = self.predictor.predict_single(historical_data)
                predictions.append(result["prediction"])
                actual_values.append(
                    data.iloc[i + self.config.sequence_length]["close"]
                )
            except Exception as e:
                self.logger.warning(f"Backtest prediction failed at {i}: {e}")

        # Calculate metrics
        import numpy as np

        if predictions and actual_values:
            predictions = np.array(predictions)
            actual_values = np.array(actual_values)

            mse = np.mean((predictions - actual_values) ** 2)
            mae = np.mean(np.abs(predictions - actual_values))
            direction_accuracy = np.mean(
                np.sign(np.diff(predictions)) == np.sign(np.diff(actual_values))
            )

            return {
                "n_predictions": len(predictions),
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(np.sqrt(mse)),
                "direction_accuracy": float(direction_accuracy),
                "predictions": predictions.tolist(),
                "actual_values": actual_values.tolist(),
                "backtest_period": test_size,
            }
        else:
            return {"error": "No valid predictions generated during backtest"}

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the current configuration and models"""
        return {
            "config": self.config.model_dump(),
            "model_name": self.config.model_name,
            "training_params": {
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "sequence_length": self.config.sequence_length,
            },
            "peft_enabled": self.config.use_peft,
            "wandb_enabled": self.config.wandb.enabled,
            "last_trained_model": self._last_trained_model_path,
            "training_history": self._training_history,
            "available_models": [model.value for model in ModelType],
            "supported_tasks": [task.value for task in TaskType],
        }

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get history of all training sessions"""
        return self._training_history.copy()

    def training_service(self, request: TrainingRequest) -> TrainingResponse:
        """
        Training service for API endpoint with improved error handling

        Args:
            request: Training request parameters

        Returns:
            Training response with results
        """
        start_time = datetime.now()

        try:
            self.logger.info(
                f"Starting training service with model: {request.model_name}"
            )

            # Validate model name
            valid_models = [model.value for model in ModelType]
            if request.model_name not in valid_models:
                return TrainingResponse(
                    success=False,
                    error_message=f"Invalid model name. Must be one of: {valid_models}",
                    training_duration=0.0,
                )

            # Create custom configuration
            config = FineTuneConfig()
            config.model_name = request.model_name
            config.task_type = TaskType.FORECASTING
            config.num_epochs = request.num_epochs
            config.batch_size = request.batch_size
            config.learning_rate = request.learning_rate
            config.sequence_length = request.sequence_length
            config.prediction_horizon = request.prediction_horizon
            config.features = request.features
            config.target_column = request.target_column
            config.use_peft = request.use_peft

            # Disable problematic features for time series models
            config.gradient_checkpointing = False
            config.use_fp16 = False
            config.dataloader_num_workers = 0

            if request.output_dir:
                config.output_dir = Path(request.output_dir)

            # Create facade and run training
            facade = FineTuneFacade(config)
            results = facade.finetune(request.data_path)

            training_duration = (datetime.now() - start_time).total_seconds()

            if results["status"] == "success":
                return TrainingResponse(
                    success=True,
                    model_path=results["model_path"],
                    training_loss=results.get("training", {}).get("training_loss"),
                    validation_metrics=results.get("evaluation", {}).get(
                        "basic_metrics"
                    ),
                    training_duration=training_duration,
                )
            else:
                return TrainingResponse(
                    success=False,
                    error_message=results.get("error", "Unknown training error"),
                    training_duration=training_duration,
                )

        except Exception as e:
            training_duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Training service failed: {str(e)}")
            return TrainingResponse(
                success=False,
                error_message=str(e),
                training_duration=training_duration,
            )


def create_default_facade() -> FineTuneFacade:
    """Create a facade with default configuration for quick start"""
    config = FineTuneConfig()

    # Set sensible defaults for financial data
    config.model_name = ModelType.TIMESFM
    config.task_type = TaskType.FORECASTING
    config.num_epochs = 3
    config.batch_size = 4
    config.learning_rate = 5e-5

    # Disable PEFT for time series models by default
    config.use_peft = False

    return FineTuneFacade(config=config)


def quick_finetune(
    data_path: str,
    model_name: str = ModelType.PATCH_TSMIXER,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
) -> Dict[str, Any]:
    """
    Quick fine-tuning function for easy API access

    Args:
        data_path: Path to the financial dataset (CSV format)
        model_name: HuggingFace model name to fine-tune
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training

    Returns:
        Dictionary containing training results and model path
    """
    facade = create_default_facade()

    # Update configuration
    facade.config.model_name = model_name
    facade.config.num_epochs = num_epochs
    facade.config.batch_size = batch_size
    facade.config.learning_rate = learning_rate

    return facade.finetune(data_path)
