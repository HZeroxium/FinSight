# finetune/main.py

"""
Main facade for the finetune module providing a simple API for fine-tuning models.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd

from ..common.logger.logger_factory import LoggerFactory, LoggerType, LogLevel
from .config import FineTuneConfig, WandBConfig
from .data_processor import FinancialDataProcessor
from .model_factory import ModelFactory
from .trainer import FineTuneTrainer
from .evaluator import FineTuneEvaluator
from .predictor import FineTunePredictor


class FineTuneFacade:
    """
    Facade class providing a simple interface for fine-tuning financial models.
    Encapsulates the complexity of the fine-tuning pipeline.
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

        self.logger.info("✓ FineTune Facade initialized")

    def prepare_data(self, data_path: str) -> Dict[str, Any]:
        """
        Prepare data for fine-tuning

        Args:
            data_path: Path to the dataset file

        Returns:
            Prepared datasets
        """
        self.logger.info(f"Preparing data from {data_path}")

        # Load and process data
        dataset_dict = self.data_processor.load_and_prepare_data(Path(data_path))

        # Create torch datasets
        torch_datasets = self.data_processor.create_torch_datasets(dataset_dict)

        self.logger.info("✓ Data preparation completed")
        return torch_datasets

    def finetune(
        self,
        data_path: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Complete fine-tuning pipeline

        Args:
            data_path: Path to the dataset file
            output_dir: Output directory for saving models

        Returns:
            Training and evaluation results
        """
        self.logger.info("🚀 Starting complete fine-tuning pipeline")

        # Update output directory if provided
        if output_dir:
            self.config.output_dir = Path(output_dir)

        try:
            # Prepare data
            datasets = self.prepare_data(data_path)

            # Initialize trainer
            self.trainer = FineTuneTrainer(
                config=self.config,
                model_factory=self.model_factory,
                data_processor=self.data_processor,
                evaluator=self.evaluator,
            )

            # Setup model
            self.trainer.setup_model()

            # Train model
            train_results = self.trainer.train(
                datasets["train"], datasets.get("validation")
            )

            # Evaluate on test set
            test_results = self.trainer.evaluate_model(datasets["test"])

            # Save model
            model_path = self.trainer.save_model()

            results = {
                "training": train_results,
                "evaluation": test_results,
                "model_path": model_path,
                "status": "success",
                "config": self.config.model_dump(),
            }

            self.logger.info("✅ Fine-tuning pipeline completed successfully")
            return results

        except Exception as e:
            error_msg = f"Fine-tuning failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {
                "status": "failed",
                "error": error_msg,
                "model_path": None,  # Ensure this key exists
                "training": {},
                "evaluation": {},
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

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current configuration and model"""
        return {
            "config": self.config.model_dump(),
            "model_name": self.config.model_name,
            "training_params": {
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
            },
            "peft_enabled": self.config.use_peft,
            "wandb_enabled": self.config.wandb.enabled,
        }


def create_default_facade() -> FineTuneFacade:
    """Create a facade with default configuration for quick start"""
    config = FineTuneConfig()

    # Set sensible defaults for financial data
    config.model_name = "google/flan-t5-small"
    config.num_epochs = 3
    config.batch_size = 4
    config.learning_rate = 5e-5

    # W&B configuration
    wandb_config = WandBConfig()
    wandb_config.enabled = False  # Disabled by default
    wandb_config.project = "finsight-finetune"
    config.wandb = wandb_config

    return FineTuneFacade(config)


def quick_finetune(
    data_path: str,
    output_dir: str = "./finetune_output",
    model_name: str = "google/flan-t5-small",
) -> Dict[str, Any]:
    """
    Quick utility function for immediate fine-tuning

    Args:
        data_path: Path to dataset
        output_dir: Output directory
        model_name: Model to fine-tune

    Returns:
        Results
    """
    facade = create_default_facade()
    return facade.quick_finetune(
        data_path=data_path,
        model_name=model_name,
    )
