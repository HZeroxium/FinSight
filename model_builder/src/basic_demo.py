# basic_demo.py

"""
Main module for AI prediction service demonstrating basic functionality.
This module provides a simple interface for training and using financial prediction models.
"""

import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from .core.config import Config, create_development_config
from .data import FinancialDataLoader, FeatureEngineering
from .models import create_model
from .training.trainer import ModelTrainer
from .utils import DeviceUtils, CommonUtils, MetricUtils
from .common.logger.logger_factory import LoggerFactory, LoggerType, LogLevel


class AIPredictor:
    """
    Main AI Predictor class providing high-level interface for financial prediction.

    This class orchestrates the entire pipeline from data loading to model training
    and prediction, making it easy to use the AI prediction system.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize AI Predictor

        Args:
            config: Optional configuration. If None, uses development config.
        """
        self.config = config or create_development_config()
        self.logger = LoggerFactory.get_logger(
            name=self.__class__.__name__,
            logger_type=LoggerType.STANDARD,
            level=LogLevel[self.config.log_level],
            use_colors=True,
        )

        # Initialize components
        self.device = None
        self.model = None
        self.trainer = None
        self.data_loader = None
        self.feature_engineering = None

        # Data storage
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

        # Setup
        self._setup()

    def _setup(self) -> None:
        """Setup AI Predictor components"""
        try:
            # Set random seed for reproducibility
            CommonUtils.set_seed(self.config.random_seed)

            # Setup device
            self.device = DeviceUtils.get_device(
                prefer_gpu=self.config.model.use_gpu, gpu_id=self.config.model.gpu_id
            )

            # Initialize data components
            self.data_loader = FinancialDataLoader(self.config)
            self.feature_engineering = FeatureEngineering(self.config)

            self.logger.info("AI Predictor setup completed successfully")

        except Exception as e:
            self.logger.error(f"Failed to setup AI Predictor: {str(e)}")
            raise

    def load_and_prepare_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and prepare data for training

        Args:
            data_path: Optional path to data file. Uses config default if None.

        Returns:
            pd.DataFrame: Processed data ready for model training
        """
        try:
            self.logger.info("Loading and preparing data...")

            # Load raw data
            if data_path:
                self.config.data.data_file = data_path

            raw_data = self.data_loader.load_data()
            self.logger.info(f"Loaded raw data with shape: {raw_data.shape}")

            # Process data through feature engineering
            processed_data = self.feature_engineering.process_data(raw_data, fit=True)
            self.logger.info(f"Processed data with shape: {processed_data.shape}")

            # Update config with actual feature dimensions
            numeric_features = processed_data.select_dtypes(
                include=["float64", "int64"]
            ).columns
            available_features = [
                col
                for col in self.config.model.features_to_use
                if col in numeric_features
            ]

            if len(available_features) != len(self.config.model.features_to_use):
                self.logger.warning(
                    f"Some configured features not found. Using: {available_features}"
                )
                self.config.model.features_to_use = available_features

            self.config.model.input_dim = len(available_features)

            return processed_data

        except Exception as e:
            self.logger.error(f"Failed to load and prepare data: {str(e)}")
            raise

    def create_data_loaders(
        self, processed_data: pd.DataFrame
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch data loaders for training

        Args:
            processed_data: Processed data from load_and_prepare_data

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        try:
            self.logger.info("Creating data loaders...")

            # Create data loaders
            train_loader, val_loader, test_loader = self.data_loader.prepare_data(
                processed_data,
                feature_columns=self.config.model.features_to_use,
                target_column=self.config.model.target_column,
            )

            # Store for later use
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader

            self.logger.info(
                f"Created data loaders - Train: {len(train_loader)} batches, "
                f"Val: {len(val_loader)} batches, Test: {len(test_loader)} batches"
            )

            return train_loader, val_loader, test_loader

        except Exception as e:
            self.logger.error(f"Failed to create data loaders: {str(e)}")
            raise

    def create_model(self) -> torch.nn.Module:
        """
        Create and initialize model

        Returns:
            torch.nn.Module: Initialized model
        """
        try:
            self.logger.info(f"Creating {self.config.model.model_type} model...")

            # Create model
            self.model = create_model(self.config.model.model_type.value, self.config)
            self.model.to(self.device)

            # Log model information
            model_info = self.model.get_model_info()
            self.logger.info(f"Model created: {model_info['model_name']}")
            self.logger.info(f"Parameters: {model_info['num_parameters']:,}")
            self.logger.info(f"Model size: {model_info['model_size_mb']:.2f} MB")

            return self.model

        except Exception as e:
            self.logger.error(f"Failed to create model: {str(e)}")
            raise

    def train_model(
        self, train_loader: DataLoader, val_loader: DataLoader, save_best: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_best: Whether to save the best model

        Returns:
            Dict containing training history and metrics
        """
        try:
            self.logger.info("Starting model training...")

            if self.model is None:
                raise ValueError("Model not created. Call create_model() first.")

            # Create trainer
            self.trainer = ModelTrainer(self.config, self.device)

            # Train model
            training_history = self.trainer.train(
                model=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
                save_best=save_best,
            )

            self.logger.info("Model training completed successfully")
            return training_history

        except Exception as e:
            self.logger.error(f"Failed to train model: {str(e)}")
            raise

    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test data

        Args:
            test_loader: Test data loader

        Returns:
            Dict containing evaluation metrics
        """
        try:
            self.logger.info("Evaluating model on test data...")

            if self.model is None:
                raise ValueError("Model not created. Call create_model() first.")

            # Get predictions
            predictions, targets = self.model.predict_batch(test_loader, self.device)

            # Calculate metrics
            metrics = MetricUtils.calculate_all_metrics(targets, predictions)

            # Log metrics
            self.logger.info("Test Results:")
            for metric_name, metric_value in metrics.items():
                self.logger.info(f"  {metric_name}: {metric_value:.4f}")

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to evaluate model: {str(e)}")
            raise

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on new data

        Args:
            data: Input data tensor

        Returns:
            torch.Tensor: Model predictions
        """
        try:
            if self.model is None:
                raise ValueError("Model not created. Call create_model() first.")

            data = data.to(self.device)
            predictions = self.model.predict(data)

            return predictions

        except Exception as e:
            self.logger.error(f"Failed to make predictions: {str(e)}")
            raise

    def save_model(self, model_path: Optional[str] = None) -> str:
        """
        Save trained model

        Args:
            model_path: Optional path to save model. Uses default if None.

        Returns:
            str: Path where model was saved
        """
        try:
            if self.model is None:
                raise ValueError("Model not created. Call create_model() first.")

            if model_path is None:
                model_name = (
                    f"{self.model.get_model_name()}_{CommonUtils.get_timestamp()}"
                )
                model_path = str(self.config.get_model_save_path(model_name))

            # Save model
            self.model.save_checkpoint(model_path)
            self.logger.info(f"Model saved to: {model_path}")

            return model_path

        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise

    def load_model(self, model_path: str) -> None:
        """
        Load trained model

        Args:
            model_path: Path to saved model
        """
        try:
            if self.model is None:
                self.create_model()

            # Load model
            checkpoint_info = self.model.load_checkpoint(model_path, device=self.device)
            self.logger.info(f"Model loaded from: {model_path}")

            if checkpoint_info.get("epoch"):
                self.logger.info(f"Loaded model from epoch: {checkpoint_info['epoch']}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def run_full_pipeline(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete ML pipeline from data loading to evaluation

        Args:
            data_path: Optional path to data file

        Returns:
            Dict containing pipeline results
        """
        try:
            self.logger.info("Starting full AI prediction pipeline...")

            # 1. Load and prepare data
            processed_data = self.load_and_prepare_data(data_path)

            # 2. Create data loaders
            train_loader, val_loader, test_loader = self.create_data_loaders(
                processed_data
            )

            # 3. Create model
            model = self.create_model()

            # 4. Train model
            training_history = self.train_model(train_loader, val_loader)

            # 5. Evaluate model
            test_metrics = self.evaluate_model(test_loader)

            # 6. Save model
            model_path = self.save_model()

            results = {
                "training_history": training_history,
                "test_metrics": test_metrics,
                "model_path": model_path,
                "config": self.config.to_dict(),
                "data_info": self.data_loader.get_data_info(),
                "model_info": model.get_model_info(),
            }

            self.logger.info("Full pipeline completed successfully!")
            return results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise


def main():
    """
    Main function demonstrating basic usage of AI Predictor
    """
    logger = LoggerFactory.get_logger(
        name="ai_main",
        logger_type=LoggerType.STANDARD,
        level=LogLevel.INFO,
        use_colors=True,
    )

    try:
        logger.info("Starting AI Prediction Demo")

        # Create development configuration
        config = create_development_config()
        logger.info(f"Using configuration: {config.environment}")

        # Initialize predictor
        predictor = AIPredictor(config)

        # Run full pipeline
        results = predictor.run_full_pipeline()

        # Display results
        logger.info("\n" + "=" * 50)
        logger.info("PIPELINE RESULTS")
        logger.info("=" * 50)

        logger.info(f"Model: {results['model_info']['model_name']}")
        logger.info(f"Parameters: {results['model_info']['num_parameters']:,}")
        logger.info(f"Model saved to: {results['model_path']}")

        logger.info("\nTest Metrics:")
        for metric, value in results["test_metrics"].items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

        logger.info("Demo completed successfully!")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
