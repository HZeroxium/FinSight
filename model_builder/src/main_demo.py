# main_demo.py

"""
Comprehensive AI Prediction Demo

This module demonstrates the full capabilities of the AI prediction system,
including data processing, model training, evaluation, and prediction.
It serves as both a usage example and a testing framework.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from .core.config import (
    Config,
    ModelConfig,
    create_development_config,
    create_production_config,
    create_lightweight_config,
    ModelType,
)
from .data import FinancialDataLoader, FeatureEngineering
from .models import create_model
from .training.trainer import ModelTrainer
from .utils import DeviceUtils, CommonUtils, FileUtils
from .common.logger.logger_factory import LoggerFactory, LoggerType, LogLevel


class AIDemo:
    """
    Comprehensive demo class showcasing AI prediction capabilities

    This class provides a complete walkthrough of the AI system including:
    - Data loading and preprocessing
    - Feature engineering
    - Model architecture comparison
    - Training with monitoring
    - Evaluation and analysis
    - Prediction and visualization
    """

    def __init__(self, config: Optional[Config] = None, verbose: bool = True):
        """
        Initialize AI Demo

        Args:
            config: Configuration to use (default: development config)
            verbose: Whether to enable verbose logging
        """
        self.config = config or create_development_config()
        self.verbose = verbose

        # Setup logging
        log_level = LogLevel.DEBUG if verbose else LogLevel.INFO
        self.logger = LoggerFactory.get_logger(
            name=self.__class__.__name__,
            logger_type=LoggerType.STANDARD,
            level=log_level,
            use_colors=True,
        )

        # Initialize components
        self.device = None
        self.data_loader = None
        self.feature_engineering = None
        self.models: Dict[str, torch.nn.Module] = {}
        self.training_results: Dict[str, Dict[str, Any]] = {}

        # Demo results
        self.demo_results: Dict[str, Any] = {
            "setup_info": {},
            "data_analysis": {},
            "model_comparisons": {},
            "training_results": {},
            "evaluation_results": {},
            "predictions": {},
            "visualizations": {},
        }

        self._setup_demo()

    def _setup_demo(self) -> None:
        """Setup demo environment and components"""
        self.logger.info("=" * 80)
        self.logger.info("🚀 AI PREDICTION SYSTEM DEMO")
        self.logger.info("=" * 80)

        # Set reproducibility
        CommonUtils.ensure_reproducibility(self.config.random_seed)

        # Setup device
        self.device = DeviceUtils.get_device(
            prefer_gpu=self.config.model.use_gpu, gpu_id=self.config.model.gpu_id
        )

        # Get environment info
        env_info = CommonUtils.get_environment_info()
        self.demo_results["setup_info"] = {
            "device": str(self.device),
            "environment": env_info,
            "config": self.config.to_dict(),
            "timestamp": CommonUtils.get_readable_timestamp(),
        }

        # Initialize data components
        self.data_loader = FinancialDataLoader(self.config)
        self.feature_engineering = FeatureEngineering(self.config)

        self.logger.info(f"✓ Demo setup completed")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Random seed: {self.config.random_seed}")
        self.logger.info(f"  Configuration: {self.config.environment}")

    def demonstrate_data_processing(self) -> Dict[str, Any]:
        """
        Demonstrate comprehensive data processing capabilities

        Returns:
            Dict containing data processing results and analysis
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 DATA PROCESSING DEMONSTRATION")
        self.logger.info("=" * 60)

        # Load raw data
        self.logger.info("Loading raw financial data...")
        raw_data = self.data_loader.load_data()

        # Basic data analysis
        data_info = self._analyze_raw_data(raw_data)
        self.logger.info(f"✓ Raw data loaded: {raw_data.shape}")

        # Feature engineering demonstration
        self.logger.info("Applying feature engineering...")
        processed_data = self.feature_engineering.process_data(raw_data, fit=True)

        # Analyze processed data
        processed_info = self._analyze_processed_data(processed_data)
        self.logger.info(f"✓ Data processed: {processed_data.shape}")

        # Feature importance analysis
        feature_names = self.feature_engineering.get_feature_importance_names()
        self.logger.info(f"✓ Generated {len(feature_names)} features")

        # Data quality validation
        quality_report = self._validate_data_quality(processed_data)

        # Create sequences for model training
        self.logger.info("Creating training sequences...")
        train_loader, val_loader, test_loader = self.data_loader.prepare_data(
            processed_data,
            feature_columns=self.config.model.features_to_use,
            target_column=self.config.model.target_column,
        )

        # Store data loaders for later use
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        data_results = {
            "raw_data_info": data_info,
            "processed_data_info": processed_info,
            "feature_names": feature_names,
            "data_quality": quality_report,
            "data_loaders": {
                "train_batches": len(train_loader),
                "val_batches": len(val_loader),
                "test_batches": len(test_loader),
            },
        }

        self.demo_results["data_analysis"] = data_results
        return data_results

    def _analyze_raw_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze raw data characteristics"""
        return {
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "dtypes": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "date_range": (
                (
                    data[self.config.data.date_column].min(),
                    data[self.config.data.date_column].max(),
                )
                if self.config.data.date_column in data.columns
                else None
            ),
            "numeric_summary": data.describe().to_dict(),
            "memory_usage": data.memory_usage(deep=True).sum() / 1024**2,  # MB
        }

    def _analyze_processed_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze processed data characteristics"""
        numeric_data = data.select_dtypes(include=[np.number])
        return {
            "shape": data.shape,
            "feature_count": len(numeric_data.columns),
            "missing_values": data.isnull().sum().sum(),
            "correlation_matrix_size": numeric_data.corr().shape,
            "high_correlation_pairs": self._find_high_correlations(numeric_data),
            "feature_variance": numeric_data.var().describe().to_dict(),
            "memory_usage": data.memory_usage(deep=True).sum() / 1024**2,  # MB
        }

    def _find_high_correlations(
        self, data: pd.DataFrame, threshold: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """Find highly correlated feature pairs"""
        corr_matrix = data.corr().abs()
        high_corr = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr.append(
                        (
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j],
                        )
                    )
        return high_corr[:10]  # Top 10 pairs

    def _validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality.
        Returns completeness %, duplicate count, infinite values, zero-variance feature count, etc.
        """
        # 1. Completeness: proportion of non-null cells
        completeness = (1 - data.isnull().sum().sum() / data.size) * 100

        # 2. Duplicates: count of fully duplicated rows
        duplicates = data.duplicated().sum()

        # 3. Infinite values: only numeric columns support isinf
        numeric_data = data.select_dtypes(include=[np.number])
        infinite_values = np.isinf(numeric_data).sum().sum()

        # 4. Zero-variance features: compute variance on numeric columns only
        #    Use numeric_only parameter where supported or pre-select numeric dtypes
        try:
            variances = numeric_data.var()  # numeric_only enforced by selecting dtypes
        except TypeError:
            # Fallback for older pandas versions
            variances = numeric_data.var(numeric_only=True)

        zero_variance_features = int((variances == 0).sum())

        # 5. Data types consistency: simplified check
        data_types_consistency = True

        # 6. Temporal consistency: only meaningful if date_column exists
        temporal_consistency = (
            True if self.config.data.date_column in data.columns else None
        )

        return {
            "completeness": completeness,
            "duplicates": duplicates,
            "infinite_values": int(infinite_values),
            "zero_variance_features": zero_variance_features,
            "data_types_consistency": data_types_consistency,
            "temporal_consistency": temporal_consistency,
        }

    def demonstrate_model_architectures(self) -> Dict[str, Any]:
        """
        Demonstrate different model architectures and their characteristics

        Returns:
            Dict containing model comparison results
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🏗️  MODEL ARCHITECTURE DEMONSTRATION")
        self.logger.info("=" * 60)

        # Model types to demonstrate
        model_configs = {
            "transformer": self.config,
            "lightweight": create_lightweight_config(),
            "hybrid": Config(
                model=ModelConfig(model_type=ModelType.HYBRID_TRANSFORMER)
            ),
        }

        model_comparisons = {}

        for model_name, config in model_configs.items():
            self.logger.info(f"\n🔧 Creating {model_name} model...")

            # Update config with current data dimensions
            config.model.input_dim = self.config.model.input_dim
            config.model.features_to_use = self.config.model.features_to_use

            # Create model
            model = create_model(config.model.model_type.value, config)
            model.to(self.device)

            # Analyze model
            model_info = model.get_model_info()
            param_count = model.count_parameters()

            # Test forward pass
            sample_input = torch.randn(
                1, config.model.sequence_length, config.model.input_dim
            ).to(self.device)

            with torch.no_grad():
                output = model(sample_input)

            model_comparisons[model_name] = {
                "model_info": model_info,
                "parameter_counts": param_count,
                "output_shape": output.shape,
                "model_size_mb": model_info["model_size_mb"],
                "architecture_type": config.model.model_type.value,
                "config": config.model.model_dump(),
            }

            # Store model for training
            self.models[model_name] = model

            self.logger.info(f"✓ {model_name.capitalize()} model created:")
            self.logger.info(f"  Parameters: {param_count['total']:,}")
            self.logger.info(f"  Size: {model_info['model_size_mb']:.2f} MB")
            self.logger.info(f"  Output shape: {output.shape}")

        self.demo_results["model_comparisons"] = model_comparisons
        return model_comparisons

    def demonstrate_training(
        self, model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Demonstrate model training with comprehensive monitoring

        Args:
            model_names: List of model names to train (None for all)

        Returns:
            Dict containing training results for all models
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🏋️  MODEL TRAINING DEMONSTRATION")
        self.logger.info("=" * 60)

        if model_names is None:
            model_names = list(self.models.keys())

        training_results = {}

        for model_name in model_names:
            if model_name not in self.models:
                self.logger.warning(f"Model {model_name} not found, skipping...")
                continue

            self.logger.info(f"\n🚂 Training {model_name} model...")

            model = self.models[model_name]

            # Create trainer
            trainer = ModelTrainer(self.config, self.device)

            # Train model
            training_result = trainer.train(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                save_best=True,
            )

            # Get training summary
            training_summary = trainer.get_training_summary()

            training_results[model_name] = {
                "training_result": training_result,
                "training_summary": training_summary,
                "trainer_config": {
                    "optimizer": self.config.model.optimizer_type.value,
                    "scheduler": self.config.model.scheduler_type.value,
                    "learning_rate": self.config.model.learning_rate,
                    "epochs": self.config.model.epochs,
                },
            }

            self.logger.info(f"✓ {model_name.capitalize()} training completed:")
            self.logger.info(f"  Best val loss: {training_result['best_val_loss']:.6f}")
            self.logger.info(
                f"  Training time: {CommonUtils.format_duration(training_result['total_training_time'])}"
            )
            self.logger.info(f"  Epochs: {training_result['epochs_completed']}")

        self.demo_results["training_results"] = training_results
        return training_results

    def demonstrate_evaluation(
        self, model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Demonstrate comprehensive model evaluation

        Args:
            model_names: List of model names to evaluate

        Returns:
            Dict containing evaluation results for all models
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📈 MODEL EVALUATION DEMONSTRATION")
        self.logger.info("=" * 60)

        if model_names is None:
            model_names = list(self.models.keys())

        evaluation_results = {}

        for model_name in model_names:
            if model_name not in self.models:
                continue

            self.logger.info(f"\n📊 Evaluating {model_name} model...")

            model = self.models[model_name]
            trainer = ModelTrainer(self.config, self.device)

            # Evaluate model
            eval_result = trainer.evaluate(model, self.test_loader)

            # Additional analysis
            predictions, targets = model.predict_batch(self.test_loader, self.device)

            # Calculate additional metrics
            additional_metrics = self._calculate_additional_metrics(
                predictions.numpy(), targets.numpy()
            )

            evaluation_results[model_name] = {
                "evaluation_result": eval_result,
                "additional_metrics": additional_metrics,
                "predictions_sample": {
                    "predictions": predictions[:10].tolist(),
                    "targets": targets[:10].tolist(),
                },
            }

            self.logger.info(f"✓ {model_name.capitalize()} evaluation completed:")
            for metric, value in eval_result["test_metrics"].items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {metric.upper()}: {value:.4f}")

        # Model comparison
        comparison = self._compare_models(evaluation_results)
        evaluation_results["model_comparison"] = comparison

        self.demo_results["evaluation_results"] = evaluation_results
        return evaluation_results

    def _calculate_additional_metrics(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate additional evaluation metrics"""
        # Ensure arrays are flattened and finite
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()

        # Remove invalid values
        valid_mask = np.isfinite(pred_flat) & np.isfinite(target_flat)
        if not valid_mask.any():
            return {
                key: 0.0
                for key in [
                    "prediction_bias",
                    "prediction_variance",
                    "correlation",
                    "max_error",
                    "quantile_loss_50",
                    "hit_rate_5pct",
                ]
            }

        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]

        try:
            correlation = (
                np.corrcoef(pred_valid, target_valid)[0, 1]
                if len(pred_valid) > 1
                else 0.0
            )
            correlation = correlation if np.isfinite(correlation) else 0.0
        except:
            correlation = 0.0

        try:
            # Safe division for hit rate calculation
            relative_errors = np.abs(pred_valid - target_valid)
            threshold_values = 0.05 * np.abs(target_valid)
            # Avoid division by zero
            safe_threshold = np.where(threshold_values == 0, 1e-8, threshold_values)
            hit_rate = np.mean(relative_errors < safe_threshold)
        except:
            hit_rate = 0.0

        return {
            "prediction_bias": float(np.mean(pred_valid - target_valid)),
            "prediction_variance": float(np.var(pred_valid - target_valid)),
            "correlation": float(correlation),
            "max_error": float(np.max(np.abs(pred_valid - target_valid))),
            "quantile_loss_50": float(
                np.mean(
                    np.maximum(
                        0.5 * (pred_valid - target_valid),
                        (0.5 - 1) * (pred_valid - target_valid),
                    )
                )
            ),
            "hit_rate_5pct": float(hit_rate),
        }

    def _compare_models(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare models based on evaluation metrics"""
        comparison = {}

        # Extract key metrics for comparison
        metrics_to_compare = ["rmse", "mae", "mape", "directional_accuracy", "r2"]

        for metric in metrics_to_compare:
            metric_values = {}
            for model_name, results in evaluation_results.items():
                if model_name != "model_comparison":
                    test_metrics = results["evaluation_result"]["test_metrics"]
                    if metric in test_metrics:
                        metric_values[model_name] = test_metrics[metric]

            if metric_values:
                best_model = min(metric_values.items(), key=lambda x: x[1])
                worst_model = max(metric_values.items(), key=lambda x: x[1])

                comparison[metric] = {
                    "values": metric_values,
                    "best_model": best_model[0],
                    "best_value": best_model[1],
                    "worst_model": worst_model[0],
                    "worst_value": worst_model[1],
                    "improvement": (worst_model[1] - best_model[1])
                    / worst_model[1]
                    * 100,
                }

        return comparison

    def demonstrate_predictions(
        self, model_name: str = "transformer"
    ) -> Dict[str, Any]:
        """
        Demonstrate prediction capabilities with visualization

        Args:
            model_name: Name of model to use for predictions

        Returns:
            Dict containing prediction results and analysis
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🔮 PREDICTION DEMONSTRATION")
        self.logger.info("=" * 60)

        if model_name not in self.models:
            self.logger.error(f"Model {model_name} not found")
            return {}

        model = self.models[model_name]

        # Get predictions on test set
        predictions, targets = model.predict_batch(self.test_loader, self.device)

        # Analyze predictions
        prediction_analysis = self._analyze_predictions(
            predictions.numpy(), targets.numpy()
        )

        # Generate future predictions (if applicable)
        future_predictions = self._generate_future_predictions(model)

        prediction_results = {
            "test_predictions": {
                "predictions": predictions.numpy(),
                "targets": targets.numpy(),
                "analysis": prediction_analysis,
            },
            "future_predictions": future_predictions,
            "model_used": model_name,
        }

        self.logger.info(f"✓ Predictions generated using {model_name} model:")
        self.logger.info(f"  Test samples: {len(predictions)}")
        self.logger.info(
            f"  Prediction accuracy: {prediction_analysis['accuracy']:.2f}%"
        )
        self.logger.info(f"  Average error: {prediction_analysis['mean_error']:.4f}")

        self.demo_results["predictions"] = prediction_results
        return prediction_results

    def _analyze_predictions(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction quality"""
        errors = predictions - targets

        return {
            "mean_error": np.mean(errors),
            "std_error": np.std(errors),
            "mae": np.mean(np.abs(errors)),
            "rmse": np.sqrt(np.mean(errors**2)),
            "accuracy": 100 - np.mean(np.abs(errors / targets)) * 100,
            "directional_accuracy": np.mean(
                (predictions[1:] - predictions[:-1]) * (targets[1:] - targets[:-1]) > 0
            )
            * 100,
            "error_distribution": {
                "min": np.min(errors),
                "max": np.max(errors),
                "q25": np.percentile(errors, 25),
                "q50": np.percentile(errors, 50),
                "q75": np.percentile(errors, 75),
            },
        }

    def _generate_future_predictions(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Generate future predictions (simplified demonstration)"""
        # Get last sequence from test data
        last_batch = next(iter(self.test_loader))
        last_sequence = last_batch[0][-1:].to(self.device)  # Take last sample

        # Generate predictions
        model.eval()
        with torch.no_grad():
            future_pred = model(last_sequence)

        return {
            "prediction": future_pred.cpu().numpy().tolist(),
            "confidence": "N/A",  # Would need uncertainty quantification
            "horizon": self.config.model.prediction_horizon,
            "note": "Simplified demonstration - real implementation would use rolling predictions",
        }

    def create_visualizations(self) -> Dict[str, str]:
        """
        Create comprehensive visualizations of results

        Returns:
            Dict containing paths to generated visualization files
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📈 CREATING VISUALIZATIONS")
        self.logger.info("=" * 60)

        viz_dir = Path("demo_visualizations")
        FileUtils.ensure_dir(viz_dir)

        visualization_paths = {}

        try:
            # 1. Training curves
            if "training_results" in self.demo_results:
                viz_path = self._plot_training_curves(viz_dir)
                visualization_paths["training_curves"] = viz_path

            # 2. Model comparison
            if "evaluation_results" in self.demo_results:
                viz_path = self._plot_model_comparison(viz_dir)
                visualization_paths["model_comparison"] = viz_path

            # 3. Prediction analysis
            if "predictions" in self.demo_results:
                viz_path = self._plot_predictions(viz_dir)
                visualization_paths["predictions"] = viz_path

            # 4. Feature importance (if available)
            viz_path = self._plot_feature_analysis(viz_dir)
            visualization_paths["feature_analysis"] = viz_path

        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")

        self.demo_results["visualizations"] = visualization_paths
        return visualization_paths

    def _plot_training_curves(self, viz_dir: Path) -> str:
        """Plot training curves for all models"""
        plt.figure(figsize=(15, 10))

        training_results = self.demo_results["training_results"]

        # Plot training and validation loss
        plt.subplot(2, 2, 1)
        for model_name, results in training_results.items():
            history = results["training_result"]["training_history"]
            epochs = range(1, len(history["train_loss"]) + 1)
            plt.plot(
                epochs,
                history["train_loss"],
                label=f"{model_name} - Train",
                linestyle="-",
            )
            plt.plot(
                epochs, history["val_loss"], label=f"{model_name} - Val", linestyle="--"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)

        # Plot learning rate
        plt.subplot(2, 2, 2)
        for model_name, results in training_results.items():
            history = results["training_result"]["training_history"]
            epochs = range(1, len(history["learning_rate"]) + 1)
            plt.plot(epochs, history["learning_rate"], label=f"{model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.legend()
        plt.yscale("log")
        plt.grid(True)

        # Plot epoch time
        plt.subplot(2, 2, 3)
        for model_name, results in training_results.items():
            history = results["training_result"]["training_history"]
            epochs = range(1, len(history["epoch_time"]) + 1)
            plt.plot(epochs, history["epoch_time"], label=f"{model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Time (seconds)")
        plt.title("Training Time per Epoch")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        save_path = viz_dir / "training_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return str(save_path)

    def _plot_model_comparison(self, viz_dir: Path) -> str:
        """Plot model comparison metrics"""
        plt.figure(figsize=(12, 8))

        evaluation_results = self.demo_results["evaluation_results"]
        model_comparison = evaluation_results.get("model_comparison", {})

        metrics = ["rmse", "mae", "mape", "directional_accuracy"]
        model_names = [
            name for name in evaluation_results.keys() if name != "model_comparison"
        ]

        # Create comparison bar plot
        x = np.arange(len(metrics))
        width = 0.25

        for i, model_name in enumerate(model_names):
            values = []
            for metric in metrics:
                if metric in model_comparison:
                    values.append(model_comparison[metric]["values"].get(model_name, 0))
                else:
                    values.append(0)
            plt.bar(x + i * width, values, width, label=model_name)

        plt.xlabel("Metrics")
        plt.ylabel("Values")
        plt.title("Model Performance Comparison")
        plt.xticks(x + width, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = viz_dir / "model_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return str(save_path)

    def _plot_predictions(self, viz_dir: Path) -> str:
        """Plot prediction analysis"""
        if "predictions" not in self.demo_results:
            return ""

        predictions_data = self.demo_results["predictions"]["test_predictions"]
        predictions = predictions_data["predictions"].flatten()
        targets = predictions_data["targets"].flatten()

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Predictions vs Targets scatter plot
        axes[0, 0].scatter(targets, predictions, alpha=0.6)
        axes[0, 0].plot(
            [targets.min(), targets.max()], [targets.min(), targets.max()], "r--", lw=2
        )
        axes[0, 0].set_xlabel("Actual Values")
        axes[0, 0].set_ylabel("Predicted Values")
        axes[0, 0].set_title("Predictions vs Actual Values")
        axes[0, 0].grid(True, alpha=0.3)

        # Time series plot
        sample_size = min(100, len(predictions))
        indices = range(sample_size)
        axes[0, 1].plot(indices, targets[:sample_size], label="Actual", linewidth=2)
        axes[0, 1].plot(
            indices, predictions[:sample_size], label="Predicted", linewidth=2
        )
        axes[0, 1].set_xlabel("Time Steps")
        axes[0, 1].set_ylabel("Values")
        axes[0, 1].set_title("Time Series Comparison")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Error distribution
        errors = predictions - targets
        axes[1, 0].hist(errors, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        axes[1, 0].set_xlabel("Prediction Error")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Error Distribution")
        axes[1, 0].grid(True, alpha=0.3)

        # Error over time
        axes[1, 1].plot(range(sample_size), errors[:sample_size])
        axes[1, 1].set_xlabel("Time Steps")
        axes[1, 1].set_ylabel("Prediction Error")
        axes[1, 1].set_title("Error Over Time")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = viz_dir / "prediction_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return str(save_path)

    def _plot_feature_analysis(self, viz_dir: Path) -> str:
        """Plot feature analysis"""
        plt.figure(figsize=(12, 6))

        # Feature count by category (simplified)
        feature_names = self.feature_engineering.get_feature_importance_names()

        # Categorize features
        categories = {
            "Price": [
                f
                for f in feature_names
                if any(
                    x in f.lower() for x in ["open", "high", "low", "close", "price"]
                )
            ],
            "Volume": [f for f in feature_names if "volume" in f.lower()],
            "Technical": [
                f
                for f in feature_names
                if any(x in f.lower() for x in ["sma", "ema", "rsi", "bb", "macd"])
            ],
            "Time": [
                f
                for f in feature_names
                if any(x in f.lower() for x in ["day", "month", "hour"])
            ],
            "Returns": [f for f in feature_names if "return" in f.lower()],
            "Other": [],
        }

        # Assign uncategorized features
        categorized = set()
        for cat_features in categories.values():
            categorized.update(cat_features)
        categories["Other"] = [f for f in feature_names if f not in categorized]

        # Plot feature count by category
        cat_names = list(categories.keys())
        cat_counts = [len(categories[cat]) for cat in cat_names]

        plt.bar(cat_names, cat_counts, color="lightblue", edgecolor="black")
        plt.xlabel("Feature Categories")
        plt.ylabel("Number of Features")
        plt.title(f"Feature Distribution by Category (Total: {len(feature_names)})")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = viz_dir / "feature_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return str(save_path)

    def save_demo_results(self, output_path: Optional[str] = None) -> str:
        """
        Save comprehensive demo results to file

        Args:
            output_path: Path to save results (auto-generated if None)

        Returns:
            str: Path to saved results file
        """
        if output_path is None:
            timestamp = CommonUtils.get_timestamp()
            output_path = f"ai_demo_results_{timestamp}.json"

        # Prepare results for serialization
        serializable_results = self._prepare_results_for_serialization()

        # Save results
        FileUtils.save_json(serializable_results, output_path)

        self.logger.info(f"✓ Demo results saved to: {output_path}")
        return output_path

    def _prepare_results_for_serialization(self) -> Dict[str, Any]:
        """Prepare results for JSON serialization"""
        results = {}

        for key, value in self.demo_results.items():
            if key == "predictions":
                # Convert numpy arrays to lists
                if "test_predictions" in value:
                    test_preds = value["test_predictions"]
                    results[key] = {
                        "test_predictions": {
                            "predictions": (
                                test_preds["predictions"].tolist()
                                if hasattr(test_preds["predictions"], "tolist")
                                else test_preds["predictions"]
                            ),
                            "targets": (
                                test_preds["targets"].tolist()
                                if hasattr(test_preds["targets"], "tolist")
                                else test_preds["targets"]
                            ),
                            "analysis": test_preds["analysis"],
                        },
                        "future_predictions": value.get("future_predictions", {}),
                        "model_used": value.get("model_used", ""),
                    }
                else:
                    results[key] = value
            else:
                results[key] = value

        return results

    def run_full_demo(self) -> Dict[str, Any]:
        """
        Run the complete AI demonstration pipeline

        Returns:
            Dict containing all demo results
        """
        self.logger.info("\n" + "🚀" * 20)
        self.logger.info("STARTING COMPREHENSIVE AI DEMO")
        self.logger.info("🚀" * 20)

        demo_start_time = time.time()

        try:
            # 1. Data Processing
            self.demonstrate_data_processing()

            # 2. Model Architectures
            self.demonstrate_model_architectures()

            # 3. Training (limited models for demo)
            self.demonstrate_training(["transformer"])  # Train only one model for demo

            # 4. Evaluation
            self.demonstrate_evaluation(["transformer"])

            # 5. Predictions
            self.demonstrate_predictions("transformer")

            # 6. Visualizations
            self.create_visualizations()

            # Calculate total demo time
            demo_time = time.time() - demo_start_time
            self.demo_results["demo_metadata"] = {
                "total_time": demo_time,
                "completion_status": "success",
                "timestamp": CommonUtils.get_readable_timestamp(),
            }

            # Save results
            results_path = self.save_demo_results()

            self.logger.info("\n" + "✅" * 20)
            self.logger.info("DEMO COMPLETED SUCCESSFULLY!")
            self.logger.info("✅" * 20)
            self.logger.info(f"Total time: {CommonUtils.format_duration(demo_time)}")
            self.logger.info(f"Results saved to: {results_path}")

            return self.demo_results

        except Exception as e:
            self.logger.error(f"Demo failed: {str(e)}")
            raise


def main():
    """Main function to run AI demo"""
    parser = argparse.ArgumentParser(description="AI Prediction System Demo")
    parser.add_argument(
        "--config",
        type=str,
        choices=["development", "production", "lightweight"],
        default="development",
        help="Configuration type to use",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--data-path", type=str, help="Path to data file (optional)")

    args = parser.parse_args()

    # Create configuration
    if args.config == "development":
        config = create_development_config()
    elif args.config == "production":
        config = create_production_config()
    elif args.config == "lightweight":
        config = create_lightweight_config()
    else:
        config = create_development_config()

    # Override data path if provided
    if args.data_path:
        config.data.data_file = args.data_path

    # Run demo
    demo = AIDemo(config=config, verbose=args.verbose)
    results = demo.run_full_demo()

    return results


if __name__ == "__main__":
    main()
