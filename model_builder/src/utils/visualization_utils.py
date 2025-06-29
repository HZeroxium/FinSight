# utils/visualization_utils.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import seaborn as sns

from ..common.logger.logger_factory import LoggerFactory
from . import FileUtils


class VisualizationUtils:
    """Utility class for creating visualizations for AI prediction results"""

    _logger = LoggerFactory.get_logger(__name__)

    @staticmethod
    def setup_plot_style() -> None:
        """Setup matplotlib style for consistent plotting"""
        plt.style.use(
            "seaborn-v0_8"
            if hasattr(plt.style, "available") and "seaborn-v0_8" in plt.style.available
            else "default"
        )
        sns.set_palette("husl")

    @staticmethod
    def plot_training_curves(
        training_results: Dict[str, Any], save_path: Optional[Path] = None
    ) -> str:
        """
        Plot training curves for all models

        Args:
            training_results: Dictionary containing training results for each model
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        VisualizationUtils.setup_plot_style()
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot training and validation loss
        for model_name, results in training_results.items():
            history = results["training_result"]["training_history"]
            epochs = range(1, len(history["train_loss"]) + 1)

            axes[0, 0].plot(
                epochs,
                history["train_loss"],
                label=f"{model_name} - Train",
                linestyle="-",
            )
            axes[0, 0].plot(
                epochs, history["val_loss"], label=f"{model_name} - Val", linestyle="--"
            )

        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot learning rate
        for model_name, results in training_results.items():
            history = results["training_result"]["training_history"]
            epochs = range(1, len(history["learning_rate"]) + 1)
            axes[0, 1].plot(epochs, history["learning_rate"], label=f"{model_name}")

        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Learning Rate")
        axes[0, 1].set_title("Learning Rate Schedule")
        axes[0, 1].legend()
        axes[0, 1].set_yscale("log")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot epoch time
        for model_name, results in training_results.items():
            history = results["training_result"]["training_history"]
            epochs = range(1, len(history["epoch_time"]) + 1)
            axes[1, 0].plot(epochs, history["epoch_time"], label=f"{model_name}")

        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Time (seconds)")
        axes[1, 0].set_title("Training Time per Epoch")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot validation metrics
        for model_name, results in training_results.items():
            history = results["training_result"]["training_history"]
            if "val_metrics" in history and len(history["val_metrics"]) > 0:
                epochs = range(1, len(history["val_metrics"]) + 1)
                rmse_values = [
                    metrics.get("rmse", 0) for metrics in history["val_metrics"]
                ]
                axes[1, 1].plot(epochs, rmse_values, label=f"{model_name} RMSE")

        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("RMSE")
        axes[1, 1].set_title("Validation RMSE")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = Path("training_curves.png")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        VisualizationUtils._logger.info(f"Training curves saved to {save_path}")
        return str(save_path)

    @staticmethod
    def plot_model_comparison(
        evaluation_results: Dict[str, Any], save_path: Optional[Path] = None
    ) -> str:
        """
        Plot model comparison metrics

        Args:
            evaluation_results: Dictionary containing evaluation results
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        VisualizationUtils.setup_plot_style()
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        model_comparison = evaluation_results.get("model_comparison", {})
        metrics = ["rmse", "mae", "mape", "directional_accuracy"]
        model_names = [
            name for name in evaluation_results.keys() if name != "model_comparison"
        ]

        # Bar plot comparison
        x = np.arange(len(metrics))
        width = 0.25

        for i, model_name in enumerate(model_names):
            values = []
            for metric in metrics:
                if metric in model_comparison:
                    values.append(model_comparison[metric]["values"].get(model_name, 0))
                else:
                    values.append(0)
            axes[0, 0].bar(x + i * width, values, width, label=model_name)

        axes[0, 0].set_xlabel("Metrics")
        axes[0, 0].set_ylabel("Values")
        axes[0, 0].set_title("Model Performance Comparison")
        axes[0, 0].set_xticks(x + width)
        axes[0, 0].set_xticklabels(metrics)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Individual metric plots
        for idx, metric in enumerate(["rmse", "mae", "directional_accuracy"]):
            if idx < 3:
                row, col = (0, 1) if idx == 0 else (1, idx - 1)
                if metric in model_comparison:
                    values = model_comparison[metric]["values"]
                    axes[row, col].bar(values.keys(), values.values())
                    axes[row, col].set_title(f"{metric.upper()}")
                    axes[row, col].set_ylabel("Value")
                    axes[row, col].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = Path("model_comparison.png")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        VisualizationUtils._logger.info(f"Model comparison saved to {save_path}")
        return str(save_path)

    @staticmethod
    def plot_predictions(
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: Optional[Path] = None,
        sample_size: int = 100,
    ) -> str:
        """
        Plot prediction analysis

        Args:
            predictions: Model predictions
            targets: True targets
            save_path: Optional path to save the plot
            sample_size: Number of samples to plot in time series

        Returns:
            str: Path to saved plot
        """
        VisualizationUtils.setup_plot_style()
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()

        # Predictions vs Targets scatter plot
        axes[0, 0].scatter(targets_flat, predictions_flat, alpha=0.6)
        axes[0, 0].plot(
            [targets_flat.min(), targets_flat.max()],
            [targets_flat.min(), targets_flat.max()],
            "r--",
            lw=2,
        )
        axes[0, 0].set_xlabel("Actual Values")
        axes[0, 0].set_ylabel("Predicted Values")
        axes[0, 0].set_title("Predictions vs Actual Values")
        axes[0, 0].grid(True, alpha=0.3)

        # Time series plot
        sample_size = min(sample_size, len(predictions_flat))
        indices = range(sample_size)
        axes[0, 1].plot(
            indices, targets_flat[:sample_size], label="Actual", linewidth=2
        )
        axes[0, 1].plot(
            indices, predictions_flat[:sample_size], label="Predicted", linewidth=2
        )
        axes[0, 1].set_xlabel("Time Steps")
        axes[0, 1].set_ylabel("Values")
        axes[0, 1].set_title("Time Series Comparison")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Error distribution
        errors = predictions_flat - targets_flat
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

        if save_path is None:
            save_path = Path("prediction_analysis.png")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        VisualizationUtils._logger.info(f"Prediction analysis saved to {save_path}")
        return str(save_path)

    @staticmethod
    def plot_feature_analysis(
        feature_names: List[str], save_path: Optional[Path] = None
    ) -> str:
        """
        Plot feature analysis

        Args:
            feature_names: List of feature names
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        VisualizationUtils.setup_plot_style()
        plt.figure(figsize=(12, 6))

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

        if save_path is None:
            save_path = Path("feature_analysis.png")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        VisualizationUtils._logger.info(f"Feature analysis saved to {save_path}")
        return str(save_path)

    @staticmethod
    def create_comprehensive_visualizations(
        training_results: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        predictions_data: Dict[str, Any],
        feature_names: List[str],
        output_dir: Path,
    ) -> Dict[str, str]:
        """
        Create all visualizations in one call

        Args:
            training_results: Training results
            evaluation_results: Evaluation results
            predictions_data: Predictions data
            feature_names: Feature names
            output_dir: Output directory

        Returns:
            Dict mapping visualization type to file path
        """
        FileUtils.ensure_dir(output_dir)
        visualization_paths = {}

        try:
            # Training curves
            if training_results:
                viz_path = VisualizationUtils.plot_training_curves(
                    training_results, output_dir / "training_curves.png"
                )
                visualization_paths["training_curves"] = viz_path

            # Model comparison
            if evaluation_results:
                viz_path = VisualizationUtils.plot_model_comparison(
                    evaluation_results, output_dir / "model_comparison.png"
                )
                visualization_paths["model_comparison"] = viz_path

            # Predictions
            if "test_predictions" in predictions_data:
                test_preds = predictions_data["test_predictions"]
                viz_path = VisualizationUtils.plot_predictions(
                    test_preds["predictions"],
                    test_preds["targets"],
                    output_dir / "prediction_analysis.png",
                )
                visualization_paths["predictions"] = viz_path

            # Feature analysis
            if feature_names:
                viz_path = VisualizationUtils.plot_feature_analysis(
                    feature_names, output_dir / "feature_analysis.png"
                )
                visualization_paths["feature_analysis"] = viz_path

        except Exception as e:
            VisualizationUtils._logger.error(f"Error creating visualizations: {str(e)}")

        return visualization_paths
