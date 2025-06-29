"""
Visualizations for model performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .base import BaseVisualizer, set_finance_style


class ModelPerformanceVisualizer(BaseVisualizer):
    """Visualizer for model training and performance metrics"""

    def plot_training_curves(
        self, training_results: Dict[str, Any], save_path: Optional[Path] = None
    ) -> str:
        """
        Enhanced plot for training curves with better layout and styling

        Args:
            training_results: Dictionary containing training results for each model
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        # Set financial visualization style
        set_finance_style()

        # Create figure with improved layout - more space for detailed metrics
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))

        # Color palette for models
        model_names = list(training_results.keys())
        colors = sns.color_palette("husl", len(model_names))

        # 1. Plot training and validation loss
        for i, model_name in enumerate(model_names):
            results = training_results[model_name]
            history = results["training_result"]["training_history"]
            epochs = range(1, len(history["train_loss"]) + 1)

            axes[0, 0].plot(
                epochs,
                history["train_loss"],
                label=f"{model_name} - Train",
                color=colors[i],
                linestyle="-",
                marker="o",
                markersize=4,
                alpha=0.8,
            )
            axes[0, 0].plot(
                epochs,
                history["val_loss"],
                label=f"{model_name} - Val",
                color=colors[i],
                linestyle="--",
                marker="s",
                markersize=4,
                alpha=0.8,
            )

        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].legend(loc="upper right")

        # 2. Plot learning rate
        has_lr_data = False
        for i, model_name in enumerate(model_names):
            results = training_results[model_name]
            history = results["training_result"]["training_history"]
            if "learning_rate" in history and len(history["learning_rate"]) > 0:
                epochs = range(1, len(history["learning_rate"]) + 1)
                axes[0, 1].plot(
                    epochs,
                    history["learning_rate"],
                    label=f"{model_name}",
                    color=colors[i],
                    marker="o",
                    markersize=4,
                )
                has_lr_data = True

        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Learning Rate")
        axes[0, 1].set_title("Learning Rate Schedule")
        if has_lr_data:
            axes[0, 1].legend(loc="upper right")
            axes[0, 1].set_yscale("log")
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "No learning rate data available",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
            )

        # 3. Plot epoch time
        has_time_data = False
        for i, model_name in enumerate(model_names):
            results = training_results[model_name]
            history = results["training_result"]["training_history"]
            if "epoch_time" in history and len(history["epoch_time"]) > 0:
                epochs = range(1, len(history["epoch_time"]) + 1)
                axes[1, 0].plot(
                    epochs,
                    history["epoch_time"],
                    label=f"{model_name}",
                    color=colors[i],
                    marker="o",
                    markersize=4,
                )
                has_time_data = True

        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Time (seconds)")
        axes[1, 0].set_title("Training Time per Epoch")
        if has_time_data:
            axes[1, 0].legend()
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "No timing data available",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )

        # 4. Plot validation metrics (RMSE)
        has_rmse_data = False
        for i, model_name in enumerate(model_names):
            results = training_results[model_name]
            history = results["training_result"]["training_history"]
            if "val_metrics" in history and len(history["val_metrics"]) > 0:
                epochs = range(1, len(history["val_metrics"]) + 1)
                rmse_values = [
                    metrics.get("rmse", 0) for metrics in history["val_metrics"]
                ]
                if any(rmse_values):  # Check if there are any non-zero values
                    axes[1, 1].plot(
                        epochs,
                        rmse_values,
                        label=f"{model_name} RMSE",
                        color=colors[i],
                        marker="o",
                        markersize=4,
                    )
                    has_rmse_data = True

        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("RMSE")
        axes[1, 1].set_title("Validation RMSE")
        if has_rmse_data:
            axes[1, 1].legend()
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No RMSE data available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )

        # 5. Plot validation metrics (Directional Accuracy)
        has_da_data = False
        for i, model_name in enumerate(model_names):
            results = training_results[model_name]
            history = results["training_result"]["training_history"]
            if "val_metrics" in history and len(history["val_metrics"]) > 0:
                epochs = range(1, len(history["val_metrics"]) + 1)
                da_values = [
                    metrics.get("directional_accuracy", 0)
                    for metrics in history["val_metrics"]
                ]
                if any(da_values):  # Check if there are any non-zero values
                    axes[2, 0].plot(
                        epochs,
                        da_values,
                        label=f"{model_name}",
                        color=colors[i],
                        marker="o",
                        markersize=4,
                    )
                    has_da_data = True

        axes[2, 0].set_xlabel("Epoch")
        axes[2, 0].set_ylabel("Directional Accuracy (%)")
        axes[2, 0].set_title("Validation Directional Accuracy")
        if has_da_data:
            axes[2, 0].legend()
        else:
            axes[2, 0].text(
                0.5,
                0.5,
                "No directional accuracy data available",
                ha="center",
                va="center",
                transform=axes[2, 0].transAxes,
            )

        # 6. Plot validation metrics (MAE)
        has_mae_data = False
        for i, model_name in enumerate(model_names):
            results = training_results[model_name]
            history = results["training_result"]["training_history"]
            if "val_metrics" in history and len(history["val_metrics"]) > 0:
                epochs = range(1, len(history["val_metrics"]) + 1)
                mae_values = [
                    metrics.get("mae", 0) for metrics in history["val_metrics"]
                ]
                if any(mae_values):  # Check if there are any non-zero values
                    axes[2, 1].plot(
                        epochs,
                        mae_values,
                        label=f"{model_name}",
                        color=colors[i],
                        marker="o",
                        markersize=4,
                    )
                    has_mae_data = True

        axes[2, 1].set_xlabel("Epoch")
        axes[2, 1].set_ylabel("MAE")
        axes[2, 1].set_title("Validation MAE")
        if has_mae_data:
            axes[2, 1].legend()
        else:
            axes[2, 1].text(
                0.5,
                0.5,
                "No MAE data available",
                ha="center",
                va="center",
                transform=axes[2, 1].transAxes,
            )

        # Add overall title
        fig.suptitle("Model Training Performance", fontsize=16, y=0.995)
        plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.3)

        # Save and return
        if save_path is None:
            save_path = Path("training_curves.png")

        return self.save_figure(fig, save_path)

    def plot_model_comparison(
        self, evaluation_results: Dict[str, Any], save_path: Optional[Path] = None
    ) -> str:
        """
        Enhanced plot for model comparison with better styling

        Args:
            evaluation_results: Dictionary containing evaluation results
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        set_finance_style()

        # Prepare metrics comparison data
        model_comparison = evaluation_results.get("model_comparison", {})
        metrics = [
            "rmse",
            "mae",
            "mape",
            "smape",
            "r2",
            "directional_accuracy",
            "true_max_drawdown",
            "pred_max_drawdown",
        ]
        available_metrics = [m for m in metrics if m in model_comparison]

        # If no comparison data is available, create a figure with individual metrics
        if not model_comparison:
            model_names = [
                name for name in evaluation_results.keys() if name != "model_comparison"
            ]

            if not model_names:
                # No data available
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(
                    0.5,
                    0.5,
                    "No evaluation data available",
                    ha="center",
                    va="center",
                    fontsize=14,
                )

                if save_path is None:
                    save_path = Path("model_comparison.png")
                return self.save_figure(fig, save_path)

            # Extract available metrics from the first model
            available_metrics = []
            for k in evaluation_results[model_names[0]].keys():
                if k not in ["test_loss", "prediction_time"]:
                    available_metrics.append(k)

            # Create a comparison data structure
            model_comparison = {metric: {"values": {}} for metric in available_metrics}
            for model_name in model_names:
                for metric in available_metrics:
                    if metric in evaluation_results[model_name]:
                        model_comparison[metric]["values"][model_name] = (
                            evaluation_results[model_name].get(metric, 0)
                        )

        # Updated metrics for improved visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Bar plot comparison for error metrics (RMSE, MAE)
        error_metrics = [m for m in ["rmse", "mae"] if m in model_comparison]
        if error_metrics:
            ax = axes[0, 0]
            x_labels = []
            x_pos = []
            bar_width = 0.25

            for i, metric in enumerate(error_metrics):
                metric_values = model_comparison[metric]["values"]
                model_names = list(metric_values.keys())
                values = list(metric_values.values())

                x = np.arange(len(model_names))
                ax.bar(x + i * bar_width, values, bar_width, label=metric.upper())

                if not x_labels:
                    x_labels = model_names
                    x_pos = x + (len(error_metrics) - 1) * bar_width / 2

            ax.set_ylabel("Value (lower is better)")
            ax.set_title("Error Metrics Comparison")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
            ax.legend()

            # Add value labels on bars
            for i, metric in enumerate(error_metrics):
                metric_values = model_comparison[metric]["values"]
                model_names = list(metric_values.keys())
                values = list(metric_values.values())
                x = np.arange(len(model_names))

                for j, v in enumerate(values):
                    ax.text(
                        x[j] + i * bar_width,
                        v + 0.05,
                        f"{v:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

        # 2. Bar plot for accuracy metrics (Directional Accuracy, R2)
        accuracy_metrics = [
            m for m in ["directional_accuracy", "r2"] if m in model_comparison
        ]
        if accuracy_metrics:
            ax = axes[0, 1]
            x_labels = []
            x_pos = []
            bar_width = 0.25

            for i, metric in enumerate(accuracy_metrics):
                metric_values = model_comparison[metric]["values"]
                model_names = list(metric_values.keys())
                values = list(metric_values.values())

                x = np.arange(len(model_names))
                ax.bar(x + i * bar_width, values, bar_width, label=metric.upper())

                if not x_labels:
                    x_labels = model_names
                    x_pos = x + (len(accuracy_metrics) - 1) * bar_width / 2

            ax.set_ylabel("Value (higher is better)")
            ax.set_title("Accuracy Metrics Comparison")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
            ax.legend()

            # Add value labels on bars
            for i, metric in enumerate(accuracy_metrics):
                metric_values = model_comparison[metric]["values"]
                model_names = list(metric_values.keys())
                values = list(metric_values.values())
                x = np.arange(len(model_names))

                for j, v in enumerate(values):
                    ax.text(
                        x[j] + i * bar_width,
                        v + 0.5,
                        f"{v:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

        # 3. Bar plot for percentage error metrics (MAPE, SMAPE)
        pct_metrics = [m for m in ["mape", "smape"] if m in model_comparison]
        if pct_metrics:
            ax = axes[1, 0]
            x_labels = []
            x_pos = []
            bar_width = 0.25

            for i, metric in enumerate(pct_metrics):
                metric_values = model_comparison[metric]["values"]
                model_names = list(metric_values.keys())
                values = list(metric_values.values())

                x = np.arange(len(model_names))
                ax.bar(x + i * bar_width, values, bar_width, label=metric.upper())

                if not x_labels:
                    x_labels = model_names
                    x_pos = x + (len(pct_metrics) - 1) * bar_width / 2

            ax.set_ylabel("Percentage Error (lower is better)")
            ax.set_title("Percentage Error Metrics")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
            ax.legend()

            # Add value labels on bars
            for i, metric in enumerate(pct_metrics):
                metric_values = model_comparison[metric]["values"]
                model_names = list(metric_values.keys())
                values = list(metric_values.values())
                x = np.arange(len(model_names))

                for j, v in enumerate(values):
                    ax.text(
                        x[j] + i * bar_width,
                        v + 2,
                        f"{v:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

        # 4. Risk metrics (Drawdown, Volatility)
        risk_metrics = [
            m
            for m in ["true_max_drawdown", "pred_max_drawdown"]
            if m in model_comparison
        ]
        if risk_metrics:
            ax = axes[1, 1]
            x_labels = []
            x_pos = []
            bar_width = 0.25

            for i, metric in enumerate(risk_metrics):
                metric_values = model_comparison[metric]["values"]
                model_names = list(metric_values.keys())
                values = list(metric_values.values())

                x = np.arange(len(model_names))
                ax.bar(
                    x + i * bar_width,
                    values,
                    bar_width,
                    label=metric.replace("_", " ").title(),
                )

                if not x_labels:
                    x_labels = model_names
                    x_pos = x + (len(risk_metrics) - 1) * bar_width / 2

            ax.set_ylabel("Value")
            ax.set_title("Risk Metrics Comparison")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
            ax.legend()

            # Add value labels on bars
            for i, metric in enumerate(risk_metrics):
                metric_values = model_comparison[metric]["values"]
                model_names = list(metric_values.keys())
                values = list(metric_values.values())
                x = np.arange(len(model_names))

                for j, v in enumerate(values):
                    ax.text(
                        x[j] + i * bar_width,
                        v - 0.5,
                        f"{v:.2f}",
                        ha="center",
                        va="top",
                        fontsize=9,
                    )

        # Add overall title
        fig.suptitle("Model Performance Comparison", fontsize=16, y=0.995)
        plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.3)

        # Save and return
        if save_path is None:
            save_path = Path("model_comparison.png")

        return self.save_figure(fig, save_path)
