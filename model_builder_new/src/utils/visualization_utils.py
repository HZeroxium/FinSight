# utils/visualization_utils.py

"""
Visualization utilities for time series models and backtesting results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set style
plt.style.use("default")
sns.set_palette("husl")


class VisualizationUtils:
    """Utilities for creating visualizations of model performance and data"""

    @staticmethod
    def plot_price_predictions(
        data: pd.DataFrame,
        predictions: List[float],
        actual_column: str = "close",
        title: str = "Price Predictions vs Actual",
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot actual vs predicted prices"""

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot actual prices
        ax.plot(data.index, data[actual_column], label="Actual", linewidth=2)

        # Plot predictions (assume they start from the last actual point)
        pred_start_idx = len(data) - len(predictions)
        pred_indices = range(pred_start_idx, len(data))
        ax.plot(
            pred_indices, predictions, label="Predicted", linewidth=2, linestyle="--"
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    @staticmethod
    def plot_backtest_results(
        backtest_results: Dict[str, Any],
        title: str = "Backtest Results",
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot comprehensive backtest results"""

        if "portfolio_values" not in backtest_results:
            print("No portfolio values found in backtest results")
            return

        portfolio_values = backtest_results["portfolio_values"]
        timestamps = backtest_results.get("timestamps", range(len(portfolio_values)))

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Portfolio value over time
        ax1.plot(timestamps, portfolio_values, linewidth=2, color="green")
        ax1.set_title("Portfolio Value Over Time", fontweight="bold")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Portfolio Value")
        ax1.grid(True, alpha=0.3)

        # Daily returns distribution
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1] * 100
            ax2.hist(returns, bins=30, alpha=0.7, color="blue", edgecolor="black")
            ax2.set_title("Daily Returns Distribution", fontweight="bold")
            ax2.set_xlabel("Return (%)")
            ax2.set_ylabel("Frequency")
            ax2.axvline(
                x=np.mean(returns),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(returns):.2f}%",
            )
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak * 100
        ax3.fill_between(timestamps, drawdown, 0, alpha=0.3, color="red")
        ax3.plot(timestamps, drawdown, color="darkred", linewidth=1)
        ax3.set_title("Drawdown (%)", fontweight="bold")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Drawdown (%)")
        ax3.grid(True, alpha=0.3)

        # Performance metrics (text)
        metrics = backtest_results.get("metrics", {})
        metrics_text = []
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    metrics_text.append(f"{key}: {value:.4f}")
                else:
                    metrics_text.append(f"{key}: {value}")

        ax4.text(
            0.1,
            0.9,
            "Performance Metrics:",
            fontsize=12,
            fontweight="bold",
            transform=ax4.transAxes,
            verticalalignment="top",
        )

        y_pos = 0.8
        for metric in metrics_text[:10]:  # Show top 10 metrics
            ax4.text(
                0.1,
                y_pos,
                metric,
                fontsize=10,
                transform=ax4.transAxes,
                verticalalignment="top",
            )
            y_pos -= 0.08

        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis("off")

        plt.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    @staticmethod
    def plot_feature_importance(
        feature_names: List[str],
        importance_scores: List[float],
        title: str = "Feature Importance",
        top_n: int = 20,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot feature importance scores"""

        # Sort features by importance
        sorted_data = sorted(
            zip(feature_names, importance_scores), key=lambda x: abs(x[1]), reverse=True
        )

        # Take top N features
        top_features = sorted_data[:top_n]
        features, scores = zip(*top_features)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create horizontal bar chart
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, scores, alpha=0.8)

        # Color bars based on positive/negative
        for i, (bar, score) in enumerate(zip(bars, scores)):
            if score > 0:
                bar.set_color("green")
            else:
                bar.set_color("red")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Top to bottom
        ax.set_xlabel("Importance Score")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    @staticmethod
    def plot_model_comparison(
        model_results: Dict[str, Dict[str, Any]],
        metric: str = "rmse",
        title: str = "Model Comparison",
        save_path: Optional[Path] = None,
    ) -> None:
        """Compare multiple models on a specific metric"""

        model_names = []
        metric_values = []

        for model_name, results in model_results.items():
            if metric in results:
                model_names.append(model_name)
                metric_values.append(results[metric])

        if not model_names:
            print(f"No results found for metric: {metric}")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(
            model_names,
            metric_values,
            alpha=0.8,
            color="skyblue",
            edgecolor="navy",
            linewidth=1.2,
        )

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_ylabel(metric.upper())
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Rotate x-axis labels if needed
        if len(max(model_names, key=len)) > 10:
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    @staticmethod
    def plot_hyperparameter_heatmap(
        results_df: pd.DataFrame,
        param1: str,
        param2: str,
        metric: str = "rmse",
        title: str = "Hyperparameter Heatmap",
        save_path: Optional[Path] = None,
    ) -> None:
        """Create heatmap of hyperparameter results"""

        # Pivot data for heatmap
        heatmap_data = results_df.pivot(index=param1, columns=param2, values=metric)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".4f",
            cmap="viridis_r",
            ax=ax,
            cbar_kws={"label": metric.upper()},
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(param2.replace("_", " ").title())
        ax.set_ylabel(param1.replace("_", " ").title())

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    @staticmethod
    def plot_training_curves(
        training_history: Dict[str, List[float]],
        title: str = "Training Curves",
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot training and validation curves"""

        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = range(1, len(list(training_history.values())[0]) + 1)

        for metric_name, values in training_history.items():
            ax.plot(epochs, values, label=metric_name, linewidth=2, marker="o")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss/Metric")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    @staticmethod
    def plot_correlation_matrix(
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        title: str = "Feature Correlation Matrix",
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot correlation matrix of features"""

        if features:
            plot_data = data[features]
        else:
            # Use only numeric columns
            plot_data = data.select_dtypes(include=[np.number])

        corr_matrix = plot_data.corr()

        fig, ax = plt.subplots(figsize=(12, 10))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            center=0,
            cmap="RdBu_r",
            ax=ax,
            square=True,
            cbar_kws={"shrink": 0.8},
        )

        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    @staticmethod
    def plot_training_metrics(
        training_results: Dict[str, Any],
        title: str = "Training Metrics",
        save_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """Plot training metrics like loss curves"""

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(title, fontsize=16, fontweight="bold")

            # Training loss
            train_loss = training_results.get("training_loss", 0)
            val_loss = training_results.get("validation_loss", 0)

            if train_loss > 0:
                axes[0, 0].bar(["Training Loss"], [train_loss])
                axes[0, 0].set_title("Training Loss")
                axes[0, 0].set_ylabel("Loss")

            # Validation loss
            if val_loss > 0:
                axes[0, 1].bar(["Validation Loss"], [val_loss])
                axes[0, 1].set_title("Validation Loss")
                axes[0, 1].set_ylabel("Loss")

            # Metrics comparison (MAE, RMSE if available)
            metrics = {}
            for key in ["train_mae", "val_mae", "train_rmse", "val_rmse"]:
                if key in training_results:
                    metrics[key] = training_results[key]

            if metrics:
                metric_names = list(metrics.keys())
                metric_values = list(metrics.values())
                axes[1, 0].bar(metric_names, metric_values)
                axes[1, 0].set_title("Training Metrics")
                axes[1, 0].tick_params(axis="x", rotation=45)
            else:
                axes[1, 0].text(
                    0.5, 0.5, "No detailed metrics available", ha="center", va="center"
                )
                axes[1, 0].set_title("Training Metrics")

            # Summary text
            summary_text = (
                f"Training completed\nSuccess: {training_results.get('success', False)}"
            )
            if "epochs_trained" in training_results:
                summary_text += f"\nEpochs: {training_results['epochs_trained']}"

            axes[1, 1].text(
                0.1,
                0.5,
                summary_text,
                fontsize=12,
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("Training Summary")
            axes[1, 1].axis("off")

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                return save_path
            else:
                plt.show()

        except Exception as e:
            print(f"Error plotting training metrics: {e}")

        return None

    @staticmethod
    def plot_prediction_analysis(
        prediction_results,  # Can be Dict[str, Any] or numpy array
        title: str = "Prediction Analysis",
        save_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """Plot prediction analysis with accuracy metrics"""

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(title, fontsize=16, fontweight="bold")

            # Handle different input formats
            if isinstance(prediction_results, (list, np.ndarray)):
                # Direct predictions array
                predictions = np.array(prediction_results)
                actuals = []
            elif isinstance(prediction_results, dict):
                predictions = prediction_results.get("predictions", [])
                actuals = prediction_results.get("actuals", [])
            else:
                predictions = []
                actuals = []

            # Predictions line plot
            if len(predictions) > 0:
                x_range = range(len(predictions))
                axes[0, 0].plot(x_range, predictions, label="Predictions", marker="o")
                if len(actuals) > 0:
                    min_len = min(len(predictions), len(actuals))
                    axes[0, 0].plot(
                        x_range[:min_len],
                        actuals[:min_len],
                        label="Actuals",
                        marker="s",
                    )
                axes[0, 0].set_title("Predictions vs Actuals")
                axes[0, 0].set_xlabel("Sample")
                axes[0, 0].set_ylabel("Value")
                axes[0, 0].legend()

                # Scatter plot (only if we have actuals)
                if len(actuals) > 0:
                    min_len = min(len(predictions), len(actuals))
                    axes[0, 1].scatter(
                        actuals[:min_len], predictions[:min_len], alpha=0.6
                    )
                    axes[0, 1].plot(
                        [min(actuals[:min_len]), max(actuals[:min_len])],
                        [min(actuals[:min_len]), max(actuals[:min_len])],
                        "r--",
                    )
                    axes[0, 1].set_title("Prediction Scatter Plot")
                    axes[0, 1].set_xlabel("Actual")
                    axes[0, 1].set_ylabel("Predicted")
                else:
                    # Just plot prediction distribution
                    axes[0, 1].hist(predictions, bins=20, alpha=0.7)
                    axes[0, 1].set_title("Prediction Distribution")
                    axes[0, 1].set_xlabel("Value")
                    axes[0, 1].set_ylabel("Frequency")

            # Accuracy metrics
            if isinstance(prediction_results, dict):
                accuracy_metrics = prediction_results.get("accuracy_metrics", {})
                if accuracy_metrics:
                    metric_names = list(accuracy_metrics.keys())
                    metric_values = list(accuracy_metrics.values())
                    axes[1, 0].bar(metric_names, metric_values)
                    axes[1, 0].set_title("Accuracy Metrics")
                    axes[1, 0].tick_params(axis="x", rotation=45)
                else:
                    axes[1, 0].text(
                        0.5, 0.5, "No metrics available", ha="center", va="center"
                    )
                    axes[1, 0].set_title("Accuracy Metrics")
            else:
                axes[1, 0].text(
                    0.5, 0.5, "No metrics available", ha="center", va="center"
                )
                axes[1, 0].set_title("Accuracy Metrics")

            # Summary info
            if isinstance(prediction_results, dict):
                current_price = prediction_results.get("current_price", 0)
                next_prediction = prediction_results.get("next_prediction", 0)
                direction = prediction_results.get("predicted_direction", "Unknown")

                summary_text = (
                    f"Current Price: ${current_price:.2f}\n"
                    f"Next Prediction: ${next_prediction:.2f}\n"
                    f"Direction: {direction}"
                )
            else:
                # For array predictions, use first prediction
                if len(predictions) > 0:
                    summary_text = (
                        f"Prediction Count: {len(predictions)}\n"
                        f"First Prediction: {predictions[0]:.2f}\n"
                        f"Mean Prediction: {np.mean(predictions):.2f}"
                    )
                else:
                    summary_text = "No predictions available"

            axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, va="center")
            axes[1, 1].set_title("Summary Information")
            axes[1, 1].axis("off")

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                return save_path
            else:
                plt.show()

        except Exception as e:
            print(f"Error plotting prediction analysis: {e}")

        return None

    @staticmethod
    def create_experiment_dashboard(
        experiment_results: Dict[str, Any], save_dir: Optional[Path] = None
    ) -> None:
        """Create comprehensive experiment dashboard"""

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        # Extract data
        models = experiment_results.get("models", {})
        predictions = experiment_results.get("predictions", {})
        backtest_results = experiment_results.get("backtest", {})

        # Model comparison
        if models:
            model_metrics = {}
            for model_name, model_data in models.items():
                if "metrics" in model_data:
                    model_metrics[model_name] = model_data["metrics"]

            if model_metrics:
                VisualizationUtils.plot_model_comparison(
                    model_metrics,
                    save_path=save_dir / "model_comparison.png" if save_dir else None,
                )

        # Prediction analysis
        if predictions is not None and len(predictions) > 0:
            for model_name, pred_data in predictions.items():
                if "actual" in pred_data and "predicted" in pred_data:
                    VisualizationUtils.plot_prediction_analysis(
                        pred_data["actual"],
                        pred_data["predicted"],
                        timestamps=pred_data.get("timestamps"),
                        title=f"Prediction Analysis - {model_name}",
                        save_path=(
                            save_dir / f"prediction_analysis_{model_name}.png"
                            if save_dir
                            else None
                        ),
                    )

        # Backtest results
        if backtest_results:
            VisualizationUtils.plot_backtest_results(
                backtest_results,
                save_path=save_dir / "backtest_results.png" if save_dir else None,
            )

    @staticmethod
    def save_experiment_json(
        experiment_results: Dict[str, Any], save_path: Path
    ) -> None:
        """Save experiment results as JSON"""

        # Convert numpy arrays and other non-serializable objects
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, pd.Timestamp):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj

        converted_results = convert_for_json(experiment_results)

        import json

        with open(save_path, "w") as f:
            json.dump(converted_results, f, indent=2, default=str)
