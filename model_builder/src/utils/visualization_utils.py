# utils/visualization_utils.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import seaborn as sns
import pandas as pd

from ..common.logger.logger_factory import LoggerFactory
from . import FileUtils
from ..utils.visualization import (
    ModelPerformanceVisualizer,
    PredictionVisualizer,
    FinancialVisualizer,
    ExplainabilityVisualizer,
    FeatureVisualizer,
    set_finance_style,
)


class VisualizationUtils:
    """Utility class for creating visualizations for AI prediction results"""

    _logger = LoggerFactory.get_logger(__name__)

    # Initialize singleton visualizer instances
    _performance_visualizer = None
    _prediction_visualizer = None
    _financial_visualizer = None
    _explainability_visualizer = None
    _feature_visualizer = None

    @classmethod
    def _get_performance_visualizer(cls):
        if cls._performance_visualizer is None:
            cls._performance_visualizer = ModelPerformanceVisualizer()
        return cls._performance_visualizer

    @classmethod
    def _get_prediction_visualizer(cls):
        if cls._prediction_visualizer is None:
            cls._prediction_visualizer = PredictionVisualizer()
        return cls._prediction_visualizer

    @classmethod
    def _get_financial_visualizer(cls):
        if cls._financial_visualizer is None:
            cls._financial_visualizer = FinancialVisualizer()
        return cls._financial_visualizer

    @classmethod
    def _get_explainability_visualizer(cls):
        if cls._explainability_visualizer is None:
            cls._explainability_visualizer = ExplainabilityVisualizer()
        return cls._explainability_visualizer

    @classmethod
    def _get_feature_visualizer(cls):
        if cls._feature_visualizer is None:
            cls._feature_visualizer = FeatureVisualizer()
        return cls._feature_visualizer

    @staticmethod
    def setup_plot_style() -> None:
        """Setup matplotlib style for consistent plotting"""
        set_finance_style()

    @staticmethod
    def plot_training_curves(
        training_results: Dict[str, Any], save_path: Optional[Path] = None
    ) -> str:
        """
        Plot training curves for all models using the specialized visualizer

        Args:
            training_results: Dictionary containing training results for each model
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        visualizer = VisualizationUtils._get_performance_visualizer()
        return visualizer.plot_training_curves(training_results, save_path)

    @staticmethod
    def plot_model_comparison(
        evaluation_results: Dict[str, Any], save_path: Optional[Path] = None
    ) -> str:
        """
        Plot model comparison metrics using the specialized visualizer

        Args:
            evaluation_results: Dictionary containing evaluation results
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        visualizer = VisualizationUtils._get_performance_visualizer()
        return visualizer.plot_model_comparison(evaluation_results, save_path)

    @staticmethod
    def plot_predictions(
        predictions: np.ndarray,
        targets: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
        sample_size: int = 100,
        model_name: str = "model",
    ) -> str:
        """
        Plot prediction analysis using the specialized visualizer

        Args:
            predictions: Model predictions
            targets: True targets
            timestamps: Optional timestamps for x-axis
            save_path: Optional path to save the plot
            sample_size: Number of samples to plot in time series
            model_name: Name of the model

        Returns:
            str: Path to saved plot
        """
        visualizer = VisualizationUtils._get_prediction_visualizer()
        return visualizer.plot_predictions(
            predictions=predictions,
            targets=targets,
            timestamps=timestamps,
            save_path=save_path,
            sample_size=sample_size,
            model_name=model_name,
        )

    @staticmethod
    def plot_forecast_comparison(
        predictions: Dict[str, np.ndarray],
        targets: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
        sample_size: int = 50,
    ) -> str:
        """
        Plot comparison of forecasts from multiple models

        Args:
            predictions: Dictionary of model predictions {model_name: predictions}
            targets: True target values
            timestamps: Optional timestamps for x-axis
            save_path: Optional path to save the plot
            sample_size: Number of samples to plot

        Returns:
            str: Path to saved plot
        """
        visualizer = VisualizationUtils._get_prediction_visualizer()
        return visualizer.plot_forecast_comparison(
            predictions=predictions,
            targets=targets,
            timestamps=timestamps,
            save_path=save_path,
            sample_size=sample_size,
        )

    @staticmethod
    def plot_trading_simulation(
        predictions: np.ndarray,
        targets: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        initial_capital: float = 10000.0,
        save_path: Optional[Path] = None,
    ) -> str:
        """
        Simulate trading based on predictions and visualize results

        Args:
            predictions: Model predictions
            targets: Actual values
            timestamps: Optional timestamps for x-axis
            initial_capital: Initial capital for trading simulation
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        visualizer = VisualizationUtils._get_prediction_visualizer()
        return visualizer.plot_trading_simulation(
            predictions=predictions,
            targets=targets,
            timestamps=timestamps,
            initial_capital=initial_capital,
            save_path=save_path,
        )

    @staticmethod
    def plot_feature_importance(
        feature_importance: np.ndarray,
        feature_names: List[str],
        title: str = "Feature Importance",
        save_path: Optional[Path] = None,
        top_n: int = 20,
    ) -> str:
        """
        Plot feature importance from model

        Args:
            feature_importance: Array of feature importance values
            feature_names: List of feature names
            title: Plot title
            save_path: Optional path to save the plot
            top_n: Number of top features to include

        Returns:
            str: Path to saved plot
        """
        visualizer = VisualizationUtils._get_explainability_visualizer()
        return visualizer.plot_feature_importance(
            feature_importance=feature_importance,
            feature_names=feature_names,
            title=title,
            save_path=save_path,
            top_n=top_n,
        )

    @staticmethod
    def plot_correlation_matrix(
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        method: str = "spearman",
        save_path: Optional[Path] = None,
        threshold: Optional[float] = None,
    ) -> str:
        """
        Plot correlation matrix of features

        Args:
            data: DataFrame containing features
            features: List of feature names to include (optional)
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            save_path: Optional path to save the plot
            threshold: Optional threshold to highlight correlations (absolute value)

        Returns:
            str: Path to saved plot
        """
        # Filter to only numeric columns to avoid string conversion errors
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            VisualizationUtils._logger.warning(
                "No numeric data available for correlation matrix"
            )
            # Create empty plot with message
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(
                0.5,
                0.5,
                "No numeric data available for correlation analysis",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title("Correlation Matrix")

            if save_path is None:
                save_path = Path("correlation_matrix.png")
            return VisualizationUtils._save_figure_safely(fig, save_path)

        visualizer = VisualizationUtils._get_feature_visualizer()
        return visualizer.plot_correlation_matrix(
            data=numeric_data,
            features=features,
            method=method,
            save_path=save_path,
            threshold=threshold,
        )

    @staticmethod
    def plot_feature_distributions(
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        n_features: int = 10,
        save_path: Optional[Path] = None,
    ) -> str:
        """
        Plot distributions of selected features

        Args:
            data: DataFrame containing features
            features: List of feature names to plot (optional)
            n_features: Number of features to plot if features not specified
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        # Filter to only numeric columns to avoid plotting issues
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            VisualizationUtils._logger.warning(
                "No numeric data available for feature distributions"
            )
            # Create empty plot with message
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(
                0.5,
                0.5,
                "No numeric data available for feature distribution analysis",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title("Feature Distributions")

            if save_path is None:
                save_path = Path("feature_distributions.png")
            return VisualizationUtils._save_figure_safely(fig, save_path)

        # Filter features to only include numeric ones
        if features:
            features = [f for f in features if f in numeric_data.columns]

        visualizer = VisualizationUtils._get_feature_visualizer()
        return visualizer.plot_feature_distributions(
            data=numeric_data,
            features=features,
            n_features=n_features,
            save_path=save_path,
        )

    @staticmethod
    def _save_figure_safely(fig, save_path: Path) -> str:
        """Safely save a figure and return the path"""
        try:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            VisualizationUtils._logger.info(f"Figure saved to {save_path}")
            return str(save_path)
        except Exception as e:
            VisualizationUtils._logger.error(
                f"Error saving figure to {save_path}: {str(e)}"
            )
            plt.close(fig)
            return str(save_path)

    @staticmethod
    def plot_price_series(
        data: pd.DataFrame,
        price_cols: List[str] = ["Close"],
        volume_col: Optional[str] = "Volume",
        date_col: Optional[str] = "Date",
        title: str = "Price Series Analysis",
        save_path: Optional[Path] = None,
        show_ma: bool = True,
    ) -> str:
        """
        Plot price series with volume and optional moving averages

        Args:
            data: DataFrame with financial data
            price_cols: Column names for price data
            volume_col: Column name for volume data
            date_col: Column name for date data
            title: Plot title
            save_path: Optional path to save the plot
            show_ma: Whether to show moving averages

        Returns:
            str: Path to saved plot
        """
        visualizer = VisualizationUtils._get_financial_visualizer()
        return visualizer.plot_price_series(
            data=data,
            price_cols=price_cols,
            volume_col=volume_col,
            date_col=date_col,
            title=title,
            save_path=save_path,
            show_ma=show_ma,
        )

    @staticmethod
    def plot_candlestick_chart(
        data: pd.DataFrame,
        date_col: str = "Date",
        open_col: str = "Open",
        high_col: str = "High",
        low_col: str = "Low",
        close_col: str = "Close",
        volume_col: Optional[str] = "Volume",
        overlay_predictions: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
    ) -> str:
        """
        Plot candlestick chart with optional predicted values overlay

        Args:
            data: DataFrame with OHLCV data
            date_col: Column name for date
            open_col: Column name for open price
            high_col: Column name for high price
            low_col: Column name for low price
            close_col: Column name for close price
            volume_col: Column name for volume
            overlay_predictions: Optional predicted values to overlay
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        visualizer = VisualizationUtils._get_financial_visualizer()
        return visualizer.plot_candlestick_chart(
            data=data,
            date_col=date_col,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            volume_col=volume_col,
            overlay_predictions=overlay_predictions,
            save_path=save_path,
        )

    @staticmethod
    def plot_attention_weights(
        attention_weights: np.ndarray,
        sequence_length: int,
        feature_names: Optional[List[str]] = None,
        timestamps: Optional[List] = None,
        save_path: Optional[Path] = None,
    ) -> str:
        """
        Plot attention weights from transformer models

        Args:
            attention_weights: Attention weight matrix
            sequence_length: Length of the input sequence
            feature_names: Optional list of feature names
            timestamps: Optional list of timestamps for x-axis labels
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        visualizer = VisualizationUtils._get_explainability_visualizer()
        return visualizer.plot_attention_weights(
            attention_weights=attention_weights,
            sequence_length=sequence_length,
            feature_names=feature_names,
            timestamps=timestamps,
            save_path=save_path,
        )

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
        raw_data: Optional[pd.DataFrame] = None,
        processed_data: Optional[pd.DataFrame] = None,
        show_advanced: bool = False,
    ) -> Dict[str, str]:
        """
        Create all visualizations in one call

        Args:
            training_results: Training results
            evaluation_results: Evaluation results
            predictions_data: Predictions data
            feature_names: Feature names
            output_dir: Output directory
            raw_data: Optional raw data DataFrame for additional visualizations
            processed_data: Optional processed data DataFrame for additional visualizations
            show_advanced: Whether to include advanced visualizations

        Returns:
            Dict mapping visualization type to file path
        """
        FileUtils.ensure_dir(output_dir)
        visualization_paths = {}

        try:
            # Basic visualizations
            # -------------------
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

            # Predictions - handle both old and new structure
            predictions_array = None
            targets_array = None

            if "predictions" in predictions_data and "targets" in predictions_data:
                # New consistent structure
                predictions_array = predictions_data["predictions"]
                targets_array = predictions_data["targets"]
            elif "test_predictions" in predictions_data:
                # Old structure for backward compatibility
                test_preds = predictions_data["test_predictions"]
                predictions_array = test_preds["predictions"]
                targets_array = test_preds["targets"]

            if predictions_array is not None and targets_array is not None:
                viz_path = VisualizationUtils.plot_predictions(
                    predictions_array,
                    targets_array,
                    save_path=output_dir / "prediction_analysis.png",
                )
                visualization_paths["predictions"] = viz_path

                # Trading simulation based on predictions
                viz_path = VisualizationUtils.plot_trading_simulation(
                    predictions_array,
                    targets_array,
                    save_path=output_dir / "trading_simulation.png",
                )
                visualization_paths["trading_simulation"] = viz_path

                # Multi-model forecast comparison if available
                if "all_model_predictions" in predictions_data:
                    all_preds = predictions_data["all_model_predictions"]
                    viz_path = VisualizationUtils.plot_forecast_comparison(
                        all_preds,
                        targets_array,
                        save_path=output_dir / "forecast_comparison.png",
                    )
                    visualization_paths["forecast_comparison"] = viz_path

            # Feature analysis
            if feature_names:
                viz_path = VisualizationUtils.plot_feature_analysis(
                    feature_names, output_dir / "feature_analysis.png"
                )
                visualization_paths["feature_analysis"] = viz_path

            # Advanced visualizations
            # ----------------------
            if show_advanced:
                # Processed data visualizations
                if processed_data is not None and not processed_data.empty:
                    try:
                        # Correlation matrix - with proper error handling
                        viz_path = VisualizationUtils.plot_correlation_matrix(
                            processed_data,
                            save_path=output_dir / "correlation_matrix.png",
                        )
                        visualization_paths["correlation_matrix"] = viz_path

                        # Feature distributions - with proper error handling
                        viz_path = VisualizationUtils.plot_feature_distributions(
                            processed_data,
                            n_features=12,
                            save_path=output_dir / "feature_distributions.png",
                        )
                        visualization_paths["feature_distributions"] = viz_path
                    except Exception as e:
                        VisualizationUtils._logger.error(
                            f"Error creating processed data visualizations: {str(e)}"
                        )

                # Raw data visualizations
                if (
                    raw_data is not None
                    and not raw_data.empty
                    and "Date" in raw_data.columns
                ):
                    try:
                        # Price series
                        if all(
                            col in raw_data.columns
                            for col in ["Open", "High", "Low", "Close"]
                        ):
                            viz_path = VisualizationUtils.plot_price_series(
                                raw_data,
                                price_cols=["Close"],
                                save_path=output_dir / "price_series.png",
                            )
                            visualization_paths["price_series"] = viz_path

                            # Candlestick with predictions overlay
                            if predictions_array is not None:
                                # Need to align predictions with raw data dates
                                # This is simplified - in practice you'd need proper alignment
                                last_n = min(len(predictions_array), 30)  # Last 30 days

                                viz_path = VisualizationUtils.plot_candlestick_chart(
                                    raw_data.tail(last_n),
                                    overlay_predictions=(
                                        predictions_array[-last_n:]
                                        if last_n > 0
                                        else None
                                    ),
                                    save_path=output_dir
                                    / "candlestick_predictions.png",
                                )
                                visualization_paths["candlestick_predictions"] = (
                                    viz_path
                                )
                    except Exception as e:
                        VisualizationUtils._logger.error(
                            f"Error creating raw data visualizations: {str(e)}"
                        )

                # Model explainability - attention weights visualization for transformer models
                if "attention_weights" in predictions_data:
                    viz_path = VisualizationUtils.plot_attention_weights(
                        predictions_data["attention_weights"],
                        sequence_length=30,  # Assuming sequence length is 30
                        save_path=output_dir / "attention_weights.png",
                    )
                    visualization_paths["attention_weights"] = viz_path

                # Feature importance if available
                if "feature_importance" in predictions_data:
                    viz_path = VisualizationUtils.plot_feature_importance(
                        predictions_data["feature_importance"],
                        feature_names=feature_names,
                        save_path=output_dir / "feature_importance.png",
                    )
                    visualization_paths["feature_importance"] = viz_path

        except Exception as e:
            VisualizationUtils._logger.error(
                f"Error creating visualizations: {str(e)}", exc_info=True
            )

        return visualization_paths
