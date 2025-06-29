"""
Visualizations for model explainability.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd

from .base import BaseVisualizer, set_finance_style


class ExplainabilityVisualizer(BaseVisualizer):
    """Visualizer for model explainability"""

    def plot_feature_importance(
        self,
        feature_importance: np.ndarray,
        feature_names: List[str],
        title: str = "Feature Importance",
        save_path: Optional[Path] = None,
        top_n: int = 20,
        ascending: bool = False,
        show_values: bool = True,
    ) -> str:
        """
        Plot feature importance from model

        Args:
            feature_importance: Array of feature importance values
            feature_names: List of feature names
            title: Plot title
            save_path: Optional path to save the plot
            top_n: Number of top features to include
            ascending: Sort in ascending order if True
            show_values: Show values on bars if True

        Returns:
            str: Path to saved plot
        """
        set_finance_style()

        # Create DataFrame for easier sorting
        df = pd.DataFrame({"importance": feature_importance, "feature": feature_names})
        df = df.sort_values("importance", ascending=ascending)

        # Select top N features
        if top_n > 0 and len(df) > top_n:
            df = df.iloc[-top_n:] if ascending else df.iloc[:top_n]

        plt.figure(figsize=(12, max(6, min(14, len(df) * 0.4))))

        # Create horizontal bar plot
        bars = plt.barh(
            df["feature"], df["importance"], color=sns.color_palette("viridis", len(df))
        )

        # Add values to the bars
        if show_values:
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width * 1.01
                plt.text(
                    label_x_pos,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.4f}",
                    va="center",
                )

        plt.title(title, fontsize=14)
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = Path("feature_importance.png")

        return self.save_figure(plt.gcf(), save_path)

    def plot_attention_weights(
        self,
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
        set_finance_style(theme="presentation")

        # Ensure attention weights has the right shape
        if attention_weights.ndim == 3:  # [batch, seq, seq]
            # Use first batch item
            attention_weights = attention_weights[0]
        elif attention_weights.ndim > 3:  # [batch, head, seq, seq]
            # Average across heads for first batch item
            attention_weights = np.mean(attention_weights[0], axis=0)

        # Handle sequence length mismatch
        if attention_weights.shape[0] != sequence_length:
            self.logger.warning(
                f"Attention weights shape {attention_weights.shape} doesn't match sequence length {sequence_length}"
            )
            attention_weights = attention_weights[:sequence_length, :sequence_length]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot heatmap
        cax = ax.matshow(attention_weights, cmap="viridis")

        # Add colorbar
        cbar = fig.colorbar(cax)
        cbar.set_label("Attention Weight", rotation=270, labelpad=15)

        # Set labels for x and y axes
        if timestamps is not None:
            # Format timestamps if they're datetime objects
            if len(timestamps) > 0 and hasattr(timestamps[0], "strftime"):
                labels = [ts.strftime("%m/%d") for ts in timestamps]
            else:
                labels = timestamps

            # Limit the number of x-ticks based on sequence length
            stride = max(1, len(labels) // 10)
            ax.set_xticks(range(0, len(labels), stride))
            ax.set_xticklabels(labels[::stride], rotation=45)

            ax.set_yticks(range(0, len(labels), stride))
            ax.set_yticklabels(labels[::stride])
        else:
            # Just show indices as ticks
            stride = max(1, sequence_length // 10)
            ax.set_xticks(range(0, sequence_length, stride))
            ax.set_yticks(range(0, sequence_length, stride))

        # Add axis labels and title
        ax.set_xlabel("Target Position", fontsize=12)
        ax.set_ylabel("Source Position", fontsize=12)
        ax.set_title("Attention Weights Visualization", fontsize=14)

        # Add time direction indicators
        ax.text(
            sequence_length - 1,
            -1,
            "Most Recent",
            horizontalalignment="right",
            verticalalignment="bottom",
            fontsize=10,
            color="black",
            weight="bold",
        )
        ax.text(
            0,
            -1,
            "Oldest",
            horizontalalignment="left",
            verticalalignment="bottom",
            fontsize=10,
            color="black",
            weight="bold",
        )

        fig.tight_layout()

        # Save figure
        if save_path is None:
            save_path = Path("attention_weights.png")

        return self.save_figure(fig, save_path)

    def plot_partial_dependence(
        self,
        values: List[np.ndarray],
        feature_values: List[np.ndarray],
        feature_names: List[str],
        save_path: Optional[Path] = None,
    ) -> str:
        """
        Plot partial dependence for multiple features

        Args:
            values: List of arrays containing predicted values for each feature
            feature_values: List of arrays containing feature values
            feature_names: List of feature names
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        set_finance_style()

        num_features = len(feature_names)

        # Determine grid dimensions
        cols = min(3, num_features)
        rows = (num_features + cols - 1) // cols

        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))

        # Flatten axes for easy indexing
        if rows > 1 or cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        # Plot each feature's partial dependence
        for i, (feat_vals, pdp_vals, feat_name) in enumerate(
            zip(feature_values, values, feature_names)
        ):
            if i < len(axes):
                ax = axes[i]

                # Plot the partial dependence
                ax.plot(feat_vals, pdp_vals, "b-", linewidth=2)

                # Add shaded region for confidence interval if available
                if (
                    hasattr(pdp_vals, "shape")
                    and len(pdp_vals.shape) > 1
                    and pdp_vals.shape[1] >= 3
                ):
                    lower_bound = pdp_vals[:, 1]
                    upper_bound = pdp_vals[:, 2]
                    ax.fill_between(
                        feat_vals, lower_bound, upper_bound, alpha=0.3, color="b"
                    )

                ax.set_xlabel(feat_name, fontsize=12)
                ax.set_ylabel("Partial Dependence", fontsize=12)
                ax.set_title(f"Partial Dependence of {feat_name}", fontsize=12)
                ax.grid(True, alpha=0.3)

                # Add horizontal line at y=0 for reference
                ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

                # Add mean prediction line
                mean_prediction = np.mean(pdp_vals)
                ax.axhline(
                    y=mean_prediction,
                    color="red",
                    linestyle="-",
                    alpha=0.3,
                    label=f"Mean Prediction: {mean_prediction:.4f}",
                )
                ax.legend()

        # Hide unused subplots
        for i in range(num_features, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = Path("partial_dependence.png")

        return self.save_figure(fig, save_path)

    def plot_prediction_explanation(
        self,
        feature_contributions: np.ndarray,
        feature_names: List[str],
        base_value: float,
        prediction: float,
        actual: Optional[float] = None,
        save_path: Optional[Path] = None,
        top_n: int = 10,
    ) -> str:
        """
        Plot contribution of each feature to a prediction (waterfall chart)

        Args:
            feature_contributions: Contribution of each feature
            feature_names: List of feature names
            base_value: Base value for prediction
            prediction: Final prediction value
            actual: Optional actual value
            save_path: Optional path to save the plot
            top_n: Number of top features to include

        Returns:
            str: Path to saved plot
        """
        set_finance_style()

        # Create DataFrame for easier manipulation
        df = pd.DataFrame(
            {"feature": feature_names, "contribution": feature_contributions}
        )

        # Sort by absolute contribution
        df["abs_contribution"] = np.abs(df["contribution"])
        df = df.sort_values("abs_contribution", ascending=False)

        # Keep top N features and group the rest
        if top_n > 0 and len(df) > top_n + 1:
            top_features = df.iloc[:top_n].copy()
            other_contribution = df.iloc[top_n:]["contribution"].sum()

            # Add "other" category
            other_row = pd.DataFrame(
                {
                    "feature": ["Other features"],
                    "contribution": [other_contribution],
                    "abs_contribution": [np.abs(other_contribution)],
                }
            )

            df = pd.concat([top_features, other_row])

        # Sort by contribution for waterfall effect
        df = df.sort_values("contribution")

        # Calculate positions for the bars
        cumulative_sum = base_value
        bottoms = []
        heights = []

        for contrib in df["contribution"]:
            if contrib >= 0:
                bottoms.append(cumulative_sum)
                heights.append(contrib)
            else:
                bottoms.append(cumulative_sum + contrib)
                heights.append(-contrib)

            cumulative_sum += contrib

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Set colors based on contribution sign
        colors = ["red" if h <= 0 else "green" for h in df["contribution"]]

        # Plot bars
        bars = ax.bar(df["feature"], heights, bottom=bottoms, color=colors)

        # Add baseline
        ax.axhline(
            y=base_value,
            color="gray",
            linestyle="--",
            label=f"Base value: {base_value:.4f}",
        )

        # Add prediction line
        ax.axhline(
            y=prediction,
            color="blue",
            linestyle="-",
            linewidth=2,
            label=f"Prediction: {prediction:.4f}",
        )

        # Add actual value if provided
        if actual is not None:
            ax.axhline(
                y=actual,
                color="black",
                linestyle=":",
                linewidth=2,
                label=f"Actual: {actual:.4f}",
            )

        # Add contribution values as text
        for bar, contrib in zip(bars, df["contribution"]):
            height = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_y() + height / 2

            # Skip small contributions to avoid clutter
            if abs(contrib) < (prediction - base_value) * 0.02:
                continue

            ax.text(
                x,
                y,
                f"{contrib:.4f}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=10,
            )

        # Set labels and title
        ax.set_xlabel("Features", fontsize=12)
        ax.set_ylabel("Contribution", fontsize=12)
        ax.set_title("Feature Contributions to Prediction", fontsize=14)

        # Rotate x-tick labels for better readability
        plt.xticks(rotation=45, ha="right")

        # Add legend
        ax.legend()

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = Path("prediction_explanation.png")

        return self.save_figure(fig, save_path)

    def plot_permutation_importance(
        self,
        importances: np.ndarray,
        feature_names: List[str],
        save_path: Optional[Path] = None,
        sort: bool = True,
    ) -> str:
        """
        Plot permutation importance for features

        Args:
            importances: Array of shape (n_repeats, n_features) or (n_features,)
            feature_names: List of feature names
            save_path: Optional path to save the plot
            sort: Whether to sort features by importance

        Returns:
            str: Path to saved plot
        """
        set_finance_style()

        # Handle different shapes of importances
        if importances.ndim > 1:
            # If we have multiple repetitions, compute mean and std
            mean_importances = np.mean(importances, axis=0)
            std_importances = np.std(importances, axis=0)
        else:
            mean_importances = importances
            std_importances = np.zeros_like(importances)

        # Create DataFrame for easier manipulation
        df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": mean_importances,
                "std": std_importances,
            }
        )

        # Sort by importance
        if sort:
            df = df.sort_values("importance", ascending=False)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(6, min(14, len(df) * 0.4))))

        # Plot importance bars
        ax.barh(
            df["feature"],
            df["importance"],
            xerr=df["std"],
            color="skyblue",
            edgecolor="blue",
            alpha=0.7,
        )

        # Add values
        for i, (importance, std) in enumerate(zip(df["importance"], df["std"])):
            value_text = f"{importance:.4f}"
            if std > 0:
                value_text += f" ± {std:.4f}"

            ax.text(importance + std + 0.005, i, value_text, va="center", fontsize=10)

        # Set labels and title
        ax.set_xlabel("Permutation Importance", fontsize=12)
        ax.set_ylabel("Features", fontsize=12)
        ax.set_title("Feature Permutation Importance", fontsize=14)

        # Add grid
        ax.grid(True, axis="x", alpha=0.3)

        # Set x-axis to start at 0
        xlim = ax.get_xlim()
        ax.set_xlim([0, xlim[1]])

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = Path("permutation_importance.png")

        return self.save_figure(fig, save_path)
