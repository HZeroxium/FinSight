"""
Visualizations for feature analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster import hierarchy
from scipy.stats import spearmanr

from .base import BaseVisualizer, set_finance_style


class FeatureVisualizer(BaseVisualizer):
    """Visualizer for feature analysis and engineering"""

    def plot_feature_distributions(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        n_features: int = 10,
        save_path: Optional[Path] = None,
    ) -> str:
        """
        Plot distributions of top features

        Args:
            data: DataFrame containing features
            features: List of feature names to plot (optional)
            n_features: Number of features to plot if features not specified
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        set_finance_style()

        # If features not specified, select top n based on variance
        if features is None:
            # Calculate variance for each feature
            variances = data.var().sort_values(ascending=False)
            features = variances.index[:n_features].tolist()
        else:
            # Use only features that exist in the data
            features = [f for f in features if f in data.columns]
            features = features[:n_features]  # Limit to n_features

        # Determine grid dimensions
        n_cols = min(3, len(features))
        n_rows = (len(features) + n_cols - 1) // n_cols

        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

        # Flatten axes array for easier indexing
        if n_rows * n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        # Plot each feature
        for i, feature in enumerate(features):
            if i < len(axes):
                ax = axes[i]

                # Check if the feature exists and has non-null values
                if feature in data.columns and not data[feature].isna().all():
                    # Create a histogram with KDE
                    sns.histplot(
                        data[feature], kde=True, ax=ax, color="#1E88E5", alpha=0.7
                    )

                    # Add vertical line for mean and median
                    mean_val = data[feature].mean()
                    median_val = data[feature].median()
                    ax.axvline(
                        mean_val,
                        color="red",
                        linestyle="--",
                        alpha=0.8,
                        label=f"Mean: {mean_val:.2f}",
                    )
                    ax.axvline(
                        median_val,
                        color="green",
                        linestyle=":",
                        alpha=0.8,
                        label=f"Median: {median_val:.2f}",
                    )

                    # Add basic statistics as text
                    stats_text = (
                        f"Std: {data[feature].std():.2f}\n"
                        f"Min: {data[feature].min():.2f}\n"
                        f"Max: {data[feature].max():.2f}"
                    )
                    ax.text(
                        0.95,
                        0.95,
                        stats_text,
                        transform=ax.transAxes,
                        fontsize=9,
                        verticalalignment="top",
                        horizontalalignment="right",
                        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                    )

                    ax.legend(fontsize=8)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for {feature}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

                ax.set_title(feature)
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        fig.suptitle("Feature Distributions", fontsize=16, y=1.02)
        plt.subplots_adjust(top=0.92)

        if save_path is None:
            save_path = Path("feature_distributions.png")

        return self.save_figure(fig, save_path)

    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        method: str = "spearman",
        save_path: Optional[Path] = None,
        threshold: Optional[float] = None,
        figsize: Tuple[int, int] = (12, 10),
    ) -> str:
        """
        Plot correlation matrix of features

        Args:
            data: DataFrame containing features
            features: List of feature names to include (optional)
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            save_path: Optional path to save the plot
            threshold: Optional threshold to highlight correlations (absolute value)
            figsize: Figure size

        Returns:
            str: Path to saved plot
        """
        set_finance_style()

        # Select features if specified
        if features is not None:
            data = data[features].copy()

        # Compute correlation matrix
        corr_matrix = data.corr(method=method)

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list(
            "blue_white_red", ["#1565C0", "#FFFFFF", "#C62828"]
        )

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Draw heatmap with mask for upper triangle
        heatmap = sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            linewidths=0.5,
            annot=True if len(corr_matrix) <= 20 else False,
            fmt=".2f" if len(corr_matrix) <= 20 else None,
            annot_kws={"size": 8},
            ax=ax,
        )

        # Highlight correlations above threshold
        if threshold is not None:
            # Create a copy without the mask for threshold filtering
            corr_filtered = corr_matrix.copy()
            threshold_mask = (corr_filtered.abs() < threshold) | np.triu(
                np.ones_like(corr_filtered, dtype=bool)
            )
            heatmap_threshold = sns.heatmap(
                corr_filtered,
                mask=threshold_mask,
                cmap=cmap,
                vmax=1.0,
                vmin=-1.0,
                center=0,
                square=True,
                linewidths=1.5,
                annot=True,
                fmt=".2f",
                annot_kws={"size": 8, "weight": "bold"},
                cbar=False,
                ax=ax,
            )

        # Set title and adjust layout
        plt.title(f"{method.capitalize()} Correlation Matrix", fontsize=14, pad=20)

        # Make sure all labels are visible
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        # Adjust figure to ensure labels are visible
        plt.tight_layout()

        if save_path is None:
            save_path = Path(f"correlation_{method}.png")

        return self.save_figure(fig, save_path)

    def plot_feature_clustering(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        method: str = "ward",
        save_path: Optional[Path] = None,
    ) -> str:
        """
        Plot hierarchical clustering of features based on correlation

        Args:
            data: DataFrame containing features
            features: List of feature names to include (optional)
            method: Linkage method for hierarchical clustering
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        set_finance_style()

        # Select features if specified
        if features is not None:
            data = data[features].copy()

        # Compute correlation matrix
        corr = data.corr()

        # Convert to distance matrix
        dissimilarity = 1 - np.abs(corr)

        # Create figure
        fig = plt.figure(figsize=(14, 10))

        # Add clustering on rows using correlation as distance
        row_linkage = hierarchy.linkage(dissimilarity, method=method)
        dendro_row = hierarchy.dendrogram(
            row_linkage, labels=corr.index, orientation="right", leaf_font_size=10
        )

        # Reorder correlation matrix based on clustering
        row_order = dendro_row["ivl"]
        corr_ordered = corr.loc[row_order, row_order]

        # Create clustered heatmap
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(20, 10), gridspec_kw={"width_ratios": [1, 5]}
        )

        # Plot dendrogram on first axis
        hierarchy.dendrogram(
            row_linkage,
            orientation="left",
            ax=ax1,
            labels=None,
            leaf_font_size=0,
        )
        ax1.set_yticks([])
        ax1.set_ylabel("")
        ax1.set_xlabel("Distance")

        # Plot clustered heatmap on second axis
        sns.heatmap(corr_ordered, cmap="coolwarm", annot=False, square=True, ax=ax2)
        ax2.set_title("Clustered Feature Correlation Matrix", fontsize=14)
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        if save_path is None:
            save_path = Path("feature_clustering.png")

        return self.save_figure(fig, save_path)

    def plot_feature_pca(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        n_components: int = 2,
        save_path: Optional[Path] = None,
    ) -> str:
        """
        Plot PCA of features to show feature relationships and importance

        Args:
            data: DataFrame containing features
            features: List of feature names to include (optional)
            n_components: Number of PCA components to compute
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        set_finance_style()

        # Select features if specified
        if features is not None:
            data_subset = data[features].copy()
        else:
            data_subset = data.copy()

        # Drop non-numeric columns
        numeric_cols = data_subset.select_dtypes(include=[np.number]).columns
        data_subset = data_subset[numeric_cols]

        # Fill NaN values with column means
        data_subset = data_subset.fillna(data_subset.mean())

        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_subset)

        # Apply PCA
        n_components = min(n_components, min(scaled_data.shape))
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(scaled_data)

        # Create figure for explained variance
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot explained variance
        explained_variance = pca.explained_variance_ratio_ * 100
        cum_explained_variance = np.cumsum(explained_variance)

        axes[0].bar(
            range(1, len(explained_variance) + 1),
            explained_variance,
            alpha=0.8,
            color="#1E88E5",
            label="Individual",
        )
        axes[0].step(
            range(1, len(cum_explained_variance) + 1),
            cum_explained_variance,
            color="#E53935",
            linewidth=2,
            label="Cumulative",
        )
        axes[0].axhline(y=80, color="gray", linestyle="--", alpha=0.7)
        axes[0].set_xlabel("Principal Components")
        axes[0].set_ylabel("Explained Variance (%)")
        axes[0].set_title("PCA Explained Variance")
        axes[0].set_xticks(range(1, len(explained_variance) + 1))
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # If we have at least 2 components, plot feature loadings
        if n_components >= 2:
            # Get feature loadings
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

            # Create a DataFrame with loadings
            loadings_df = pd.DataFrame(
                loadings[:, :2], columns=["PC1", "PC2"], index=data_subset.columns
            )

            # Plot feature loadings
            for i, feature in enumerate(loadings_df.index):
                axes[1].arrow(
                    0,
                    0,
                    loadings_df.loc[feature, "PC1"],
                    loadings_df.loc[feature, "PC2"],
                    head_width=0.05,
                    head_length=0.05,
                    fc="#2E7D32",
                    ec="#2E7D32",
                    alpha=0.8,
                )
                axes[1].text(
                    loadings_df.loc[feature, "PC1"] * 1.15,
                    loadings_df.loc[feature, "PC2"] * 1.15,
                    feature,
                    fontsize=8,
                )

            # Plot PC1 and PC2 limits
            axes[1].axvline(x=0, color="gray", linestyle="-", alpha=0.3)
            axes[1].axhline(y=0, color="gray", linestyle="-", alpha=0.3)

            # Set equal aspect ratio
            axes[1].set_aspect("equal")
            axes[1].set_xlabel(f"PC1 ({explained_variance[0]:.1f}%)")
            axes[1].set_ylabel(f"PC2 ({explained_variance[1]:.1f}%)")
            axes[1].set_title("Feature Loadings (PC1 vs PC2)")

            # Set limits with some padding
            max_val = (
                max(abs(loadings_df.values.min()), abs(loadings_df.values.max())) * 1.2
            )
            axes[1].set_xlim(-max_val, max_val)
            axes[1].set_ylim(-max_val, max_val)

            # Add circle
            circle = plt.Circle(
                (0, 0), 1, color="gray", fill=False, linestyle="--", alpha=0.3
            )
            axes[1].add_patch(circle)
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = Path("feature_pca.png")

        return self.save_figure(fig, save_path)

    def plot_feature_importance_comparison(
        self,
        feature_importance_dict: Dict[str, np.ndarray],
        feature_names: List[str],
        save_path: Optional[Path] = None,
        top_n: int = 15,
    ) -> str:
        """
        Plot and compare feature importance from multiple sources/models

        Args:
            feature_importance_dict: Dictionary mapping source name to feature importance arrays
            feature_names: List of feature names
            save_path: Optional path to save the plot
            top_n: Number of top features to show

        Returns:
            str: Path to saved plot
        """
        set_finance_style()

        # Create a DataFrame to store all importance values
        all_importances = pd.DataFrame(index=feature_names)

        # Add importance from each source
        for source, importance in feature_importance_dict.items():
            if len(importance) == len(feature_names):
                all_importances[source] = importance
            else:
                self.logger.warning(
                    f"Feature importance length mismatch for {source}. Expected {len(feature_names)}, got {len(importance)}"
                )

        # Calculate mean importance across all sources
        if len(all_importances.columns) > 0:
            all_importances["Mean"] = all_importances.mean(axis=1)

            # Sort by mean importance
            all_importances = all_importances.sort_values("Mean", ascending=False)

            # Select top N features
            top_features = all_importances.head(top_n)

            # Create a plot
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot importance for each source
            bar_width = 0.8 / len(feature_importance_dict)
            x = np.arange(len(top_features))

            for i, source in enumerate(feature_importance_dict.keys()):
                if source in top_features.columns:
                    ax.bar(
                        x + i * bar_width - 0.4 + bar_width / 2,
                        top_features[source],
                        width=bar_width,
                        label=source,
                        alpha=0.7,
                    )

            # Add mean importance as a line
            if "Mean" in top_features.columns:
                ax.plot(
                    x,
                    top_features["Mean"],
                    "ko-",
                    linewidth=2,
                    label="Mean Importance",
                    alpha=0.8,
                )

            # Set chart properties
            ax.set_ylabel("Importance")
            ax.set_title("Feature Importance Comparison")
            ax.set_xticks(x)
            ax.set_xticklabels(top_features.index, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path is None:
                save_path = Path("feature_importance_comparison.png")

            return self.save_figure(fig, save_path)
        else:
            # Create empty figure with message if no valid importances
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No valid feature importance data available",
                ha="center",
                va="center",
                fontsize=14,
            )
            ax.axis("off")

            if save_path is None:
                save_path = Path("feature_importance_comparison.png")

            return self.save_figure(fig, save_path)

    def plot_time_series_features(
        self,
        data: pd.DataFrame,
        date_col: str,
        feature_cols: List[str],
        target_col: Optional[str] = None,
        save_path: Optional[Path] = None,
        max_features: int = 5,
        figsize: Tuple[int, int] = (14, 10),
    ) -> str:
        """
        Plot time series of selected features with optional target

        Args:
            data: DataFrame with time series data
            date_col: Column name containing dates
            feature_cols: List of feature column names to plot
            target_col: Optional target column to highlight
            save_path: Optional path to save the plot
            max_features: Maximum number of features to plot
            figsize: Figure size

        Returns:
            str: Path to saved plot
        """
        set_finance_style()

        # Ensure date column is datetime
        if date_col in data.columns:
            data = data.copy()
            if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
                data[date_col] = pd.to_datetime(data[date_col])

            # Set date as index for easier plotting
            data.set_index(date_col, inplace=True)

        # Limit to max_features
        if len(feature_cols) > max_features:
            self.logger.info(
                f"Limiting plot to {max_features} features out of {len(feature_cols)}"
            )
            feature_cols = feature_cols[:max_features]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot each feature
        for feature in feature_cols:
            if feature in data.columns:
                # Normalize to better compare trends
                normalized = (data[feature] - data[feature].mean()) / data[
                    feature
                ].std()
                ax.plot(data.index, normalized, label=feature, alpha=0.7, linewidth=1.5)

        # Add target if specified
        if target_col and target_col in data.columns:
            # Normalize target
            normalized_target = (data[target_col] - data[target_col].mean()) / data[
                target_col
            ].std()
            ax.plot(
                data.index,
                normalized_target,
                label=target_col,
                color="black",
                linewidth=2.5,
                alpha=0.8,
            )

        # Set chart properties
        ax.set_title("Normalized Time Series Features", fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Normalized Value", fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Format x-axis date ticks
        plt.gcf().autofmt_xdate()

        plt.tight_layout()

        if save_path is None:
            save_path = Path("time_series_features.png")

        return self.save_figure(fig, save_path)
