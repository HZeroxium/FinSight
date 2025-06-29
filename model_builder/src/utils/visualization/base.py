"""
Base visualization module with shared functionality.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

from ...common.logger.logger_factory import LoggerFactory


def set_finance_style(theme: str = "default") -> None:
    """
    Set a finance-themed style for matplotlib visualizations.

    Args:
        theme: Style theme name ("default", "dark", "presentation", "paper")
    """
    # Base style using seaborn
    sns.set_style("whitegrid" if theme != "dark" else "darkgrid")

    # Color palette selection
    if theme == "dark":
        # Dark theme with high contrast colors
        palette = "viridis"
        plt.rcParams.update(
            {
                "figure.facecolor": "#212121",
                "axes.facecolor": "#212121",
                "axes.edgecolor": "#757575",
                "axes.labelcolor": "white",
                "xtick.color": "white",
                "ytick.color": "white",
                "grid.color": "#424242",
                "text.color": "white",
                "legend.facecolor": "#313131",
                "legend.edgecolor": "#757575",
            }
        )
    elif theme == "presentation":
        # Bright, high-contrast colors for presentations
        palette = "Set1"
        plt.rcParams["figure.figsize"] = (12, 7)  # Larger figures for presentations
        plt.rcParams["font.size"] = 14
        plt.rcParams["axes.titlesize"] = 18
        plt.rcParams["axes.labelsize"] = 16
    elif theme == "paper":
        # Professional style for papers/publications
        palette = "colorblind"
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["figure.figsize"] = (8, 6)
    else:
        # Default financial theme
        palette = "muted"

    # Set palette
    sns.set_palette(palette)

    # Common settings
    plt.rcParams.update(
        {
            "lines.linewidth": 2,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


class BaseVisualizer:
    """Base class for visualizers with common functionality"""

    def __init__(self):
        """Initialize the base visualizer"""
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)

    def save_figure(self, fig: plt.Figure, filepath: Path) -> str:
        """
        Save a figure with proper formatting

        Args:
            fig: Matplotlib figure to save
            filepath: Path to save the figure

        Returns:
            str: Path to the saved figure
        """
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Add tight layout and save with high DPI
        fig.tight_layout()
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"Figure saved to {filepath}")
        return str(filepath)

    def create_subplot_grid(
        self,
        nplots: int,
        max_cols: int = 2,
        figsize_per_plot: Tuple[float, float] = (6, 4),
    ) -> Tuple[plt.Figure, Any]:
        """
        Create a grid of subplots

        Args:
            nplots: Number of plots
            max_cols: Maximum number of columns
            figsize_per_plot: Figure size per subplot

        Returns:
            Tuple of (fig, axes)
        """
        ncols = min(nplots, max_cols)
        nrows = (nplots + ncols - 1) // ncols

        figsize = (figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        # Ensure axes is always a flattened array
        if nplots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Hide unused subplots
        for i in range(nplots, len(axes)):
            axes[i].set_visible(False)

        return fig, axes
