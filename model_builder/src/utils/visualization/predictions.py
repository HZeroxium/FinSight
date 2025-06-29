"""
Visualizations for prediction analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .base import BaseVisualizer, set_finance_style


class PredictionVisualizer(BaseVisualizer):
    """Visualizer for model predictions"""

    def plot_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
        sample_size: int = 100,
        model_name: str = "model",
    ) -> str:
        """
        Enhanced plot for prediction analysis

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
        set_finance_style()

        # Create a 2x3 grid for more detailed analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Flatten arrays for analysis
        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()

        # 1. Predictions vs Targets scatter plot
        ax = axes[0, 0]
        ax.scatter(targets_flat, predictions_flat, alpha=0.6, color="#2196F3")

        # Add regression line
        min_val = min(targets_flat.min(), predictions_flat.min())
        max_val = max(targets_flat.max(), predictions_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

        # Add perfect prediction line
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="black",
            linestyle="-",
            alpha=0.3,
        )

        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Predictions vs Actual Values")

        # Add R² value
        errors = predictions_flat - targets_flat
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((targets_flat - np.mean(targets_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        ax.annotate(
            f"R² = {r2:.4f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.2),
        )

        # 2. Time series plot (with optional confidence intervals)
        ax = axes[0, 1]
        sample_size = min(sample_size, len(predictions_flat))

        if timestamps is not None:
            timestamps = timestamps[:sample_size]
            indices = timestamps
        else:
            indices = range(sample_size)

        ax.plot(
            indices,
            targets_flat[:sample_size],
            label="Actual",
            linewidth=2,
            color="#4CAF50",
        )
        ax.plot(
            indices,
            predictions_flat[:sample_size],
            label="Predicted",
            linewidth=2,
            color="#FF9800",
            alpha=0.8,
        )

        # Calculate moving average of predictions for trend line
        window = min(5, sample_size // 5)
        if window > 1:
            trend = np.convolve(
                predictions_flat[:sample_size], np.ones(window) / window, mode="valid"
            )
            trend_indices = (
                indices[window - 1 :]
                if isinstance(indices, np.ndarray)
                else range(window - 1, sample_size)
            )
            ax.plot(
                trend_indices,
                trend,
                label=f"{window}-period Trend",
                linewidth=2,
                color="#E91E63",
                linestyle="--",
            )

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Values")
        ax.set_title("Time Series Comparison")
        ax.legend()

        # 3. Error distribution
        ax = axes[0, 2]
        errors = predictions_flat - targets_flat

        # Create a histogram with KDE overlay
        sns.histplot(errors, bins=30, kde=True, ax=ax, color="#9C27B0", alpha=0.6)

        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Frequency")
        ax.set_title("Error Distribution")

        # Add statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        ax.axvline(mean_error, color="red", linestyle="--", alpha=0.8)
        ax.axvline(mean_error + std_error, color="gray", linestyle=":", alpha=0.8)
        ax.axvline(mean_error - std_error, color="gray", linestyle=":", alpha=0.8)

        stats_text = (
            f"Mean: {mean_error:.4f}\nStd: {std_error:.4f}\n"
            f"Min: {np.min(errors):.4f}\nMax: {np.max(errors):.4f}"
        )
        ax.annotate(
            stats_text,
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.2),
            verticalalignment="top",
        )

        # 4. Error over time
        ax = axes[1, 0]

        if sample_size > 0:
            ax.plot(
                indices[:sample_size],
                errors[:sample_size],
                color="#F44336",
                marker="o",
                markersize=4,
                linestyle="-",
                linewidth=1,
                alpha=0.7,
            )

            # Add a zero line
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

            # Add moving average of errors
            window = min(5, sample_size // 5)
            if window > 1:
                error_ma = np.convolve(
                    errors[:sample_size], np.ones(window) / window, mode="valid"
                )
                ma_indices = (
                    indices[window - 1 : sample_size]
                    if isinstance(indices, np.ndarray)
                    else range(window - 1, sample_size)
                )
                ax.plot(
                    ma_indices,
                    error_ma,
                    label=f"{window}-period MA",
                    linewidth=2,
                    color="blue",
                )

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Prediction Error")
        ax.set_title("Error Over Time")
        if window > 1:
            ax.legend()

        # 5. Cumulative error
        ax = axes[1, 1]
        if sample_size > 0:
            cumulative_errors = np.cumsum(np.abs(errors[:sample_size]))
            ax.plot(
                indices[:sample_size], cumulative_errors, color="#795548", linewidth=2
            )

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Cumulative Absolute Error")
        ax.set_title("Cumulative Error")

        # 6. Prediction accuracy by magnitude
        ax = axes[1, 2]

        # Bin the targets and compute average error in each bin
        if len(targets_flat) > 10:  # Only if we have enough data
            bins = min(10, len(targets_flat) // 5)
            target_bins = np.linspace(targets_flat.min(), targets_flat.max(), bins + 1)
            binned_errors = []
            bin_centers = []

            for i in range(bins):
                mask = (targets_flat >= target_bins[i]) & (
                    targets_flat < target_bins[i + 1]
                )
                if np.sum(mask) > 0:
                    bin_errors = np.abs(errors[mask])
                    binned_errors.append(np.mean(bin_errors))
                    bin_centers.append((target_bins[i] + target_bins[i + 1]) / 2)

            if bin_centers:
                ax.bar(
                    bin_centers,
                    binned_errors,
                    width=(target_bins[1] - target_bins[0]) * 0.8,
                    color="#009688",
                    alpha=0.7,
                )

                ax.set_xlabel("Target Value")
                ax.set_ylabel("Mean Absolute Error")
                ax.set_title("Error by Target Magnitude")
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for binning",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        # Add overall title
        fig.suptitle(f"Prediction Analysis: {model_name}", fontsize=16, y=0.995)
        plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.3)

        # Save figure
        if save_path is None:
            save_path = Path("prediction_analysis.png")

        return self.save_figure(fig, save_path)

    def plot_forecast_comparison(
        self,
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
        set_finance_style()
        fig, axes = plt.subplots(
            2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]}
        )

        # Prepare data
        targets_flat = targets.flatten()
        sample_size = min(sample_size, len(targets_flat))

        if timestamps is not None:
            timestamps = timestamps[:sample_size]
            indices = timestamps
        else:
            indices = range(sample_size)

        # Plot actual values
        axes[0].plot(
            indices,
            targets_flat[:sample_size],
            label="Actual",
            linewidth=2,
            color="black",
        )

        # Plot predictions for each model
        colors = sns.color_palette("Set1", len(predictions))
        error_data = []

        for i, (model_name, preds) in enumerate(predictions.items()):
            preds_flat = preds.flatten()[:sample_size]
            axes[0].plot(
                indices,
                preds_flat,
                label=f"{model_name}",
                linewidth=1.5,
                color=colors[i],
                alpha=0.8,
            )

            # Calculate errors for the second plot
            errors = preds_flat - targets_flat[:sample_size]
            error_data.append((model_name, errors, colors[i]))

        axes[0].set_title("Forecast Comparison Across Models")
        axes[0].set_xlabel("Time Steps")
        axes[0].set_ylabel("Values")
        axes[0].legend(loc="upper left")

        # Plot errors for each model
        for model_name, errors, color in error_data:
            axes[1].plot(
                indices, errors, label=f"{model_name} error", color=color, alpha=0.8
            )

        axes[1].set_title("Error Comparison")
        axes[1].set_xlabel("Time Steps")
        axes[1].set_ylabel("Error (Predicted - Actual)")
        axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
        axes[1].legend(loc="upper left")

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = Path("forecast_comparison.png")

        return self.save_figure(fig, save_path)

    def plot_trading_simulation(
        self,
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
        set_finance_style()
        fig, axes = plt.subplots(
            3, 1, figsize=(14, 16), gridspec_kw={"height_ratios": [2, 1, 1]}
        )

        # Flatten arrays
        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()

        # Create indices for x-axis
        if timestamps is not None:
            indices = timestamps
        else:
            indices = range(len(targets_flat))

        # 1. Price chart with predictions
        ax = axes[0]
        ax.plot(indices, targets_flat, label="Actual Price", color="blue", linewidth=2)
        ax.plot(
            indices,
            predictions_flat,
            label="Predicted Price",
            color="orange",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
        )

        # Signal generation (simple strategy: buy if predicted > actual, sell otherwise)
        signals = np.sign(predictions_flat[1:] - targets_flat[:-1])
        buy_indices = [indices[i + 1] for i in range(len(signals)) if signals[i] > 0]
        sell_indices = [indices[i + 1] for i in range(len(signals)) if signals[i] < 0]

        buy_prices = [
            targets_flat[i + 1] for i in range(len(signals)) if signals[i] > 0
        ]
        sell_prices = [
            targets_flat[i + 1] for i in range(len(signals)) if signals[i] < 0
        ]

        # Plot buy/sell signals
        ax.scatter(
            buy_indices,
            buy_prices,
            color="green",
            label="Buy Signal",
            marker="^",
            s=100,
            alpha=0.7,
        )
        ax.scatter(
            sell_indices,
            sell_prices,
            color="red",
            label="Sell Signal",
            marker="v",
            s=100,
            alpha=0.7,
        )

        ax.set_title("Price Chart with Trading Signals")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()

        # 2. Portfolio value simulation
        ax = axes[1]

        # Simple trading simulation
        portfolio_value = [initial_capital]
        cash = initial_capital
        holdings = 0

        for i in range(1, len(targets_flat)):
            if i < len(signals) + 1:
                # Execute trade based on previous signal
                price = targets_flat[i]

                # Simplified trading logic
                if signals[i - 1] > 0 and cash > price:  # Buy signal
                    units_to_buy = cash / price  # Simplified: use all cash
                    holdings += units_to_buy
                    cash = 0
                elif signals[i - 1] < 0 and holdings > 0:  # Sell signal
                    cash += holdings * price
                    holdings = 0

            # Calculate portfolio value
            current_value = cash + holdings * targets_flat[i]
            portfolio_value.append(current_value)

        # Plot portfolio value
        ax.plot(
            indices,
            portfolio_value,
            color="purple",
            linewidth=2,
            label="Portfolio Value",
        )

        # Plot buy-hold baseline
        units_buyhold = initial_capital / targets_flat[0]
        buyhold_value = [units_buyhold * price for price in targets_flat]
        ax.plot(
            indices,
            buyhold_value,
            color="gray",
            linewidth=1.5,
            linestyle="--",
            label="Buy & Hold",
        )

        ax.set_title("Portfolio Value Simulation")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value ($)")
        ax.legend()

        # 3. Performance metrics
        ax = axes[2]

        # Calculate returns
        portfolio_returns = [0]
        for i in range(1, len(portfolio_value)):
            ret = (portfolio_value[i] - portfolio_value[i - 1]) / portfolio_value[i - 1]
            portfolio_returns.append(ret)

        buyhold_returns = [0]
        for i in range(1, len(buyhold_value)):
            ret = (buyhold_value[i] - buyhold_value[i - 1]) / buyhold_value[i - 1]
            buyhold_returns.append(ret)

        # Calculate cumulative returns
        cum_portfolio_returns = np.cumprod(np.array(portfolio_returns) + 1) - 1
        cum_buyhold_returns = np.cumprod(np.array(buyhold_returns) + 1) - 1

        # Plot cumulative returns
        ax.plot(
            indices,
            cum_portfolio_returns * 100,
            color="purple",
            linewidth=2,
            label="Strategy Returns",
        )
        ax.plot(
            indices,
            cum_buyhold_returns * 100,
            color="gray",
            linewidth=1.5,
            linestyle="--",
            label="Buy & Hold Returns",
        )

        # Calculate performance metrics
        final_portfolio_return = portfolio_value[-1] / initial_capital - 1
        final_buyhold_return = buyhold_value[-1] / initial_capital - 1

        # Add performance text
        ax.text(
            0.02,
            0.95,
            f"Strategy Return: {final_portfolio_return:.2%}\n"
            f"Buy & Hold Return: {final_buyhold_return:.2%}\n"
            f"Outperformance: {final_portfolio_return - final_buyhold_return:.2%}",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.5"),
            verticalalignment="top",
        )

        ax.set_title("Cumulative Returns (%)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative Return (%)")
        ax.legend()

        # Add overall title
        fig.suptitle(
            "Trading Strategy Simulation Based on Model Predictions",
            fontsize=16,
            y=0.995,
        )
        plt.subplots_adjust(top=0.95, hspace=0.3)

        # Save figure
        if save_path is None:
            save_path = Path("trading_simulation.png")

        return self.save_figure(fig, save_path)
