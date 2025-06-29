"""
Financial data visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import mplfinance as mpf

from .base import BaseVisualizer, set_finance_style


class FinancialVisualizer(BaseVisualizer):
    """Visualizer for financial data and market analysis."""

    def plot_price_series(
        self,
        data: pd.DataFrame,
        price_cols: List[str] = ["Close"],
        volume_col: Optional[str] = "Volume",
        date_col: Optional[str] = "Date",
        title: str = "Price Series Analysis",
        save_path: Optional[Path] = None,
        show_ma: bool = True,
        ma_periods: List[int] = [20, 50],
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
            ma_periods: Periods for moving averages

        Returns:
            str: Path to saved plot
        """
        set_finance_style()

        # Create figure with primary and secondary y-axis
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # Check if date column exists and convert to datetime if needed
        if date_col in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
                data = data.copy()
                data[date_col] = pd.to_datetime(data[date_col])
            x = data[date_col]
        else:
            x = (
                data.index
                if isinstance(data.index, pd.DatetimeIndex)
                else range(len(data))
            )

        # Plot price series
        for i, col in enumerate(price_cols):
            if col in data.columns:
                color = f"C{i}"
                ax1.plot(x, data[col], label=col, linewidth=2, color=color)

        # Add moving averages if requested
        if show_ma:
            for i, period in enumerate(ma_periods):
                for j, col in enumerate(
                    price_cols[:1]
                ):  # Only for the first price column
                    if col in data.columns:
                        ma_col = f"MA{period}"
                        data[ma_col] = data[col].rolling(window=period).mean()
                        ax1.plot(
                            x,
                            data[ma_col],
                            label=f"{period}-day MA",
                            linewidth=1.5,
                            linestyle="--",
                            color=f"C{len(price_cols)+i}",
                        )

        # Create secondary y-axis for volume
        if volume_col and volume_col in data.columns:
            ax2 = ax1.twinx()
            ax2.bar(
                x, data[volume_col], alpha=0.3, width=0.8, color="gray", label="Volume"
            )
            ax2.set_ylabel("Volume", fontsize=12)

            # Format y-axis for volume (K, M, B)
            def volume_formatter(x, pos):
                if x >= 1e9:
                    return f"{x/1e9:.1f}B"
                elif x >= 1e6:
                    return f"{x/1e6:.1f}M"
                elif x >= 1e3:
                    return f"{x/1e3:.1f}K"
                else:
                    return f"{x:.0f}"

            ax2.yaxis.set_major_formatter(FuncFormatter(volume_formatter))

            # Ensure volume doesn't dominate the chart
            ax2.set_ylim(0, data[volume_col].max() * 5)

            # Add volume legend to the combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        else:
            ax1.legend(loc="best")

        # Format x-axis for dates
        if isinstance(x, pd.DatetimeIndex) or (
            date_col in data.columns
            and pd.api.types.is_datetime64_any_dtype(data[date_col])
        ):
            plt.gcf().autofmt_xdate()
            if len(x) > 20:  # If many data points, limit ticks
                ax1.xaxis.set_major_locator(mdates.MonthLocator())
                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

        # Set labels and title
        ax1.set_xlabel("Date", fontsize=12)
        ax1.set_ylabel("Price", fontsize=12)
        ax1.set_title(title, fontsize=16)

        # Grid and style refinements
        ax1.grid(True, alpha=0.3)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # Save and return
        if save_path is None:
            save_path = Path("price_series.png")

        return self.save_figure(fig, save_path)

    def plot_candlestick_chart(
        self,
        data: pd.DataFrame,
        date_col: str = "Date",
        open_col: str = "Open",
        high_col: str = "High",
        low_col: str = "Low",
        close_col: str = "Close",
        volume_col: Optional[str] = "Volume",
        title: str = "Candlestick Chart",
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
            title: Plot title
            overlay_predictions: Optional predicted values to overlay
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        # Check if required columns exist
        required_cols = [date_col, open_col, high_col, low_col, close_col]
        if not all(col in data.columns for col in required_cols):
            self.logger.error("Missing required columns for candlestick chart")
            # Create a simple error figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "Missing required columns for candlestick chart",
                ha="center",
                va="center",
                fontsize=14,
            )

            if save_path is None:
                save_path = Path("candlestick_error.png")
            return self.save_figure(fig, save_path)

        # Prepare data in the format expected by mplfinance
        plot_data = data.copy()

        # Set date as index if not already
        if not isinstance(plot_data.index, pd.DatetimeIndex):
            plot_data[date_col] = pd.to_datetime(plot_data[date_col])
            plot_data.set_index(date_col, inplace=True)

        # Rename columns to match mplfinance expected format
        column_map = {
            open_col: "Open",
            high_col: "High",
            low_col: "Low",
            close_col: "Close",
        }

        if volume_col and volume_col in plot_data.columns:
            column_map[volume_col] = "Volume"

        plot_data = plot_data.rename(columns=column_map)

        # Keep only necessary columns
        needed_columns = ["Open", "High", "Low", "Close"]
        if "Volume" in plot_data.columns:
            needed_columns.append("Volume")

        plot_data = plot_data[needed_columns]

        # Set up plot style
        mc = mpf.make_marketcolors(
            up="#00c853",
            down="#ff5252",
            edge={"up": "#00c853", "down": "#ff5252"},
            wick={"up": "#00c853", "down": "#ff5252"},
            volume={"up": "#a5d6a7", "down": "#ef9a9a"},
        )

        style = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle="--",
            y_on_right=False,
            facecolor="white",
            figcolor="white",
            gridcolor="lightgray",
        )

        # Prepare plot arguments
        kwargs = {
            "type": "candle",
            "volume": True if "Volume" in plot_data.columns else False,
            "title": title,
            "figratio": (14, 9),
            "figscale": 1.2,
            "style": style,
        }

        # Add predictions as overlay if provided
        if overlay_predictions is not None and len(overlay_predictions) > 0:
            try:
                # Ensure predictions is a 1D array
                if isinstance(overlay_predictions, np.ndarray):
                    if overlay_predictions.ndim > 1:
                        overlay_predictions = overlay_predictions.flatten()
                else:
                    overlay_predictions = np.asarray(overlay_predictions).flatten()

                # Align predictions with plot data length
                prediction_length = min(len(overlay_predictions), len(plot_data))
                aligned_predictions = overlay_predictions[:prediction_length]
                aligned_index = plot_data.index[:prediction_length]

                prediction_series = pd.Series(
                    aligned_predictions, index=aligned_index, name="Predicted Close"
                )

                kwargs["addplot"] = [
                    mpf.make_addplot(
                        prediction_series,
                        color="purple",
                        width=2,
                        linestyle="--",
                        panel=0,
                        secondary_y=False,
                    )
                ]

                self.logger.info(
                    f"Added predictions overlay with {len(aligned_predictions)} points"
                )

            except Exception as e:
                self.logger.warning(f"Could not add predictions overlay: {str(e)}")
                # Continue without predictions overlay

        # Create a temporary file path if none provided
        if save_path is None:
            save_path = Path("candlestick_chart.png")

        try:
            # Plot and save
            fig, axes = mpf.plot(plot_data, **kwargs, returnfig=True)

            # Add legend for predictions if applicable
            if overlay_predictions is not None and "addplot" in kwargs:
                if hasattr(axes, "__len__") and len(axes) > 0:
                    axes[0].legend(["Predicted Close"], loc="upper left")
                else:
                    axes.legend(["Predicted Close"], loc="upper left")

            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            self.logger.info(f"Candlestick chart saved to {save_path}")
            return str(save_path)

        except Exception as e:
            self.logger.error(f"Error creating candlestick chart: {str(e)}")
            # Create fallback simple chart
            fig, ax = plt.subplots(figsize=(12, 8))

            # Simple line chart as fallback
            ax.plot(
                plot_data.index, plot_data["Close"], label="Close Price", linewidth=2
            )

            if overlay_predictions is not None:
                try:
                    aligned_predictions = overlay_predictions[: len(plot_data)]
                    ax.plot(
                        plot_data.index[: len(aligned_predictions)],
                        aligned_predictions,
                        label="Predicted Close",
                        linestyle="--",
                        color="purple",
                        linewidth=2,
                    )
                except:
                    pass

            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Format dates
            plt.gcf().autofmt_xdate()

            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            self.logger.info(f"Fallback chart saved to {save_path}")
            return str(save_path)

    def plot_volatility_analysis(
        self,
        returns: Union[pd.Series, np.ndarray],
        predicted_returns: Optional[Union[pd.Series, np.ndarray]] = None,
        window_sizes: List[int] = [5, 10, 20],
        dates: Optional[Union[pd.DatetimeIndex, List]] = None,
        save_path: Optional[Path] = None,
    ) -> str:
        """
        Plot volatility analysis with rolling standard deviation

        Args:
            returns: Series or array of returns
            predicted_returns: Optional predicted returns for comparison
            window_sizes: List of window sizes for calculating rolling volatility
            dates: Optional dates for x-axis
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        set_finance_style()

        # Convert inputs to pandas Series if they're not already
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        if predicted_returns is not None and isinstance(predicted_returns, np.ndarray):
            predicted_returns = pd.Series(predicted_returns)

        # Set up index
        if dates is not None:
            returns.index = dates
            if predicted_returns is not None:
                predicted_returns.index = dates[: len(predicted_returns)]

        # Create figure
        fig, axes = plt.subplots(
            2, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [1, 2]}
        )

        # Plot returns
        axes[0].plot(returns, label="Actual Returns", color="blue", alpha=0.7)
        if predicted_returns is not None:
            axes[0].plot(
                predicted_returns,
                label="Predicted Returns",
                color="red",
                linestyle="--",
                alpha=0.7,
            )

        axes[0].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        axes[0].set_title("Returns Over Time", fontsize=14)
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Return (%)", fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot rolling volatility for different window sizes
        colors = sns.color_palette("viridis", len(window_sizes))

        for i, window in enumerate(window_sizes):
            # Calculate rolling volatility (annualized)
            # Assuming daily data, multiply by sqrt(252) to annualize
            vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
            axes[1].plot(
                vol, label=f"{window}-day Volatility", color=colors[i], linewidth=2
            )

            if predicted_returns is not None and len(predicted_returns) >= window:
                pred_vol = (
                    predicted_returns.rolling(window=window).std() * np.sqrt(252) * 100
                )
                axes[1].plot(
                    pred_vol,
                    label=f"{window}-day Predicted Volatility",
                    color=colors[i],
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.7,
                )

        # Add historical volatility (full period)
        hist_vol = returns.std() * np.sqrt(252) * 100
        axes[1].axhline(
            y=hist_vol,
            color="black",
            linestyle="-",
            label=f"Historical Volatility ({hist_vol:.2f}%)",
        )

        axes[1].set_title("Volatility Analysis", fontsize=14)
        axes[1].set_xlabel("Date", fontsize=12)
        axes[1].set_ylabel("Annualized Volatility (%)", fontsize=12)
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.3)

        # Format dates if appropriate
        if isinstance(returns.index, pd.DatetimeIndex):
            fig.autofmt_xdate()

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = Path("volatility_analysis.png")

        return self.save_figure(fig, save_path)

    def plot_drawdown_analysis(
        self,
        prices: Union[pd.Series, np.ndarray],
        predicted_prices: Optional[Union[pd.Series, np.ndarray]] = None,
        dates: Optional[Union[pd.DatetimeIndex, List]] = None,
        title: str = "Drawdown Analysis",
        save_path: Optional[Path] = None,
    ) -> str:
        """
        Plot drawdown analysis showing periods of decline from peaks

        Args:
            prices: Series or array of prices
            predicted_prices: Optional predicted prices for comparison
            dates: Optional dates for x-axis
            title: Plot title
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        set_finance_style()

        # Convert inputs to pandas Series if they're not already
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)

        if predicted_prices is not None and isinstance(predicted_prices, np.ndarray):
            predicted_prices = pd.Series(predicted_prices)

        # Set up index
        if dates is not None:
            prices.index = dates
            if predicted_prices is not None:
                predicted_prices.index = dates[: len(predicted_prices)]

        # Create figure
        fig, axes = plt.subplots(
            3, 1, figsize=(14, 15), gridspec_kw={"height_ratios": [2, 1, 1]}
        )

        # 1. Plot prices
        axes[0].plot(prices, label="Actual Price", color="blue", linewidth=2)
        if predicted_prices is not None:
            axes[0].plot(
                predicted_prices,
                label="Predicted Price",
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )

        axes[0].set_title("Price Series", fontsize=14)
        axes[0].set_ylabel("Price", fontsize=12)
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)

        # 2. Calculate and plot drawdowns
        # For actual prices
        rolling_max = prices.cummax()
        drawdowns = (prices - rolling_max) / rolling_max * 100

        axes[1].fill_between(drawdowns.index, drawdowns, 0, color="red", alpha=0.3)
        axes[1].plot(drawdowns, color="red", linewidth=1)

        # Mark the maximum drawdown point
        max_drawdown = drawdowns.min()
        max_drawdown_idx = drawdowns.idxmin()

        axes[1].scatter(
            max_drawdown_idx, max_drawdown, color="darkred", s=100, zorder=5
        )
        axes[1].annotate(
            f"Max Drawdown: {max_drawdown:.2f}%",
            xy=(max_drawdown_idx, max_drawdown),
            xytext=(max_drawdown_idx, max_drawdown - 5),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
            ha="center",
            fontsize=10,
        )

        # For predicted prices if available
        if predicted_prices is not None:
            pred_rolling_max = predicted_prices.cummax()
            pred_drawdowns = (
                (predicted_prices - pred_rolling_max) / pred_rolling_max * 100
            )

            axes[2].fill_between(
                pred_drawdowns.index, pred_drawdowns, 0, color="blue", alpha=0.3
            )
            axes[2].plot(pred_drawdowns, color="blue", linewidth=1)

            # Mark the maximum drawdown point for predictions
            pred_max_drawdown = pred_drawdowns.min()
            pred_max_drawdown_idx = pred_drawdowns.idxmin()

            axes[2].scatter(
                pred_max_drawdown_idx,
                pred_max_drawdown,
                color="darkblue",
                s=100,
                zorder=5,
            )
            axes[2].annotate(
                f"Predicted Max Drawdown: {pred_max_drawdown:.2f}%",
                xy=(pred_max_drawdown_idx, pred_max_drawdown),
                xytext=(pred_max_drawdown_idx, pred_max_drawdown - 5),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
                ha="center",
                fontsize=10,
            )

            axes[2].set_title("Predicted Drawdowns", fontsize=14)
            axes[2].set_ylabel("Drawdown (%)", fontsize=12)
            axes[2].set_xlabel("Date", fontsize=12)
            axes[2].grid(True, alpha=0.3)

            # Adjust titles and labels for clarity when showing both
            axes[1].set_title("Actual Drawdowns", fontsize=14)
            axes[1].set_ylabel("Drawdown (%)", fontsize=12)
            axes[1].set_xlabel("")
        else:
            # If no predictions, use the second subplot for drawdowns
            axes[1].set_title("Drawdowns", fontsize=14)
            axes[1].set_ylabel("Drawdown (%)", fontsize=12)
            axes[1].set_xlabel("Date", fontsize=12)

            # Hide the third subplot
            axes[2].axis("off")

        axes[1].grid(True, alpha=0.3)

        # Format dates if appropriate
        if isinstance(prices.index, pd.DatetimeIndex):
            fig.autofmt_xdate()

        # Add overall title
        fig.suptitle(title, fontsize=16, y=0.995)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        # Save figure
        if save_path is None:
            save_path = Path("drawdown_analysis.png")

        return self.save_figure(fig, save_path)

    def plot_risk_return_analysis(
        self,
        returns_dict: Dict[str, np.ndarray],
        risk_free_rate: float = 0.0,
        save_path: Optional[Path] = None,
    ) -> str:
        """
        Plot risk-return analysis comparing different investment strategies

        Args:
            returns_dict: Dictionary mapping strategy names to return arrays
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            save_path: Optional path to save the plot

        Returns:
            str: Path to saved plot
        """
        set_finance_style()

        # Calculate risk and return metrics for each strategy
        metrics = {}
        for strategy, returns in returns_dict.items():
            # Remove NaNs
            cleaned_returns = returns[~np.isnan(returns)]

            # Annualize metrics (assuming daily returns)
            annual_return = np.mean(cleaned_returns) * 252 * 100
            annual_volatility = np.std(cleaned_returns) * np.sqrt(252) * 100
            sharpe_ratio = (
                (annual_return - risk_free_rate) / annual_volatility
                if annual_volatility > 0
                else 0
            )
            max_drawdown = 0

            # Calculate maximum drawdown
            cumulative_returns = np.cumprod(1 + cleaned_returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = np.min(drawdown) * 100

            metrics[strategy] = {
                "return": annual_return,
                "volatility": annual_volatility,
                "sharpe": sharpe_ratio,
                "max_drawdown": max_drawdown,
            }

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 1. Risk-Return Scatter plot
        for strategy, metric in metrics.items():
            ax1.scatter(
                metric["volatility"], metric["return"], s=100, label=strategy, alpha=0.7
            )

            # Add strategy name as annotation
            ax1.annotate(
                strategy,
                xy=(metric["volatility"], metric["return"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
            )

        # Add risk-free rate horizontal line
        if risk_free_rate > 0:
            ax1.axhline(
                y=risk_free_rate,
                color="gray",
                linestyle="--",
                label=f"Risk-free rate ({risk_free_rate:.2f}%)",
            )

        # Add constant Sharpe ratio lines
        max_vol = max([m["volatility"] for m in metrics.values()]) * 1.2
        for sharpe in [0.5, 1.0, 1.5, 2.0]:
            x = np.linspace(0, max_vol, 100)
            y = risk_free_rate + sharpe * x
            ax1.plot(x, y, "k--", alpha=0.2)
            ax1.annotate(
                f"Sharpe = {sharpe}",
                xy=(max_vol * 0.7, risk_free_rate + sharpe * max_vol * 0.7),
                fontsize=8,
                color="gray",
            )

        ax1.set_title("Risk-Return Analysis", fontsize=14)
        ax1.set_xlabel("Annualized Volatility (%)", fontsize=12)
        ax1.set_ylabel("Annualized Return (%)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Performance metrics comparison
        strategies = list(metrics.keys())
        x = np.arange(len(strategies))
        width = 0.2

        # Plot return, volatility, max drawdown bars
        ax2.bar(
            x - width,
            [metrics[s]["return"] for s in strategies],
            width,
            label="Return (%)",
        )
        ax2.bar(
            x,
            [metrics[s]["volatility"] for s in strategies],
            width,
            label="Volatility (%)",
        )
        ax2.bar(
            x + width,
            [-metrics[s]["max_drawdown"] for s in strategies],
            width,
            label="Max Drawdown (%)",
        )

        # Add Sharpe ratio as a line on secondary axis
        ax3 = ax2.twinx()
        ax3.plot(
            x,
            [metrics[s]["sharpe"] for s in strategies],
            "ro-",
            linewidth=2,
            label="Sharpe Ratio",
        )
        ax3.set_ylabel("Sharpe Ratio", fontsize=12, color="r")
        ax3.tick_params(axis="y", labelcolor="r")

        ax2.set_title("Performance Metrics Comparison", fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies, rotation=30, ha="right")
        ax2.set_ylabel("Percentage (%)", fontsize=12)

        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = Path("risk_return_analysis.png")

        return self.save_figure(fig, save_path)
