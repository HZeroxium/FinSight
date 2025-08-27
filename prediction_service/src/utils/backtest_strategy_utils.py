# utils/backtest_utils.py

"""
Backtesting utilities for time series trading strategies.
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class SignalType(Enum):
    """Trading signal types"""

    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class Trade:
    """Represents a single trade"""

    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    signal_type: SignalType
    pnl: Optional[float] = None

    def close_trade(self, exit_time: pd.Timestamp, exit_price: float):
        """Close the trade and calculate PnL"""
        self.exit_time = exit_time
        self.exit_price = exit_price

        if self.signal_type == SignalType.BUY:
            self.pnl = (exit_price - self.entry_price) * self.quantity
        else:  # SELL/SHORT
            self.pnl = (self.entry_price - exit_price) * self.quantity


class BacktestEngine:
    """Engine for backtesting trading strategies"""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,  # 0.1% commission
        slippage: float = 0.0001,  # 0.01% slippage
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        # State tracking
        self.current_capital = initial_capital
        self.positions = []
        self.trades = []
        self.portfolio_values = []
        self.timestamps = []

    def run_backtest(
        self,
        data: pd.DataFrame,
        predictions: List[float],
        price_column: str = "close",
        strategy: str = "simple_threshold",
        **strategy_params,
    ) -> Dict[str, Any]:
        """Run backtest with given predictions and strategy"""

        if len(predictions) != len(data):
            raise ValueError("Predictions length must match data length")

        # Reset state
        self._reset_state()

        # Generate signals based on strategy
        signals = self._generate_signals(data, predictions, strategy, **strategy_params)

        # Execute trades
        for i, (timestamp, price, signal, prediction) in enumerate(
            zip(data.index, data[price_column], signals, predictions)
        ):
            self._process_signal(timestamp, price, signal, prediction)

            # Track portfolio value
            portfolio_value = self._calculate_portfolio_value(price)
            self.portfolio_values.append(portfolio_value)
            self.timestamps.append(timestamp)

        # Close any remaining positions
        if self.positions:
            final_price = data[price_column].iloc[-1]
            final_timestamp = data.index[-1]
            for position in self.positions:
                position.close_trade(final_timestamp, final_price)
                self.trades.append(position)
            self.positions = []

        # Calculate metrics
        metrics = self._calculate_metrics(data, price_column)

        return {
            "portfolio_values": self.portfolio_values,
            "timestamps": self.timestamps,
            "trades": self.trades,
            "signals": signals,
            "metrics": metrics,
            "final_capital": self.current_capital,
        }

    def _reset_state(self):
        """Reset backtest state"""
        self.current_capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.portfolio_values = []
        self.timestamps = []

    def _generate_signals(
        self,
        data: pd.DataFrame,
        predictions: List[float],
        strategy: str,
        **strategy_params,
    ) -> List[SignalType]:
        """Generate trading signals based on strategy"""

        if strategy == "simple_threshold":
            return self._simple_threshold_strategy(data, predictions, **strategy_params)
        elif strategy == "momentum":
            return self._momentum_strategy(data, predictions, **strategy_params)
        elif strategy == "mean_reversion":
            return self._mean_reversion_strategy(data, predictions, **strategy_params)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _simple_threshold_strategy(
        self,
        data: pd.DataFrame,
        predictions: List[float],
        price_column: str = "close",
        buy_threshold: float = 0.02,  # 2% predicted increase
        sell_threshold: float = -0.02,  # 2% predicted decrease
    ) -> List[SignalType]:
        """Simple threshold-based strategy"""

        signals = []
        current_prices = data[price_column].values

        for i, (current_price, predicted_price) in enumerate(
            zip(current_prices, predictions)
        ):
            if i == 0:
                signals.append(SignalType.HOLD)
                continue

            # Calculate predicted return
            predicted_return = (predicted_price - current_price) / current_price

            if predicted_return > buy_threshold:
                signals.append(SignalType.BUY)
            elif predicted_return < sell_threshold:
                signals.append(SignalType.SELL)
            else:
                signals.append(SignalType.HOLD)

        return signals

    def _momentum_strategy(
        self,
        data: pd.DataFrame,
        predictions: List[float],
        price_column: str = "close",
        lookback: int = 5,
        threshold: float = 0.01,
    ) -> List[SignalType]:
        """Momentum-based strategy"""

        signals = []
        current_prices = data[price_column].values

        for i in range(len(predictions)):
            if i < lookback:
                signals.append(SignalType.HOLD)
                continue

            # Calculate momentum
            price_momentum = (
                current_prices[i] - current_prices[i - lookback]
            ) / current_prices[i - lookback]
            pred_momentum = (predictions[i] - current_prices[i]) / current_prices[i]

            # Align with momentum
            if price_momentum > threshold and pred_momentum > 0:
                signals.append(SignalType.BUY)
            elif price_momentum < -threshold and pred_momentum < 0:
                signals.append(SignalType.SELL)
            else:
                signals.append(SignalType.HOLD)

        return signals

    def _mean_reversion_strategy(
        self,
        data: pd.DataFrame,
        predictions: List[float],
        price_column: str = "close",
        window: int = 20,
        std_threshold: float = 2.0,
    ) -> List[SignalType]:
        """Mean reversion strategy"""

        signals = []
        current_prices = data[price_column].values

        # Calculate rolling mean and std
        price_series = pd.Series(current_prices)
        rolling_mean = price_series.rolling(window=window).mean()
        rolling_std = price_series.rolling(window=window).std()

        for i, (current_price, predicted_price, mean_price, std_price) in enumerate(
            zip(current_prices, predictions, rolling_mean, rolling_std)
        ):
            if i < window or pd.isna(mean_price) or pd.isna(std_price):
                signals.append(SignalType.HOLD)
                continue

            # Calculate z-score
            z_score = (current_price - mean_price) / std_price
            pred_direction = 1 if predicted_price > current_price else -1

            # Mean reversion logic
            if z_score > std_threshold and pred_direction < 0:
                signals.append(SignalType.SELL)  # Price too high, expecting reversion
            elif z_score < -std_threshold and pred_direction > 0:
                signals.append(SignalType.BUY)  # Price too low, expecting reversion
            else:
                signals.append(SignalType.HOLD)

        return signals

    def _process_signal(
        self,
        timestamp: pd.Timestamp,
        price: float,
        signal: SignalType,
        prediction: float,
    ):
        """Process a trading signal"""

        if signal == SignalType.HOLD:
            return

        # Apply slippage
        adjusted_price = (
            price * (1 + self.slippage)
            if signal == SignalType.BUY
            else price * (1 - self.slippage)
        )

        # Calculate position size (simple: use fixed percentage of capital)
        position_size = self.current_capital * 0.1  # 10% of capital per trade
        quantity = position_size / adjusted_price

        # Calculate commission
        commission_cost = position_size * self.commission

        if signal == SignalType.BUY:
            # Open long position
            if self.current_capital >= position_size + commission_cost:
                trade = Trade(
                    entry_time=timestamp,
                    exit_time=None,
                    entry_price=adjusted_price,
                    exit_price=None,
                    quantity=quantity,
                    signal_type=signal,
                )
                self.positions.append(trade)
                self.current_capital -= position_size + commission_cost

        elif signal == SignalType.SELL:
            # Close long positions or open short (simplified: just close longs)
            if self.positions:
                for position in self.positions[:]:
                    if position.signal_type == SignalType.BUY:
                        # Close long position
                        position.close_trade(timestamp, adjusted_price)

                        # Calculate proceeds
                        proceeds = position.quantity * adjusted_price
                        commission_cost = proceeds * self.commission

                        self.current_capital += proceeds - commission_cost
                        self.trades.append(position)
                        self.positions.remove(position)
                        break

    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value"""

        portfolio_value = self.current_capital

        # Add value of open positions
        for position in self.positions:
            if position.signal_type == SignalType.BUY:
                position_value = position.quantity * current_price
                portfolio_value += position_value

        return portfolio_value

    def _calculate_metrics(
        self, data: pd.DataFrame, price_column: str
    ) -> Dict[str, float]:
        """Calculate performance metrics"""

        if not self.portfolio_values:
            return {}

        portfolio_series = pd.Series(self.portfolio_values)

        # Basic metrics
        total_return = (
            portfolio_series.iloc[-1] - portfolio_series.iloc[0]
        ) / portfolio_series.iloc[0]

        # Volatility (annualized)
        returns = portfolio_series.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Assuming daily data

        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe_ratio = (
            returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        )

        # Maximum drawdown
        peak = portfolio_series.expanding().max()
        drawdown = (portfolio_series - peak) / peak
        max_drawdown = drawdown.min()

        # Win rate
        winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0

        # Average win/loss
        wins = [t.pnl for t in self.trades if t.pnl and t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl and t.pnl < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        # Buy and hold comparison
        buy_hold_return = (
            data[price_column].iloc[-1] - data[price_column].iloc[0]
        ) / data[price_column].iloc[0]

        return {
            "total_return": total_return,
            "buy_hold_return": buy_hold_return,
            "excess_return": total_return - buy_hold_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_trades": len(self.trades),
            "final_portfolio_value": portfolio_series.iloc[-1],
        }

    def run_comprehensive_backtest(
        self,
        model_type,
        data: pd.DataFrame,
        lookback_periods: List[int] = [7, 14, 30],
        metrics_to_calculate: List[str] = None,
    ) -> Dict[str, Any]:
        """Run comprehensive backtest for a model type with various lookback periods"""

        if metrics_to_calculate is None:
            metrics_to_calculate = [
                "sharpe_ratio",
                "max_drawdown",
                "win_rate",
                "profit_factor",
                "total_return",
            ]

        # Generate dummy predictions for now (in real implementation, use actual model)
        predictions = [
            data["close"].iloc[i] * (1 + np.random.normal(0, 0.01))
            for i in range(len(data))
        ]

        # Run basic backtest
        results = self.run_backtest(
            data=data,
            predictions=predictions,
            price_column="close",
            strategy="simple_threshold",
            threshold=0.01,
        )

        # Add lookback-specific analysis
        for period in lookback_periods:
            if len(data) >= period:
                lookback_data = data.tail(period)
                period_return = (
                    lookback_data["close"].iloc[-1] / lookback_data["close"].iloc[0] - 1
                ) * 100
                results[f"return_{period}d"] = period_return

        return results


class HyperparameterTuner:
    """Utility for running hyperparameter experiments"""

    def __init__(self, backtest_engine: BacktestEngine):
        self.backtest_engine = backtest_engine
        self.results = []

    def run_grid_search(
        self,
        data: pd.DataFrame,
        predictions_func,  # Function that returns predictions given parameters
        param_grid: Dict[str, List[Any]],
        strategy: str = "simple_threshold",
        **fixed_strategy_params,
    ) -> pd.DataFrame:
        """Run grid search over hyperparameters"""

        from itertools import product

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for param_combo in product(*param_values):
            param_dict = dict(zip(param_names, param_combo))

            try:
                # Generate predictions with these parameters
                predictions = predictions_func(**param_dict)

                # Run backtest
                backtest_result = self.backtest_engine.run_backtest(
                    data=data,
                    predictions=predictions,
                    strategy=strategy,
                    **fixed_strategy_params,
                )

                # Store results
                result_row = param_dict.copy()
                result_row.update(backtest_result["metrics"])
                self.results.append(result_row)

            except Exception as e:
                print(f"Error with parameters {param_dict}: {e}")
                continue

        return pd.DataFrame(self.results)

    def get_best_params(
        self, metric: str = "total_return", maximize: bool = True
    ) -> Dict[str, Any]:
        """Get best parameters based on a metric"""

        if not self.results:
            raise ValueError("No results available. Run grid search first.")

        results_df = pd.DataFrame(self.results)

        if maximize:
            best_idx = results_df[metric].idxmax()
        else:
            best_idx = results_df[metric].idxmin()

        return results_df.loc[best_idx].to_dict()


def simple_moving_average_strategy(
    data: pd.DataFrame,
    short_window: int = 10,
    long_window: int = 30,
    price_column: str = "close",
) -> List[float]:
    """Simple moving average crossover strategy predictions"""

    prices = data[price_column]
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()

    # Generate "predictions" based on MA crossover
    predictions = []
    for i in range(len(prices)):
        if i < long_window:
            predictions.append(prices.iloc[i])
        else:
            # If short MA > long MA, predict price increase
            if short_ma.iloc[i] > long_ma.iloc[i]:
                predictions.append(prices.iloc[i] * 1.01)  # 1% increase
            else:
                predictions.append(prices.iloc[i] * 0.99)  # 1% decrease

    return predictions


class HyperparameterTuner:
    """Hyperparameter tuning utility for time series models"""

    def __init__(self):
        pass

    def grid_search(
        self,
        model_type,
        data: pd.DataFrame,
        param_grid: Dict[str, List],
        cv_folds: int = 3,
        metric: str = "rmse",
    ) -> Dict[str, Any]:
        """Run grid search over hyperparameters"""

        from itertools import product

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        results = []

        for param_combo in product(*param_values):
            param_dict = dict(zip(param_names, param_combo))

            # For now, return dummy results (in real implementation, train model)
            # This would normally involve training the model with these params
            dummy_score = np.random.random() * 100  # Dummy RMSE score

            results.append(
                {
                    "params": param_dict,
                    "score": dummy_score,
                    "cv_scores": [
                        dummy_score + np.random.normal(0, 5) for _ in range(cv_folds)
                    ],
                }
            )

        # Sort by score (assuming lower is better for RMSE)
        results.sort(key=lambda x: x["score"])

        return {
            "best_params": results[0]["params"],
            "best_score": results[0]["score"],
            "all_results": results,
            "param_grid": param_grid,
        }
