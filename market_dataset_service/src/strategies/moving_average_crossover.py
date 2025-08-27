# strategies/moving_average_crossover.py

"""
Moving Average Crossover Strategy Implementation.
"""

from typing import Any, Dict, List

from ..schemas.ohlcv_schemas import OHLCVSchema
from .base_strategy import BaseStrategy


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy.

    Generates buy signals when fast MA crosses above slow MA,
    and sell signals when fast MA crosses below slow MA.
    """

    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this strategy."""
        return ["fast_period", "slow_period"]

    def get_parameter_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter constraints for validation."""
        return {
            "fast_period": {
                "type": int,
                "min": 1,
                "max": 200,
                "description": "Fast moving average period",
            },
            "slow_period": {
                "type": int,
                "min": 2,
                "max": 500,
                "description": "Slow moving average period",
            },
            "ma_type": {
                "type": str,
                "choices": ["sma", "ema"],
                "description": "Moving average type (simple or exponential)",
            },
        }

    def generate_signals(
        self, data: List[OHLCVSchema], current_index: int
    ) -> Dict[str, Any]:
        """
        Generate trading signals based on MA crossover.

        Args:
            data: Historical OHLCV data
            current_index: Current position in data

        Returns:
            Signal dictionary with action and reasoning
        """
        # Validate we have enough data
        fast_period = self.parameters["fast_period"]
        slow_period = self.parameters["slow_period"]
        ma_type = self.parameters.get("ma_type", "sma")

        if current_index < slow_period:
            return {
                "action": "hold",
                "reason": f"Insufficient data (need {slow_period} periods)",
                "confidence": 0.0,
            }

        # Extract closing prices up to current index
        prices = [candle.close for candle in data[: current_index + 1]]

        # Calculate moving averages
        if ma_type == "ema":
            fast_ma = self.calculate_ema(prices, fast_period)
            slow_ma = self.calculate_ema(prices, slow_period)
        else:  # sma
            fast_ma = self.calculate_sma(prices, fast_period)
            slow_ma = self.calculate_sma(prices, slow_period)

        # Store MA values for crossover detection
        if not hasattr(self, "fast_ma_history"):
            self.fast_ma_history = []
            self.slow_ma_history = []

        self.fast_ma_history.append(fast_ma)
        self.slow_ma_history.append(slow_ma)

        # Check for crossover signals
        if len(self.fast_ma_history) >= 2:
            # Bullish crossover: fast MA crosses above slow MA
            if self.is_bullish_crossover(self.fast_ma_history, self.slow_ma_history):
                if self.current_position == 0:  # Not in position
                    return {
                        "action": "buy",
                        "reason": f"Bullish crossover: Fast MA ({fast_ma:.4f}) > Slow MA ({slow_ma:.4f})",
                        "confidence": 0.8,
                        "fast_ma": fast_ma,
                        "slow_ma": slow_ma,
                    }

            # Bearish crossover: fast MA crosses below slow MA
            if self.is_bearish_crossover(self.fast_ma_history, self.slow_ma_history):
                if self.current_position > 0:  # In long position
                    return {
                        "action": "sell",
                        "reason": f"Bearish crossover: Fast MA ({fast_ma:.4f}) < Slow MA ({slow_ma:.4f})",
                        "confidence": 0.8,
                        "fast_ma": fast_ma,
                        "slow_ma": slow_ma,
                    }

        # No signal
        return {
            "action": "hold",
            "reason": f"No crossover signal (Fast MA: {fast_ma:.4f}, Slow MA: {slow_ma:.4f})",
            "confidence": 0.0,
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
        }
