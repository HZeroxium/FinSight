# strategies/rsi_strategy.py

"""
RSI-based Trading Strategy Implementation.
"""

from typing import Dict, Any, List
from ..schemas.ohlcv_schemas import OHLCVSchema
from .base_strategy import BaseStrategy


class RSIStrategy(BaseStrategy):
    """
    RSI-based Trading Strategy.

    Generates buy signals when RSI is oversold (< oversold_threshold)
    and sell signals when RSI is overbought (> overbought_threshold).
    """

    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this strategy."""
        return ["rsi_period", "oversold_threshold", "overbought_threshold"]

    def get_parameter_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter constraints for validation."""
        return {
            "rsi_period": {
                "type": int,
                "min": 2,
                "max": 100,
                "description": "RSI calculation period",
            },
            "oversold_threshold": {
                "type": (int, float),
                "min": 0,
                "max": 50,
                "description": "RSI level considered oversold (buy signal)",
            },
            "overbought_threshold": {
                "type": (int, float),
                "min": 50,
                "max": 100,
                "description": "RSI level considered overbought (sell signal)",
            },
        }

    def generate_signals(
        self, data: List[OHLCVSchema], current_index: int
    ) -> Dict[str, Any]:
        """
        Generate trading signals based on RSI levels.

        Args:
            data: Historical OHLCV data
            current_index: Current position in data

        Returns:
            Signal dictionary with action and reasoning
        """
        rsi_period = self.parameters["rsi_period"]
        oversold_threshold = self.parameters["oversold_threshold"]
        overbought_threshold = self.parameters["overbought_threshold"]

        # Need at least rsi_period + 1 data points for RSI calculation
        if current_index < rsi_period:
            return {
                "action": "hold",
                "reason": f"Insufficient data for RSI calculation (need {rsi_period + 1} periods)",
                "confidence": 0.0,
            }

        # Extract closing prices up to current index
        prices = [candle.close for candle in data[: current_index + 1]]

        # Calculate RSI
        rsi = self.calculate_rsi(prices, rsi_period)

        # Store RSI history for trend analysis
        if not hasattr(self, "rsi_history"):
            self.rsi_history = []

        self.rsi_history.append(rsi)

        # Generate signals based on RSI levels
        if rsi <= oversold_threshold:
            if self.current_position == 0:  # Not in position
                # Additional confirmation: check if RSI is starting to turn up
                confidence = 0.7
                if (
                    len(self.rsi_history) >= 2
                    and self.rsi_history[-1] > self.rsi_history[-2]
                ):
                    confidence = 0.9  # Higher confidence if RSI is turning up

                return {
                    "action": "buy",
                    "reason": f"RSI oversold signal: RSI ({rsi:.2f}) <= {oversold_threshold}",
                    "confidence": confidence,
                    "rsi": rsi,
                    "threshold": oversold_threshold,
                }

        elif rsi >= overbought_threshold:
            if self.current_position > 0:  # In long position
                # Additional confirmation: check if RSI is starting to turn down
                confidence = 0.7
                if (
                    len(self.rsi_history) >= 2
                    and self.rsi_history[-1] < self.rsi_history[-2]
                ):
                    confidence = 0.9  # Higher confidence if RSI is turning down

                return {
                    "action": "sell",
                    "reason": f"RSI overbought signal: RSI ({rsi:.2f}) >= {overbought_threshold}",
                    "confidence": confidence,
                    "rsi": rsi,
                    "threshold": overbought_threshold,
                }

        # No signal
        return {
            "action": "hold",
            "reason": f"RSI in neutral zone: {rsi:.2f} (oversold: {oversold_threshold}, overbought: {overbought_threshold})",
            "confidence": 0.0,
            "rsi": rsi,
        }
