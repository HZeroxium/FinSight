# strategies/macd_strategy.py

"""
MACD (Moving Average Convergence Divergence) Trading Strategy Implementation.
"""

from typing import Any, Dict, List

from ..schemas.ohlcv_schemas import OHLCVSchema
from .base_strategy import BaseStrategy


class MACDStrategy(BaseStrategy):
    """
    MACD Trading Strategy.

    Generates buy signals when MACD line crosses above signal line,
    and sell signals when MACD line crosses below signal line.
    """

    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this strategy."""
        return ["fast_period", "slow_period", "signal_period"]

    def get_parameter_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter constraints for validation."""
        return {
            "fast_period": {
                "type": int,
                "min": 5,
                "max": 50,
                "description": "Fast EMA period for MACD calculation",
            },
            "slow_period": {
                "type": int,
                "min": 10,
                "max": 100,
                "description": "Slow EMA period for MACD calculation",
            },
            "signal_period": {
                "type": int,
                "min": 5,
                "max": 30,
                "description": "Signal line EMA period",
            },
            "histogram_threshold": {
                "type": (int, float),
                "min": 0.0,
                "max": 1.0,
                "description": "Histogram threshold for signal confirmation",
            },
        }

    def generate_signals(
        self, data: List[OHLCVSchema], current_index: int
    ) -> Dict[str, Any]:
        """
        Generate trading signals based on MACD.

        Args:
            data: Historical OHLCV data
            current_index: Current position in data

        Returns:
            Signal dictionary with action and reasoning
        """
        # Validate we have enough data
        fast_period = self.parameters["fast_period"]
        slow_period = self.parameters["slow_period"]
        signal_period = self.parameters["signal_period"]
        histogram_threshold = self.parameters.get("histogram_threshold", 0.0)

        min_periods = max(slow_period, signal_period) + signal_period
        if current_index < min_periods:
            return {
                "action": "hold",
                "reason": f"Insufficient data (need {min_periods} periods)",
                "confidence": 0.0,
            }

        # Extract closing prices up to current index
        prices = [candle.close for candle in data[: current_index + 1]]

        # Calculate MACD components
        macd_line, signal_line, histogram = self.calculate_macd(
            prices, fast_period, slow_period, signal_period
        )

        # Store MACD values for crossover detection
        if not hasattr(self, "macd_history"):
            self.macd_history = []
            self.signal_history = []
            self.histogram_history = []

        self.macd_history.append(macd_line)
        self.signal_history.append(signal_line)
        self.histogram_history.append(histogram)

        # Check for crossover signals
        if len(self.macd_history) >= 2:
            prev_macd = self.macd_history[-2]
            prev_signal = self.signal_history[-2]
            prev_histogram = self.histogram_history[-2]

            # Bullish crossover: MACD crosses above signal line
            if self._is_bullish_crossover(
                prev_macd, prev_signal, macd_line, signal_line
            ):
                # Additional confirmation with histogram
                if abs(histogram) >= histogram_threshold:
                    if self.current_position == 0:
                        return {
                            "action": "buy",
                            "reason": f"MACD bullish crossover: MACD ({macd_line:.6f}) > Signal ({signal_line:.6f})",
                            "confidence": 0.8,
                            "macd": macd_line,
                            "signal": signal_line,
                            "histogram": histogram,
                        }

            # Bearish crossover: MACD crosses below signal line
            elif self._is_bearish_crossover(
                prev_macd, prev_signal, macd_line, signal_line
            ):
                # Additional confirmation with histogram
                if abs(histogram) >= histogram_threshold:
                    if self.current_position > 0:
                        return {
                            "action": "sell",
                            "reason": f"MACD bearish crossover: MACD ({macd_line:.6f}) < Signal ({signal_line:.6f})",
                            "confidence": 0.8,
                            "macd": macd_line,
                            "signal": signal_line,
                            "histogram": histogram,
                        }

        # No clear signal
        return {
            "action": "hold",
            "reason": f"No clear MACD signal (MACD: {macd_line:.6f}, Signal: {signal_line:.6f})",
            "confidence": 0.5,
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        }

    def _is_bullish_crossover(
        self, prev_macd: float, prev_signal: float, curr_macd: float, curr_signal: float
    ) -> bool:
        """Check if MACD has crossed above signal line."""
        return prev_macd <= prev_signal and curr_macd > curr_signal

    def _is_bearish_crossover(
        self, prev_macd: float, prev_signal: float, curr_macd: float, curr_signal: float
    ) -> bool:
        """Check if MACD has crossed below signal line."""
        return prev_macd >= prev_signal and curr_macd < curr_signal

    def calculate_macd(
        self,
        prices: List[float],
        fast_period: int,
        slow_period: int,
        signal_period: int,
    ) -> tuple:
        """
        Calculate MACD indicator.

        Args:
            prices: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        if len(prices) < slow_period:
            return 0.0, 0.0, 0.0

        # Calculate EMAs
        fast_ema = self.calculate_ema(prices, fast_period)
        slow_ema = self.calculate_ema(prices, slow_period)

        # MACD line = Fast EMA - Slow EMA
        macd_line = fast_ema - slow_ema

        # Calculate signal line (EMA of MACD line)
        # For signal line calculation, we need MACD history
        if not hasattr(self, "_macd_values"):
            self._macd_values = []

        self._macd_values.append(macd_line)

        # Keep only the values we need for signal calculation
        if len(self._macd_values) > signal_period * 2:
            self._macd_values = self._macd_values[-signal_period * 2 :]

        if len(self._macd_values) >= signal_period:
            signal_line = self.calculate_ema(self._macd_values, signal_period)
        else:
            signal_line = macd_line  # Use MACD line as signal if not enough data

        # Histogram = MACD line - Signal line
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram
