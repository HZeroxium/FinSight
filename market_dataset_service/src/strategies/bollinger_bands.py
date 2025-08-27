# strategies/bollinger_bands.py

"""
Bollinger Bands Trading Strategy Implementation.
"""

from typing import Any, Dict, List

from ..schemas.ohlcv_schemas import OHLCVSchema
from .base_strategy import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Trading Strategy.

    Generates buy signals when price touches or crosses below lower band,
    and sell signals when price touches or crosses above upper band.
    """

    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this strategy."""
        return ["bb_period", "bb_std_dev"]

    def get_parameter_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter constraints for validation."""
        return {
            "bb_period": {
                "type": int,
                "min": 5,
                "max": 100,
                "description": "Bollinger Bands period for moving average",
            },
            "bb_std_dev": {
                "type": (int, float),
                "min": 1.0,
                "max": 3.0,
                "description": "Standard deviation multiplier for bands",
            },
            "bb_position": {
                "type": str,
                "choices": ["mean_reversion", "breakout"],
                "description": "Strategy mode: mean reversion or breakout",
            },
        }

    def generate_signals(
        self, data: List[OHLCVSchema], current_index: int
    ) -> Dict[str, Any]:
        """
        Generate trading signals based on Bollinger Bands.

        Args:
            data: Historical OHLCV data
            current_index: Current position in data

        Returns:
            Signal dictionary with action and reasoning
        """
        # Validate we have enough data
        bb_period = self.parameters["bb_period"]
        bb_std_dev = self.parameters["bb_std_dev"]
        bb_position = self.parameters.get("bb_position", "mean_reversion")

        if current_index < bb_period:
            return {
                "action": "hold",
                "reason": f"Insufficient data (need {bb_period} periods)",
                "confidence": 0.0,
            }

        # Extract closing prices up to current index
        prices = [candle.close for candle in data[: current_index + 1]]
        current_price = prices[-1]

        # Calculate Bollinger Bands
        bb_middle, bb_upper, bb_lower = self.calculate_bollinger_bands(
            prices, bb_period, bb_std_dev
        )

        # Calculate price position within bands
        bb_position_percent = (current_price - bb_lower) / (bb_upper - bb_lower) * 100

        # Store BB values for trend analysis
        if not hasattr(self, "bb_history"):
            self.bb_history = []

        self.bb_history.append(
            {
                "upper": bb_upper,
                "middle": bb_middle,
                "lower": bb_lower,
                "price": current_price,
                "position_percent": bb_position_percent,
            }
        )

        # Generate signals based on strategy mode
        if bb_position == "mean_reversion":
            return self._mean_reversion_signal(
                current_price, bb_upper, bb_middle, bb_lower, bb_position_percent
            )
        else:  # breakout
            return self._breakout_signal(
                current_price, bb_upper, bb_middle, bb_lower, bb_position_percent
            )

    def _mean_reversion_signal(
        self,
        price: float,
        upper: float,
        middle: float,
        lower: float,
        position_percent: float,
    ) -> Dict[str, Any]:
        """Generate mean reversion signals."""

        # Buy when price is near lower band (oversold)
        if position_percent <= 10:  # Price is in bottom 10% of band
            if self.current_position == 0:
                return {
                    "action": "buy",
                    "reason": f"Mean reversion: Price ({price:.4f}) near lower band ({lower:.4f})",
                    "confidence": 0.7,
                    "bb_upper": upper,
                    "bb_middle": middle,
                    "bb_lower": lower,
                    "position_percent": position_percent,
                }

        # Sell when price is near upper band (overbought)
        elif position_percent >= 90:  # Price is in top 10% of band
            if self.current_position > 0:
                return {
                    "action": "sell",
                    "reason": f"Mean reversion: Price ({price:.4f}) near upper band ({upper:.4f})",
                    "confidence": 0.7,
                    "bb_upper": upper,
                    "bb_middle": middle,
                    "bb_lower": lower,
                    "position_percent": position_percent,
                }

        # Hold in middle range
        return {
            "action": "hold",
            "reason": f"Price in middle range ({position_percent:.1f}% of bands)",
            "confidence": 0.5,
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
            "position_percent": position_percent,
        }

    def _breakout_signal(
        self,
        price: float,
        upper: float,
        middle: float,
        lower: float,
        position_percent: float,
    ) -> Dict[str, Any]:
        """Generate breakout signals."""

        # Buy on upward breakout
        if price > upper:
            if self.current_position == 0:
                return {
                    "action": "buy",
                    "reason": f"Breakout: Price ({price:.4f}) above upper band ({upper:.4f})",
                    "confidence": 0.8,
                    "bb_upper": upper,
                    "bb_middle": middle,
                    "bb_lower": lower,
                    "position_percent": position_percent,
                }

        # Sell on downward breakout
        elif price < lower:
            if self.current_position > 0:
                return {
                    "action": "sell",
                    "reason": f"Breakout: Price ({price:.4f}) below lower band ({lower:.4f})",
                    "confidence": 0.8,
                    "bb_upper": upper,
                    "bb_middle": middle,
                    "bb_lower": lower,
                    "position_percent": position_percent,
                }

        # Hold within bands
        return {
            "action": "hold",
            "reason": f"Price within bands ({position_percent:.1f}%)",
            "confidence": 0.5,
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
            "position_percent": position_percent,
        }

    def calculate_bollinger_bands(
        self, prices: List[float], period: int, std_dev: float
    ) -> tuple:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier

        Returns:
            Tuple of (middle_band, upper_band, lower_band)
        """
        if len(prices) < period:
            return 0.0, 0.0, 0.0

        # Calculate simple moving average (middle band)
        recent_prices = prices[-period:]
        middle_band = sum(recent_prices) / len(recent_prices)

        # Calculate standard deviation
        variance = sum((price - middle_band) ** 2 for price in recent_prices) / len(
            recent_prices
        )
        std_deviation = variance**0.5

        # Calculate bands
        upper_band = middle_band + (std_dev * std_deviation)
        lower_band = middle_band - (std_dev * std_deviation)

        return middle_band, upper_band, lower_band
