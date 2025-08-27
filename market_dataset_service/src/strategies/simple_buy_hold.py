# strategies/simple_buy_hold.py

"""
Simple Buy and Hold Strategy Implementation.
"""

from typing import Any, Dict, List

from ..schemas.ohlcv_schemas import OHLCVSchema
from .base_strategy import BaseStrategy


class SimpleBuyHoldStrategy(BaseStrategy):
    """
    Simple Buy and Hold Strategy.

    Buys at the beginning and holds until the end of the backtest period.
    Useful as a baseline for comparing other strategies.
    """

    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this strategy."""
        return []  # No parameters required for buy and hold

    def get_parameter_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter constraints for validation."""
        return {
            "entry_delay": {
                "type": int,
                "min": 0,
                "max": 100,
                "description": "Number of periods to wait before buying (default: 0)",
            }
        }

    def generate_signals(
        self, data: List[OHLCVSchema], current_index: int
    ) -> Dict[str, Any]:
        """
        Generate trading signals for buy and hold strategy.

        Args:
            data: Historical OHLCV data
            current_index: Current position in data

        Returns:
            Signal dictionary with action and reasoning
        """
        entry_delay = self.parameters.get("entry_delay", 0)

        # Buy once at the beginning (after any specified delay)
        if current_index == entry_delay and self.current_position == 0:
            return {
                "action": "buy",
                "reason": f"Buy and hold entry at index {current_index}",
                "confidence": 1.0,
                "entry_point": True,
            }

        # Hold for the rest of the period
        return {
            "action": "hold",
            "reason": "Buy and hold - maintaining position",
            "confidence": 1.0,
            "holding": True,
        }
