# strategies/base_strategy.py

"""
Base strategy implementation providing common functionality.
"""

from typing import Dict, Any, List
from ..interfaces.strategy import Strategy, StrategyError
from ..schemas.backtesting_schemas import StrategyConfig


class BaseStrategy(Strategy):
    """
    Base strategy class with common functionality.

    Provides basic implementation of Strategy interface that can be extended
    by concrete strategy implementations.
    """

    def __init__(self, config: StrategyConfig):
        """Initialize base strategy."""
        super().__init__(config)
        self.indicators_cache: Dict[str, Any] = {}

    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters against requirements.

        Returns:
            True if all required parameters are present and valid

        Raises:
            StrategyError: If validation fails
        """
        required_params = self.get_required_parameters()
        constraints = self.get_parameter_constraints()

        # Check required parameters
        for param in required_params:
            if param not in self.parameters:
                raise StrategyError(f"Missing required parameter: {param}")

        # Validate parameter constraints
        for param, value in self.parameters.items():
            if param in constraints:
                constraint = constraints[param]

                # Type validation
                expected_type = constraint.get("type")
                if expected_type and not isinstance(value, expected_type):
                    raise StrategyError(
                        f"Parameter '{param}' must be of type {expected_type.__name__}"
                    )

                # Range validation for numeric types
                if isinstance(value, (int, float)):
                    min_val = constraint.get("min")
                    max_val = constraint.get("max")

                    if min_val is not None and value < min_val:
                        raise StrategyError(f"Parameter '{param}' must be >= {min_val}")

                    if max_val is not None and value > max_val:
                        raise StrategyError(f"Parameter '{param}' must be <= {max_val}")

                # Choice validation
                choices = constraint.get("choices")
                if choices and value not in choices:
                    raise StrategyError(
                        f"Parameter '{param}' must be one of: {choices}"
                    )

        return True

    def calculate_sma(self, prices: List[float], period: int) -> float:
        """
        Calculate Simple Moving Average.

        Args:
            prices: List of prices
            period: SMA period

        Returns:
            SMA value or 0 if insufficient data
        """
        if len(prices) < period:
            return 0.0

        return sum(prices[-period:]) / period

    def calculate_ema(self, prices: List[float], period: int) -> float:
        """
        Calculate Exponential Moving Average.

        Args:
            prices: List of prices
            period: EMA period

        Returns:
            EMA value or 0 if insufficient data
        """
        if len(prices) < period:
            return 0.0

        # Calculate initial SMA for first EMA value
        if len(prices) == period:
            return self.calculate_sma(prices, period)

        # Get previous EMA from cache or calculate
        cache_key = f"ema_{period}_{len(prices)-1}"
        prev_ema = self.indicators_cache.get(
            cache_key, self.calculate_sma(prices[:-1], period)
        )

        # Calculate EMA
        multiplier = 2 / (period + 1)
        ema = (prices[-1] * multiplier) + (prev_ema * (1 - multiplier))

        # Cache result
        cache_key = f"ema_{period}_{len(prices)}"
        self.indicators_cache[cache_key] = ema

        return ema

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """
        Calculate Relative Strength Index.

        Args:
            prices: List of prices
            period: RSI period (default 14)

        Returns:
            RSI value (0-100) or 50 if insufficient data
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI

        # Calculate price changes
        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        if len(changes) < period:
            return 50.0

        # Separate gains and losses
        gains = [change if change > 0 else 0 for change in changes[-period:]]
        losses = [-change if change < 0 else 0 for change in changes[-period:]]

        # Calculate average gain and loss
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_bollinger_bands(
        self, prices: List[float], period: int = 20, std_dev: float = 2.0
    ) -> Dict[str, float]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: List of prices
            period: Moving average period
            std_dev: Standard deviation multiplier

        Returns:
            Dictionary with 'upper', 'middle', 'lower' band values
        """
        if len(prices) < period:
            current_price = prices[-1] if prices else 0
            return {
                "upper": current_price,
                "middle": current_price,
                "lower": current_price,
            }

        # Calculate middle band (SMA)
        middle = self.calculate_sma(prices, period)

        # Calculate standard deviation
        recent_prices = prices[-period:]
        variance = sum((price - middle) ** 2 for price in recent_prices) / period
        std = variance**0.5

        # Calculate bands
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        return {"upper": upper, "middle": middle, "lower": lower}

    def is_bullish_crossover(self, fast_ma: List[float], slow_ma: List[float]) -> bool:
        """
        Check if fast MA crosses above slow MA (bullish signal).

        Args:
            fast_ma: Fast moving average values
            slow_ma: Slow moving average values

        Returns:
            True if bullish crossover detected
        """
        if len(fast_ma) < 2 or len(slow_ma) < 2:
            return False

        return fast_ma[-2] <= slow_ma[-2] and fast_ma[-1] > slow_ma[-1]

    def is_bearish_crossover(self, fast_ma: List[float], slow_ma: List[float]) -> bool:
        """
        Check if fast MA crosses below slow MA (bearish signal).

        Args:
            fast_ma: Fast moving average values
            slow_ma: Slow moving average values

        Returns:
            True if bearish crossover detected
        """
        if len(fast_ma) < 2 or len(slow_ma) < 2:
            return False

        return fast_ma[-2] >= slow_ma[-2] and fast_ma[-1] < slow_ma[-1]
