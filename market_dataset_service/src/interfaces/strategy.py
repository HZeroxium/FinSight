# interfaces/strategy.py

"""
Strategy interface for trading strategy implementations.
Part of Strategy Pattern implementation.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..schemas.backtesting_schemas import StrategyConfig, TradeResult
from ..schemas.ohlcv_schemas import OHLCVSchema


class StrategyError(Exception):
    """Base exception for strategy errors."""

    pass


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Implements Strategy Pattern to allow pluggable trading algorithms.
    Each strategy defines its own logic for generating buy/sell signals.
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize strategy with configuration.

        Args:
            config: Strategy configuration including parameters
        """
        self.config = config
        self.parameters = config.parameters
        self.name = config.strategy_type.value

        # Strategy state
        self.current_position = 0.0  # Current position size
        self.entry_price = 0.0  # Entry price for current position
        self.cash = 0.0  # Available cash
        self.portfolio_value = 0.0  # Total portfolio value

        # Performance tracking
        self.trades: List[TradeResult] = []
        self.signals_history: List[Dict[str, Any]] = []

    @abstractmethod
    def generate_signals(
        self, data: List[OHLCVSchema], current_index: int
    ) -> Dict[str, Any]:
        """
        Generate trading signals based on market data.

        Args:
            data: Historical OHLCV data up to current point
            current_index: Current position in the data series

        Returns:
            Dictionary containing signal information:
            - action: 'buy', 'sell', 'hold'
            - size: Position size (optional)
            - price: Target price (optional)
            - reason: Signal reasoning (for debugging)
            - confidence: Signal confidence 0.0-1.0 (optional)

        Raises:
            StrategyError: If signal generation fails
        """
        pass

    @abstractmethod
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.

        Returns:
            True if parameters are valid

        Raises:
            StrategyError: If parameters are invalid
        """
        pass

    @abstractmethod
    def get_required_parameters(self) -> List[str]:
        """
        Get list of required parameter names.

        Returns:
            List of required parameter names
        """
        pass

    @abstractmethod
    def get_parameter_constraints(self) -> Dict[str, Dict[str, Any]]:
        """
        Get parameter constraints for validation.

        Returns:
            Dictionary mapping parameter names to their constraints:
            - type: Parameter type (int, float, str, bool)
            - min: Minimum value (for numeric types)
            - max: Maximum value (for numeric types)
            - choices: Valid choices (for categorical types)
            - description: Parameter description
        """
        pass

    def initialize(self, initial_capital: float) -> None:
        """
        Initialize strategy state for backtesting.

        Args:
            initial_capital: Starting capital amount
        """
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        self.current_position = 0.0
        self.entry_price = 0.0
        self.trades.clear()
        self.signals_history.clear()

    def update_portfolio_value(self, current_price: float) -> None:
        """
        Update current portfolio value based on market price.

        Args:
            current_price: Current market price
        """
        position_value = self.current_position * current_price
        self.portfolio_value = self.cash + position_value

    def execute_trade(
        self,
        action: str,
        price: float,
        timestamp: datetime,
        size: Optional[float] = None,
        reason: str = "signal",
    ) -> Optional[TradeResult]:
        """
        Execute a trade and update portfolio state.

        Args:
            action: 'buy' or 'sell'
            price: Execution price
            timestamp: Trade timestamp
            size: Position size (if None, use default sizing)
            reason: Reason for trade

        Returns:
            TradeResult if trade executed, None otherwise
        """
        if action not in ["buy", "sell"]:
            return None

        # Default position sizing if not specified
        if size is None:
            size = self._calculate_position_size(price)

        if action == "buy" and self.current_position == 0:
            # Open long position
            cost = size * price
            if cost <= self.cash:
                self.current_position = size
                self.entry_price = price
                self.cash -= cost

                trade = TradeResult(
                    entry_date=timestamp,
                    entry_price=price,
                    position_side="long",
                    quantity=size,
                    entry_reason=reason,
                )
                self.trades.append(trade)
                return trade

        elif action == "sell" and self.current_position > 0:
            # Close long position
            proceeds = self.current_position * price
            self.cash += proceeds

            # Update the last trade
            if self.trades:
                last_trade = self.trades[-1]
                last_trade.exit_date = timestamp
                last_trade.exit_price = price
                last_trade.pnl = proceeds - (
                    last_trade.quantity * last_trade.entry_price
                )
                last_trade.pnl_percentage = (
                    last_trade.pnl / (last_trade.quantity * last_trade.entry_price)
                ) * 100
                last_trade.exit_reason = reason
                last_trade.is_open = False

            self.current_position = 0.0
            self.entry_price = 0.0

            return self.trades[-1] if self.trades else None

        return None

    def _calculate_position_size(self, price: float) -> float:
        """
        Calculate position size based on strategy configuration.

        Args:
            price: Current market price

        Returns:
            Position size
        """
        sizing_method = self.config.position_sizing or "fixed"
        size_value = self.config.position_size

        if sizing_method == "fixed":
            return size_value
        elif sizing_method == "percentage":
            # Use percentage of available cash
            available_cash = self.cash
            return (available_cash * size_value) / price
        else:
            # Default to fixed size
            return size_value

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information and current state.

        Returns:
            Dictionary containing strategy metadata
        """
        return {
            "name": self.name,
            "parameters": self.parameters,
            "current_position": self.current_position,
            "entry_price": self.entry_price,
            "cash": self.cash,
            "portfolio_value": self.portfolio_value,
            "total_trades": len(self.trades),
            "open_trades": len([t for t in self.trades if t.is_open]),
        }
