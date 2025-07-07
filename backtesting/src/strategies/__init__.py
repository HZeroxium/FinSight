# strategies/__init__.py

"""
Trading strategies package.
Contains implementations of different trading strategies following Strategy Pattern.
"""

from .base_strategy import BaseStrategy
from .moving_average_crossover import MovingAverageCrossoverStrategy
from .rsi_strategy import RSIStrategy
from .simple_buy_hold import SimpleBuyHoldStrategy
from .bollinger_bands import BollingerBandsStrategy
from .macd_strategy import MACDStrategy
from .strategy_factory import StrategyFactory

__all__ = [
    "BaseStrategy",
    "MovingAverageCrossoverStrategy",
    "RSIStrategy",
    "SimpleBuyHoldStrategy",
    "BollingerBandsStrategy",
    "MACDStrategy",
    "StrategyFactory",
]

"""
Trading strategies package.
Contains implementations of different trading strategies following Strategy Pattern.
"""

from .base_strategy import BaseStrategy
from .moving_average_crossover import MovingAverageCrossoverStrategy
from .rsi_strategy import RSIStrategy
from .simple_buy_hold import SimpleBuyHoldStrategy

__all__ = [
    "BaseStrategy",
    "MovingAverageCrossoverStrategy",
    "RSIStrategy",
    "SimpleBuyHoldStrategy",
]
