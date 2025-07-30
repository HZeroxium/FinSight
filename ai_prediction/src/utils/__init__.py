# utils/__init__.py

"""
Utility modules for the FinSight Model Builder system.
"""

from .backtest_strategy_utils import BacktestEngine, HyperparameterTuner, SignalType, Trade
# Conditional import for visualization utils due to matplotlib dependency
try:
    from .visualization_utils import VisualizationUtils
except ImportError:
    VisualizationUtils = None

__all__ = [
    "BacktestEngine",
    "HyperparameterTuner",
    "SignalType",
    "Trade",
    "VisualizationUtils",
]
