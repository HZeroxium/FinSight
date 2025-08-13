# adapters/__init__.py

"""
Adapters package for external backtesting engines.
Implements Adapter Pattern for different backtesting libraries.
"""

from .backtrader_adapter import BacktraderAdapter

__all__ = ["BacktraderAdapter"]
