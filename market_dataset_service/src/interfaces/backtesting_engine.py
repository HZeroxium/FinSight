# interfaces/backtesting_engine.py

"""
Backtesting engine interface for pluggable backtesting implementations.
Part of Ports & Adapters (Hexagonal Architecture) pattern.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..schemas.backtesting_schemas import (BacktestRequest, BacktestResult,
                                           StrategyConfig)
from ..schemas.ohlcv_schemas import OHLCVSchema


class BacktestingEngineError(Exception):
    """Base exception for backtesting engine errors."""

    pass


class BacktestingEngine(ABC):
    """
    Abstract backtesting engine interface.

    This port defines the contract for backtesting engines,
    allowing different implementations (Backtrader, VectorBT) to be plugged in.
    """

    @abstractmethod
    async def run_backtest(
        self,
        request: BacktestRequest,
        market_data: List[OHLCVSchema],
        strategy_config: StrategyConfig,
    ) -> BacktestResult:
        """
        Execute a backtest with given parameters.

        Args:
            request: Backtest configuration and parameters
            market_data: Historical OHLCV data for backtesting
            strategy_config: Strategy configuration and parameters

        Returns:
            BacktestResult: Comprehensive backtesting results

        Raises:
            BacktestingEngineError: If backtest execution fails
        """
        pass

    @abstractmethod
    def validate_strategy_config(self, strategy_config: StrategyConfig) -> bool:
        """
        Validate strategy configuration for this engine.

        Args:
            strategy_config: Strategy configuration to validate

        Returns:
            True if configuration is valid

        Raises:
            BacktestingEngineError: If configuration is invalid
        """
        pass

    @abstractmethod
    def get_supported_strategies(self) -> List[str]:
        """
        Get list of supported strategy types.

        Returns:
            List of strategy type names supported by this engine
        """
        pass

    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get engine information and capabilities.

        Returns:
            Dictionary containing engine metadata
        """
        pass
