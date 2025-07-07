# services/backtesting_service.py

"""
Backtesting Service - Main business logic for backtesting operations.
Implements Aggregator pattern to coordinate between strategies, engines, and data.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from ..interfaces.backtesting_engine import BacktestingEngine
from ..schemas.backtesting_schemas import (
    BacktestRequest,
    BacktestResult,
    StrategyConfig,
    StrategyType,
    BacktestSummary,
)
from ..schemas.ohlcv_schemas import OHLCVSchema
from ..services.market_data_service import MarketDataService
from ..services.backtesting_data_service import BacktestingDataService
from ..strategies.strategy_factory import StrategyFactory
from ..common.logger import LoggerFactory


class BacktestingServiceError(Exception):
    """Base exception for backtesting service errors."""

    pass


class BacktestingService:
    """
    Backtesting service coordinating the entire backtesting workflow.

    This service acts as an aggregator, orchestrating:
    - Market data retrieval
    - Strategy instantiation
    - Engine execution
    - Result processing
    - Result storage
    """

    def __init__(
        self,
        market_data_service: MarketDataService,
        backtesting_engine: BacktestingEngine,
        backtesting_data_service: Optional[BacktestingDataService] = None,
    ):
        """
        Initialize backtesting service.

        Args:
            market_data_service: Service for market data operations
            backtesting_engine: Backtesting engine implementation
            backtesting_data_service: Optional service for result storage
        """
        self.market_data_service = market_data_service
        self.backtesting_engine = backtesting_engine
        self.backtesting_data_service = backtesting_data_service
        self.logger = LoggerFactory.get_logger(name="backtesting_service")

        self.logger.info("BacktestingService initialized")

    async def run_backtest(self, request: BacktestRequest) -> BacktestResult:
        """
        Execute a complete backtest workflow.

        Args:
            request: Backtest request with all parameters

        Returns:
            Comprehensive backtest results

        Raises:
            BacktestingServiceError: If backtest execution fails
        """
        self.logger.info(
            f"Starting backtest for {request.symbol} ({request.timeframe}) "
            f"from {request.start_date} to {request.end_date}"
        )

        try:
            # Step 1: Validate request
            self._validate_backtest_request(request)

            # Step 2: Retrieve market data
            market_data = await self._get_market_data(request)

            if not market_data:
                raise BacktestingServiceError(
                    "No market data available for the specified period"
                )

            self.logger.info(
                f"Retrieved {len(market_data)} data points for backtesting"
            )

            # Step 3: Create strategy configuration
            strategy_config = StrategyConfig(
                strategy_type=request.strategy_type,
                parameters=request.strategy_params,
                position_sizing=request.strategy_params.get("position_sizing", "fixed"),
                position_size=request.strategy_params.get("position_size", 1.0),
            )

            # Step 4: Validate strategy configuration
            self._validate_strategy_config(strategy_config)

            # Step 5: Execute backtest using engine
            result = await self.backtesting_engine.run_backtest(
                request=request,
                market_data=market_data,
                strategy_config=strategy_config,
            )

            # Step 6: Post-process results
            self._post_process_results(result, request)

            # Step 7: Save results if data service is available
            if self.backtesting_data_service:
                try:
                    backtest_id = (
                        await self.backtesting_data_service.save_backtest_result(
                            result=result,
                            metadata={
                                "request_timestamp": datetime.now(),
                                "execution_time_seconds": getattr(
                                    result, "execution_time_seconds", 0
                                ),
                                "engine_used": "backtrader",
                            },
                        )
                    )
                    # Add the backtest ID to the result
                    result.backtest_id = backtest_id
                    self.logger.info(f"Saved backtest result with ID: {backtest_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to save backtest result: {e}")
                    # Continue without saving - don't fail the entire backtest

            self.logger.info(
                f"Backtest completed successfully. Total return: {result.metrics.total_return:.2f}%"
            )

            return result

        except Exception as e:
            self.logger.error(f"Backtest execution failed: {e}")
            raise BacktestingServiceError(f"Backtest failed: {str(e)}")

    async def run_multiple_backtests(
        self, requests: List[BacktestRequest]
    ) -> List[BacktestResult]:
        """
        Execute multiple backtests in sequence.

        Args:
            requests: List of backtest requests

        Returns:
            List of backtest results
        """
        results = []

        for i, request in enumerate(requests):
            self.logger.info(f"Executing backtest {i+1}/{len(requests)}")

            try:
                result = await self.run_backtest(request)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Backtest {i+1} failed: {e}")
                # Continue with other backtests
                continue

        return results

    async def compare_strategies(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        strategy_configs: List[Dict[str, Any]],
        **common_params,
    ) -> List[BacktestResult]:
        """
        Compare multiple strategies on the same data.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Backtest start date
            end_date: Backtest end date
            strategy_configs: List of strategy configurations
            **common_params: Common parameters for all backtests

        Returns:
            List of backtest results for comparison
        """
        requests = []

        for config in strategy_configs:
            request = BacktestRequest(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                strategy_type=StrategyType(config["strategy_type"]),
                strategy_params=config.get("parameters", {}),
                **common_params,
            )
            requests.append(request)

        return await self.run_multiple_backtests(requests)

    def get_strategy_recommendations(
        self, symbol: str, market_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get strategy recommendations based on symbol and market conditions.

        Args:
            symbol: Trading symbol
            market_conditions: Current market conditions (optional)

        Returns:
            List of recommended strategy configurations
        """
        recommendations = []

        # Get all supported strategies
        supported_strategies = StrategyFactory.get_supported_strategies()

        for strategy_type in supported_strategies:
            try:
                # Get default parameters
                default_params = StrategyFactory.get_default_parameters(strategy_type)
                strategy_info = StrategyFactory.get_strategy_info(strategy_type)

                recommendation = {
                    "strategy_type": strategy_type.value,
                    "name": strategy_info.get("name", strategy_type.value),
                    "description": strategy_info.get(
                        "description", "No description available"
                    ),
                    "default_parameters": default_params,
                    "parameter_constraints": strategy_info.get(
                        "parameter_constraints", {}
                    ),
                    "recommended_for": self._get_strategy_market_fit(
                        strategy_type, market_conditions
                    ),
                }

                recommendations.append(recommendation)

            except Exception as e:
                self.logger.warning(
                    f"Could not get recommendation for {strategy_type}: {e}"
                )
                continue

        return recommendations

    async def _get_market_data(self, request: BacktestRequest) -> List[OHLCVSchema]:
        """
        Retrieve market data for backtesting.

        Args:
            request: Backtest request

        Returns:
            List of OHLCV data points
        """
        try:
            # Query market data using the market data service
            market_data = self.market_data_service.get_ohlcv_data(
                exchange=request.exchange,
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=request.start_date.isoformat(),
                end_date=request.end_date.isoformat(),
            )

            return market_data

        except Exception as e:
            raise BacktestingServiceError(f"Failed to retrieve market data: {str(e)}")

    def _validate_backtest_request(self, request: BacktestRequest) -> None:
        """
        Validate backtest request parameters.

        Args:
            request: Backtest request to validate

        Raises:
            BacktestingServiceError: If request is invalid
        """
        # Date validation
        if request.start_date >= request.end_date:
            raise BacktestingServiceError("Start date must be before end date")

        # Ensure reasonable time period
        max_period = timedelta(days=365 * 10)  # 10 years max
        if request.end_date - request.start_date > max_period:
            raise BacktestingServiceError("Backtest period too long (max 10 years)")

        # Capital validation
        if request.initial_capital <= 0:
            raise BacktestingServiceError("Initial capital must be positive")

        # Commission validation
        if request.commission < 0 or request.commission > 1:
            raise BacktestingServiceError("Commission must be between 0 and 1")

        # Strategy type validation
        supported_strategies = self.backtesting_engine.get_supported_strategies()
        if request.strategy_type.value not in supported_strategies:
            raise BacktestingServiceError(
                f"Strategy {request.strategy_type} not supported by engine"
            )

    def _validate_strategy_config(self, config: StrategyConfig) -> None:
        """
        Validate strategy configuration.

        Args:
            config: Strategy configuration to validate

        Raises:
            BacktestingServiceError: If configuration is invalid
        """
        try:
            self.backtesting_engine.validate_strategy_config(config)
        except Exception as e:
            raise BacktestingServiceError(f"Invalid strategy configuration: {str(e)}")

    def _post_process_results(
        self, result: BacktestResult, request: BacktestRequest
    ) -> None:
        """
        Post-process backtest results for additional insights.

        Args:
            result: Backtest result to process
            request: Original request parameters
        """
        # Add any warnings based on results
        if result.metrics.total_trades == 0:
            result.warnings.append("No trades executed during backtest period")

        if result.metrics.max_drawdown > 50:
            result.warnings.append("High maximum drawdown detected (>50%)")

        if result.metrics.total_return < -90:
            result.warnings.append("Significant losses detected (>90% loss)")

        # Validate result consistency
        if (
            abs(
                result.final_capital
                - result.initial_capital * (1 + result.metrics.total_return / 100)
            )
            > 0.01
        ):
            result.warnings.append("Inconsistency detected in capital calculations")

    def _get_strategy_market_fit(
        self, strategy_type: StrategyType, market_conditions: Optional[Dict[str, Any]]
    ) -> List[str]:
        """
        Get market conditions where strategy performs well.

        Args:
            strategy_type: Strategy type
            market_conditions: Current market conditions

        Returns:
            List of market condition descriptions
        """
        # Simple heuristics for strategy recommendations
        recommendations = {
            StrategyType.MOVING_AVERAGE_CROSSOVER: [
                "Trending markets",
                "Medium to low volatility",
                "Clear directional moves",
            ],
            StrategyType.RSI_STRATEGY: [
                "Range-bound markets",
                "Mean-reverting conditions",
                "High volatility with clear support/resistance",
            ],
            StrategyType.SIMPLE_BUY_HOLD: [
                "Long-term bull markets",
                "Low volatility",
                "Strong fundamental growth",
            ],
        }

        return recommendations.get(strategy_type, ["General market conditions"])

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service information and capabilities.

        Returns:
            Dictionary containing service metadata
        """
        return {
            "name": "BacktestingService",
            "version": "1.0.0",
            "description": "Comprehensive backtesting service for crypto trading strategies",
            "engine_info": self.backtesting_engine.get_engine_info(),
            "supported_strategies": self.backtesting_engine.get_supported_strategies(),
            "features": [
                "Multiple strategy support",
                "Comprehensive performance metrics",
                "Risk analysis",
                "Strategy comparison",
                "Market data integration",
                "Detailed trade analysis",
            ],
        }

    def create_backtest_summary(self, result: BacktestResult) -> BacktestSummary:
        """
        Create a summary view of backtest results.

        Args:
            result: Full backtest result

        Returns:
            Backtest summary
        """
        return BacktestSummary(
            symbol=result.symbol,
            timeframe=result.timeframe,
            strategy_type=result.strategy_type,
            total_return=result.metrics.total_return,
            annual_return=result.metrics.annual_return,
            max_drawdown=result.metrics.max_drawdown,
            sharpe_ratio=result.metrics.sharpe_ratio,
            total_trades=result.metrics.total_trades,
            win_rate=result.metrics.win_rate,
            start_date=result.start_date,
            end_date=result.end_date,
        )
