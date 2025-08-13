# routers/backtesting_router.py

"""
Backtesting Router

Provides endpoints for backtesting operations including:
- Strategy execution and analysis
- Backtest configuration and parameters
- Results retrieval and visualization data
- Strategy performance metrics
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, status

from ..services.backtesting_service import BacktestingService
from ..services.backtesting_data_service import BacktestingDataService
from ..schemas.backtesting_schemas import (
    BacktestRequest,
    BacktestResult as BacktestResponse,
    StrategyListResponse,
    StrategyConfigSchema,
    BacktestHistoryResponse,
    BacktestDeletionResponse,
    BacktestEnginesResponse,
    BacktestEngineInfo,
)
from ..strategies.strategy_factory import StrategyFactory
from common.logger import LoggerFactory
from ..factories.backtesting_factory import (
    get_backtesting_service,
    get_backtesting_data_service,
)


# Router configuration
router = APIRouter(
    prefix="/backtesting",
    tags=["backtesting"],
    responses={
        400: {"description": "Bad request - Invalid parameters"},
        404: {"description": "Not found - Strategy or data not available"},
        500: {"description": "Internal server error"},
    },
)

# Logger
logger = LoggerFactory.get_logger(name="backtesting_router")


@router.get("/strategies", response_model=StrategyListResponse)
async def get_available_strategies() -> StrategyListResponse:
    """
    Get list of available trading strategies.

    Returns all registered strategies with their descriptions,
    parameters, and configuration options.
    """
    try:
        logger.info("Available strategies requested")

        # Get strategies from factory
        strategy_factory = StrategyFactory()
        strategies_info = strategy_factory.get_all_strategies_info()

        return StrategyListResponse(
            strategies=strategies_info,
            count=len(strategies_info),
        )

    except Exception as e:
        logger.error(f"Failed to get available strategies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available strategies: {str(e)}",
        )


@router.get("/strategies/{strategy_name}/config", response_model=StrategyConfigSchema)
async def get_strategy_config(
    strategy_name: str,
) -> StrategyConfigSchema:
    """
    Get configuration schema for a specific strategy.

    Returns the parameter definitions, default values,
    and validation rules for the strategy.
    """
    try:
        logger.info(f"Strategy config requested: {strategy_name}")

        strategy_factory = StrategyFactory()

        if not strategy_factory.is_strategy_available(strategy_name):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy '{strategy_name}' not found",
            )

        config = strategy_factory.get_strategy_config(strategy_name)
        return StrategyConfigSchema(**config)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get strategy config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get strategy configuration: {str(e)}",
        )


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    backtesting_service: BacktestingService = Depends(get_backtesting_service),
) -> BacktestResponse:
    """
    Execute a backtest with specified parameters.

    Runs the specified strategy against historical market data
    and returns comprehensive performance metrics and analysis.
    """
    try:
        logger.info(
            f"Backtest requested: {request.strategy_type} on {request.symbol} "
            f"({request.timeframe}) from {request.start_date} to {request.end_date}"
        )

        # Validate strategy exists
        strategy_factory = StrategyFactory()
        if not strategy_factory.is_strategy_available(request.strategy_type.value):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy '{request.strategy_type.value}' not found",
            )

        # Execute backtest
        response = await backtesting_service.run_backtest(request)

        logger.info(
            f"Backtest completed: {response.metrics.total_return:.2%} return, "
            f"{response.metrics.total_trades} trades"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backtest execution failed: {str(e)}",
        )


@router.get("/results/{backtest_id}", response_model=BacktestResponse)
async def get_backtest_results(
    backtest_id: str,
    include_trades: bool = Query(default=True, description="Include trade details"),
    include_equity_curve: bool = Query(
        default=True, description="Include equity curve data"
    ),
    backtesting_data_service: BacktestingDataService = Depends(
        get_backtesting_data_service
    ),
) -> BacktestResponse:
    """
    Retrieve results from a previous backtest.

    Returns stored backtest results by ID, with options to include
    or exclude detailed trade and equity curve data.
    """
    try:
        logger.info(f"Backtest results requested: {backtest_id}")

        result = await backtesting_data_service.get_backtest_result(
            backtest_id=backtest_id,
            include_trades=include_trades,
            include_equity_curve=include_equity_curve,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Backtest result not found: {backtest_id}",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve backtest results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve backtest results: {str(e)}",
        )


@router.get("/history", response_model=BacktestHistoryResponse)
async def get_backtest_history(
    page: int = Query(default=1, description="Page number", ge=1),
    per_page: int = Query(default=10, description="Items per page", ge=1, le=100),
    strategy_filter: Optional[str] = Query(
        default=None, description="Filter by strategy name"
    ),
    symbol_filter: Optional[str] = Query(default=None, description="Filter by symbol"),
    backtesting_data_service: BacktestingDataService = Depends(
        get_backtesting_data_service
    ),
) -> BacktestHistoryResponse:
    """
    Get history of recent backtests.

    Returns a paginated list of recent backtest executions with basic
    metadata and performance summaries.
    """
    try:
        logger.info(
            f"Backtest history requested: page={page}, per_page={per_page}, "
            f"strategy={strategy_filter}, symbol={symbol_filter}"
        )

        history = await backtesting_data_service.get_backtest_history(
            page=page,
            per_page=per_page,
            strategy_filter=strategy_filter,
            symbol_filter=symbol_filter,
        )

        return history

    except Exception as e:
        logger.error(f"Failed to get backtest history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get backtest history: {str(e)}",
        )


@router.delete("/results/{backtest_id}", response_model=BacktestDeletionResponse)
async def delete_backtest_results(
    backtest_id: str,
    backtesting_data_service: BacktestingDataService = Depends(
        get_backtesting_data_service
    ),
) -> BacktestDeletionResponse:
    """
    Delete stored backtest results.

    Removes backtest results and associated data from storage.
    This operation cannot be undone.
    """
    try:
        logger.info(f"Backtest deletion requested: {backtest_id}")

        response = await backtesting_data_service.delete_backtest_result(backtest_id)

        if response.success:
            logger.info(f"Successfully deleted backtest: {backtest_id}")
        else:
            logger.warning(
                f"Failed to delete backtest: {backtest_id} - {response.message}"
            )

        return response

    except Exception as e:
        logger.error(f"Failed to delete backtest results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete backtest results: {str(e)}",
        )


@router.get("/engines", response_model=BacktestEnginesResponse)
async def get_available_engines() -> BacktestEnginesResponse:
    """
    Get list of available backtesting engines.

    Returns information about supported backtesting engines
    and their capabilities.
    """
    try:
        logger.info("Available engines requested")

        engines = [
            BacktestEngineInfo(
                name="backtrader",
                description="Feature-rich Python backtesting framework",
                version="1.9.78",
                supported_strategies=[
                    "moving_average_crossover",
                    "rsi_strategy",
                    "bollinger_bands",
                    "macd_strategy",
                    "simple_buy_hold",
                ],
                features=[
                    "Advanced order types",
                    "Portfolio analytics",
                    "Multiple data feeds",
                    "Custom indicators",
                    "Strategy optimization",
                ],
            )
        ]

        return BacktestEnginesResponse(
            engines=engines,
            count=len(engines),
            current_engine="backtrader",
        )

    except Exception as e:
        logger.error(f"Failed to get available engines: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available engines: {str(e)}",
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_backtesting_stats(
    backtesting_data_service: BacktestingDataService = Depends(
        get_backtesting_data_service
    ),
) -> Dict[str, Any]:
    """
    Get backtesting system statistics.

    Returns statistics about stored backtests, performance metrics,
    and system utilization.
    """
    try:
        logger.info("Backtesting stats requested")

        stats = await backtesting_data_service.get_storage_stats()

        # Get performance stats for the last 30 days
        performance_stats = (
            await backtesting_data_service.get_strategy_performance_stats(days_back=30)
        )

        return {
            "storage_stats": stats,
            "performance_stats": performance_stats,
            "system_info": {
                "supported_strategies": len(StrategyFactory.get_available_strategies()),
                "supported_engines": 1,
                "current_engine": "backtrader",
            },
        }

    except Exception as e:
        logger.error(f"Failed to get backtesting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get backtesting stats: {str(e)}",
        )
