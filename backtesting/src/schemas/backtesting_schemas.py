# schemas/backtesting_schemas.py

"""
Backtesting API schemas for request/response DTOs.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class StrategyType(str, Enum):
    """Supported strategy types."""

    MOVING_AVERAGE_CROSSOVER = "moving_average_crossover"
    RSI_STRATEGY = "rsi_strategy"
    BOLLINGER_BANDS = "bollinger_bands"
    MACD_STRATEGY = "macd_strategy"
    SIMPLE_BUY_HOLD = "simple_buy_hold"


class OrderType(str, Enum):
    """Order types for strategy execution."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class PositionSide(str, Enum):
    """Position side for trades."""

    LONG = "long"
    SHORT = "short"


class BacktestRequest(BaseModel):
    """Request schema for backtesting operations."""

    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    timeframe: str = Field(..., description="Timeframe (e.g., 1h, 1d)")
    exchange: str = Field(default="binance", description="Exchange name")

    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")

    strategy_type: StrategyType = Field(..., description="Strategy type to use")
    strategy_params: Dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific parameters"
    )

    initial_capital: float = Field(
        default=10000.0, gt=0, description="Initial capital for backtesting"
    )
    commission: float = Field(
        default=0.001, ge=0, le=1, description="Commission rate (0.001 = 0.1%)"
    )

    # Risk management parameters
    max_position_size: Optional[float] = Field(
        default=None, description="Maximum position size as fraction of capital"
    )
    stop_loss: Optional[float] = Field(
        default=None, description="Stop loss percentage (0.05 = 5%)"
    )
    take_profit: Optional[float] = Field(
        default=None, description="Take profit percentage (0.10 = 10%)"
    )

    model_config = ConfigDict(use_enum_values=True)


class StrategyConfig(BaseModel):
    """Strategy configuration schema."""

    strategy_type: StrategyType
    parameters: Dict[str, Any] = Field(default_factory=dict)

    # Common strategy settings
    position_sizing: Optional[str] = Field(
        default="fixed", description="Position sizing method: fixed, percentage, kelly"
    )
    position_size: float = Field(
        default=1.0,
        description="Position size (interpretation depends on sizing method)",
    )

    model_config = ConfigDict(use_enum_values=True)


class TradeResult(BaseModel):
    """Individual trade result."""

    entry_date: datetime
    exit_date: Optional[datetime] = None
    entry_price: float
    exit_price: Optional[float] = None

    position_side: PositionSide
    quantity: float

    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    commission_paid: float = 0.0

    entry_reason: str = Field(default="signal", description="Reason for entry")
    exit_reason: Optional[str] = Field(default=None, description="Reason for exit")

    is_open: bool = Field(default=True, description="Whether position is still open")

    model_config = ConfigDict(use_enum_values=True)


class PerformanceMetrics(BaseModel):
    """Comprehensive performance metrics."""

    # Basic metrics
    total_return: float = Field(description="Total return percentage")
    annual_return: float = Field(description="Annualized return percentage")
    total_trades: int = Field(description="Total number of trades")
    winning_trades: int = Field(description="Number of winning trades")
    losing_trades: int = Field(description="Number of losing trades")

    # Risk metrics
    max_drawdown: float = Field(description="Maximum drawdown percentage")
    sharpe_ratio: Optional[float] = Field(description="Sharpe ratio")
    sortino_ratio: Optional[float] = Field(description="Sortino ratio")
    calmar_ratio: Optional[float] = Field(description="Calmar ratio")

    # Trade statistics
    win_rate: float = Field(description="Win rate percentage")
    average_win: float = Field(description="Average winning trade percentage")
    average_loss: float = Field(description="Average losing trade percentage")
    profit_factor: float = Field(
        description="Profit factor (gross profit / gross loss)"
    )

    # Additional metrics
    volatility: float = Field(description="Strategy volatility (annualized)")
    var_95: Optional[float] = Field(description="95% Value at Risk")

    # Execution metrics
    total_commission: float = Field(description="Total commission paid")

    # Time-based metrics
    holding_period_avg: Optional[float] = Field(
        description="Average holding period in days"
    )


class EquityCurvePoint(BaseModel):
    """Single point in equity curve."""

    timestamp: datetime
    portfolio_value: float
    cash: float
    position_value: float
    drawdown: float = Field(description="Drawdown from peak")


class BacktestResult(BaseModel):
    """Comprehensive backtesting result."""

    # Unique identifier (set after saving)
    backtest_id: Optional[str] = Field(
        default=None, description="Unique backtest identifier"
    )

    # Request metadata
    symbol: str
    timeframe: str
    exchange: str
    strategy_type: StrategyType

    # Time period
    start_date: datetime
    end_date: datetime
    duration_days: int

    # Capital information
    initial_capital: float
    final_capital: float

    # Performance metrics
    metrics: PerformanceMetrics

    # Trade details
    trades: List[TradeResult]

    # Equity curve
    equity_curve: List[EquityCurvePoint]

    # Strategy-specific data
    strategy_data: Dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific output data"
    )

    # Execution metadata
    execution_time_seconds: float
    engine_used: str

    # Validation
    is_valid: bool = Field(default=True, description="Whether results are valid")
    warnings: List[str] = Field(default_factory=list, description="Execution warnings")
    errors: List[str] = Field(default_factory=list, description="Execution errors")

    model_config = ConfigDict(use_enum_values=True)


class BacktestSummary(BaseModel):
    """Summary view of backtest results."""

    symbol: str
    timeframe: str
    strategy_type: StrategyType

    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: Optional[float]

    total_trades: int
    win_rate: float

    start_date: datetime
    end_date: datetime

    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))

    model_config = ConfigDict(use_enum_values=True)


class StrategyInfo(BaseModel):
    """Information about a specific strategy."""

    name: str = Field(..., description="Strategy name")
    description: str = Field(..., description="Strategy description")
    required_parameters: List[str] = Field(
        default_factory=list, description="Required parameters"
    )
    optional_parameters: List[str] = Field(
        default_factory=list, description="Optional parameters"
    )
    parameter_descriptions: Dict[str, str] = Field(
        default_factory=dict, description="Parameter descriptions"
    )
    default_values: Dict[str, Any] = Field(
        default_factory=dict, description="Default parameter values"
    )

    model_config = ConfigDict(use_enum_values=True)


class StrategyListResponse(BaseModel):
    """Response schema for strategy list endpoint."""

    strategies: List[Dict[str, Any]] = Field(
        ..., description="List of available strategies"
    )
    count: int = Field(..., description="Total number of strategies")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(use_enum_values=True)


class ParameterSchema(BaseModel):
    """Schema for strategy parameter definition."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(
        ..., description="Parameter type (string, integer, float, boolean)"
    )
    description: str = Field(..., description="Parameter description")
    required: bool = Field(..., description="Whether parameter is required")
    default_value: Optional[Any] = Field(default=None, description="Default value")
    min_value: Optional[float] = Field(
        default=None, description="Minimum value (for numeric types)"
    )
    max_value: Optional[float] = Field(
        default=None, description="Maximum value (for numeric types)"
    )
    choices: Optional[List[str]] = Field(
        default=None, description="Valid choices (for string types)"
    )

    model_config = ConfigDict(use_enum_values=True)


class StrategyConfigSchema(BaseModel):
    """Response schema for strategy configuration endpoint."""

    strategy_name: str = Field(..., description="Strategy name")
    description: str = Field(..., description="Strategy description")
    parameters: List[ParameterSchema] = Field(..., description="Strategy parameters")
    examples: Dict[str, Any] = Field(
        default_factory=dict, description="Example configurations"
    )

    model_config = ConfigDict(use_enum_values=True)


class BacktestHistoryItem(BaseModel):
    """Single item in backtest history."""

    backtest_id: str = Field(..., description="Unique backtest identifier")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    strategy_type: StrategyType = Field(..., description="Strategy type")

    # Performance summary
    total_return: float = Field(..., description="Total return percentage")
    sharpe_ratio: Optional[float] = Field(default=None, description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    win_rate: float = Field(..., description="Win rate percentage")

    # Execution info
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    executed_at: datetime = Field(..., description="Execution timestamp")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")

    # Status
    status: str = Field(default="completed", description="Backtest status")

    model_config = ConfigDict(use_enum_values=True)


class BacktestHistoryResponse(BaseModel):
    """Response schema for backtest history endpoint."""

    history: List[BacktestHistoryItem] = Field(
        ..., description="List of backtest history items"
    )
    count: int = Field(..., description="Total number of items")
    page: int = Field(default=1, description="Current page number")
    total_pages: int = Field(default=1, description="Total number of pages")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(use_enum_values=True)


class BacktestDeletionResponse(BaseModel):
    """Response schema for backtest deletion endpoint."""

    success: bool = Field(..., description="Whether deletion was successful")
    backtest_id: str = Field(..., description="Deleted backtest identifier")
    message: str = Field(..., description="Status message")
    deleted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(use_enum_values=True)


class BacktestEngineInfo(BaseModel):
    """Information about a backtesting engine."""

    name: str = Field(..., description="Engine name")
    description: str = Field(..., description="Engine description")
    version: str = Field(..., description="Engine version")
    supported_strategies: List[str] = Field(..., description="Supported strategies")
    features: List[str] = Field(..., description="Engine features")

    model_config = ConfigDict(use_enum_values=True)


class BacktestEnginesResponse(BaseModel):
    """Response schema for backtesting engines endpoint."""

    engines: List[BacktestEngineInfo] = Field(
        ..., description="List of available engines"
    )
    count: int = Field(..., description="Total number of engines")
    current_engine: str = Field(..., description="Currently active engine")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(use_enum_values=True)
