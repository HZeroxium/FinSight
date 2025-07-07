# schemas/__init__.py

"""
API schemas for request/response DTOs.
"""

from .ohlcv_schemas import (
    OHLCVSchema,
    OHLCVBatchSchema,
    OHLCVQuerySchema,
    OHLCVResponseSchema,
    OHLCVStatsSchema,
)
from .backtesting_schemas import (
    BacktestRequest,
    BacktestResult,
    StrategyConfig,
    TradeResult,
    PerformanceMetrics,
    EquityCurvePoint,
    BacktestSummary,
    StrategyType,
    OrderType,
    PositionSide,
)
from .enums import (
    RepositoryType,
    Exchange,
    CryptoSymbol,
    TimeFrame,
    MarketDataType,
    TimeFrameMultiplier,
)
from .admin_schemas import (
    DataEnsureRequest,
    DataEnsureResponse,
    TimeframeConvertRequest,
    TimeframeConvertResponse,
    AdminStatsResponse,
    SystemHealthResponse,
    CleanupRequest,
    CleanupResponse,
    APIKeyValidationRequest,
    APIKeyValidationResponse,
)

__all__ = [
    # OHLCV schemas
    "OHLCVSchema",
    "OHLCVBatchSchema",
    "OHLCVQuerySchema",
    "OHLCVResponseSchema",
    "OHLCVStatsSchema",
    # Backtesting schemas
    "BacktestRequest",
    "BacktestResult",
    "StrategyConfig",
    "TradeResult",
    "PerformanceMetrics",
    "EquityCurvePoint",
    "BacktestSummary",
    "StrategyType",
    "OrderType",
    "PositionSide",
    # Admin schemas
    "DataEnsureRequest",
    "DataEnsureResponse",
    "TimeframeConvertRequest",
    "TimeframeConvertResponse",
    "AdminStatsResponse",
    "SystemHealthResponse",
    "CleanupRequest",
    "CleanupResponse",
    "APIKeyValidationRequest",
    "APIKeyValidationResponse",
    # Enums
    "RepositoryType",
    "Exchange",
    "CryptoSymbol",
    "TimeFrame",
    "MarketDataType",
    "TimeFrameMultiplier",
]
