# schemas/__init__.py

"""
API schemas for request/response DTOs.
"""

from .admin_schemas import (AdminStatsResponse, APIKeyValidationRequest,
                            APIKeyValidationResponse, CleanupRequest,
                            CleanupResponse, DataEnsureRequest,
                            DataEnsureResponse, SystemHealthResponse,
                            TimeframeConvertRequest, TimeframeConvertResponse)
from .backtesting_schemas import (BacktestRequest, BacktestResult,
                                  BacktestSummary, EquityCurvePoint, OrderType,
                                  PerformanceMetrics, PositionSide,
                                  StrategyConfig, StrategyType, TradeResult)
from .enums import (CryptoSymbol, Exchange, MarketDataType, RepositoryType,
                    TimeFrame, TimeFrameMultiplier)
from .job_schemas import (DataCollectionJobRequest, DataCollectionJobResponse,
                          HealthCheckResponse, JobConfigUpdateRequest,
                          JobOperationResponse, JobPriority, JobStartRequest,
                          JobStatsModel, JobStatus, JobStatusResponse,
                          JobStopRequest, ManualJobRequest, ManualJobResponse,
                          MarketDataJobConfigModel, NotificationLevel)
from .ohlcv_schemas import (OHLCVBatchSchema, OHLCVQuerySchema,
                            OHLCVResponseSchema, OHLCVSchema, OHLCVStatsSchema)

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
    # Job schemas
    "JobStatus",
    "JobPriority",
    "NotificationLevel",
    "MarketDataJobConfigModel",
    "JobStatsModel",
    "JobStatusResponse",
    "JobStartRequest",
    "JobStopRequest",
    "ManualJobRequest",
    "ManualJobResponse",
    "JobConfigUpdateRequest",
    "JobOperationResponse",
    "DataCollectionJobRequest",
    "DataCollectionJobResponse",
    "HealthCheckResponse",
]
