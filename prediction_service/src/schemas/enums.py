# schemas/enums.py

from enum import Enum


class ModelType(str, Enum):
    """Supported time series model types for fine-tuning"""

    PATCHTST = "ibm/patchtst-forecasting"
    PATCHTSMIXER = "ibm/patchtsmixer-forecasting"
    PYTORCH_TRANSFORMER = "pytorch-lightning/time-series-transformer"
    ENHANCED_TRANSFORMER = "enhanced-transformer"  # New enhanced transformer


class TaskType(str, Enum):
    """Task types for time series models"""

    FORECASTING = "forecasting"
    CLASSIFICATION = "classification"


class CryptoSymbol(str, Enum):
    """Supported cryptocurrency symbols"""

    BTCUSDT = "BTCUSDT"
    ETHUSDT = "ETHUSDT"
    BNBUSDT = "BNBUSDT"
    SOLUSDT = "SOLUSDT"


class TimeFrame(str, Enum):
    """Supported timeframes for data"""

    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


class DataLoaderType(str, Enum):
    """Supported data loader types"""

    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"


class ExperimentTrackerType(str, Enum):
    """Supported experiment tracker types"""

    SIMPLE = "simple"
    MLFLOW = "mlflow"


class ServingAdapterType(str, Enum):
    """Supported serving adapter types"""

    SIMPLE = "simple"
    TRITON = "triton"
    TORCHSERVE = "torchserve"
    TORCHSCRIPT = "torchscript"


class StorageProviderType(str, Enum):
    """Supported storage provider types"""

    MINIO = "minio"
    DIGITALOCEAN = "digitalocean"
    AWS = "aws"
    S3 = "s3"


class Exchange(str, Enum):
    """Supported exchanges"""

    BINANCE = "binance"
    KUCOIN = "kucoin"
    OKX = "okx"
    HUOBI = "huobi"


class DataFormat(str, Enum):
    CSV = "csv"
    PARQUET = "parquet"


class DeviceType(str, Enum):
    """Supported compute device types.

    Use these enum values instead of magic strings like 'cpu' or 'cuda'.
    """

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon Metal Performance Shaders (if available)


class FallbackStrategy(str, Enum):
    """Fallback strategies for model selection"""

    NONE = "none"  # No fallback, fail if exact model not found
    TIMEFRAME_ONLY = "timeframe_only"  # Fallback to different timeframe only
    SYMBOL_ONLY = "symbol_only"  # Fallback to different symbol only
    TIMEFRAME_AND_SYMBOL = "timeframe_and_symbol"  # Full fallback strategy


class ModelSelectionPriority(str, Enum):
    """Priority order for model selection during fallback"""

    EXACT_MATCH = "exact_match"  # Exact symbol + timeframe + model_type match
    TIMEFRAME_FALLBACK = "timeframe_fallback"  # Same symbol, fallback timeframe
    SYMBOL_FALLBACK = "symbol_fallback"  # Fallback symbol, same timeframe
    FULL_FALLBACK = "full_fallback"  # Fallback both symbol and timeframe


class PredictionTrend(str, Enum):
    """Prediction trend analysis results"""

    BULLISH = "bullish"  # Overall positive trend
    BEARISH = "bearish"  # Overall negative trend
    NEUTRAL = "neutral"  # No clear trend
    VOLATILE = "volatile"  # High volatility, unclear trend


class PercentageCalculationMethod(str, Enum):
    """Methods for calculating percentage changes"""

    CURRENT_PRICE_BASED = "current_price_based"  # Use current market price as reference
    FIRST_PREDICTION_BASED = (
        "first_prediction_based"  # Use first prediction as reference
    )
    CUSTOM_BASE_PRICE = "custom_base_price"  # Use custom base price
    ROLLING_BASE = "rolling_base"  # Use previous prediction as base for next


class PredictionConfidenceLevel(str, Enum):
    """Confidence levels for predictions"""

    VERY_HIGH = "very_high"  # 90-100% confidence
    HIGH = "high"  # 75-89% confidence
    MEDIUM = "medium"  # 50-74% confidence
    LOW = "low"  # 25-49% confidence
    VERY_LOW = "very_low"  # 0-24% confidence


class DataSelectionPriority(str, Enum):
    """Priority order for data selection during fallback"""

    EXACT_MATCH = "exact_match"  # Exact symbol + timeframe match
    TIMEFRAME_FALLBACK = "timeframe_fallback"  # Same symbol, fallback timeframe
    SYMBOL_FALLBACK = "symbol_fallback"  # Fallback symbol, same timeframe
    FULL_FALLBACK = "full_fallback"  # Fallback both symbol and timeframe
    BEST_AVAILABLE = "best_available"  # Best available option as last resort


class DataAvailabilityStatus(str, Enum):
    """Status of data availability"""

    AVAILABLE = "available"  # Data is available and accessible
    NOT_FOUND = "not_found"  # Data not found
    ACCESS_ERROR = "access_error"  # Data exists but cannot be accessed
    EMPTY = "empty"  # Data exists but is empty
    CORRUPTED = "corrupted"  # Data exists but is corrupted


class FallbackReason(str, Enum):
    """Reasons for applying fallback strategies"""

    EXACT_MATCH_NOT_FOUND = "exact_match_not_found"
    TIMEFRAME_NOT_AVAILABLE = "timeframe_not_available"
    SYMBOL_NOT_AVAILABLE = "symbol_not_available"
    DATA_LOAD_ERROR = "data_load_error"
    INSUFFICIENT_DATA = "insufficient_data"
    BEST_AVAILABLE_OPTION = "best_available_option"
