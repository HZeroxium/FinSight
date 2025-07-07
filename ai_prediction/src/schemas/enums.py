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
