# schemas/enums.py

from enum import Enum


class RepositoryType(str, Enum):
    """Available repository types"""

    CSV = "csv"
    MONGODB = "mongodb"
    INFLUXDB = "influxdb"
    TIMESCALEDB = "timescaledb"
    PARQUET = "parquet"


class Exchange(str, Enum):
    """Supported cryptocurrency exchanges"""

    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BYBIT = "bybit"
    KUCOIN = "kucoin"
    GATE = "gate"
    HUOBI = "huobi"
    OKX = "okx"
    BITGET = "bitget"
    MEXC = "mexc"


class CryptoSymbol(str, Enum):
    """Supported cryptocurrency trading pairs"""

    # Major BTC pairs
    BTCUSDT = "BTCUSDT"
    BTCBUSD = "BTCBUSD"
    BTCUSDC = "BTCUSDC"

    # Major ETH pairs
    ETHUSDT = "ETHUSDT"
    ETHBUSD = "ETHBUSD"
    ETHUSDC = "ETHUSDC"
    ETHBTC = "ETHBTC"

    # Top Market Cap Coins
    BNBUSDT = "BNBUSDT"
    XRPUSDT = "XRPUSDT"
    ADAUSDT = "ADAUSDT"
    SOLUSDT = "SOLUSDT"
    DOGEUSDT = "DOGEUSDT"
    DOTUSDT = "DOTUSDT"
    MATICUSDT = "MATICUSDT"
    LTCUSDT = "LTCUSDT"
    BCHUSDT = "BCHUSDT"
    LINKUSDT = "LINKUSDT"

    # Popular Altcoins
    AVAXUSDT = "AVAXUSDT"
    ATOMUSDT = "ATOMUSDT"
    UNIUSDT = "UNIUSDT"
    FILUSDT = "FILUSDT"
    TRXUSDT = "TRXUSDT"
    ETCUSDT = "ETCUSDT"
    XLMUSDT = "XLMUSDT"
    VETUSDT = "VETUSDT"
    ICPUSDT = "ICPUSDT"
    NEARUSDT = "NEARUSDT"
    ALGOUSDT = "ALGOUSDT"
    HBARUSDT = "HBARUSDT"

    # DeFi Tokens
    AAVEUSDT = "AAVEUSDT"
    COMPUSDT = "COMPUSDT"
    MKRUSDT = "MKRUSDT"
    SUSHIUSDT = "SUSHIUSDT"
    CRVUSDT = "CRVUSDT"
    YFIUSDT = "YFIUSDT"
    SNXUSDT = "SNXUSDT"
    BALUSDT = "BALUSDT"

    # Gaming & Metaverse
    AXSUSDT = "AXSUSDT"
    SANDUSDT = "SANDUSDT"
    MANAUSDT = "MANAUSDT"
    ENJUSDT = "ENJUSDT"
    GALAUSDT = "GALAUSDT"

    # Layer 2 & Scaling Solutions
    OPUSDT = "OPUSDT"
    ARBUSDT = "ARBUSDT"
    IMXUSDT = "IMXUSDT"
    LRCUSDT = "LRCUSDT"

    # Meme Coins
    SHIBUSDT = "SHIBUSDT"
    PEPEUSDT = "PEPEUSDT"
    FLOKIUSDT = "FLOKIUSDT"

    # Infrastructure & AI
    FETUSDT = "FETUSDT"
    OCEUSDT = "OCEUSDT"
    RENDERUSDT = "RENDERUSDT"

    # Additional coins for comprehensive coverage
    APTUSDT = "APTUSDT"
    SUIUSDT = "SUIUSDT"


class TimeFrame(str, Enum):
    """Supported timeframes for market data"""

    # Minutes
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"

    # Hours
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"

    # Days and above
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


class MarketDataType(str, Enum):
    """Types of market data"""

    OHLCV = "ohlcv"
    TRADES = "trades"
    ORDERBOOK = "orderbook"
    TICKER = "ticker"
    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"


class DataQuality(str, Enum):
    """Data quality levels"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class CollectionStatus(str, Enum):
    """Status of data collection operations"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TimeFrameMultiplier(Enum):
    """Multiplier values for timeframe conversions"""

    MINUTE_1 = 1
    MINUTE_3 = 3
    MINUTE_5 = 5
    MINUTE_15 = 15
    MINUTE_30 = 30
    HOUR_1 = 60
    HOUR_2 = 120
    HOUR_4 = 240
    HOUR_6 = 360
    HOUR_8 = 480
    HOUR_12 = 720
    DAY_1 = 1440
    DAY_3 = 4320
    WEEK_1 = 10080
    MONTH_1 = 43200  # Approximate: 30 days
