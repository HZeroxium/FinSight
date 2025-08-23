# core/enums.py

"""Enums for the sentiment analysis inference engine."""

from enum import Enum, IntEnum


class LogLevel(str, Enum):
    """Log levels."""

    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class ServerStatus(str, Enum):
    """Server status states."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class SentimentLabel(str, Enum):
    """Sentiment labels."""

    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"


class SentimentScore(IntEnum):
    """Sentiment score mapping."""

    NEGATIVE = 0
    NEUTRAL = 1
    POSITIVE = 2


class ModelStatus(str, Enum):
    """Model status in Triton."""

    READY = "READY"
    UNAVAILABLE = "UNAVAILABLE"
    LOADING = "LOADING"
    UNLOADING = "UNLOADING"


class RequestStatus(str, Enum):
    """Request processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
