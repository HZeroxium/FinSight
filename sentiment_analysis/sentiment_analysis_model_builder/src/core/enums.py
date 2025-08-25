# core/enums.py

"""Enums for the sentiment analysis model builder."""

from enum import Enum


class ModelBackbone(str, Enum):
    """Available model backbones for sentiment analysis."""

    FINBERT = "ProsusAI/finbert"
    DISTILBERT_SST2 = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    ALBERT_BASE_V2 = "albert-base-v2"
    # ROBERTA_BASE = "roberta-base"
    # DEBERTA_BASE = "microsoft/deberta-base"
    # BERT_BASE = "bert-base-uncased"


class ExportFormat(str, Enum):
    """Available export formats for the trained model."""

    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    BOTH = "both"


class ModelStage(str, Enum):
    """Model stages for registry management."""

    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class DataFormat(str, Enum):
    """Supported data input formats."""

    CSV = "csv"
    JSONL = "jsonl"
    PARQUET = "parquet"
    JSON = "json"


class SentimentLabel(str, Enum):
    """Sentiment labels for classification."""

    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EvaluationStrategy(str, Enum):
    """Evaluation strategies for training."""

    STEPS = "steps"
    EPOCH = "epoch"
    NO_EVAL = "no"


class SaveStrategy(str, Enum):
    """Model saving strategies."""

    STEPS = "steps"
    EPOCH = "epoch"
    NO_SAVE = "no"


class PaddingStrategy(str, Enum):
    """Text padding strategies."""

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TruncationStrategy(str, Enum):
    """Text truncation strategies."""

    LONGEST_FIRST = "longest_first"
    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    DO_NOT_TRUNCATE = "do_not_truncate"


class MetricType(str, Enum):
    """Evaluation metric types."""

    ACCURACY = "accuracy"
    F1_MACRO = "f1_macro"
    F1_WEIGHTED = "f1_weighted"
    PRECISION = "precision"
    RECALL = "recall"


class DeviceType(str, Enum):
    """Device types for model training."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


class DataSplit(str, Enum):
    """Data split names."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    DEV = "dev"


class FileExtension(str, Enum):
    """Common file extensions."""

    JSON = ".json"
    JSONL = ".jsonl"
    CSV = ".csv"
    PARQUET = ".parquet"
    TXT = ".txt"
    YAML = ".yaml"
    YML = ".yml"


class APIEndpoint(str, Enum):
    """API endpoint paths."""

    HEALTH = "/health"
    PREDICT = "/predict"
    BATCH_PREDICT = "/batch-predict"
    MODEL_INFO = "/model-info"
    METRICS = "/metrics"


class ResponseStatus(str, Enum):
    """API response status."""

    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
