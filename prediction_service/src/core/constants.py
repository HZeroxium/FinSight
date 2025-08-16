# core/constants.py

"""
Constants for the AI prediction service
"""

from enum import Enum
from typing import Dict, Any


# Training Job Status Constants
class TrainingJobStatus(str, Enum):
    """Training job status enumeration"""

    PENDING = "pending"
    QUEUED = "queued"
    INITIALIZING = "initializing"
    LOADING_DATA = "loading_data"
    PREPARING_FEATURES = "preparing_features"
    TRAINING = "training"
    VALIDATING = "validating"
    SAVING_MODEL = "saving_model"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def get_active_statuses(cls) -> list:
        """Get list of active (non-terminal) statuses"""
        return [
            cls.PENDING,
            cls.QUEUED,
            cls.INITIALIZING,
            cls.LOADING_DATA,
            cls.PREPARING_FEATURES,
            cls.TRAINING,
            cls.VALIDATING,
            cls.SAVING_MODEL,
        ]

    @classmethod
    def get_terminal_statuses(cls) -> list:
        """Get list of terminal (final) statuses"""
        return [
            cls.COMPLETED,
            cls.FAILED,
            cls.CANCELLED,
        ]

    @classmethod
    def is_active(cls, status: str) -> bool:
        """Check if status is active (non-terminal)"""
        return status in cls.get_active_statuses()

    @classmethod
    def is_terminal(cls, status: str) -> bool:
        """Check if status is terminal (final)"""
        return status in cls.get_terminal_statuses()


# Training Configuration Constants
class TrainingConstants:
    """Training-related constants"""

    # Default timeouts (in seconds)
    DEFAULT_TRAINING_TIMEOUT = 3600  # 1 hour
    MAX_TRAINING_TIMEOUT = 7200  # 2 hours
    STATUS_UPDATE_INTERVAL = 5  # seconds
    CLEANUP_INTERVAL = 300  # 5 minutes

    # Concurrency limits
    MAX_CONCURRENT_TRAININGS = 3
    MAX_QUEUE_SIZE = 10

    STATS_CACHE_TTL = 3600 * 24 * 7  # 7 days

    # Progress tracking
    PROGRESS_STAGES = {
        TrainingJobStatus.PENDING: 0.0,
        TrainingJobStatus.QUEUED: 0.05,
        TrainingJobStatus.INITIALIZING: 0.10,
        TrainingJobStatus.LOADING_DATA: 0.20,
        TrainingJobStatus.PREPARING_FEATURES: 0.30,
        TrainingJobStatus.TRAINING: 0.70,  # Training takes most time
        TrainingJobStatus.VALIDATING: 0.85,
        TrainingJobStatus.SAVING_MODEL: 0.95,
        TrainingJobStatus.COMPLETED: 1.0,
        TrainingJobStatus.FAILED: 0.0,
        TrainingJobStatus.CANCELLED: 0.0,
    }

    # Error codes
    ERROR_CODES = {
        "DATA_NOT_FOUND": "E001",
        "INVALID_CONFIG": "E002",
        "TRAINING_FAILED": "E003",
        "MODEL_SAVE_FAILED": "E004",
        "TIMEOUT": "E005",
        "CANCELLED": "E006",
        "RESOURCE_LIMIT": "E007",
        "UNKNOWN_ERROR": "E999",
    }


# Cache and Storage Constants
class StorageConstants:
    """Storage-related constants"""

    # Job persistence
    JOB_STORAGE_TTL = 86400  # 24 hours
    COMPLETED_JOB_TTL = 3600  # 1 hour for completed jobs
    FAILED_JOB_TTL = 7200  # 2 hours for failed jobs
    PROGRESS_TTL = 3600  # 1 hour for progress updates

    # Model cache
    MODEL_CACHE_SIZE = 10
    MODEL_CACHE_TTL = 1800  # 30 minutes

    # File patterns
    JOB_FILE_PATTERN = "training_job_{job_id}.json"
    PROGRESS_FILE_PATTERN = "progress_{job_id}.json"

    # Redis-specific constants
    REDIS_KEY_PREFIX = "finsight:training_job:"
    REDIS_ACTIVE_JOBS_SET = "finsight:active_jobs"
    REDIS_PROGRESS_PREFIX = "finsight:progress:"
    REDIS_STATS_KEY = "finsight:training_stats"
    REDIS_JOB_HASH_PREFIX = "finsight:job:"


# Validation Constants
class ValidationConstants:
    """Validation-related constants"""

    # Data requirements
    MIN_DATA_ROWS = 100
    MIN_TRAINING_RATIO = 0.5
    MAX_TRAINING_RATIO = 0.9
    MIN_VALIDATION_RATIO = 0.05
    MAX_VALIDATION_RATIO = 0.3

    # Model configuration limits
    MIN_CONTEXT_LENGTH = 10
    MAX_CONTEXT_LENGTH = 1000
    MIN_PREDICTION_LENGTH = 1
    MAX_PREDICTION_LENGTH = 100
    MIN_EPOCHS = 1
    MAX_EPOCHS = 200
    MIN_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 512
    MIN_LEARNING_RATE = 1e-6
    MAX_LEARNING_RATE = 1.0


# API Response Messages
class ResponseMessages:
    """Standard API response messages"""

    # Success messages
    TRAINING_STARTED = "Model training started successfully"
    TRAINING_COMPLETED = "Model training completed successfully"
    TRAINING_CANCELLED = "Model training cancelled successfully"

    # Error messages
    TRAINING_NOT_FOUND = "Training job not found"
    TRAINING_ALREADY_EXISTS = "Training job already exists for this configuration"
    MAX_CONCURRENT_REACHED = "Maximum concurrent training jobs reached"
    QUEUE_FULL = "Training queue is full"
    INVALID_REQUEST = "Invalid training request"
    DATA_NOT_AVAILABLE = "Training data not available"
    INTERNAL_ERROR = "Internal server error occurred"

    # Status messages
    JOB_STATUS_RETRIEVED = "Training job status retrieved successfully"
    ACTIVE_JOBS_RETRIEVED = "Active training jobs retrieved successfully"
    JOB_HISTORY_RETRIEVED = "Training job history retrieved successfully"


# Logging Constants
class LoggingConstants:
    """Logging-related constants"""

    # Log levels for different components
    COMPONENT_LOG_LEVELS = {
        "TrainingService": "INFO",
        "BackgroundTaskManager": "INFO",
        "JobRepository": "DEBUG",
        "TaskRunner": "INFO",
        "UnifiedModelFacade": "INFO",
        "ModelTrainingFacade": "INFO",
        "ModelServingFacade": "INFO",
    }

    # Log format patterns
    TRAINING_LOG_FORMAT = "[{timestamp}] [{level}] [{job_id}] {message}"
    PROGRESS_LOG_FORMAT = "[{timestamp}] [{job_id}] Progress: {progress:.1%} - {status}"

    # Log file patterns
    TRAINING_LOG_FILE = "logs/training_{job_id}.log"
    BACKGROUND_LOG_FILE = "logs/background_tasks.log"
    JOB_MANAGER_LOG_FILE = "logs/job_manager.log"


# Facade Constants
class FacadeConstants:
    """Facade-related constants and configuration"""

    # Facade types
    TRAINING_FACADE = "training"
    SERVING_FACADE = "serving"
    UNIFIED_FACADE = "unified"

    # Serving adapter types
    ADAPTER_SIMPLE = "simple"
    ADAPTER_TORCHSCRIPT = "torchscript"
    ADAPTER_TORCHSERVE = "torchserve"
    ADAPTER_TRITON = "triton"

    # All supported adapter types
    SUPPORTED_ADAPTERS = [
        ADAPTER_SIMPLE,
        ADAPTER_TORCHSCRIPT,
        ADAPTER_TORCHSERVE,
        ADAPTER_TRITON,
    ]

    # Model cache settings
    DEFAULT_MODEL_CACHE_SIZE = 10
    MODEL_CACHE_TTL_SECONDS = 1800  # 30 minutes

    # Serving adapter settings
    DEFAULT_SERVING_TIMEOUT = 30.0
    MAX_SERVING_RETRIES = 3
    SERVING_HEALTH_CHECK_INTERVAL = 60.0

    # Training settings
    DEFAULT_TRAINING_TIMEOUT = 3600.0  # 1 hour
    TRAINING_PROGRESS_UPDATE_INTERVAL = 10.0  # 10 seconds

    # File extensions and patterns
    MODEL_METADATA_SUFFIX = "_metadata.json"
    MODEL_CHECKPOINT_PATTERN = "checkpoint-*"

    # Error handling
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY_SECONDS = 1.0

    # Performance monitoring
    METRICS_COLLECTION_ENABLED = True
    PERFORMANCE_LOG_INTERVAL = 300.0  # 5 minutes


# Background Task Configuration
class BackgroundTaskConfig:
    """Background task configuration"""

    # Task execution settings
    TASK_RETRY_ATTEMPTS = 3
    TASK_RETRY_DELAY = 60  # seconds
    TASK_CLEANUP_BATCH_SIZE = 50
    TASK_HEARTBEAT_INTERVAL = 30  # seconds

    # Resource monitoring
    MEMORY_THRESHOLD_MB = 2048  # 2GB
    CPU_THRESHOLD_PERCENT = 80
    DISK_THRESHOLD_PERCENT = 90

    # Health check settings
    HEALTH_CHECK_INTERVAL = 60  # seconds
    HEALTH_CHECK_TIMEOUT = 10  # seconds


# Model Fallback Configuration
class FallbackConstants:
    """Constants for model fallback strategy"""

    # Default fallback strategy
    DEFAULT_FALLBACK_STRATEGY = "timeframe_and_symbol"

    # Fallback priority order for timeframes (most preferred first)
    TIMEFRAME_FALLBACK_PRIORITY = [
        "1d",  # DAY_1 - most preferred fallback
        "4h",  # HOUR_4
        "1h",  # HOUR_1
        "15m",  # MINUTE_15
        "5m",  # MINUTE_5
        "1m",  # MINUTE_1
        "12h",  # HOUR_12
        "1w",  # WEEK_1
    ]

    # Fallback priority order for symbols (most preferred first)
    SYMBOL_FALLBACK_PRIORITY = [
        "BTCUSDT",  # Most preferred fallback symbol
        "ETHUSDT",  # Second choice
        "BNBUSDT",  # Third choice
    ]

    # Model type priority for fallback (most preferred first)
    MODEL_TYPE_FALLBACK_PRIORITY = [
        "ibm/patchtst-forecasting",  # PATCHTST
        "ibm/patchtsmixer-forecasting",  # PATCHTSMIXER
        "pytorch-lightning/time-series-transformer",  # PYTORCH_TRANSFORMER
        "enhanced-transformer",  # ENHANCED_TRANSFORMER
    ]

    # Maximum fallback attempts
    MAX_FALLBACK_ATTEMPTS = 5

    # Fallback timeout (seconds)
    FALLBACK_TIMEOUT = 30.0

    # Enable/disable fallback features
    ENABLE_TIMEFRAME_FALLBACK = True
    ENABLE_SYMBOL_FALLBACK = True
    ENABLE_MODEL_TYPE_FALLBACK = True

    # Logging for fallback operations
    LOG_FALLBACK_ATTEMPTS = True
    LOG_FALLBACK_SUCCESS = True
    LOG_FALLBACK_FAILURES = True


# Default Configurations
DEFAULT_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "patchtst": {
        "context_length": 64,
        "prediction_length": 1,
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "d_model": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "dropout": 0.1,
    },
    "patchtsmixer": {
        "context_length": 64,
        "prediction_length": 1,
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "d_model": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "dropout": 0.1,
    },
    "pytorch_lightning_transformer": {
        "context_length": 64,
        "prediction_length": 1,
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.1,
    },
}
