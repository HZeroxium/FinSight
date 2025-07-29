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

    # Model cache
    MODEL_CACHE_SIZE = 10
    MODEL_CACHE_TTL = 1800  # 30 minutes

    # File patterns
    JOB_FILE_PATTERN = "training_job_{job_id}.json"
    PROGRESS_FILE_PATTERN = "progress_{job_id}.json"


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
        "ModelFacade": "INFO",
    }

    # Log format patterns
    TRAINING_LOG_FORMAT = "[{timestamp}] [{level}] [{job_id}] {message}"
    PROGRESS_LOG_FORMAT = "[{timestamp}] [{job_id}] Progress: {progress:.1%} - {status}"

    # Log file patterns
    TRAINING_LOG_FILE = "logs/training_{job_id}.log"
    BACKGROUND_LOG_FILE = "logs/background_tasks.log"
    JOB_MANAGER_LOG_FILE = "logs/job_manager.log"


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
