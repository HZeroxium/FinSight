# schemas/__init__.py

"""
API schemas for request/response DTOs.
"""

from .base_schemas import *
from .dataset_schemas import *
from .enums import *
from .model_schemas import *
from .training_schemas import *

__all__ = [
    # Base schemas
    "BaseResponse",
    "ErrorResponse",
    "HealthResponse",
    "ModelInfoResponse",
    # Enums
    "ModelType",
    "TaskType",
    "CryptoSymbol",
    "TimeFrame",
    "DataLoaderType",
    "ExperimentTrackerType",
    "ServingAdapterType",
    "StorageProviderType",
    "DeviceType",
    "FallbackStrategy",
    "ModelSelectionPriority",
    "PredictionTrend",
    "PercentageCalculationMethod",
    "PredictionConfidenceLevel",
    # Model schemas
    "ModelConfig",
    "TrainingRequest",
    "TrainingResponse",
    "PredictionRequest",
    "PredictionResponse",
    "ModelInfo",
    # Training schemas
    "TrainingJobPriority",
    "AsyncTrainingRequest",
    "TrainingJobInfo",
    "AsyncTrainingResponse",
    "TrainingJobStatusResponse",
    "TrainingJobListResponse",
    "TrainingJobCancelRequest",
    "TrainingJobCancelResponse",
    "TrainingQueueInfo",
    "TrainingQueueResponse",
    "TrainingProgressUpdate",
    "TrainingJobFilter",
    "BackgroundTaskHealth",
    "BackgroundTaskHealthResponse",
    # Dataset schemas
    "DatasetInfo",
    "DatasetListRequest",
    "DatasetListResponse",
    "DatasetAvailabilityRequest",
    "DatasetAvailabilityResponse",
    "DatasetDownloadRequest",
    "DatasetDownloadResponse",
    "CacheInfo",
    "CacheListRequest",
    "CacheListResponse",
    "CacheInvalidateRequest",
    "CacheInvalidateResponse",
    "DatasetStatistics",
    "DatasetHealthCheck",
    "BulkDatasetOperation",
    "BulkOperationResponse",
]
