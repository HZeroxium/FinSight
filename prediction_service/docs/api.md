# FinSight Prediction Service - API Documentation

> **Complete API Reference for Financial Time Series Prediction Service**

## 🌐 Overview

The FinSight Prediction Service provides a comprehensive REST API for:

- **Model Training**: Asynchronous training of time series models
- **Predictions**: Real-time cryptocurrency price forecasting
- **Model Management**: Model lifecycle and versioning
- **Data Management**: Dataset availability and cloud storage
- **Service Discovery**: Eureka client integration
- **System Maintenance**: Automated cleanup and monitoring

## 🔐 Authentication

Currently, the service operates without authentication for development purposes. In production, implement:

- API Key authentication for admin endpoints
- Rate limiting for public endpoints
- JWT tokens for user-specific operations

## 🌍 Base URL

- **Local Development**: `http://localhost:8000`
- **Docker**: `http://localhost:8001`
- **Production**: Configure via environment variables

## 📊 Common Response Formats

### Success Response

```json
{
  "success": true,
  "message": "Operation completed successfully",
  "data": {},
  "metadata": {}
}
```

### Error Response

```json
{
  "success": false,
  "message": "Operation failed",
  "error": "Detailed error description",
  "error_code": "VALIDATION_ERROR",
  "error_details": {}
}
```

### Health Response

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "dependencies": {
    "data_dir": "available",
    "models_dir": "available",
    "eureka_client": "healthy"
  }
}
```

## ❌ Error Handling

### HTTP Status Codes

| Code | Description           | Usage                         |
| ---- | --------------------- | ----------------------------- |
| 200  | OK                    | Successful operation          |
| 201  | Created               | Resource created successfully |
| 400  | Bad Request           | Invalid input parameters      |
| 404  | Not Found             | Resource not found            |
| 422  | Unprocessable Entity  | Validation errors             |
| 500  | Internal Server Error | Server-side errors            |

### Error Codes

| Code                  | Description                   | HTTP Status |
| --------------------- | ----------------------------- | ----------- |
| `VALIDATION_ERROR`    | Input validation failed       | 422         |
| `MODEL_NOT_FOUND`     | Requested model not available | 404         |
| `TRAINING_FAILED`     | Model training failed         | 500         |
| `SERVING_ERROR`       | Model serving error           | 500         |
| `DATA_UNAVAILABLE`    | Training data not found       | 404         |
| `RATE_LIMIT_EXCEEDED` | API rate limit exceeded       | 429         |

## 🚀 API Endpoints

### Root & Health

#### GET `/`

Get service information and available endpoints.

**Response:**

```json
{
  "name": "FinSight Prediction Service",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "endpoints": {
    "training": "/training",
    "prediction": "/prediction",
    "models": "/models",
    "serving": "/serving",
    "datasets": "/datasets",
    "cloud_storage": "/cloud-storage",
    "eureka": "/eureka",
    "cleanup": "/cleanup",
    "health": "/health"
  },
  "features": {
    "model_training": true,
    "model_serving": true,
    "dataset_management": true,
    "cloud_storage": true,
    "eureka_client": true,
    "background_cleanup": true
  }
}
```

#### GET `/health`

Health check endpoint for monitoring and load balancers.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "dependencies": {
    "data_dir": "available",
    "models_dir": "available",
    "logs_dir": "available",
    "eureka_client": "healthy"
  }
}
```

### Training

#### POST `/training/train`

Start synchronous model training.

**Request:**

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1d",
  "model_type": "ibm/patchtst-forecasting",
  "config": {
    "context_length": 64,
    "prediction_length": 1,
    "num_epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

**Response:**

```json
{
  "success": true,
  "message": "Model training started successfully",
  "training_id": "train_abc123",
  "model_path": "/models/BTCUSDT_1d_patchtst/model.pt",
  "training_metrics": {
    "train_loss": 0.0234,
    "val_loss": 0.0289
  },
  "training_duration": 125.6
}
```

#### POST `/training/train-async`

Start asynchronous model training with job management.

**Request:**

```json
{
  "symbol": "ETHUSDT",
  "timeframe": "4h",
  "model_type": "ibm/patchtsmixer-forecasting",
  "config": {
    "context_length": 96,
    "prediction_length": 3,
    "num_epochs": 20,
    "batch_size": 64,
    "learning_rate": 0.0005
  },
  "priority": "high"
}
```

**Response:**

```json
{
  "success": true,
  "message": "Asynchronous training started",
  "job_id": "job_xyz789",
  "status": "queued",
  "estimated_duration": 1800,
  "queue_position": 1
}
```

#### GET `/training/status/{job_id}`

Get training job status and progress.

**Response:**

```json
{
  "success": true,
  "job_id": "job_xyz789",
  "status": "training",
  "progress": 0.65,
  "current_stage": "training",
  "started_at": "2024-01-15T10:00:00Z",
  "estimated_completion": "2024-01-15T10:30:00Z",
  "metrics": {
    "current_epoch": 13,
    "train_loss": 0.0189,
    "val_loss": 0.0221
  }
}
```

#### GET `/training/jobs`

List training jobs with filtering and pagination.

**Query Parameters:**

- `statuses`: Filter by job statuses (comma-separated)
- `symbols`: Filter by trading symbols
- `timeframes`: Filter by timeframes
- `limit`: Maximum results (default: 50)
- `offset`: Result offset (default: 0)
- `sort_by`: Sort field (default: created_at)
- `sort_order`: Sort order (asc/desc)

**Response:**

```json
{
  "success": true,
  "jobs": [
    {
      "job_id": "job_abc123",
      "symbol": "BTCUSDT",
      "timeframe": "1d",
      "model_type": "ibm/patchtst-forecasting",
      "status": "completed",
      "created_at": "2024-01-15T09:00:00Z",
      "completed_at": "2024-01-15T09:25:00Z",
      "progress": 1.0
    }
  ],
  "total_count": 15,
  "has_more": false
}
```

#### DELETE `/training/jobs/{job_id}/cancel`

Cancel a running training job.

**Response:**

```json
{
  "success": true,
  "message": "Training job cancelled successfully",
  "job_id": "job_xyz789",
  "cancelled_at": "2024-01-15T10:15:00Z"
}
```

### Prediction

#### POST `/prediction/predict`

Make predictions using trained models with intelligent fallback.

**Request:**

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1d",
  "model_type": "ibm/patchtst-forecasting",
  "n_steps": 3,
  "enable_fallback": true,
  "input_data": {
    "close": [45000, 45100, 45200, 45300, 45400],
    "volume": [1000, 1100, 1200, 1300, 1400],
    "timestamp": [
      "2024-01-10",
      "2024-01-11",
      "2024-01-12",
      "2024-01-13",
      "2024-01-14"
    ]
  }
}
```

**Response:**

```json
{
  "success": true,
  "predictions": [45500.5, 45620.3, 45780.1],
  "prediction_percentages": [2.2, 1.15, 0.35],
  "prediction_timestamps": ["2024-01-15", "2024-01-16", "2024-01-17"],
  "current_price": 45400,
  "predicted_change_pct": 2.2,
  "confidence_score": 0.87,
  "model_info": {
    "model_path": "/models/BTCUSDT_1d_patchtst/model.pt",
    "training_date": "2024-01-10",
    "performance_metrics": {
      "mae": 0.0234,
      "rmse": 0.0345
    }
  },
  "fallback_info": {
    "fallback_applied": false,
    "original_request": {},
    "selected_model": {},
    "confidence_score": 0.87
  }
}
```

### Models

#### GET `/models/info`

Get information about available models and system capabilities.

**Response:**

```json
{
  "success": true,
  "available_models": [
    "ibm/patchtst-forecasting",
    "ibm/patchtsmixer-forecasting",
    "pytorch-lightning/time-series-transformer",
    "enhanced-transformer"
  ],
  "trained_models": {
    "BTCUSDT": {
      "1d": {
        "ibm/patchtst-forecasting": {
          "model_path": "/models/BTCUSDT_1d_patchtst/",
          "created_at": "2024-01-10T09:00:00Z",
          "performance": {
            "mae": 0.0234,
            "rmse": 0.0345
          }
        }
      }
    }
  },
  "supported_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d", "1w"],
  "supported_symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
}
```

#### GET `/models/available`

List all available trained models with metadata.

**Response:**

```json
{
  "success": true,
  "models": [
    {
      "symbol": "BTCUSDT",
      "timeframe": "1d",
      "model_type": "ibm/patchtst-forecasting",
      "model_path": "/models/BTCUSDT_1d_patchtst/",
      "created_at": "2024-01-10T09:00:00Z",
      "file_size_mb": 45.2,
      "is_available": true,
      "config": {
        "context_length": 64,
        "prediction_length": 1,
        "num_epochs": 10
      }
    }
  ],
  "total_count": 12
}
```

#### GET `/models/check`

Check if a specific model exists.

**Query Parameters:**

- `symbol`: Trading symbol (required)
- `timeframe`: Data timeframe (required)
- `model_type`: Model type (optional)

**Response:**

```json
{
  "success": true,
  "exists": true,
  "model_info": {
    "symbol": "BTCUSDT",
    "timeframe": "1d",
    "model_type": "ibm/patchtst-forecasting",
    "model_path": "/models/BTCUSDT_1d_patchtst/",
    "created_at": "2024-01-10T09:00:00Z"
  }
}
```

### Serving

#### GET `/serving/health`

Get serving adapter health status.

**Response:**

```json
{
  "success": true,
  "adapter_type": "simple",
  "status": "healthy",
  "models_loaded": 3,
  "total_memory_usage_mb": 156.7,
  "uptime_seconds": 86400,
  "last_health_check": "2024-01-15T10:30:00Z"
}
```

#### GET `/serving/stats`

Get serving adapter statistics.

**Response:**

```json
{
  "success": true,
  "total_predictions": 1250,
  "successful_predictions": 1234,
  "failed_predictions": 16,
  "average_inference_time_ms": 45.2,
  "models_loaded": 3,
  "total_memory_usage_mb": 156.7,
  "uptime_seconds": 86400
}
```

#### GET `/serving/models`

List models currently loaded in the serving adapter.

**Response:**

```json
{
  "success": true,
  "models": [
    {
      "model_id": "BTCUSDT_1d_patchtst",
      "symbol": "BTCUSDT",
      "timeframe": "1d",
      "model_type": "ibm/patchtst-forecasting",
      "is_loaded": true,
      "loaded_at": "2024-01-15T09:00:00Z",
      "memory_usage_mb": 45.2,
      "version": "1.0"
    }
  ]
}
```

#### POST `/serving/models/load`

Load a model into the serving adapter.

**Request:**

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1d",
  "model_type": "ibm/patchtst-forecasting"
}
```

**Response:**

```json
{
  "success": true,
  "message": "Model loaded successfully",
  "model_id": "BTCUSDT_1d_patchtst",
  "memory_usage_mb": 45.2,
  "load_time_seconds": 1.8
}
```

#### DELETE `/serving/models/unload`

Unload a model from the serving adapter.

**Request:**

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1d",
  "model_type": "ibm/patchtst-forecasting"
}
```

**Response:**

```json
{
  "success": true,
  "message": "Model unloaded successfully",
  "model_id": "BTCUSDT_1d_patchtst",
  "freed_memory_mb": 45.2
}
```

#### POST `/serving/models/predict`

Make predictions using the serving adapter.

**Request:**

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1d",
  "model_type": "ibm/patchtst-forecasting",
  "input_data": {
    "close": [45000, 45100, 45200, 45300, 45400],
    "volume": [1000, 1100, 1200, 1300, 1400]
  },
  "n_steps": 1
}
```

**Response:**

```json
{
  "success": true,
  "predictions": [45500.5],
  "inference_time_ms": 23.4,
  "model_info": {
    "model_id": "BTCUSDT_1d_patchtst",
    "symbol": "BTCUSDT",
    "timeframe": "1d",
    "model_type": "ibm/patchtst-forecasting"
  }
}
```

### Datasets

#### GET `/datasets/`

List available datasets with filtering and pagination.

**Query Parameters:**

- `exchange_filter`: Filter by exchange (default: binance)
- `symbol_filter`: Filter by trading symbol
- `timeframe_filter`: Filter by timeframe
- `format_filter`: Filter by data format
- `include_cached`: Include cached datasets (default: true)
- `include_cloud`: Include cloud datasets (default: true)
- `limit`: Maximum results (default: 100)
- `offset`: Result offset (default: 0)

**Response:**

```json
{
  "success": true,
  "datasets": [
    {
      "exchange": "binance",
      "symbol": "BTCUSDT",
      "timeframe": "1h",
      "format_type": "csv",
      "size_bytes": 1048576,
      "last_modified": "2024-01-15T10:00:00Z",
      "is_cached": true,
      "cache_age_hours": 2.5
    }
  ],
  "total_count": 45,
  "filtered_count": 12,
  "has_more": false
}
```

#### GET `/datasets/availability/{symbol}/{timeframe}`

Check dataset availability for a specific symbol and timeframe.

**Response:**

```json
{
  "success": true,
  "exists": true,
  "available_sources": ["cloud", "cache", "local"],
  "cloud_available": true,
  "cache_available": true,
  "local_available": true,
  "dataset_info": {
    "exchange": "binance",
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "size_bytes": 1048576,
    "last_modified": "2024-01-15T10:00:00Z"
  },
  "recommended_action": "use_cache"
}
```

#### POST `/datasets/download`

Download a dataset to local storage or cache.

**Request:**

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "exchange": "binance",
  "target_format": "csv",
  "force_download": false,
  "update_cache": true
}
```

**Response:**

```json
{
  "success": true,
  "download_id": "dl_abc123",
  "dataset_info": {
    "exchange": "binance",
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "size_bytes": 1048576
  },
  "local_path": "/data/binance/BTCUSDT_1h.csv",
  "cached_path": "/tmp/cloud_cache/BTCUSDT_1h.csv",
  "download_duration_seconds": 12.5,
  "download_speed_mbps": 8.2
}
```

### Cloud Storage

#### GET `/cloud-storage/health`

Check cloud storage health and connectivity.

**Response:**

```json
{
  "success": true,
  "provider": "minio",
  "status": "healthy",
  "endpoint": "http://localhost:9000",
  "bucket": "market-data",
  "connectivity": "connected",
  "last_check": "2024-01-15T10:30:00Z"
}
```

#### POST `/cloud-storage/models/sync-to-cloud`

Sync a local model to cloud storage.

**Request:**

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1d",
  "model_type": "ibm/patchtst-forecasting",
  "adapter_type": "simple",
  "force_upload": false
}
```

**Response:**

```json
{
  "success": true,
  "message": "Model synced to cloud successfully",
  "cloud_path": "finsight/models/BTCUSDT_1d_patchtst/",
  "local_path": "/models/BTCUSDT_1d_patchtst/",
  "sync_duration_seconds": 8.5,
  "uploaded_files": 5
}
```

#### POST `/cloud-storage/models/sync-from-cloud`

Sync a model from cloud storage to local storage.

**Request:**

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1d",
  "model_type": "ibm/patchtst-forecasting",
  "adapter_type": "simple",
  "force_download": false
}
```

**Response:**

```json
{
  "success": true,
  "message": "Model synced from cloud successfully",
  "local_path": "/models/BTCUSDT_1d_patchtst/",
  "cloud_path": "finsight/models/BTCUSDT_1d_patchtst/",
  "sync_duration_seconds": 12.3,
  "downloaded_files": 5
}
```

### Eureka Client

#### GET `/eureka/status`

Get Eureka client status and registration information.

**Response:**

```json
{
  "success": true,
  "status": "registered",
  "server_url": "http://localhost:8761",
  "app_name": "prediction-service",
  "instance_id": "prediction-service-1",
  "registration_time": "2024-01-15T09:00:00Z",
  "last_heartbeat": "2024-01-15T10:30:00Z",
  "lease_info": {
    "renewal_interval": 30,
    "expiration_duration": 90
  }
}
```

#### POST `/eureka/register`

Manually register with Eureka server.

**Response:**

```json
{
  "success": true,
  "message": "Successfully registered with Eureka",
  "registration_id": "reg_abc123",
  "server_url": "http://localhost:8761"
}
```

#### POST `/eureka/deregister`

Manually deregister from Eureka server.

**Response:**

```json
{
  "success": true,
  "message": "Successfully deregistered from Eureka",
  "server_url": "http://localhost:8761"
}
```

### Cleanup

#### POST `/cleanup/cloud-cache`

Clean up expired cloud cache files.

**Response:**

```json
{
  "success": true,
  "message": "Cloud cache cleanup completed",
  "cleaned_files": 15,
  "freed_space_bytes": 104857600,
  "cleanup_duration_seconds": 8.5
}
```

#### POST `/cleanup/datasets`

Clean up old dataset files.

**Response:**

```json
{
  "success": true,
  "message": "Dataset cleanup completed",
  "cleaned_files": 8,
  "freed_space_bytes": 52428800,
  "cleanup_duration_seconds": 12.3
}
```

#### POST `/cleanup/models`

Clean up old model files.

**Response:**

```json
{
  "success": true,
  "message": "Model cleanup completed",
  "cleaned_files": 3,
  "freed_space_bytes": 157286400,
  "cleanup_duration_seconds": 5.2
}
```

#### POST `/cleanup/all`

Perform comprehensive cleanup of all targets.

**Response:**

```json
{
  "success": true,
  "message": "Comprehensive cleanup completed",
  "total_cleaned_files": 26,
  "total_freed_space_bytes": 314572800,
  "cleanup_duration_seconds": 25.8,
  "results": {
    "cloud_cache": {
      "cleaned_files": 15,
      "freed_space_bytes": 104857600
    },
    "datasets": {
      "cleaned_files": 8,
      "freed_space_bytes": 52428800
    },
    "models": {
      "cleaned_files": 3,
      "freed_space_bytes": 157286400
    }
  }
}
```

## 🔧 Rate Limiting

The service implements configurable rate limiting:

- **Public Endpoints**: 100 requests per minute
- **Training Endpoints**: 10 requests per minute
- **Admin Endpoints**: 50 requests per minute

Rate limit headers are included in responses:

```bash
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642233600
```

## 📝 Request/Response Examples

### Complete Training Workflow

1. **Start Training**

   ```bash
   curl -X POST "http://localhost:8000/training/train-async" \
     -H "Content-Type: application/json" \
     -d '{
       "symbol": "BTCUSDT",
       "timeframe": "1d",
       "model_type": "ibm/patchtst-forecasting",
       "config": {
         "context_length": 64,
         "prediction_length": 1,
         "num_epochs": 10,
         "batch_size": 32,
         "learning_rate": 0.001
       }
     }'
   ```

2. **Check Status**

   ```bash
   curl "http://localhost:8000/training/status/job_abc123"
   ```

3. **Make Prediction**

   ```bash
   curl -X POST "http://localhost:8000/prediction/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "symbol": "BTCUSDT",
       "timeframe": "1d",
       "model_type": "ibm/patchtst-forecasting",
       "n_steps": 1,
       "enable_fallback": true
     }'
   ```

## 🚨 Troubleshooting

### Common Issues

1. **Model Not Found**

   - Check if model exists: `GET /models/check`
   - Verify model path and permissions
   - Check training job status

2. **Training Failures**

   - Verify data availability: `GET /datasets/availability/{symbol}/{timeframe}`
   - Check system resources and GPU availability
   - Review training logs

3. **Serving Errors**
   - Check serving health: `GET /serving/health`
   - Verify model loading: `GET /serving/models`
   - Check memory usage and model cache

### Debug Endpoints

- **Health Check**: `GET /health` - Overall system health
- **Serving Health**: `GET /serving/health` - Model serving status
- **Eureka Status**: `GET /eureka/status` - Service discovery status
- **Cleanup Status**: `GET /cleanup/status` - Background maintenance status

---

**For more information, see the [Architecture Documentation](architecture.md) and [Configuration Guide](configuration.md).**
