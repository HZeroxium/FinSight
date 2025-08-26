# FinSight Prediction Service - Configuration Guide

> **Complete Configuration Reference for Environment Variables and Settings**

## 🌐 Overview

The FinSight Prediction Service uses environment variables for configuration, following the **12-Factor App** methodology. All configuration is externalized and can be overridden at runtime without code changes.

Configuration is loaded using Pydantic's `BaseSettings` with automatic environment variable parsing and validation.

## 🏗️ Configuration Structure

The service configuration is organized into logical groups:

```bash
Configuration
├── Service Configuration
├── Eureka Client
├── Model Management
├── Training & Fallback
├── Device & Performance
├── Serving Adapters
├── Cloud Storage
├── Data Management
├── Experiment Tracking
└── Cleanup & Maintenance
```

## 🔧 Environment Variables

### Service Configuration

| Variable      | Default                         | Required | Description              |
| ------------- | ------------------------------- | -------- | ------------------------ |
| `APP_NAME`    | `"FinSight Prediction Service"` | ❌       | Application display name |
| `APP_VERSION` | `"1.0.0"`                       | ❌       | Application version      |
| `DEBUG`       | `false`                         | ❌       | Enable debug mode        |
| `API_HOST`    | `"0.0.0.0"`                     | ❌       | API binding host         |
| `API_PORT`    | `8000`                          | ❌       | API binding port         |

**Example:**

```bash
APP_NAME="FinSight AI Prediction Service"
APP_VERSION="2.0.0"
DEBUG=true
API_HOST="127.0.0.1"
API_PORT=8080
```

### Eureka Client Configuration

| Variable                     | Default                   | Required | Description                       |
| ---------------------------- | ------------------------- | -------- | --------------------------------- |
| `ENABLE_EUREKA_CLIENT`       | `true`                    | ❌       | Enable Eureka client registration |
| `EUREKA_SERVER_URL`          | `"http://localhost:8761"` | ❌       | Eureka server URL                 |
| `EUREKA_APP_NAME`            | `"prediction-service"`    | ❌       | Application name for registration |
| `EUREKA_INSTANCE_ID`         | `None`                    | ❌       | Custom instance ID                |
| `EUREKA_HOST_NAME`           | `None`                    | ❌       | Custom hostname                   |
| `EUREKA_IP_ADDRESS`          | `None`                    | ❌       | Custom IP address                 |
| `EUREKA_PORT`                | `8000`                    | ❌       | Service port for registration     |
| `EUREKA_SECURE_PORT`         | `8443`                    | ❌       | Secure port for registration      |
| `EUREKA_SECURE_PORT_ENABLED` | `false`                   | ❌       | Enable secure port                |

**Timing Configuration:**

| Variable                                      | Default | Required | Description             |
| --------------------------------------------- | ------- | -------- | ----------------------- |
| `EUREKA_LEASE_RENEWAL_INTERVAL_IN_SECONDS`    | `30`    | ❌       | Heartbeat interval      |
| `EUREKA_LEASE_EXPIRATION_DURATION_IN_SECONDS` | `90`    | ❌       | Lease expiration time   |
| `EUREKA_REGISTRY_FETCH_INTERVAL_SECONDS`      | `30`    | ❌       | Registry fetch interval |
| `EUREKA_HEARTBEAT_INTERVAL_SECONDS`           | `30`    | ❌       | Heartbeat interval      |

**Retry Configuration:**

| Variable                                  | Default | Required | Description                    |
| ----------------------------------------- | ------- | -------- | ------------------------------ |
| `EUREKA_REGISTRATION_RETRY_ATTEMPTS`      | `3`     | ❌       | Registration retry attempts    |
| `EUREKA_REGISTRATION_RETRY_DELAY_SECONDS` | `5`     | ❌       | Initial retry delay            |
| `EUREKA_HEARTBEAT_RETRY_ATTEMPTS`         | `3`     | ❌       | Heartbeat retry attempts       |
| `EUREKA_RETRY_BACKOFF_MULTIPLIER`         | `2.0`   | ❌       | Exponential backoff multiplier |
| `EUREKA_MAX_RETRY_DELAY_SECONDS`          | `60`    | ❌       | Maximum retry delay            |

**Example:**

```bash
ENABLE_EUREKA_CLIENT=true
EUREKA_SERVER_URL="http://eureka-server:8761"
EUREKA_APP_NAME="prediction-service-prod"
EUREKA_INSTANCE_ID="prediction-service-1"
EUREKA_PORT=8000
EUREKA_LEASE_RENEWAL_INTERVAL_IN_SECONDS=30
EUREKA_LEASE_EXPIRATION_DURATION_IN_SECONDS=90
```

### Model Management

| Variable     | Default             | Required | Description                    |
| ------------ | ------------------- | -------- | ------------------------------ |
| `BASE_DIR`   | `Auto-detected`     | ❌       | Base application directory     |
| `DATA_DIR`   | `{BASE_DIR}/data`   | ❌       | Data storage directory         |
| `MODELS_DIR` | `{BASE_DIR}/models` | ❌       | Model storage directory        |
| `LOGS_DIR`   | `{BASE_DIR}/logs`   | ❌       | Log storage directory          |
| `JOBS_DIR`   | `{BASE_DIR}/jobs`   | ❌       | Training job storage directory |

**File Patterns:**

| Variable              | Default                               | Required | Description                  |
| --------------------- | ------------------------------------- | -------- | ---------------------------- |
| `MODEL_NAME_PATTERN`  | `"{symbol}_{timeframe}_{model_type}"` | ❌       | Model naming pattern         |
| `CHECKPOINT_FILENAME` | `"model.pt"`                          | ❌       | Model checkpoint filename    |
| `METADATA_FILENAME`   | `"metadata.json"`                     | ❌       | Model metadata filename      |
| `CONFIG_FILENAME`     | `"config.json"`                       | ❌       | Model configuration filename |

**Example:**

```bash
BASE_DIR="/app"
DATA_DIR="/app/data"
MODELS_DIR="/app/models"
LOGS_DIR="/app/logs"
JOBS_DIR="/app/jobs"
MODEL_NAME_PATTERN="{symbol}_{timeframe}_{model_type}_v{version}"
```

### Training Configuration

| Variable                    | Default | Required | Description                   |
| --------------------------- | ------- | -------- | ----------------------------- |
| `DEFAULT_CONTEXT_LENGTH`    | `64`    | ❌       | Default input sequence length |
| `DEFAULT_PREDICTION_LENGTH` | `1`     | ❌       | Default prediction horizon    |
| `DEFAULT_NUM_EPOCHS`        | `10`    | ❌       | Default training epochs       |
| `DEFAULT_BATCH_SIZE`        | `32`    | ❌       | Default batch size            |
| `DEFAULT_LEARNING_RATE`     | `0.001` | ❌       | Default learning rate         |

**Model Limits:**

| Variable                | Default | Required | Description               |
| ----------------------- | ------- | -------- | ------------------------- |
| `MAX_CONTEXT_LENGTH`    | `512`   | ❌       | Maximum context length    |
| `MAX_PREDICTION_LENGTH` | `24`    | ❌       | Maximum prediction length |
| `MAX_NUM_EPOCHS`        | `100`   | ❌       | Maximum training epochs   |

**Example:**

```bash
DEFAULT_CONTEXT_LENGTH=128
DEFAULT_PREDICTION_LENGTH=3
DEFAULT_NUM_EPOCHS=20
DEFAULT_BATCH_SIZE=64
DEFAULT_LEARNING_RATE=0.0005
MAX_CONTEXT_LENGTH=1024
MAX_PREDICTION_LENGTH=48
```

### Fallback Strategy

| Variable                | Default                  | Required | Description                 |
| ----------------------- | ------------------------ | -------- | --------------------------- |
| `ENABLE_MODEL_FALLBACK` | `true`                   | ❌       | Enable intelligent fallback |
| `FALLBACK_STRATEGY`     | `"timeframe_and_symbol"` | ❌       | Fallback strategy type      |
| `MAX_FALLBACK_ATTEMPTS` | `5`                      | ❌       | Maximum fallback attempts   |
| `FALLBACK_TIMEOUT`      | `30.0`                   | ❌       | Fallback operation timeout  |

**Priority Lists (comma-separated):**

| Variable                       | Default                                                                                                                  | Required | Description                  |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | -------- | ---------------------------- |
| `FALLBACK_TIMEFRAME_PRIORITY`  | `"1d,4h,1h,15m,5m,1m,12h,1w"`                                                                                            | ❌       | Timeframe fallback priority  |
| `FALLBACK_SYMBOL_PRIORITY`     | `"BTCUSDT,ETHUSDT,BNBUSDT"`                                                                                              | ❌       | Symbol fallback priority     |
| `FALLBACK_MODEL_TYPE_PRIORITY` | `"ibm/patchtst-forecasting,ibm/patchtsmixer-forecasting,pytorch-lightning/time-series-transformer,enhanced-transformer"` | ❌       | Model type fallback priority |

**Example:**

```bash
ENABLE_MODEL_FALLBACK=true
FALLBACK_STRATEGY="timeframe_and_symbol"
MAX_FALLBACK_ATTEMPTS=3
FALLBACK_TIMEOUT=15.0
FALLBACK_TIMEFRAME_PRIORITY="1d,4h,1h,15m"
FALLBACK_SYMBOL_PRIORITY="BTCUSDT,ETHUSDT"
```

### Device Configuration

| Variable                      | Default | Required | Description                     |
| ----------------------------- | ------- | -------- | ------------------------------- |
| `FORCE_CPU`                   | `false` | ❌       | Force CPU usage (overrides GPU) |
| `CUDA_VISIBLE_DEVICES`        | `None`  | ❌       | CUDA device selection           |
| `CUDA_DEVICE_MEMORY_FRACTION` | `0.8`   | ❌       | GPU memory usage fraction       |
| `ENABLE_MIXED_PRECISION`      | `true`  | ❌       | Enable mixed precision training |

**Example:**

```bash
FORCE_CPU=false
CUDA_VISIBLE_DEVICES="0,1"
CUDA_DEVICE_MEMORY_FRACTION=0.9
ENABLE_MIXED_PRECISION=true
```

### Serving Adapters

| Variable               | Default    | Required | Description             |
| ---------------------- | ---------- | -------- | ----------------------- |
| `SERVING_ADAPTER_TYPE` | `"simple"` | ❌       | Primary serving backend |

**Simple Adapter:**

| Variable                       | Default | Required | Description              |
| ------------------------------ | ------- | -------- | ------------------------ |
| `SIMPLE_MAX_MODELS_IN_MEMORY`  | `5`     | ❌       | Maximum models in memory |
| `SIMPLE_MODEL_TIMEOUT_SECONDS` | `3600`  | ❌       | Model timeout in seconds |

**Triton Adapter:**

| Variable                  | Default            | Required | Description                |
| ------------------------- | ------------------ | -------- | -------------------------- |
| `TRITON_SERVER_URL`       | `"localhost:8000"` | ❌       | Triton HTTP server URL     |
| `TRITON_SERVER_GRPC_URL`  | `"localhost:8001"` | ❌       | Triton gRPC server URL     |
| `TRITON_USE_GRPC`         | `false`            | ❌       | Use gRPC instead of HTTP   |
| `TRITON_SSL`              | `false`            | ❌       | Enable SSL/TLS             |
| `TRITON_INSECURE`         | `true`             | ❌       | Allow insecure connections |
| `TRITON_MODEL_REPOSITORY` | `"/models"`        | ❌       | Model repository path      |
| `TRITON_MAX_BATCH_SIZE`   | `8`                | ❌       | Maximum batch size         |
| `TRITON_TIMEOUT_SECONDS`  | `30`               | ❌       | Request timeout            |

**TorchServe Adapter:**

| Variable                     | Default                   | Required | Description               |
| ---------------------------- | ------------------------- | -------- | ------------------------- |
| `TORCHSERVE_INFERENCE_URL`   | `"http://localhost:8080"` | ❌       | Inference API URL         |
| `TORCHSERVE_MANAGEMENT_URL`  | `"http://localhost:8081"` | ❌       | Management API URL        |
| `TORCHSERVE_MODEL_STORE`     | `"./model_store"`         | ❌       | Model store directory     |
| `TORCHSERVE_BATCH_SIZE`      | `1`                       | ❌       | Batch size for inference  |
| `TORCHSERVE_MAX_BATCH_DELAY` | `100`                     | ❌       | Maximum batch delay (ms)  |
| `TORCHSERVE_TIMEOUT_SECONDS` | `30`                      | ❌       | Response timeout          |
| `TORCHSERVE_INITIAL_WORKERS` | `1`                       | ❌       | Initial workers per model |
| `TORCHSERVE_MAX_WORKERS`     | `4`                       | ❌       | Maximum workers per model |

**Example:**

```bash
SERVING_ADAPTER_TYPE="triton"
TRITON_SERVER_URL="triton-server:8000"
TRITON_SERVER_GRPC_URL="triton-server:8001"
TRITON_USE_GRPC=true
TRITON_SSL=true
TRITON_INSECURE=false
TRITON_MODEL_REPOSITORY="/models"
TRITON_MAX_BATCH_SIZE=16
TRITON_TIMEOUT_SECONDS=60
```

### Cloud Storage

| Variable                  | Default   | Required | Description                                 |
| ------------------------- | --------- | -------- | ------------------------------------------- |
| `STORAGE_PROVIDER`        | `"minio"` | ❌       | Storage provider (minio, digitalocean, aws) |
| `ENABLE_CLOUD_STORAGE`    | `true`    | ❌       | Enable cloud storage                        |
| `ENABLE_MODEL_CLOUD_SYNC` | `true`    | ❌       | Enable model cloud sync                     |

**S3-Compatible Storage (MinIO, AWS S3):**

| Variable                  | Default                   | Required | Description             |
| ------------------------- | ------------------------- | -------- | ----------------------- |
| `S3_ENDPOINT_URL`         | `"http://localhost:9000"` | ❌       | S3 endpoint URL         |
| `S3_ACCESS_KEY`           | `"minioadmin"`            | ❌       | Access key ID           |
| `S3_SECRET_KEY`           | `"minioadmin"`            | ❌       | Secret access key       |
| `S3_REGION_NAME`          | `"us-east-1"`             | ❌       | AWS region              |
| `S3_BUCKET_NAME`          | `"market-data"`           | ❌       | S3 bucket name          |
| `S3_USE_SSL`              | `false`                   | ❌       | Use HTTPS               |
| `S3_VERIFY_SSL`           | `true`                    | ❌       | Verify SSL certificates |
| `S3_SIGNATURE_VERSION`    | `"s3v4"`                  | ❌       | S3 signature version    |
| `S3_MAX_POOL_CONNECTIONS` | `50`                      | ❌       | Connection pool size    |

**DigitalOcean Spaces:**

| Variable              | Default                                 | Required | Description         |
| --------------------- | --------------------------------------- | -------- | ------------------- |
| `SPACES_ENDPOINT_URL` | `"https://nyc3.digitaloceanspaces.com"` | ❌       | Spaces endpoint URL |
| `SPACES_ACCESS_KEY`   | `""`                                    | ❌       | Spaces access key   |
| `SPACES_SECRET_KEY`   | `""`                                    | ❌       | Spaces secret key   |
| `SPACES_REGION_NAME`  | `"nyc3"`                                | ❌       | Spaces region       |
| `SPACES_BUCKET_NAME`  | `"finsight-market-data"`                | ❌       | Spaces bucket name  |

**AWS S3:**

| Variable                | Default                  | Required | Description           |
| ----------------------- | ------------------------ | -------- | --------------------- |
| `AWS_ACCESS_KEY_ID`     | `""`                     | ❌       | AWS access key ID     |
| `AWS_SECRET_ACCESS_KEY` | `""`                     | ❌       | AWS secret access key |
| `AWS_REGION_NAME`       | `"us-east-1"`            | ❌       | AWS region            |
| `AWS_BUCKET_NAME`       | `"finsight-market-data"` | ❌       | S3 bucket name        |

**Storage Paths:**

| Variable                 | Default                           | Required | Description            |
| ------------------------ | --------------------------------- | -------- | ---------------------- |
| `DATASET_STORAGE_PREFIX` | `"finsight/market_data/datasets"` | ❌       | Dataset storage prefix |
| `MODEL_STORAGE_PREFIX`   | `"finsight/models"`               | ❌       | Model storage prefix   |
| `STORAGE_SEPARATOR`      | `"/"`                             | ❌       | Path separator         |

**Example:**

```bash
STORAGE_PROVIDER="aws"
ENABLE_CLOUD_STORAGE=true
ENABLE_MODEL_CLOUD_SYNC=true
AWS_ACCESS_KEY_ID="AKIA..."
AWS_SECRET_ACCESS_KEY="..."
AWS_REGION_NAME="us-west-2"
AWS_BUCKET_NAME="finsight-prod-models"
DATASET_STORAGE_PREFIX="finsight/market_data/datasets"
MODEL_STORAGE_PREFIX="finsight/models"
```

### Data Management

| Variable                     | Default    | Required | Description                             |
| ---------------------------- | ---------- | -------- | --------------------------------------- |
| `DATA_LOADER_TYPE`           | `"hybrid"` | ❌       | Data loader type (local, cloud, hybrid) |
| `ENABLE_CLOUD_STORAGE`       | `true`     | ❌       | Enable cloud data loading               |
| `CLOUD_DATA_CACHE_TTL_HOURS` | `24`       | ❌       | Cloud data cache TTL                    |

**Default Data Sources:**

| Variable             | Default                     | Required | Description             |
| -------------------- | --------------------------- | -------- | ----------------------- |
| `DEFAULT_SYMBOLS`    | `"BTCUSDT,ETHUSDT,BNBUSDT"` | ❌       | Default trading symbols |
| `DEFAULT_TIMEFRAMES` | `"1h,4h,1d"`                | ❌       | Default data timeframes |

**Example:**

```bash
DATA_LOADER_TYPE="hybrid"
ENABLE_CLOUD_STORAGE=true
CLOUD_DATA_CACHE_TTL_HOURS=48
DEFAULT_SYMBOLS="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT"
DEFAULT_TIMEFRAMES="1h,4h,1d,1w"
```

### Experiment Tracking

| Variable                      | Default    | Required | Description                   |
| ----------------------------- | ---------- | -------- | ----------------------------- |
| `EXPERIMENT_TRACKER_TYPE`     | `"simple"` | ❌       | Tracker type (simple, mlflow) |
| `EXPERIMENT_TRACKER_FALLBACK` | `"simple"` | ❌       | Fallback tracker type         |

**MLflow Configuration:**

| Variable                 | Default                   | Required | Description            |
| ------------------------ | ------------------------- | -------- | ---------------------- |
| `MLFLOW_TRACKING_URI`    | `"http://localhost:5000"` | ❌       | MLflow tracking server |
| `MLFLOW_EXPERIMENT_NAME` | `"finsight-ml"`           | ❌       | MLflow experiment name |

**Example:**

```bash
EXPERIMENT_TRACKER_TYPE="mlflow"
EXPERIMENT_TRACKER_FALLBACK="simple"
MLFLOW_TRACKING_URI="http://mlflow-server:5000"
MLFLOW_EXPERIMENT_NAME="finsight-production-models"
```

### Cleanup Configuration

| Variable                     | Default | Required | Description                 |
| ---------------------------- | ------- | -------- | --------------------------- |
| `CLEANUP_INTERVAL`           | `"1d"`  | ❌       | Background cleanup interval |
| `ENABLE_CLOUD_CACHE_CLEANUP` | `true`  | ❌       | Enable cloud cache cleanup  |
| `ENABLE_DATASETS_CLEANUP`    | `true`  | ❌       | Enable dataset cleanup      |
| `ENABLE_MODELS_CLEANUP`      | `true`  | ❌       | Enable model cleanup        |

**Cleanup Thresholds:**

| Variable                    | Default | Required | Description              |
| --------------------------- | ------- | -------- | ------------------------ |
| `CLOUD_CACHE_MAX_AGE_HOURS` | `24`    | ❌       | Cloud cache max age      |
| `DATASETS_MAX_AGE_HOURS`    | `168`   | ❌       | Dataset max age (7 days) |
| `MODELS_MAX_AGE_HOURS`      | `720`   | ❌       | Model max age (30 days)  |

**Example:**

```bash
CLEANUP_INTERVAL="6h"
ENABLE_CLOUD_CACHE_CLEANUP=true
ENABLE_DATASETS_CLEANUP=true
ENABLE_MODELS_CLEANUP=true
CLOUD_CACHE_MAX_AGE_HOURS=12
DATASETS_MAX_AGE_HOURS=72
MODELS_MAX_AGE_HOURS=360
```

## 📝 Configuration Examples

### Development Environment

```bash
# .env.development
DEBUG=true
API_PORT=8000
ENABLE_EUREKA_CLIENT=false
STORAGE_PROVIDER=minio
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
SERVING_ADAPTER_TYPE=simple
EXPERIMENT_TRACKER_TYPE=simple
CLEANUP_INTERVAL=1h
```

### Production Environment

```bash
# .env.production
DEBUG=false
API_PORT=8000
ENABLE_EUREKA_CLIENT=true
EUREKA_SERVER_URL=http://eureka-server:8761
STORAGE_PROVIDER=aws
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION_NAME=us-west-2
AWS_BUCKET_NAME=finsight-prod
SERVING_ADAPTER_TYPE=triton
TRITON_SERVER_URL=triton-server:8000
EXPERIMENT_TRACKER_TYPE=mlflow
MLFLOW_TRACKING_URI=http://mlflow-server:5000
CLEANUP_INTERVAL=1d
```

### Docker Compose Environment

```bash
# docker-compose.yml environment section
environment:
  - STORAGE_PROVIDER=minio
  - S3_ENDPOINT_URL=http://minio:9000
  - S3_ACCESS_KEY=${MINIO_ACCESS_KEY}
  - S3_SECRET_KEY=${MINIO_SECRET_KEY}
  - EUREKA_SERVER_URL=http://eureka-server:8761
  - FORCE_CPU=true
  - SERVING_ADAPTER_TYPE=simple
```

## ✅ Validation Rules

The service includes comprehensive configuration validation:

### Numeric Ranges

- **Ports**: 1-65535
- **Context Length**: 10-1000
- **Prediction Length**: 1-100
- **Epochs**: 1-200
- **Batch Size**: 1-512
- **Learning Rate**: 1e-6 to 1.0
- **CUDA Memory Fraction**: 0.0 to 1.0

### String Patterns

- **Timeframes**: Must match supported values (1m, 5m, 15m, 1h, 4h, 1d, 1w)
- **Symbols**: Must be valid trading pairs (e.g., BTCUSDT)
- **Model Types**: Must match supported model identifiers
- **Storage Providers**: Must be one of (minio, digitalocean, aws, s3)

### Required Dependencies

- **Eureka Client**: Requires valid server URL if enabled
- **Cloud Storage**: Requires access keys and endpoint URLs
- **MLflow**: Requires tracking server URI if enabled
- **Triton**: Requires server URLs and model repository path

## 🎯 Best Practices

### Security

1. **Never commit secrets** to version control
2. **Use environment variables** for all sensitive data
3. **Rotate credentials** regularly
4. **Limit access** to production configurations

### Performance

1. **Optimize batch sizes** for your hardware
2. **Configure appropriate timeouts** for external services
3. **Set reasonable cleanup intervals** to balance storage and performance
4. **Use connection pooling** for database and storage connections

### Monitoring

1. **Enable health checks** for all dependencies
2. **Configure appropriate logging levels**
3. **Set up metrics collection** for serving adapters
4. **Monitor cleanup operations** for storage efficiency

### Scalability

1. **Use external storage** for models and data
2. **Configure appropriate cache sizes** for your memory constraints
3. **Set up service discovery** for multi-instance deployments
4. **Use load balancing** for high-traffic scenarios

## 🔍 Configuration Validation

The service validates configuration at startup:

```python
# Configuration validation example
try:
    settings = get_settings()
    settings.validate_device_config()
    settings.validate_storage_config()
    logger.info("Configuration validation passed")
except ValidationError as e:
    logger.error(f"Configuration validation failed: {e}")
    sys.exit(1)
```

## 🚨 Troubleshooting

### Common Configuration Issues

1. **Invalid Port Numbers**

   ```bash
   Error: Port must be between 1 and 65535
   Solution: Check API_PORT and EUREKA_PORT values
   ```

2. **Missing Cloud Storage Credentials**

   ```bash
   Error: Cloud storage enabled but credentials missing
   Solution: Set appropriate access keys and secret keys
   ```

3. **Invalid Timeframe Values**

   ```bash
   Error: Invalid timeframe '2h' (must be one of: 1m, 5m, 15m, 1h, 4h, 1d, 1w)
   Solution: Use supported timeframe values
   ```

4. **Eureka Connection Failures**

   ```bash
   Error: Failed to connect to Eureka server
   Solution: Verify EUREKA_SERVER_URL and network connectivity
   ```

### Configuration Debugging

Enable debug logging to see configuration loading:

```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

Check configuration at runtime:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/eureka/config
```

---

**For more information, see the [Architecture Documentation](architecture.md) and [API Documentation](api.md).**
