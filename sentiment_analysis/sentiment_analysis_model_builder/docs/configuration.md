# FinSight Sentiment Analysis Model Builder - Configuration Guide

> **Complete Configuration Reference for Model Training, Export, and Registry Management**  
> Comprehensive guide to environment variables, configuration files, and deployment settings

## 🔧 Configuration Overview

The FinSight Sentiment Analysis Model Builder uses a hierarchical configuration system based on Pydantic v2 with environment variable support. Configuration is organized into logical sections for different aspects of the service.

### **Configuration Hierarchy**

1. **Environment Variables** (highest priority)
2. **Configuration Files** (YAML/JSON)
3. **Default Values** (lowest priority)

### **Configuration Sections**

- **Data Configuration**: Input formats, preprocessing settings, validation rules
- **Training Configuration**: Model backbones, hyperparameters, optimization settings
- **Export Configuration**: Output formats, validation settings, optimization options
- **Registry Configuration**: MLflow settings, S3/MinIO integration
- **Service Configuration**: Logging, monitoring, security settings

## 🌍 Environment Variables

### **Service Configuration**

| Variable       | Type      | Default   | Required | Description                                 |
| -------------- | --------- | --------- | -------- | ------------------------------------------- |
| `LOG_LEVEL`    | `string`  | `INFO`    | No       | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FORMAT`   | `string`  | `json`    | No       | Log format (json, text)                     |
| `LOG_FILE`     | `string`  | `None`    | No       | Log file path (optional)                    |
| `API_HOST`     | `string`  | `0.0.0.0` | No       | API server host                             |
| `API_PORT`     | `integer` | `8000`    | No       | API server port                             |
| `API_WORKERS`  | `integer` | `1`       | No       | Number of API workers                       |
| `API_KEY`      | `string`  | `None`    | No       | API key for authentication                  |
| `CORS_ORIGINS` | `string`  | `*`       | No       | CORS allowed origins (comma-separated)      |
| `RATE_LIMIT`   | `integer` | `100`     | No       | Requests per minute per client              |

### **Data Configuration**

| Variable              | Type      | Default        | Required | Description                                    |
| --------------------- | --------- | -------------- | -------- | ---------------------------------------------- |
| `DATA_INPUT_PATH`     | `string`  | `None`         | Yes\*    | Input data file path                           |
| `DATA_INPUT_FORMAT`   | `string`  | `auto`         | No       | Input format (json, jsonl, csv, parquet, auto) |
| `DATA_TEXT_COLUMN`    | `string`  | `text`         | No       | Text column name                               |
| `DATA_LABEL_COLUMN`   | `string`  | `label`        | No       | Label column name                              |
| `DATA_ID_COLUMN`      | `string`  | `id`           | No       | ID column name                                 |
| `DATA_TITLE_COLUMN`   | `string`  | `title`        | No       | Title column name                              |
| `DATA_DATE_COLUMN`    | `string`  | `published_at` | No       | Date column name                               |
| `DATA_TICKERS_COLUMN` | `string`  | `tickers`      | No       | Tickers column name                            |
| `DATA_SPLIT_COLUMN`   | `string`  | `split`        | No       | Split column name                              |
| `DATA_TRAIN_SPLIT`    | `float`   | `0.8`          | No       | Training split ratio                           |
| `DATA_VAL_SPLIT`      | `float`   | `0.1`          | No       | Validation split ratio                         |
| `DATA_TEST_SPLIT`     | `float`   | `0.1`          | No       | Test split ratio                               |
| `DATA_RANDOM_SEED`    | `integer` | `42`           | No       | Random seed for splitting                      |

### **Preprocessing Configuration**

| Variable                             | Type      | Default                                        | Required | Description              |
| ------------------------------------ | --------- | ---------------------------------------------- | -------- | ------------------------ |
| `PREPROCESSING_MAX_LENGTH`           | `integer` | `512`                                          | No       | Maximum sequence length  |
| `PREPROCESSING_TRUNCATION`           | `boolean` | `true`                                         | No       | Enable text truncation   |
| `PREPROCESSING_PADDING`              | `boolean` | `true`                                         | No       | Enable padding           |
| `PREPROCESSING_LOWER_CASE`           | `boolean` | `false`                                        | No       | Convert to lowercase     |
| `PREPROCESSING_REMOVE_URLS`          | `boolean` | `true`                                         | No       | Remove URLs from text    |
| `PREPROCESSING_REMOVE_HTML`          | `boolean` | `true`                                         | No       | Remove HTML tags         |
| `PREPROCESSING_REMOVE_EMOJIS`        | `boolean` | `false`                                        | No       | Remove emojis            |
| `PREPROCESSING_NORMALIZE_WHITESPACE` | `boolean` | `true`                                         | No       | Normalize whitespace     |
| `PREPROCESSING_LABEL_MAPPING`        | `string`  | `{"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}` | No       | Label to integer mapping |

### **Training Configuration**

| Variable                               | Type      | Default            | Required | Description                           |
| -------------------------------------- | --------- | ------------------ | -------- | ------------------------------------- |
| `TRAINING_BACKBONE`                    | `string`  | `ProsusAI/finbert` | No       | Model backbone                        |
| `TRAINING_BATCH_SIZE`                  | `integer` | `16`               | No       | Training batch size                   |
| `TRAINING_EVAL_BATCH_SIZE`             | `integer` | `32`               | No       | Evaluation batch size                 |
| `TRAINING_LEARNING_RATE`               | `float`   | `2e-5`             | No       | Learning rate                         |
| `TRAINING_NUM_EPOCHS`                  | `integer` | `3`                | No       | Number of training epochs             |
| `TRAINING_WARMUP_STEPS`                | `integer` | `0`                | No       | Number of warmup steps                |
| `TRAINING_WEIGHT_DECAY`                | `float`   | `0.01`             | No       | Weight decay                          |
| `TRAINING_GRADIENT_ACCUMULATION_STEPS` | `integer` | `1`                | No       | Gradient accumulation steps           |
| `TRAINING_MAX_GRAD_NORM`               | `float`   | `1.0`              | No       | Maximum gradient norm                 |
| `TRAINING_FP16`                        | `boolean` | `false`            | No       | Enable mixed precision training       |
| `TRAINING_GRADIENT_CHECKPOINTING`      | `boolean` | `false`            | No       | Enable gradient checkpointing         |
| `TRAINING_DATALOADER_NUM_WORKERS`      | `integer` | `4`                | No       | Number of data loader workers         |
| `TRAINING_RANDOM_SEED`                 | `integer` | `42`               | No       | Random seed for training              |
| `TRAINING_EARLY_STOPPING_PATIENCE`     | `integer` | `3`                | No       | Early stopping patience               |
| `TRAINING_EARLY_STOPPING_DELTA`        | `float`   | `0.001`            | No       | Early stopping minimum delta          |
| `TRAINING_SAVE_STEPS`                  | `integer` | `500`              | No       | Save checkpoint every N steps         |
| `TRAINING_EVAL_STEPS`                  | `integer` | `500`              | No       | Evaluate every N steps                |
| `TRAINING_SAVE_TOTAL_LIMIT`            | `integer` | `3`                | No       | Maximum number of checkpoints to save |

### **Export Configuration**

| Variable                        | Type      | Default | Required | Description                                     |
| ------------------------------- | --------- | ------- | -------- | ----------------------------------------------- |
| `EXPORT_FORMAT`                 | `string`  | `onnx`  | No       | Export format (onnx, torchscript, triton, both) |
| `EXPORT_ONNX_OPSET_VERSION`     | `integer` | `17`    | No       | ONNX opset version                              |
| `EXPORT_ONNX_DYNAMIC_AXES`      | `boolean` | `true`  | No       | Enable dynamic axes for ONNX                    |
| `EXPORT_TORCHSCRIPT_OPTIMIZE`   | `boolean` | `true`  | No       | Enable TorchScript optimization                 |
| `EXPORT_VALIDATE_EXPORT`        | `boolean` | `true`  | No       | Validate exported models                        |
| `EXPORT_QUANTIZE`               | `boolean` | `false` | No       | Enable model quantization                       |
| `EXPORT_QUANTIZATION_TYPE`      | `string`  | `int8`  | No       | Quantization type (int8, fp16)                  |
| `EXPORT_OPTIMIZE_FOR_INFERENCE` | `boolean` | `true`  | No       | Optimize for inference                          |

### **Registry Configuration**

| Variable                         | Type     | Default                 | Required | Description                 |
| -------------------------------- | -------- | ----------------------- | -------- | --------------------------- |
| `REGISTRY_TRACKING_URI`          | `string` | `sqlite:///mlruns.db`   | No       | MLflow tracking URI         |
| `REGISTRY_ARTIFACT_LOCATION`     | `string` | `None`                  | No       | MLflow artifact location    |
| `REGISTRY_MODEL_NAME`            | `string` | `crypto-news-sentiment` | No       | Default model name          |
| `REGISTRY_MODEL_STAGE`           | `string` | `Staging`               | No       | Default model stage         |
| `REGISTRY_AWS_ACCESS_KEY_ID`     | `string` | `None`                  | No       | AWS access key for S3       |
| `REGISTRY_AWS_SECRET_ACCESS_KEY` | `string` | `None`                  | No       | AWS secret key for S3       |
| `REGISTRY_AWS_REGION`            | `string` | `us-east-1`             | No       | AWS region                  |
| `REGISTRY_S3_BUCKET`             | `string` | `None`                  | No       | S3 bucket for artifacts     |
| `REGISTRY_S3_ENDPOINT_URL`       | `string` | `None`                  | No       | S3 endpoint URL (for MinIO) |

### **GPU Configuration**

| Variable              | Type      | Default | Required | Description                            |
| --------------------- | --------- | ------- | -------- | -------------------------------------- |
| `GPU_DEVICE`          | `string`  | `auto`  | No       | GPU device (auto, cuda:0, cuda:1, cpu) |
| `GPU_MEMORY_FRACTION` | `float`   | `0.9`   | No       | GPU memory fraction to use             |
| `GPU_MIXED_PRECISION` | `boolean` | `false` | No       | Enable mixed precision training        |
| `GPU_DETERMINISTIC`   | `boolean` | `false` | No       | Enable deterministic training          |

### **Security Configuration**

| Variable                       | Type      | Default | Required | Description                     |
| ------------------------------ | --------- | ------- | -------- | ------------------------------- |
| `SECURITY_API_KEY`             | `string`  | `None`  | No       | API key for authentication      |
| `SECURITY_JWT_SECRET`          | `string`  | `None`  | No       | JWT secret for token generation |
| `SECURITY_JWT_EXPIRY`          | `integer` | `3600`  | No       | JWT token expiry in seconds     |
| `SECURITY_RATE_LIMIT_ENABLED`  | `boolean` | `true`  | No       | Enable rate limiting            |
| `SECURITY_RATE_LIMIT_REQUESTS` | `integer` | `100`   | No       | Requests per minute             |
| `SECURITY_RATE_LIMIT_WINDOW`   | `integer` | `60`    | No       | Rate limit window in seconds    |

## 📄 Configuration Files

### **YAML Configuration Format**

```yaml
# config.yaml
service:
  log_level: INFO
  api_host: 0.0.0.0
  api_port: 8000
  api_workers: 1

data:
  input_path: data/news_dataset.json
  input_format: json
  text_column: text
  label_column: label
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  random_seed: 42

preprocessing:
  max_length: 512
  truncation: true
  padding: true
  lower_case: false
  remove_urls: true
  remove_html: true
  label_mapping:
    NEGATIVE: 0
    NEUTRAL: 1
    POSITIVE: 2

training:
  backbone: ProsusAI/finbert
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 3
  fp16: true
  gradient_checkpointing: true
  early_stopping_patience: 3
  save_steps: 500
  eval_steps: 500

export:
  format: onnx
  onnx_opset_version: 17
  validate_export: true
  optimize_for_inference: true

registry:
  tracking_uri: sqlite:///mlruns.db
  model_name: crypto-news-sentiment
  model_stage: Staging

gpu:
  device: auto
  memory_fraction: 0.9
  mixed_precision: true

security:
  api_key: your-api-key-here
  rate_limit_enabled: true
  rate_limit_requests: 100
```

### **JSON Configuration Format**

```json
{
  "service": {
    "log_level": "INFO",
    "api_host": "0.0.0.0",
    "api_port": 8000,
    "api_workers": 1
  },
  "data": {
    "input_path": "data/news_dataset.json",
    "input_format": "json",
    "text_column": "text",
    "label_column": "label",
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "random_seed": 42
  },
  "preprocessing": {
    "max_length": 512,
    "truncation": true,
    "padding": true,
    "lower_case": false,
    "remove_urls": true,
    "remove_html": true,
    "label_mapping": {
      "NEGATIVE": 0,
      "NEUTRAL": 1,
      "POSITIVE": 2
    }
  },
  "training": {
    "backbone": "ProsusAI/finbert",
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "fp16": true,
    "gradient_checkpointing": true,
    "early_stopping_patience": 3,
    "save_steps": 500,
    "eval_steps": 500
  },
  "export": {
    "format": "onnx",
    "onnx_opset_version": 17,
    "validate_export": true,
    "optimize_for_inference": true
  },
  "registry": {
    "tracking_uri": "sqlite:///mlruns.db",
    "model_name": "crypto-news-sentiment",
    "model_stage": "Staging"
  },
  "gpu": {
    "device": "auto",
    "memory_fraction": 0.9,
    "mixed_precision": true
  },
  "security": {
    "api_key": "your-api-key-here",
    "rate_limit_enabled": true,
    "rate_limit_requests": 100
  }
}
```

## 🌍 Environment-Specific Settings

### **Development Environment**

```bash
# .env.development
LOG_LEVEL=DEBUG
API_HOST=localhost
API_PORT=8000
DATA_INPUT_PATH=data/news_dataset_sample.json
TRAINING_BATCH_SIZE=8
TRAINING_NUM_EPOCHS=1
REGISTRY_TRACKING_URI=sqlite:///mlruns_dev.db
SECURITY_API_KEY=dev-api-key-123
```

### **Staging Environment**

```bash
# .env.staging
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
DATA_INPUT_PATH=/data/news_dataset.json
TRAINING_BATCH_SIZE=16
TRAINING_NUM_EPOCHS=3
REGISTRY_TRACKING_URI=http://mlflow-staging:5000
REGISTRY_ARTIFACT_LOCATION=s3://staging-models
REGISTRY_AWS_ACCESS_KEY_ID=staging-key
REGISTRY_AWS_SECRET_ACCESS_KEY=staging-secret
REGISTRY_S3_BUCKET=staging-models
SECURITY_API_KEY=staging-api-key-456
```

### **Production Environment**

```bash
# .env.production
LOG_LEVEL=WARNING
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
DATA_INPUT_PATH=/data/news_dataset.json
TRAINING_BATCH_SIZE=32
TRAINING_NUM_EPOCHS=5
TRAINING_FP16=true
TRAINING_GRADIENT_CHECKPOINTING=true
REGISTRY_TRACKING_URI=http://mlflow-prod:5000
REGISTRY_ARTIFACT_LOCATION=s3://prod-models
REGISTRY_AWS_ACCESS_KEY_ID=prod-key
REGISTRY_AWS_SECRET_ACCESS_KEY=prod-secret
REGISTRY_S3_BUCKET=prod-models
GPU_MIXED_PRECISION=true
SECURITY_API_KEY=prod-api-key-789
SECURITY_RATE_LIMIT_REQUESTS=50
```

### **Docker Environment**

```bash
# .env.docker
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
DATA_INPUT_PATH=/app/data/news_dataset.json
TRAINING_BATCH_SIZE=16
REGISTRY_TRACKING_URI=http://mlflow:5000
REGISTRY_ARTIFACT_LOCATION=s3://models
REGISTRY_AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
REGISTRY_AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
REGISTRY_S3_BUCKET=${S3_BUCKET}
SECURITY_API_KEY=${API_KEY}
```

## ✅ Validation Rules

### **Data Validation**

- **Input Path**: Must be a valid file path and file must exist
- **Input Format**: Must be one of: `json`, `jsonl`, `csv`, `parquet`, `auto`
- **Split Ratios**: Must sum to 1.0 and each must be between 0.0 and 1.0
- **Column Names**: Must be valid Python identifiers

### **Training Validation**

- **Batch Size**: Must be positive integer
- **Learning Rate**: Must be positive float
- **Epochs**: Must be positive integer
- **Backbone**: Must be a valid Hugging Face model identifier
- **Random Seed**: Must be integer

### **Export Validation**

- **Format**: Must be one of: `onnx`, `torchscript`, `triton`, `both`
- **ONNX Opset**: Must be between 11 and 17
- **Quantization Type**: Must be one of: `int8`, `fp16`

### **Registry Validation**

- **Tracking URI**: Must be valid URI format
- **Model Name**: Must be valid Python identifier
- **Model Stage**: Must be one of: `None`, `Staging`, `Production`, `Archived`
- **AWS Credentials**: Must be provided if using S3

### **Security Validation**

- **API Key**: Must be non-empty string if provided
- **Rate Limit**: Must be positive integer
- **JWT Secret**: Must be non-empty string if JWT is enabled

## 📝 Configuration Examples

### **Basic Training Configuration**

```bash
# Environment variables for basic training
export DATA_INPUT_PATH=data/news_dataset.json
export TRAINING_BACKBONE=ProsusAI/finbert
export TRAINING_BATCH_SIZE=16
export TRAINING_LEARNING_RATE=2e-5
export TRAINING_NUM_EPOCHS=3
export REGISTRY_TRACKING_URI=sqlite:///mlruns.db
```

### **Advanced Training with GPU**

```bash
# Environment variables for GPU training
export TRAINING_BATCH_SIZE=32
export TRAINING_FP16=true
export TRAINING_GRADIENT_CHECKPOINTING=true
export GPU_DEVICE=cuda:0
export GPU_MEMORY_FRACTION=0.9
export GPU_MIXED_PRECISION=true
export TRAINING_GRADIENT_ACCUMULATION_STEPS=2
```

### **Production Deployment**

```bash
# Environment variables for production
export LOG_LEVEL=WARNING
export API_WORKERS=4
export TRAINING_BATCH_SIZE=32
export TRAINING_FP16=true
export REGISTRY_TRACKING_URI=http://mlflow-prod:5000
export REGISTRY_ARTIFACT_LOCATION=s3://prod-models
export REGISTRY_AWS_ACCESS_KEY_ID=prod-key
export REGISTRY_AWS_SECRET_ACCESS_KEY=prod-secret
export REGISTRY_S3_BUCKET=prod-models
export SECURITY_API_KEY=prod-api-key
export SECURITY_RATE_LIMIT_REQUESTS=50
```

### **Docker Compose Configuration**

```yaml
# docker-compose.yml
version: "3.8"
services:
  sentiment-model-builder:
    image: finsight/sentiment-model-builder:latest
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - DATA_INPUT_PATH=/app/data/news_dataset.json
      - TRAINING_BATCH_SIZE=16
      - REGISTRY_TRACKING_URI=http://mlflow:5000
      - REGISTRY_ARTIFACT_LOCATION=s3://models
      - REGISTRY_AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - REGISTRY_AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - REGISTRY_S3_BUCKET=${S3_BUCKET}
      - SECURITY_API_KEY=${API_KEY}
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    depends_on:
      - mlflow
```

### **Kubernetes ConfigMap**

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sentiment-model-builder-config
  namespace: sentiment-analysis
data:
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  API_WORKERS: "4"
  TRAINING_BATCH_SIZE: "32"
  TRAINING_FP16: "true"
  TRAINING_GRADIENT_CHECKPOINTING: "true"
  REGISTRY_TRACKING_URI: "http://mlflow-service:5000"
  REGISTRY_ARTIFACT_LOCATION: "s3://prod-models"
  REGISTRY_MODEL_NAME: "crypto-news-sentiment"
  REGISTRY_MODEL_STAGE: "Staging"
  GPU_MIXED_PRECISION: "true"
  SECURITY_RATE_LIMIT_ENABLED: "true"
  SECURITY_RATE_LIMIT_REQUESTS: "50"
```

### **Helm Values**

```yaml
# helm-charts/sentiment-model-builder/values.yaml
service:
  logLevel: INFO
  apiHost: 0.0.0.0
  apiPort: 8000
  apiWorkers: 4

data:
  inputPath: /app/data/news_dataset.json
  inputFormat: json
  textColumn: text
  labelColumn: label
  trainSplit: 0.8
  valSplit: 0.1
  testSplit: 0.1

training:
  backbone: ProsusAI/finbert
  batchSize: 32
  learningRate: 2e-5
  numEpochs: 5
  fp16: true
  gradientCheckpointing: true
  earlyStoppingPatience: 3

export:
  format: onnx
  onnxOpsetVersion: 17
  validateExport: true
  optimizeForInference: true

registry:
  trackingUri: http://mlflow-service:5000
  artifactLocation: s3://prod-models
  modelName: crypto-news-sentiment
  modelStage: Staging

gpu:
  device: auto
  memoryFraction: 0.9
  mixedPrecision: true

security:
  apiKey: ""
  rateLimitEnabled: true
  rateLimitRequests: 50

aws:
  accessKeyId: ""
  secretAccessKey: ""
  region: us-east-1
  s3Bucket: prod-models
```

## 🎯 Best Practices

### **Configuration Management**

1. **Use Environment Variables for Secrets**: Never commit API keys or credentials to version control
2. **Separate Configuration by Environment**: Use different `.env` files for development, staging, and production
3. **Validate Configuration at Startup**: Ensure all required settings are present and valid
4. **Use Sensible Defaults**: Provide reasonable defaults for optional settings
5. **Document Configuration**: Keep configuration documentation up to date

### **Security Best Practices**

1. **Rotate API Keys Regularly**: Implement key rotation policies
2. **Use Strong Secrets**: Generate cryptographically secure API keys
3. **Limit Permissions**: Use principle of least privilege for API keys
4. **Monitor Access**: Log and monitor API key usage
5. **Secure Storage**: Use secure secret management systems

### **Performance Optimization**

1. **GPU Configuration**: Optimize GPU memory usage and enable mixed precision
2. **Batch Size Tuning**: Adjust batch size based on available memory
3. **Data Loading**: Use appropriate number of workers for data loading
4. **Caching**: Enable caching for frequently accessed data
5. **Monitoring**: Monitor resource usage and performance metrics

### **Production Deployment**

1. **Health Checks**: Implement comprehensive health checks
2. **Logging**: Use structured logging with appropriate levels
3. **Monitoring**: Set up monitoring and alerting
4. **Backup**: Implement backup strategies for models and data
5. **Rollback**: Plan for rollback procedures

### **Development Workflow**

1. **Local Development**: Use local configuration for development
2. **Testing**: Test configuration changes in staging environment
3. **Validation**: Validate configuration before deployment
4. **Documentation**: Document configuration changes
5. **Version Control**: Version control configuration templates

---

**For more information, see the [API Documentation](api.md) and [Architecture Documentation](architecture.md).**
