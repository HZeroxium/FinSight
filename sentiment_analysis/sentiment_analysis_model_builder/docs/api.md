# FinSight Sentiment Analysis Model Builder - API Documentation

> **Complete API Reference for Model Training, Export, and Registry Management**  
> Comprehensive documentation for CLI commands, REST endpoints, and integration patterns

## 🔐 Authentication

### **API Key Authentication**

The Model Builder service supports API key authentication for secure access:

```bash
# Set API key in environment
export MODEL_BUILDER_API_KEY=your-api-key-here

# Use in requests
curl -H "Authorization: Bearer your-api-key-here" \
     http://localhost:8000/api/v1/health
```

### **Authentication Headers**

| Header          | Description                              | Required                      |
| --------------- | ---------------------------------------- | ----------------------------- |
| `Authorization` | Bearer token: `Bearer <api-key>`         | Yes (for protected endpoints) |
| `Content-Type`  | Request content type: `application/json` | Yes (for POST/PUT requests)   |
| `Accept`        | Response format: `application/json`      | No (defaults to JSON)         |

### **API Key Management**

```bash
# Generate new API key
curl -X POST http://localhost:8000/api/v1/auth/generate-key \
     -H "Content-Type: application/json" \
     -d '{"name": "training-service", "permissions": ["train", "export", "register"]}'

# List active API keys
curl -H "Authorization: Bearer admin-key" \
     http://localhost:8000/api/v1/auth/keys

# Revoke API key
curl -X DELETE http://localhost:8000/api/v1/auth/keys/key-id-123 \
     -H "Authorization: Bearer admin-key"
```

## 📄 Common Response Formats

### **Success Response**

```json
{
  "success": true,
  "data": {
    "id": "training-run-123",
    "status": "completed",
    "metrics": {
      "accuracy": 0.89,
      "f1_score": 0.87,
      "precision": 0.88,
      "recall": 0.86
    },
    "created_at": "2025-01-15T10:30:00Z",
    "updated_at": "2025-01-15T11:45:00Z"
  },
  "message": "Training completed successfully",
  "correlation_id": "req-123456"
}
```

### **Error Response**

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid data format",
    "details": {
      "field": "text_column",
      "issue": "Column 'text_column' not found in dataset"
    }
  },
  "correlation_id": "req-123456",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### **Paginated Response**

```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": "model-1",
        "name": "crypto-sentiment-v1",
        "version": "1.0.0",
        "stage": "Production"
      }
    ],
    "pagination": {
      "page": 1,
      "per_page": 20,
      "total": 45,
      "total_pages": 3,
      "has_next": true,
      "has_prev": false
    }
  },
  "message": "Models retrieved successfully"
}
```

## ❌ Error Codes

### **HTTP Status Codes**

| Code  | Category         | Description                    |
| ----- | ---------------- | ------------------------------ |
| `200` | Success          | Request completed successfully |
| `201` | Created          | Resource created successfully  |
| `400` | Bad Request      | Invalid request parameters     |
| `401` | Unauthorized     | Authentication required        |
| `403` | Forbidden        | Insufficient permissions       |
| `404` | Not Found        | Resource not found             |
| `409` | Conflict         | Resource conflict              |
| `422` | Validation Error | Data validation failed         |
| `429` | Rate Limited     | Too many requests              |
| `500` | Internal Error   | Server error                   |

### **Application Error Codes**

| Code                   | Description                       | HTTP Status |
| ---------------------- | --------------------------------- | ----------- |
| `VALIDATION_ERROR`     | Input validation failed           | 422         |
| `AUTHENTICATION_ERROR` | Invalid or missing credentials    | 401         |
| `AUTHORIZATION_ERROR`  | Insufficient permissions          | 403         |
| `RESOURCE_NOT_FOUND`   | Requested resource not found      | 404         |
| `RESOURCE_CONFLICT`    | Resource already exists           | 409         |
| `TRAINING_ERROR`       | Model training failed             | 500         |
| `EXPORT_ERROR`         | Model export failed               | 500         |
| `REGISTRY_ERROR`       | Model registry operation failed   | 500         |
| `DATA_ERROR`           | Data loading or processing failed | 500         |
| `CONFIGURATION_ERROR`  | Configuration validation failed   | 500         |

## 🖥️ CLI Commands

### **Training Commands**

#### **Train Model**

```bash
sentiment-train [OPTIONS]

# Basic training
sentiment-train \
    --data data/news_dataset.json \
    --output outputs/training_run_$(date +%Y%m%d_%H%M%S) \
    --experiment crypto-sentiment-v1

# Advanced training with configuration
sentiment-train \
    --config configs/training_config.yaml \
    --data data/news_dataset.json \
    --output outputs/advanced_run \
    --experiment crypto-sentiment-v2 \
    --log-level DEBUG
```

**Options:**

| Option               | Type   | Default              | Description             |
| -------------------- | ------ | -------------------- | ----------------------- |
| `--data`, `-d`       | `PATH` | Required             | Input data file path    |
| `--output`, `-o`     | `PATH` | `./outputs`          | Output directory        |
| `--experiment`, `-e` | `TEXT` | `sentiment-analysis` | MLflow experiment name  |
| `--config`, `-c`     | `PATH` | None                 | Configuration file path |
| `--log-level`, `-l`  | `TEXT` | `INFO`               | Logging level           |

**Configuration File Format (YAML):**

```yaml
# configs/training_config.yaml
training:
  backbone: "ProsusAI/finbert"
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 3
  random_seed: 42
  fp16: true
  gradient_checkpointing: true

preprocessing:
  text_column: "text"
  label_column: "label"
  max_length: 512
  truncation: true
  padding: true

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  random_seed: 42
```

#### **Resume Training**

```bash
# Resume from checkpoint
sentiment-train \
    --data data/news_dataset.json \
    --output outputs/resumed_run \
    --resume-from outputs/previous_run/checkpoint-2 \
    --experiment crypto-sentiment-v1
```

### **Export Commands**

#### **Export Model**

```bash
sentiment-export MODEL_PATH [OPTIONS]

# Export to ONNX format
sentiment-export outputs/training_run_*/model \
    --output models/exported \
    --format onnx \
    --validate

# Export to multiple formats
sentiment-export outputs/training_run_*/model \
    --output models/exported \
    --format onnx,torchscript \
    --validate \
    --optimize
```

**Options:**

| Option                     | Type   | Default      | Description              |
| -------------------------- | ------ | ------------ | ------------------------ |
| `MODEL_PATH`               | `PATH` | Required     | Path to trained model    |
| `--output`, `-o`           | `PATH` | `./exported` | Output directory         |
| `--format`, `-f`           | `TEXT` | `onnx`       | Export format(s)         |
| `--validate/--no-validate` | `FLAG` | `True`       | Validate exported models |
| `--optimize/--no-optimize` | `FLAG` | `False`      | Apply optimizations      |

**Supported Formats:**

- `onnx`: ONNX format for cross-platform deployment
- `torchscript`: TorchScript for PyTorch serving
- `triton`: NVIDIA Triton Inference Server format
- `both`: Export to both ONNX and TorchScript

#### **Validate Exported Model**

```bash
# Validate exported model
sentiment-validate models/exported/model.onnx \
    --test-data data/test_samples.json \
    --output validation_report.json
```

### **Registry Commands**

#### **Register Model**

```bash
sentiment-register MODEL_PATH [OPTIONS]

# Register model in MLflow
sentiment-register outputs/training_run_*/model \
    --run-id mlflow-run-123 \
    --description "Crypto news sentiment model v1.0" \
    --stage Staging

# Register with custom metadata
sentiment-register outputs/training_run_*/model \
    --run-id mlflow-run-123 \
    --description "Production-ready sentiment model" \
    --stage Production \
    --tags "crypto,news,sentiment" \
    --metadata '{"accuracy": 0.89, "f1_score": 0.87}'
```

**Options:**

| Option          | Type   | Default   | Description             |
| --------------- | ------ | --------- | ----------------------- |
| `MODEL_PATH`    | `PATH` | Required  | Path to model directory |
| `--run-id`      | `TEXT` | Required  | MLflow run ID           |
| `--description` | `TEXT` | None      | Model description       |
| `--stage`       | `TEXT` | `Staging` | Initial model stage     |
| `--tags`        | `TEXT` | None      | Comma-separated tags    |
| `--metadata`    | `JSON` | None      | Additional metadata     |

#### **List Models**

```bash
# List all models
sentiment-list-models

# List models by stage
sentiment-list-models --stage Production

# List models with details
sentiment-list-models --detailed --format json
```

#### **Model Operations**

```bash
# Transition model stage
sentiment-transition-stage crypto-sentiment-model 1 Production

# Delete model version
sentiment-delete-model crypto-sentiment-model 1

# Get model info
sentiment-model-info crypto-sentiment-model 1
```

### **Utility Commands**

#### **Configuration**

```bash
# Show current configuration
sentiment-config

# Validate configuration
sentiment-config --validate

# Generate configuration template
sentiment-config --template > config_template.yaml
```

#### **Data Validation**

```bash
# Validate dataset format
sentiment-validate-data data/news_dataset.json \
    --format json \
    --text-column text \
    --label-column label

# Generate data report
sentiment-data-report data/news_dataset.json \
    --output data_report.html
```

#### **Health Check**

```bash
# Check service health
sentiment-health

# Check MLflow connection
sentiment-health --mlflow

# Check GPU availability
sentiment-health --gpu
```

## 🌐 REST API Endpoints

### **Health & Status**

#### **GET /api/v1/health**

Check service health and dependencies.

```bash
curl http://localhost:8000/api/v1/health
```

**Response:**

```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2025-01-15T10:30:00Z",
    "version": "1.0.0",
    "dependencies": {
      "mlflow": "connected",
      "gpu": "available",
      "storage": "accessible"
    }
  }
}
```

#### **GET /api/v1/status**

Get detailed service status.

```bash
curl http://localhost:8000/api/v1/status
```

**Response:**

```json
{
  "success": true,
  "data": {
    "service": "sentiment-model-builder",
    "version": "1.0.0",
    "uptime": "2h 15m 30s",
    "active_training_jobs": 1,
    "gpu_utilization": 85.5,
    "memory_usage": "4.2GB / 16GB",
    "disk_usage": "12.5GB / 100GB"
  }
}
```

### **Training Endpoints**

#### **POST /api/v1/training/start**

Start a new training job.

```bash
curl -X POST http://localhost:8000/api/v1/training/start \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "data_path": "data/news_dataset.json",
       "output_dir": "outputs/training_run_123",
       "experiment_name": "crypto-sentiment-v1",
       "config": {
         "training": {
           "backbone": "ProsusAI/finbert",
           "batch_size": 16,
           "learning_rate": 2e-5,
           "num_epochs": 3
         }
       }
     }'
```

**Request Body:**

```json
{
  "data_path": "string (required)",
  "output_dir": "string (optional)",
  "experiment_name": "string (optional)",
  "config": {
    "training": {
      "backbone": "string",
      "batch_size": "integer",
      "learning_rate": "number",
      "num_epochs": "integer"
    },
    "preprocessing": {
      "text_column": "string",
      "label_column": "string",
      "max_length": "integer"
    }
  }
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "job_id": "training-job-123",
    "status": "started",
    "estimated_duration": "2h 30m",
    "mlflow_run_id": "mlflow-run-456",
    "created_at": "2025-01-15T10:30:00Z"
  }
}
```

#### **GET /api/v1/training/{job_id}**

Get training job status and progress.

```bash
curl http://localhost:8000/api/v1/training/training-job-123 \
     -H "Authorization: Bearer your-api-key"
```

**Response:**

```json
{
  "success": true,
  "data": {
    "job_id": "training-job-123",
    "status": "running",
    "progress": {
      "current_epoch": 2,
      "total_epochs": 3,
      "current_step": 150,
      "total_steps": 225,
      "percentage": 66.7
    },
    "metrics": {
      "train_loss": 0.45,
      "val_loss": 0.52,
      "val_accuracy": 0.87
    },
    "estimated_completion": "2025-01-15T12:45:00Z"
  }
}
```

#### **GET /api/v1/training/{job_id}/logs**

Get training job logs.

```bash
curl http://localhost:8000/api/v1/training/training-job-123/logs \
     -H "Authorization: Bearer your-api-key"
```

**Query Parameters:**

| Parameter | Type      | Description                                    |
| --------- | --------- | ---------------------------------------------- |
| `level`   | `string`  | Log level filter (DEBUG, INFO, WARNING, ERROR) |
| `limit`   | `integer` | Number of log entries to return (default: 100) |
| `offset`  | `integer` | Number of log entries to skip                  |

#### **DELETE /api/v1/training/{job_id}**

Cancel a training job.

```bash
curl -X DELETE http://localhost:8000/api/v1/training/training-job-123 \
     -H "Authorization: Bearer your-api-key"
```

### **Export Endpoints**

#### **POST /api/v1/export**

Export a trained model.

```bash
curl -X POST http://localhost:8000/api/v1/export \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "model_path": "outputs/training_run_123/model",
       "output_dir": "models/exported",
       "format": "onnx",
       "validate": true
     }'
```

**Request Body:**

```json
{
  "model_path": "string (required)",
  "output_dir": "string (optional)",
  "format": "string (required)",
  "validate": "boolean (optional)",
  "optimize": "boolean (optional)"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "export_id": "export-job-123",
    "model_path": "outputs/training_run_123/model",
    "output_path": "models/exported/model.onnx",
    "format": "onnx",
    "status": "completed",
    "validation_passed": true,
    "file_size": "245MB",
    "exported_at": "2025-01-15T11:45:00Z"
  }
}
```

#### **GET /api/v1/export/{export_id}**

Get export job status.

```bash
curl http://localhost:8000/api/v1/export/export-job-123 \
     -H "Authorization: Bearer your-api-key"
```

### **Registry Endpoints**

#### **POST /api/v1/registry/register**

Register a model in MLflow.

```bash
curl -X POST http://localhost:8000/api/v1/registry/register \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "model_path": "outputs/training_run_123/model",
       "run_id": "mlflow-run-456",
       "name": "crypto-sentiment-model",
       "description": "Crypto news sentiment analysis model",
       "stage": "Staging"
     }'
```

**Request Body:**

```json
{
  "model_path": "string (required)",
  "run_id": "string (required)",
  "name": "string (required)",
  "description": "string (optional)",
  "stage": "string (optional)",
  "tags": "object (optional)",
  "metadata": "object (optional)"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "model_name": "crypto-sentiment-model",
    "version": 1,
    "stage": "Staging",
    "model_uri": "models:/crypto-sentiment-model/1",
    "registered_at": "2025-01-15T12:00:00Z"
  }
}
```

#### **GET /api/v1/registry/models**

List registered models.

```bash
curl "http://localhost:8000/api/v1/registry/models?stage=Production&limit=10" \
     -H "Authorization: Bearer your-api-key"
```

**Query Parameters:**

| Parameter | Type      | Description                |
| --------- | --------- | -------------------------- |
| `stage`   | `string`  | Filter by model stage      |
| `name`    | `string`  | Filter by model name       |
| `limit`   | `integer` | Number of models to return |
| `offset`  | `integer` | Number of models to skip   |

#### **PUT /api/v1/registry/models/{name}/versions/{version}/stage**

Transition model stage.

```bash
curl -X PUT http://localhost:8000/api/v1/registry/models/crypto-sentiment-model/versions/1/stage \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "stage": "Production",
       "comment": "Promoting to production after validation"
     }'
```

#### **DELETE /api/v1/registry/models/{name}/versions/{version}**

Delete model version.

```bash
curl -X DELETE http://localhost:8000/api/v1/registry/models/crypto-sentiment-model/versions/1 \
     -H "Authorization: Bearer your-api-key"
```

### **Data Endpoints**

#### **POST /api/v1/data/validate**

Validate dataset format and content.

```bash
curl -X POST http://localhost:8000/api/v1/data/validate \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "data_path": "data/news_dataset.json",
       "format": "json",
       "text_column": "text",
       "label_column": "label"
     }'
```

#### **POST /api/v1/data/report**

Generate dataset analysis report.

```bash
curl -X POST http://localhost:8000/api/v1/data/report \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "data_path": "data/news_dataset.json",
       "output_path": "reports/data_analysis.html"
     }'
```

## 🔌 WebSocket API

### **Training Progress Updates**

Connect to WebSocket for real-time training progress:

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/training/training-job-123");

ws.onmessage = function (event) {
  const data = JSON.parse(event.data);
  console.log("Training progress:", data);
};

ws.onclose = function () {
  console.log("WebSocket connection closed");
};
```

**Message Format:**

```json
{
  "type": "progress_update",
  "job_id": "training-job-123",
  "data": {
    "epoch": 2,
    "step": 150,
    "total_steps": 225,
    "train_loss": 0.45,
    "val_loss": 0.52,
    "val_accuracy": 0.87
  },
  "timestamp": "2025-01-15T11:30:00Z"
}
```

### **Export Status Updates**

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/export/export-job-123");

ws.onmessage = function (event) {
  const data = JSON.parse(event.data);
  console.log("Export status:", data);
};
```

## 🔗 Integration Examples

### **Python Client**

```python
import requests
import json

class ModelBuilderClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

    def start_training(self, data_path: str, config: dict) -> dict:
        """Start a training job."""
        payload = {
            'data_path': data_path,
            'config': config
        }
        response = requests.post(
            f'{self.base_url}/api/v1/training/start',
            headers=self.headers,
            json=payload
        )
        return response.json()

    def get_training_status(self, job_id: str) -> dict:
        """Get training job status."""
        response = requests.get(
            f'{self.base_url}/api/v1/training/{job_id}',
            headers=self.headers
        )
        return response.json()

    def export_model(self, model_path: str, format: str = 'onnx') -> dict:
        """Export a trained model."""
        payload = {
            'model_path': model_path,
            'format': format,
            'validate': True
        }
        response = requests.post(
            f'{self.base_url}/api/v1/export',
            headers=self.headers,
            json=payload
        )
        return response.json()

# Usage example
client = ModelBuilderClient('http://localhost:8000', 'your-api-key')

# Start training
config = {
    'training': {
        'backbone': 'ProsusAI/finbert',
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_epochs': 3
    }
}

result = client.start_training('data/news_dataset.json', config)
job_id = result['data']['job_id']

# Monitor training
while True:
    status = client.get_training_status(job_id)
    if status['data']['status'] == 'completed':
        break
    time.sleep(30)
```

### **JavaScript/Node.js Client**

```javascript
class ModelBuilderClient {
  constructor(baseUrl, apiKey) {
    this.baseUrl = baseUrl;
    this.headers = {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    };
  }

  async startTraining(dataPath, config) {
    const response = await fetch(`${this.baseUrl}/api/v1/training/start`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify({
        data_path: dataPath,
        config: config,
      }),
    });
    return response.json();
  }

  async getTrainingStatus(jobId) {
    const response = await fetch(`${this.baseUrl}/api/v1/training/${jobId}`, {
      headers: this.headers,
    });
    return response.json();
  }

  async exportModel(modelPath, format = "onnx") {
    const response = await fetch(`${this.baseUrl}/api/v1/export`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify({
        model_path: modelPath,
        format: format,
        validate: true,
      }),
    });
    return response.json();
  }
}

// Usage example
const client = new ModelBuilderClient("http://localhost:8000", "your-api-key");

async function trainModel() {
  const config = {
    training: {
      backbone: "ProsusAI/finbert",
      batch_size: 16,
      learning_rate: 2e-5,
      num_epochs: 3,
    },
  };

  const result = await client.startTraining("data/news_dataset.json", config);
  const jobId = result.data.job_id;

  // Monitor training
  const interval = setInterval(async () => {
    const status = await client.getTrainingStatus(jobId);
    if (status.data.status === "completed") {
      clearInterval(interval);
      console.log("Training completed!");
    }
  }, 30000);
}
```

### **cURL Examples**

```bash
# Start training
curl -X POST http://localhost:8000/api/v1/training/start \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "data_path": "data/news_dataset.json",
       "config": {
         "training": {
           "backbone": "ProsusAI/finbert",
           "batch_size": 16,
           "learning_rate": 2e-5,
           "num_epochs": 3
         }
       }
     }'

# Check training status
curl http://localhost:8000/api/v1/training/training-job-123 \
     -H "Authorization: Bearer your-api-key"

# Export model
curl -X POST http://localhost:8000/api/v1/export \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "model_path": "outputs/training_run_123/model",
       "format": "onnx",
       "validate": true
     }'

# Register model
curl -X POST http://localhost:8000/api/v1/registry/register \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "model_path": "outputs/training_run_123/model",
       "run_id": "mlflow-run-456",
       "name": "crypto-sentiment-model",
       "description": "Crypto news sentiment analysis model",
       "stage": "Staging"
     }'
```

---

**For more information, see the [Configuration Guide](configuration.md) and [Architecture Documentation](architecture.md).**
