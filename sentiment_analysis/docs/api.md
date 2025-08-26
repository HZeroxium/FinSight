# FinSight Sentiment Analysis Platform - API Documentation

> **Complete API Reference for Financial Sentiment Analysis Platform**

## 🌐 Overview

The FinSight Sentiment Analysis platform provides three main services with distinct APIs:

1. **Model Builder Service**: Training and model management endpoints
2. **Inference Engine**: High-performance sentiment analysis API
3. **Sentiment Analysis Service**: Content processing and analysis endpoints

### **Service Endpoints**

| Service               | Base URL                | Port | Purpose                            |
| --------------------- | ----------------------- | ---- | ---------------------------------- |
| **Model Builder**     | `http://localhost:8000` | 8000 | Model training, export, registry   |
| **Inference Engine**  | `http://localhost:8080` | 8080 | Real-time sentiment analysis       |
| **Sentiment Service** | `http://localhost:8001` | 8001 | Content processing, batch analysis |

## 🔐 Authentication

### **API Key Authentication**

Most endpoints require API key authentication:

```bash
# Add to request headers
Authorization: Bearer YOUR_API_KEY

# Or as query parameter
?api_key=YOUR_API_KEY
```

### **Authentication Levels**

- **Public**: Health checks, model info
- **Authenticated**: Training, inference, analysis
- **Admin**: Model management, system configuration

## 🌍 Base URLs

### **Development Environment**

- Model Builder: `http://localhost:8000`
- Inference Engine: `http://localhost:8080`
- Sentiment Service: `http://localhost:8001`

### **Production Environment**

- Model Builder: `https://api.finsight.ai/sentiment/builder`
- Inference Engine: `https://api.finsight.ai/sentiment/inference`
- Sentiment Service: `https://api.finsight.ai/sentiment/analysis`

## 📊 Common Response Formats

### **Success Response**

```json
{
  "success": true,
  "data": {
    // Response data
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

### **Error Response**

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "text",
      "issue": "Text cannot be empty"
    }
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

## ❌ Error Handling

### **HTTP Status Codes**

| Code  | Description           | Usage                             |
| ----- | --------------------- | --------------------------------- |
| `200` | Success               | Request completed successfully    |
| `201` | Created               | Resource created successfully     |
| `400` | Bad Request           | Invalid input data                |
| `401` | Unauthorized          | Missing or invalid authentication |
| `403` | Forbidden             | Insufficient permissions          |
| `404` | Not Found             | Resource not found                |
| `422` | Validation Error      | Data validation failed            |
| `429` | Too Many Requests     | Rate limit exceeded               |
| `500` | Internal Server Error | Server-side error                 |
| `503` | Service Unavailable   | Service temporarily unavailable   |

### **Error Codes**

| Code                   | Description                    | HTTP Status |
| ---------------------- | ------------------------------ | ----------- |
| `VALIDATION_ERROR`     | Input validation failed        | 422         |
| `AUTHENTICATION_ERROR` | Invalid or missing credentials | 401         |
| `AUTHORIZATION_ERROR`  | Insufficient permissions       | 403         |
| `RESOURCE_NOT_FOUND`   | Requested resource not found   | 404         |
| `RATE_LIMIT_EXCEEDED`  | Too many requests              | 429         |
| `MODEL_NOT_READY`      | Model not loaded or ready      | 503         |
| `TRAINING_FAILED`      | Model training failed          | 500         |
| `INFERENCE_ERROR`      | Model inference failed         | 500         |

## 🚀 API Endpoints

## 🧠 Model Builder Service

### **Health & Status**

#### **Health Check**

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "service": "sentiment-model-builder",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### **Service Status**

```http
GET /status
```

**Response:**

```json
{
  "status": "running",
  "uptime": "2h 15m 30s",
  "active_jobs": 2,
  "models_trained": 15,
  "last_training": "2024-01-15T08:00:00Z"
}
```

### **Model Training**

#### **Start Training**

```http
POST /train
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

**Request Body:**

```json
{
  "data_path": "data/news_dataset.json",
  "output_dir": "outputs/training_run_001",
  "experiment_name": "crypto-sentiment-v2",
  "config": {
    "backbone": "ProsusAI/finbert",
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 5,
    "max_length": 512
  }
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "job_id": "train_123456789",
    "status": "started",
    "experiment_name": "crypto-sentiment-v2",
    "estimated_duration": "2h 30m",
    "output_dir": "outputs/training_run_001"
  }
}
```

#### **Training Status**

```http
GET /train/{job_id}
Authorization: Bearer YOUR_API_KEY
```

**Response:**

```json
{
  "success": true,
  "data": {
    "job_id": "train_123456789",
    "status": "training",
    "progress": 0.65,
    "current_epoch": 3,
    "total_epochs": 5,
    "current_metrics": {
      "train_loss": 0.234,
      "val_loss": 0.289,
      "val_f1": 0.856
    },
    "started_at": "2024-01-15T08:00:00Z",
    "estimated_completion": "2024-01-15T10:30:00Z"
  }
}
```

### **Model Export**

#### **Export Model**

```http
POST /export
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

**Request Body:**

```json
{
  "model_path": "outputs/training_run_001/model",
  "output_dir": "models/exported",
  "format": "onnx",
  "config": {
    "onnx_opset_version": 17,
    "dynamic_axes": true,
    "validate_export": true
  }
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "export_id": "export_123456789",
    "status": "completed",
    "output_path": "models/exported/finbert_sentiment.onnx",
    "model_size": "245.6 MB",
    "export_time": "45.2s"
  }
}
```

### **Model Registry**

#### **Register Model**

```http
POST /register
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

**Request Body:**

```json
{
  "model_path": "outputs/training_run_001/model",
  "run_id": "mlflow_run_123456789",
  "stage": "Staging",
  "description": "Fine-tuned FinBERT for crypto sentiment analysis"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "model_name": "crypto-news-sentiment",
    "version": "v1.2.0",
    "stage": "Staging",
    "run_id": "mlflow_run_123456789",
    "registered_at": "2024-01-15T10:30:00Z"
  }
}
```

## ⚡ Inference Engine

### **Health & Status**

#### **Health Check**

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "service": "sentiment-inference-engine",
  "version": "1.0.0",
  "triton_status": "running",
  "model_status": "ready",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### **Model Information**

```http
GET /model/info
```

**Response:**

```json
{
  "success": true,
  "data": {
    "model_name": "finbert_sentiment",
    "version": "v1.2.0",
    "status": "ready",
    "input_shape": [1, 512],
    "output_shape": [1, 3],
    "max_batch_size": 32,
    "device": "cuda:0",
    "loaded_at": "2024-01-15T08:00:00Z"
  }
}
```

### **Sentiment Analysis**

#### **Single Text Analysis**

```http
POST /predict
Content-Type: application/json
```

**Request Body:**

```json
{
  "text": "Bitcoin shows strong bullish signals today with increasing adoption and institutional interest."
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "text": "Bitcoin shows strong bullish signals today with increasing adoption and institutional interest.",
    "label": "POSITIVE",
    "confidence": 0.8923,
    "scores": {
      "negative": 0.0456,
      "neutral": 0.0621,
      "positive": 0.8923
    },
    "processing_time_ms": 45.2,
    "model_version": "v1.2.0"
  }
}
```

#### **Batch Analysis**

```http
POST /predict/batch
Content-Type: application/json
```

**Request Body:**

```json
{
  "texts": [
    "Bitcoin shows strong bullish signals today.",
    "Market uncertainty leads to bearish sentiment.",
    "Neutral market conditions with mixed signals."
  ]
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "text": "Bitcoin shows strong bullish signals today.",
        "label": "POSITIVE",
        "confidence": 0.8923,
        "scores": {
          "negative": 0.0456,
          "neutral": 0.0621,
          "positive": 0.8923
        }
      },
      {
        "text": "Market uncertainty leads to bearish sentiment.",
        "label": "NEGATIVE",
        "confidence": 0.7845,
        "scores": {
          "negative": 0.7845,
          "neutral": 0.1567,
          "positive": 0.0588
        }
      },
      {
        "text": "Neutral market conditions with mixed signals.",
        "label": "NEUTRAL",
        "confidence": 0.6234,
        "scores": {
          "negative": 0.2345,
          "neutral": 0.6234,
          "positive": 0.1421
        }
      }
    ],
    "batch_processing_time_ms": 128.7,
    "total_texts": 3,
    "model_version": "v1.2.0"
  }
}
```

### **Triton Server Management**

#### **Server Status**

```http
GET /triton/status
```

**Response:**

```json
{
  "success": true,
  "data": {
    "status": "running",
    "container_id": "triton_container_123",
    "uptime": "2h 15m 30s",
    "gpu_utilization": 0.45,
    "memory_usage": "2.3 GB",
    "requests_processed": 15420,
    "average_latency_ms": 23.4
  }
}
```

#### **Restart Server**

```http
POST /triton/restart
Authorization: Bearer YOUR_API_KEY
```

**Response:**

```json
{
  "success": true,
  "data": {
    "message": "Triton server restart initiated",
    "status": "restarting",
    "estimated_downtime": "30s"
  }
}
```

## 🔍 Sentiment Analysis Service

### **Health & Status**

#### **Health Check**

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "service": "sentiment-analysis-service",
  "version": "1.0.0",
  "uptime": "2h 15m 30s",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### **Content Processing**

#### **Preprocess Text**

```http
POST /preprocess
Content-Type: application/json
```

**Request Body:**

```json
{
  "text": "Bitcoin (BTC) shows strong bullish signals today! 🚀",
  "config": {
    "remove_html": true,
    "normalize_unicode": true,
    "lowercase": true,
    "remove_urls": true,
    "max_length": 512
  }
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "original_text": "Bitcoin (BTC) shows strong bullish signals today! 🚀",
    "processed_text": "bitcoin btc shows strong bullish signals today",
    "processing_config": {
      "remove_html": true,
      "normalize_unicode": true,
      "lowercase": true,
      "remove_urls": true,
      "max_length": 512
    },
    "processing_time_ms": 12.3
  }
}
```

#### **Batch Preprocessing**

```http
POST /preprocess/batch
Content-Type: application/json
```

**Request Body:**

```json
{
  "texts": [
    "Bitcoin (BTC) shows strong bullish signals today! 🚀",
    "Market uncertainty leads to bearish sentiment.",
    "Neutral market conditions with mixed signals."
  ],
  "config": {
    "remove_html": true,
    "normalize_unicode": true,
    "lowercase": true
  }
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "original_text": "Bitcoin (BTC) shows strong bullish signals today! 🚀",
        "processed_text": "bitcoin btc shows strong bullish signals today"
      },
      {
        "original_text": "Market uncertainty leads to bearish sentiment.",
        "processed_text": "market uncertainty leads to bearish sentiment"
      },
      {
        "original_text": "Neutral market conditions with mixed signals.",
        "processed_text": "neutral market conditions with mixed signals"
      }
    ],
    "total_processed": 3,
    "batch_processing_time_ms": 45.6
  }
}
```

### **Sentiment Aggregation**

#### **Aggregate Sentiments**

```http
POST /aggregate
Content-Type: application/json
```

**Request Body:**

```json
{
  "sentiments": [
    {
      "text": "Bitcoin shows strong bullish signals",
      "label": "POSITIVE",
      "confidence": 0.8923
    },
    {
      "text": "Market uncertainty leads to bearish sentiment",
      "label": "NEGATIVE",
      "confidence": 0.7845
    },
    {
      "text": "Neutral market conditions",
      "label": "NEUTRAL",
      "confidence": 0.6234
    }
  ],
  "aggregation_method": "weighted_average"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "aggregate_sentiment": "NEUTRAL",
    "aggregate_score": 0.1234,
    "sentiment_distribution": {
      "positive": 0.2974,
      "negative": 0.2615,
      "neutral": 0.2078
    },
    "confidence_metrics": {
      "average_confidence": 0.7667,
      "confidence_std": 0.1344
    },
    "total_texts": 3
  }
}
```

## 📊 Data Models

### **Sentiment Request**

```json
{
  "text": "string",
  "config": {
    "max_length": "integer (optional)",
    "preprocessing": "object (optional)"
  }
}
```

### **Batch Sentiment Request**

```json
{
  "texts": ["string"],
  "config": {
    "max_batch_size": "integer (optional)",
    "batch_timeout_ms": "integer (optional)"
  }
}
```

### **Sentiment Result**

```json
{
  "text": "string",
  "label": "POSITIVE | NEGATIVE | NEUTRAL",
  "confidence": "float (0.0-1.0)",
  "scores": {
    "negative": "float",
    "neutral": "float",
    "positive": "float"
  },
  "processing_time_ms": "float",
  "model_version": "string"
}
```

### **Training Request**

```json
{
  "data_path": "string",
  "output_dir": "string",
  "experiment_name": "string",
  "config": {
    "backbone": "string",
    "batch_size": "integer",
    "learning_rate": "float",
    "num_epochs": "integer",
    "max_length": "integer"
  }
}
```

## 🚦 Rate Limiting

### **Rate Limits**

| Service              | Endpoint         | Limit        | Window   |
| -------------------- | ---------------- | ------------ | -------- |
| **Inference Engine** | `/predict`       | 100 requests | 1 minute |
| **Inference Engine** | `/predict/batch` | 50 requests  | 1 minute |
| **Model Builder**    | `/train`         | 10 requests  | 1 hour   |
| **Model Builder**    | `/export`        | 20 requests  | 1 hour   |

### **Rate Limit Headers**

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1642234567
```

### **Rate Limit Exceeded Response**

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 45 seconds.",
    "retry_after": 45
  }
}
```

## 💡 Examples

### **Complete Sentiment Analysis Workflow**

#### **1. Preprocess Text**

```bash
curl -X POST "http://localhost:8001/preprocess" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Bitcoin (BTC) shows strong bullish signals today! 🚀",
       "config": {
         "remove_html": true,
         "normalize_unicode": true,
         "lowercase": true
       }
     }'
```

#### **2. Analyze Sentiment**

```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "bitcoin btc shows strong bullish signals today"
     }'
```

#### **3. Register Model (Admin)**

```bash
curl -X POST "http://localhost:8000/register" \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "model_path": "outputs/training_run_001/model",
       "run_id": "mlflow_run_123456789",
       "stage": "Production",
       "description": "Production-ready FinBERT model"
     }'
```

### **Batch Processing Example**

```bash
# Process multiple texts
curl -X POST "http://localhost:8080/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "texts": [
         "Bitcoin shows strong bullish signals today.",
         "Market uncertainty leads to bearish sentiment.",
         "Neutral market conditions with mixed signals.",
         "Ethereum adoption continues to grow rapidly.",
         "Regulatory concerns impact market confidence."
       ]
     }'
```

### **Model Training Example**

```bash
# Start training job
curl -X POST "http://localhost:8000/train" \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "data_path": "data/crypto_news_dataset.json",
       "output_dir": "outputs/training_run_002",
       "experiment_name": "crypto-sentiment-v3",
       "config": {
         "backbone": "ProsusAI/finbert",
         "batch_size": 32,
         "learning_rate": 1e-5,
         "num_epochs": 10,
         "max_length": 512,
         "early_stopping_patience": 5
       }
     }'

# Check training status
curl -X GET "http://localhost:8000/train/train_123456789" \
     -H "Authorization: Bearer YOUR_API_KEY"
```

---

**For more information, see the [Configuration Guide](configuration.md) and [Architecture Documentation](architecture.md).**
