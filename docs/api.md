# FinSight Platform API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Common Response Formats](#common-response-formats)
4. [Market Dataset Service API](#market-dataset-service-api)
5. [News Service API](#news-service-api)
6. [Sentiment Analysis Service API](#sentiment-analysis-service-api)
7. [Prediction Service API](#prediction-service-api)
8. [Error Handling](#error-handling)
9. [Rate Limiting](#rate-limiting)
10. [API Examples](#api-examples)

## Overview

The FinSight platform provides a comprehensive set of APIs for financial data collection, news analysis, sentiment analysis, and AI-driven predictions. All services follow RESTful principles and use JSON for data exchange.

### Base URLs

| Service                        | Development             | Production                             | Description                               |
| ------------------------------ | ----------------------- | -------------------------------------- | ----------------------------------------- |
| **Market Dataset Service**     | `http://localhost:8000` | `https://api.finsight.com/market-data` | Financial data collection and backtesting |
| **News Service**               | `http://localhost:8001` | `https://api.finsight.com/news`        | News aggregation and processing           |
| **Sentiment Analysis Service** | `http://localhost:8002` | `https://api.finsight.com/sentiment`   | Sentiment analysis and inference          |
| **Prediction Service**         | `http://localhost:8003` | `https://api.finsight.com/prediction`  | AI model training and prediction          |

### API Versioning

All APIs use URL path versioning:

- Current version: `v1`
- Example: `https://api.finsight.com/market-data/v1`

## Authentication

### API Key Authentication

Most endpoints require API key authentication using the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" \
     https://api.finsight.com/market-data/v1/health
```

### Authentication Levels

| Level             | Description                | Required Endpoints                     |
| ----------------- | -------------------------- | -------------------------------------- |
| **Public**        | No authentication required | Health checks, public data             |
| **Authenticated** | API key required           | Most business endpoints                |
| **Admin**         | Admin API key required     | System administration, data management |

## Common Response Formats

### Success Response

```json
{
  "success": true,
  "data": {
    // Response data
  },
  "message": "Operation completed successfully",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

### Error Response

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "symbol",
      "issue": "Symbol is required"
    }
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

### Paginated Response

```json
{
  "success": true,
  "data": {
    "items": [
      // Array of items
    ],
    "pagination": {
      "page": 1,
      "per_page": 10,
      "total": 100,
      "total_pages": 10
    }
  },
  "message": "Data retrieved successfully",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

## Market Dataset Service API

### Base URL: `http://localhost:8000`

### Health & Status

#### GET `/health`

Get service health status.

**Response:**

```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "service": "market-dataset-service",
    "version": "1.0.0",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### GET `/admin/health`

Get detailed health status (requires admin API key).

**Response:**

```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "dependencies": {
      "mongodb": "connected",
      "redis": "connected",
      "rabbitmq": "connected",
      "binance_api": "connected"
    },
    "data_fresh": true,
    "last_collection": "2024-01-15T10:00:00Z"
  }
}
```

### Market Data

#### GET `/market-data/ohlcv`

Get OHLCV data for a specific symbol and timeframe.

**Parameters:**

- `exchange` (string, required): Exchange name (e.g., "binance")
- `symbol` (string, required): Trading symbol (e.g., "BTCUSDT")
- `timeframe` (string, required): Timeframe (e.g., "1h", "1d")
- `start_date` (string, required): Start date in ISO format
- `end_date` (string, required): End date in ISO format
- `limit` (integer, optional): Maximum number of records (default: 1000)

**Example:**

```bash
curl -H "X-API-Key: your-api-key" \
     "http://localhost:8000/market-data/ohlcv?exchange=binance&symbol=BTCUSDT&timeframe=1h&start_date=2024-01-01T00:00:00Z&end_date=2024-01-02T00:00:00Z"
```

**Response:**

```json
{
  "success": true,
  "data": {
    "exchange": "binance",
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "data": [
      {
        "timestamp": "2024-01-01T00:00:00Z",
        "open": 45000.0,
        "high": 45100.0,
        "low": 44900.0,
        "close": 45050.0,
        "volume": 1000.5
      }
    ],
    "count": 24,
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-02T00:00:00Z"
  }
}
```

#### GET `/market-data/ohlcv/stats`

Get statistics for a symbol and timeframe.

**Parameters:**

- `exchange` (string, required): Exchange name
- `symbol` (string, required): Trading symbol
- `timeframe` (string, required): Timeframe

**Response:**

```json
{
  "success": true,
  "data": {
    "exchange": "binance",
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "statistics": {
      "total_records": 8760,
      "date_range": {
        "start": "2023-01-01T00:00:00Z",
        "end": "2024-01-15T10:00:00Z"
      },
      "price_stats": {
        "current_price": 45000.0,
        "avg_price": 42000.0,
        "max_price": 52000.0,
        "min_price": 30000.0
      },
      "volume_stats": {
        "avg_volume": 1000.5,
        "max_volume": 5000.0,
        "min_volume": 100.0
      }
    }
  }
}
```

### Backtesting

#### POST `/backtesting/run`

Run a backtest strategy.

**Request Body:**

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "strategy_type": "moving_average_crossover",
  "strategy_params": {
    "short_window": 10,
    "long_window": 30
  },
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-01-31T00:00:00Z",
  "initial_capital": 10000.0,
  "commission": 0.001
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "backtest_id": "bt_123456789",
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "strategy_type": "moving_average_crossover",
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-31T00:00:00Z",
    "initial_capital": 10000.0,
    "final_capital": 10500.0,
    "total_return": 0.05,
    "annual_return": 0.6,
    "max_drawdown": 0.02,
    "sharpe_ratio": 1.2,
    "total_trades": 25,
    "win_rate": 0.6,
    "execution_time": 2.5,
    "trades": [
      {
        "timestamp": "2024-01-01T10:00:00Z",
        "action": "BUY",
        "price": 45000.0,
        "size": 0.222,
        "commission": 0.045,
        "reason": "MA crossover signal"
      }
    ],
    "equity_curve": [
      {
        "timestamp": "2024-01-01T00:00:00Z",
        "equity": 10000.0,
        "drawdown": 0.0
      }
    ]
  }
}
```

#### GET `/backtesting/results/{backtest_id}`

Get backtest results by ID.

**Parameters:**

- `include_trades` (boolean, optional): Include trade details (default: true)
- `include_equity_curve` (boolean, optional): Include equity curve (default: true)

#### GET `/backtesting/history`

Get backtest history with pagination.

**Parameters:**

- `page` (integer, optional): Page number (default: 1)
- `per_page` (integer, optional): Items per page (default: 10)
- `strategy_filter` (string, optional): Filter by strategy
- `symbol_filter` (string, optional): Filter by symbol

**Response:**

```json
{
  "success": true,
  "data": {
    "items": [
      {
        "backtest_id": "bt_123456789",
        "symbol": "BTCUSDT",
        "strategy_type": "moving_average_crossover",
        "executed_at": "2024-01-15T10:30:00Z",
        "total_return": 0.05,
        "sharpe_ratio": 1.2
      }
    ],
    "pagination": {
      "page": 1,
      "per_page": 10,
      "total": 100,
      "total_pages": 10
    }
  }
}
```

### Storage Management

#### GET `/storage/list`

List available datasets.

**Parameters:**

- `symbol` (string, optional): Filter by symbol
- `timeframe` (string, optional): Filter by timeframe
- `exchange` (string, optional): Filter by exchange
- `format_type` (string, optional): Filter by format (csv, parquet)

#### POST `/storage/upload/{symbol}`

Upload dataset for a symbol.

**Parameters:**

- `timeframe` (string, optional): Timeframe (default: "1h")
- `exchange` (string, optional): Exchange (default: "binance")
- `target_format` (string, optional): Target format (default: "parquet")
- `compress` (boolean, optional): Compress dataset (default: true)

#### GET `/storage/download/{symbol}`

Download dataset for a symbol.

**Parameters:**

- `timeframe` (string, optional): Timeframe (default: "1h")
- `exchange` (string, optional): Exchange (default: "binance")
- `format_type` (string, optional): Format (default: "csv")

### Admin Endpoints

#### POST `/admin/data/ensure`

Ensure data availability for specified symbols and timeframes.

**Request Body:**

```json
{
  "symbols": ["BTCUSDT", "ETHUSDT"],
  "timeframes": ["1h", "1d"],
  "exchange": "binance",
  "force_collection": false
}
```

#### GET `/admin/stats`

Get system statistics (requires admin API key).

**Response:**

```json
{
  "success": true,
  "data": {
    "storage_stats": {
      "total_records": 1000000,
      "total_size_gb": 50.5,
      "symbols_count": 100,
      "timeframes_count": 5
    },
    "collection_stats": {
      "last_collection": "2024-01-15T10:00:00Z",
      "collection_success_rate": 0.98,
      "active_jobs": 2
    },
    "backtesting_stats": {
      "total_backtests": 150,
      "successful_backtests": 145,
      "avg_execution_time": 3.2
    }
  }
}
```

## News Service API

### Base URL: `http://localhost:8001`

### Health & Status

#### GET `/health`

Get service health status.

#### GET `/admin/health`

Get detailed health status (requires admin API key).

### News Collection

#### GET `/news`

Get collected news articles.

**Parameters:**

- `source` (string, optional): News source filter
- `category` (string, optional): News category
- `start_date` (string, optional): Start date filter
- `end_date` (string, optional): End date filter
- `limit` (integer, optional): Maximum articles (default: 50)
- `offset` (integer, optional): Pagination offset (default: 0)

**Response:**

```json
{
  "success": true,
  "data": {
    "articles": [
      {
        "id": "news_123456789",
        "title": "Bitcoin Reaches New All-Time High",
        "content": "Bitcoin has reached a new all-time high...",
        "source": "coindesk",
        "category": "cryptocurrency",
        "published_at": "2024-01-15T10:00:00Z",
        "collected_at": "2024-01-15T10:05:00Z",
        "url": "https://coindesk.com/article/123",
        "sentiment_score": 0.8,
        "sentiment_label": "positive"
      }
    ],
    "total": 1000,
    "offset": 0,
    "limit": 50
  }
}
```

#### POST `/news/collect`

Trigger news collection for specified sources.

**Request Body:**

```json
{
  "sources": ["coindesk", "cointelegraph"],
  "categories": ["cryptocurrency", "markets"],
  "priority": "high"
}
```

### Search Integration

#### GET `/search`

Search news using Tavily search engine.

**Parameters:**

- `query` (string, required): Search query
- `sources` (string, optional): Comma-separated sources
- `max_results` (integer, optional): Maximum results (default: 10)

**Response:**

```json
{
  "success": true,
  "data": {
    "query": "bitcoin price prediction",
    "results": [
      {
        "title": "Bitcoin Price Prediction 2024",
        "url": "https://example.com/article",
        "snippet": "Analysts predict Bitcoin could reach...",
        "source": "example.com",
        "published_at": "2024-01-15T10:00:00Z"
      }
    ],
    "total_results": 1000
  }
}
```

### Job Management

#### GET `/jobs/status`

Get job status and statistics.

#### POST `/jobs/collect`

Start news collection job.

**Request Body:**

```json
{
  "sources": ["coindesk", "cointelegraph"],
  "parallel_processing": true,
  "max_workers": 5
}
```

## Sentiment Analysis Service API

### Base URL: `http://localhost:8002`

### Health & Status

#### GET `/health`

Get service health status.

#### GET `/admin/health`

Get detailed health status (requires admin API key).

### Sentiment Analysis

#### POST `/sentiment/analyze`

Analyze sentiment for single text.

**Request Body:**

```json
{
  "text": "Bitcoin shows strong bullish signals today.",
  "language": "en",
  "model": "finbert"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "text": "Bitcoin shows strong bullish signals today.",
    "sentiment": {
      "label": "positive",
      "score": 0.85,
      "confidence": 0.92
    },
    "model": "finbert",
    "processing_time": 0.15
  }
}
```

#### POST `/sentiment/analyze/batch`

Analyze sentiment for multiple texts.

**Request Body:**

```json
{
  "texts": [
    "Bitcoin is bullish today.",
    "Market shows bearish signals.",
    "Ethereum remains stable."
  ],
  "language": "en",
  "model": "finbert"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "text": "Bitcoin is bullish today.",
        "sentiment": {
          "label": "positive",
          "score": 0.78
        }
      },
      {
        "text": "Market shows bearish signals.",
        "sentiment": {
          "label": "negative",
          "score": -0.65
        }
      },
      {
        "text": "Ethereum remains stable.",
        "sentiment": {
          "label": "neutral",
          "score": 0.02
        }
      }
    ],
    "batch_size": 3,
    "processing_time": 0.45
  }
}
```

### Model Management

#### GET `/models/available`

Get available sentiment analysis models.

**Response:**

```json
{
  "success": true,
  "data": {
    "models": [
      {
        "name": "finbert",
        "description": "Financial BERT for sentiment analysis",
        "version": "1.0.0",
        "supported_languages": ["en"],
        "accuracy": 0.89
      },
      {
        "name": "bert-base",
        "description": "Base BERT model",
        "version": "1.0.0",
        "supported_languages": ["en"],
        "accuracy": 0.85
      }
    ]
  }
}
```

#### POST `/models/load`

Load a specific model into memory.

**Request Body:**

```json
{
  "model_name": "finbert",
  "force_reload": false
}
```

### Training & Evaluation

#### POST `/models/train`

Start model training.

**Request Body:**

```json
{
  "model_name": "finbert",
  "training_data": "path/to/training/data.json",
  "hyperparameters": {
    "learning_rate": 0.00001,
    "batch_size": 16,
    "epochs": 3
  },
  "evaluation_data": "path/to/eval/data.json"
}
```

#### GET `/models/evaluate`

Get model evaluation results.

**Parameters:**

- `model_name` (string, required): Model to evaluate
- `dataset` (string, required): Evaluation dataset

## Prediction Service API

### Base URL: `http://localhost:8003`

### Health & Status

#### GET `/health`

Get service health status.

#### GET `/admin/health`

Get detailed health status (requires admin API key).

### Predictions

#### POST `/prediction/predict`

Get price prediction for a symbol.

**Request Body:**

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "n_steps": 24,
  "enable_fallback": true,
  "include_confidence": true
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "model_used": "patchedtst_v1",
    "predictions": [
      {
        "timestamp": "2024-01-16T10:00:00Z",
        "price": 45100.0,
        "confidence": 0.85
      },
      {
        "timestamp": "2024-01-16T11:00:00Z",
        "price": 45200.0,
        "confidence": 0.82
      }
    ],
    "metrics": {
      "prediction_horizon": "24h",
      "model_accuracy": 0.78,
      "processing_time": 0.5
    }
  }
}
```

#### POST `/prediction/predict/batch`

Get predictions for multiple symbols.

**Request Body:**

```json
{
  "requests": [
    {
      "symbol": "BTCUSDT",
      "timeframe": "1h",
      "n_steps": 24
    },
    {
      "symbol": "ETHUSDT",
      "timeframe": "1h",
      "n_steps": 24
    }
  ]
}
```

### Model Training

#### POST `/training/start`

Start model training.

**Request Body:**

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "model_type": "patchedtst",
  "hyperparameters": {
    "context_length": 168,
    "epochs": 100,
    "learning_rate": 0.001,
    "batch_size": 32
  },
  "data_config": {
    "start_date": "2023-01-01T00:00:00Z",
    "end_date": "2024-01-01T00:00:00Z",
    "validation_split": 0.2
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
    "estimated_duration": 3600,
    "model_path": "models/patchedtst/btcusdt_1h_v1"
  }
}
```

#### GET `/training/status/{job_id}`

Get training job status.

**Response:**

```json
{
  "success": true,
  "data": {
    "job_id": "train_123456789",
    "status": "training",
    "progress": {
      "current_epoch": 45,
      "total_epochs": 100,
      "current_loss": 0.023,
      "validation_loss": 0.025
    },
    "started_at": "2024-01-15T10:00:00Z",
    "estimated_completion": "2024-01-15T11:00:00Z"
  }
}
```

### Model Management

#### GET `/models/available`

Get available models for prediction.

**Response:**

```json
{
  "success": true,
  "data": {
    "models": [
      {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "model_type": "patchedtst",
        "version": "v1",
        "accuracy": 0.78,
        "last_updated": "2024-01-15T09:00:00Z",
        "status": "active"
      }
    ]
  }
}
```

#### POST `/models/deploy`

Deploy a trained model.

**Request Body:**

```json
{
  "model_path": "models/patchedtst/btcusdt_1h_v1",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "version": "v2"
}
```

## Error Handling

### Error Codes

| Code                   | Description                     | HTTP Status |
| ---------------------- | ------------------------------- | ----------- |
| `VALIDATION_ERROR`     | Input validation failed         | 400         |
| `AUTHENTICATION_ERROR` | Invalid API key                 | 401         |
| `AUTHORIZATION_ERROR`  | Insufficient permissions        | 403         |
| `NOT_FOUND`            | Resource not found              | 404         |
| `RATE_LIMIT_EXCEEDED`  | Rate limit exceeded             | 429         |
| `SERVICE_UNAVAILABLE`  | Service temporarily unavailable | 503         |
| `INTERNAL_ERROR`       | Internal server error           | 500         |

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "symbol",
      "issue": "Symbol is required",
      "value": null
    }
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

## Rate Limiting

### Rate Limits

| Endpoint Type     | Rate Limit           | Window   |
| ----------------- | -------------------- | -------- |
| **Public**        | 100 requests/minute  | 1 minute |
| **Authenticated** | 1000 requests/minute | 1 minute |
| **Admin**         | 5000 requests/minute | 1 minute |

### Rate Limit Headers

Response headers include rate limit information:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1642233600
```

### Rate Limit Exceeded Response

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 1000,
      "window": "1 minute",
      "reset_time": "2024-01-15T10:31:00Z"
    }
  }
}
```

## API Examples

### Complete Workflow Example

Here's a complete example of analyzing market data, news sentiment, and generating predictions:

```python
import requests
import json

# Configuration
BASE_URLS = {
    "market_data": "http://localhost:8000",
    "news": "http://localhost:8001",
    "sentiment": "http://localhost:8002",
    "prediction": "http://localhost:8003"
}

API_KEY = "your-api-key"
HEADERS = {"X-API-Key": API_KEY}

# 1. Get market data
market_response = requests.get(
    f"{BASE_URLS['market_data']}/market-data/ohlcv",
    params={
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "start_date": "2024-01-01T00:00:00Z",
        "end_date": "2024-01-15T00:00:00Z"
    },
    headers=HEADERS
)

market_data = market_response.json()

# 2. Get recent news
news_response = requests.get(
    f"{BASE_URLS['news']}/news",
    params={
        "symbol": "BTCUSDT",
        "limit": 10,
        "start_date": "2024-01-14T00:00:00Z"
    },
    headers=HEADERS
)

news_data = news_response.json()

# 3. Analyze sentiment for news
sentiment_response = requests.post(
    f"{BASE_URLS['sentiment']}/sentiment/analyze/batch",
    json={
        "texts": [article["title"] for article in news_data["data"]["articles"]]
    },
    headers=HEADERS
)

sentiment_data = sentiment_response.json()

# 4. Get price prediction
prediction_response = requests.post(
    f"{BASE_URLS['prediction']}/prediction/predict",
    json={
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "n_steps": 24,
        "include_confidence": True
    },
    headers=HEADERS
)

prediction_data = prediction_response.json()

# 5. Print results
print("Market Data Points:", len(market_data["data"]["data"]))
print("Recent News Articles:", len(news_data["data"]["articles"]))
print("Average Sentiment Score:", sum(r["sentiment"]["score"] for r in sentiment_data["data"]["results"]) / len(sentiment_data["data"]["results"]))
print("Price Predictions:", len(prediction_data["data"]["predictions"]))
```

### cURL Examples

#### Get OHLCV Data

```bash
curl -H "X-API-Key: your-api-key" \
     "http://localhost:8000/market-data/ohlcv?exchange=binance&symbol=BTCUSDT&timeframe=1h&start_date=2024-01-01T00:00:00Z&end_date=2024-01-02T00:00:00Z"
```

#### Run Backtest

```bash
curl -X POST -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "symbol": "BTCUSDT",
       "timeframe": "1h",
       "strategy_type": "moving_average_crossover",
       "strategy_params": {"short_window": 10, "long_window": 30},
       "start_date": "2024-01-01T00:00:00Z",
       "end_date": "2024-01-31T00:00:00Z",
       "initial_capital": 10000.0
     }' \
     "http://localhost:8000/backtesting/run"
```

#### Analyze Sentiment

```bash
curl -X POST -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Bitcoin shows strong bullish signals today.",
       "model": "finbert"
     }' \
     "http://localhost:8002/sentiment/analyze"
```

#### Get Prediction

```bash
curl -X POST -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "symbol": "BTCUSDT",
       "timeframe": "1h",
       "n_steps": 24,
       "enable_fallback": true
     }' \
     "http://localhost:8003/prediction/predict"
```

---

_For more detailed examples and service-specific documentation, refer to the individual service documentation in their respective directories._
