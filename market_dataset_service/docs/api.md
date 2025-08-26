# API Documentation

## Overview

The FinSight Market Dataset Service provides a comprehensive REST API for market data management, backtesting, and storage operations. This document covers all available endpoints, request/response formats, and authentication requirements.

## Base URL

```bash
http://localhost:8000
```

## Authentication

### API Key Authentication

Most endpoints require API key authentication via the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/admin/stats
```

### Public Endpoints

The following endpoints are publicly accessible:

- `GET /` - Service information
- `GET /health` - Health check
- `GET /docs` - API documentation

## Common Response Formats

### Success Response

```json
{
  "success": true,
  "data": {},
  "message": "Operation completed successfully",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Error Response

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {}
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Endpoints

### 1. Service Information

#### GET /

Get service information and status.

**Response:**

```json
{
  "service": "market-dataset-service",
  "version": "1.0.0",
  "status": "operational",
  "environment": "production",
  "timestamp": "2024-01-01T00:00:00Z",
  "features": {
    "market_data": true,
    "backtesting": true,
    "storage": true,
    "job_management": true
  }
}
```

#### GET /health

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "components": {
    "database": "healthy",
    "storage": "healthy",
    "external_apis": "healthy"
  }
}
```

### 2. Market Data Operations

#### GET /market-data/ohlcv

Retrieve OHLCV data for a specific symbol and timeframe.

**Parameters:**

- `exchange` (string): Exchange name (default: "binance")
- `symbol` (string): Trading symbol (default: "BTCUSDT")
- `timeframe` (string): Timeframe (default: "1h")
- `start_date` (string): Start date in ISO format
- `end_date` (string): End date in ISO format
- `limit` (integer): Maximum number of records (max: 10000)

**Example Request:**

```bash
curl "http://localhost:8000/market-data/ohlcv?symbol=BTCUSDT&timeframe=1h&start_date=2024-01-01T00:00:00Z&end_date=2024-01-02T00:00:00Z"
```

**Response:**

```json
{
  "data": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "open": 45000.0,
      "high": 45100.0,
      "low": 44900.0,
      "close": 45050.0,
      "volume": 1000.5,
      "symbol": "BTCUSDT",
      "exchange": "binance",
      "timeframe": "1h"
    }
  ],
  "count": 1,
  "exchange": "binance",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-01-02T00:00:00Z",
  "has_more": false
}
```

#### GET /market-data/ohlcv/stats

Get statistics for OHLCV data.

**Parameters:**

- `exchange` (string): Exchange name
- `symbol` (string): Trading symbol
- `timeframe` (string): Timeframe

**Response:**

```json
{
  "exchange": "binance",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "total_records": 1000,
  "date_range": {
    "start": "2023-01-01T00:00:00Z",
    "end": "2024-01-01T00:00:00Z"
  },
  "price_range": {
    "min": 30000.0,
    "max": 50000.0
  },
  "volume_stats": {
    "min": 100.0,
    "max": 5000.0,
    "avg": 1500.0
  }
}
```

#### GET /market-data/exchanges

Get available exchanges.

**Response:**

```json
{
  "exchanges": ["binance", "coinbase", "kraken"],
  "count": 3,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### GET /market-data/symbols

Get available symbols for an exchange.

**Parameters:**

- `exchange` (string): Exchange name

**Response:**

```json
{
  "exchange": "binance",
  "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
  "count": 3,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### 3. Backtesting Operations

#### GET /backtesting/strategies

Get available backtesting strategies.

**Response:**

```json
{
  "strategies": [
    {
      "name": "moving_average_crossover",
      "description": "Simple moving average crossover strategy",
      "parameters": {
        "short_window": { "type": "integer", "default": 10 },
        "long_window": { "type": "integer", "default": 30 }
      }
    }
  ],
  "count": 1,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### POST /backtesting/run

Run a backtest.

**Request Body:**

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "exchange": "binance",
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-01-31T00:00:00Z",
  "strategy_type": "moving_average_crossover",
  "strategy_params": {
    "short_window": 10,
    "long_window": 30
  },
  "initial_capital": 10000.0,
  "commission": 0.001
}
```

**Response:**

```json
{
  "backtest_id": "bt_123456789",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "strategy_type": "moving_average_crossover",
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-01-31T00:00:00Z",
  "initial_capital": 10000.0,
  "final_capital": 10500.0,
  "metrics": {
    "total_return": 5.0,
    "annual_return": 60.0,
    "max_drawdown": 2.5,
    "sharpe_ratio": 1.2,
    "total_trades": 15,
    "win_rate": 0.67
  },
  "execution_time_seconds": 2.5,
  "status": "completed"
}
```

#### GET /backtesting/results/{backtest_id}

Get backtest results.

**Parameters:**

- `include_trades` (boolean): Include trade details (default: true)
- `include_equity_curve` (boolean): Include equity curve (default: true)

**Response:**

```json
{
  "backtest_id": "bt_123456789",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "strategy_type": "moving_average_crossover",
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-01-31T00:00:00Z",
  "initial_capital": 10000.0,
  "final_capital": 10500.0,
  "metrics": {
    "total_return": 5.0,
    "annual_return": 60.0,
    "max_drawdown": 2.5,
    "sharpe_ratio": 1.2,
    "total_trades": 15,
    "win_rate": 0.67
  },
  "trades": [
    {
      "entry_date": "2024-01-01T10:00:00Z",
      "exit_date": "2024-01-01T15:00:00Z",
      "entry_price": 45000.0,
      "exit_price": 45200.0,
      "position_side": "long",
      "quantity": 0.22,
      "pnl": 44.0,
      "pnl_percentage": 0.98
    }
  ],
  "equity_curve": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "portfolio_value": 10000.0,
      "cash": 10000.0,
      "position_value": 0.0,
      "drawdown": 0.0
    }
  ],
  "execution_time_seconds": 2.5,
  "status": "completed"
}
```

#### GET /backtesting/history

Get backtest history.

**Parameters:**

- `page` (integer): Page number (default: 1)
- `per_page` (integer): Items per page (default: 10, max: 100)
- `strategy_filter` (string): Filter by strategy name
- `symbol_filter` (string): Filter by symbol

**Response:**

```json
{
  "history": [
    {
      "backtest_id": "bt_123456789",
      "symbol": "BTCUSDT",
      "timeframe": "1h",
      "strategy_type": "moving_average_crossover",
      "total_return": 5.0,
      "sharpe_ratio": 1.2,
      "max_drawdown": 2.5,
      "win_rate": 0.67,
      "start_date": "2024-01-01T00:00:00Z",
      "end_date": "2024-01-31T00:00:00Z",
      "executed_at": "2024-01-01T12:00:00Z",
      "execution_time_seconds": 2.5,
      "status": "completed"
    }
  ],
  "count": 1,
  "page": 1,
  "total_pages": 1,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### 4. Storage Operations

#### GET /storage/query/{symbol}

Query dataset by symbol.

**Parameters:**

- `timeframe` (string): Timeframe (default: "1h")
- `exchange` (string): Exchange name (default: "binance")
- `format_type` (string): Data format (default: "csv")

**Response:**

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "exchange": "binance",
  "format_type": "csv",
  "object_key": "finsight/market_data/datasets/binance/BTCUSDT/1h/csv",
  "file_size_bytes": 1048576,
  "record_count": 1000,
  "date_range": {
    "start": "2023-01-01T00:00:00Z",
    "end": "2024-01-01T00:00:00Z"
  },
  "storage_info": {
    "provider": "minio",
    "bucket": "market-data",
    "compressed": false
  }
}
```

#### GET /storage/download/{symbol}

Download dataset by symbol.

**Parameters:**

- `timeframe` (string): Timeframe (default: "1h")
- `exchange` (string): Exchange name (default: "binance")
- `format_type` (string): Data format (default: "csv")
- `extract_archive` (boolean): Extract archive files (default: false)

**Response:**

```json
{
  "success": true,
  "download_url": "http://localhost:9000/market-data/finsight/market_data/datasets/binance/BTCUSDT/1h/csv",
  "file_size_bytes": 1048576,
  "expires_at": "2024-01-01T13:00:00Z",
  "message": "Download URL generated successfully"
}
```

#### POST /storage/upload/{symbol}

Upload dataset by symbol.

**Parameters:**

- `timeframe` (string): Timeframe (default: "1h")
- `exchange` (string): Exchange name (default: "binance")
- `source_format` (string): Source format (default: "csv")
- `target_format` (string): Target format (default: "parquet")
- `compress` (boolean): Compress dataset (default: true)

**Response:**

```json
{
  "success": true,
  "object_key": "finsight/market_data/datasets/binance/BTCUSDT/1h/parquet",
  "file_size_bytes": 524288,
  "compression_ratio": 0.5,
  "upload_time_seconds": 2.5,
  "message": "Dataset uploaded successfully"
}
```

#### POST /storage/convert/timeframes

Convert timeframes for a symbol.

**Parameters:**

- `exchange` (string): Exchange name
- `symbol` (string): Trading symbol
- `source_timeframe` (string): Source timeframe
- `target_timeframes` (array): Target timeframes
- `source_format` (string): Source format (default: "csv")
- `target_format` (string): Target format (default: "parquet")
- `overwrite_existing` (boolean): Overwrite existing data (default: false)

**Response:**

```json
{
  "success": true,
  "symbol": "BTCUSDT",
  "source_timeframe": "1h",
  "target_timeframes": ["4h", "1d"],
  "converted_records": 2000,
  "processing_time_seconds": 5.2,
  "message": "Timeframe conversion completed successfully"
}
```

### 5. Job Management

#### GET /jobs/status

Get job service status.

**Response:**

```json
{
  "service": "market-data-job",
  "version": "1.0.0",
  "status": "running",
  "is_running": true,
  "pid": 12345,
  "scheduler_running": true,
  "next_run": "2024-01-01T13:00:00Z",
  "uptime_seconds": 3600,
  "stats": {
    "total_jobs": 10,
    "successful_jobs": 9,
    "failed_jobs": 1,
    "last_run": "2024-01-01T12:00:00Z",
    "average_runtime_seconds": 30.5
  }
}
```

#### POST /jobs/start

Start the job service.

**Request Body:**

```json
{
  "config": {
    "cron_schedule": "0 */4 * * *",
    "exchange": "binance",
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "timeframes": ["1h", "4h"],
    "max_lookback_days": 30
  },
  "force_restart": false,
  "background_mode": true
}
```

**Response:**

```json
{
  "success": true,
  "message": "Job service started successfully",
  "status": "running",
  "job_id": "job_123456789",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### POST /jobs/stop

Stop the job service.

**Request Body:**

```json
{
  "graceful": true,
  "timeout_seconds": 30,
  "force_kill": false
}
```

**Response:**

```json
{
  "success": true,
  "message": "Job service stopped successfully",
  "status": "stopped",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### POST /jobs/run

Run a manual data collection job.

**Request Body:**

```json
{
  "symbols": ["BTCUSDT", "ETHUSDT"],
  "timeframes": ["1h", "4h"],
  "max_lookback_days": 30,
  "exchange": "binance",
  "repository_type": "csv",
  "job_priority": "normal",
  "async_execution": true
}
```

**Response:**

```json
{
  "status": "started",
  "job_id": "manual_job_123456789",
  "symbols": ["BTCUSDT", "ETHUSDT"],
  "timeframes": ["1h", "4h"],
  "exchange": "binance",
  "max_lookback_days": 30,
  "start_time": "2024-01-01T12:00:00Z",
  "async_execution": true,
  "message": "Manual job started successfully"
}
```

### 6. Admin Operations

#### GET /admin/stats

Get system statistics.

**Response:**

```json
{
  "total_records": 1000000,
  "unique_symbols": 50,
  "unique_exchanges": 3,
  "available_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
  "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
  "exchanges": ["binance", "coinbase", "kraken"],
  "storage_info": {
    "total_size_bytes": 1073741824,
    "available_space_bytes": 2147483648,
    "utilization_percent": 50.0
  },
  "uptime_seconds": 86400,
  "server_timestamp": "2024-01-01T12:00:00Z"
}
```

#### GET /admin/health

Get system health status.

**Response:**

```json
{
  "status": "healthy",
  "repository_connected": true,
  "data_fresh": true,
  "memory_usage_percent": 45.2,
  "disk_usage_percent": 60.1,
  "checks_timestamp": "2024-01-01T12:00:00Z"
}
```

#### POST /admin/data/ensure

Ensure data availability for a symbol.

**Request Body:**

```json
{
  "exchange": "binance",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-01-02T00:00:00Z",
  "force_refresh": false
}
```

**Response:**

```json
{
  "success": true,
  "data_was_missing": false,
  "records_fetched": 0,
  "records_saved": 0,
  "data_statistics": {
    "total_records": 1000,
    "date_range": {
      "start": "2024-01-01T00:00:00Z",
      "end": "2024-01-02T00:00:00Z"
    }
  },
  "operation_timestamp": "2024-01-01T12:00:00Z"
}
```

#### POST /admin/pipeline/quick-run

Run quick collect → convert → upload pipeline.

**Response:**

```json
{
  "exchange": "binance",
  "symbols": ["BTCUSDT", "ETHUSDT"],
  "source_timeframe": "1h",
  "target_timeframes": ["4h", "1d"],
  "source_format": "csv",
  "target_format": "parquet",
  "started_at": "2024-01-01T12:00:00Z",
  "finished_at": "2024-01-01T12:05:00Z",
  "duration_seconds": 300.0,
  "success": true,
  "message": "Pipeline completed successfully",
  "results_by_symbol": [
    {
      "symbol": "BTCUSDT",
      "collection_status": "completed",
      "collection_records": 1000,
      "conversion_status": "completed",
      "converted_timeframes": ["4h", "1d"],
      "upload_results": [
        {
          "timeframe": "4h",
          "success": true,
          "object_key": "finsight/market_data/datasets/binance/BTCUSDT/4h/parquet"
        }
      ]
    }
  ]
}
```

### 7. Eureka Service Discovery

#### GET /eureka/status

Get Eureka client status.

**Response:**

```json
{
  "registered": true,
  "instance_id": "market-dataset-service-12345",
  "app_name": "market-dataset-service",
  "host_name": "localhost",
  "ip_address": "192.168.1.100",
  "port": 8000,
  "secure_port": 8443,
  "home_page_url": "http://localhost:8000",
  "status_page_url": "http://localhost:8000/health",
  "health_check_url": "http://localhost:8000/health",
  "vip_address": "market-dataset-service",
  "secure_vip_address": "market-dataset-service",
  "lease_renewal_interval_in_seconds": 30,
  "lease_expiration_duration_in_seconds": 90,
  "last_updated_timestamp": "2024-01-01T12:00:00Z",
  "last_dirty_timestamp": "2024-01-01T12:00:00Z"
}
```

#### POST /eureka/register

Register with Eureka server.

**Response:**

```json
{
  "success": true,
  "message": "Successfully registered with Eureka server",
  "instance_id": "market-dataset-service-12345",
  "registration_time": "2024-01-01T12:00:00Z"
}
```

#### POST /eureka/deregister

Deregister from Eureka server.

**Response:**

```json
{
  "success": true,
  "message": "Successfully deregistered from Eureka server",
  "deregistration_time": "2024-01-01T12:00:00Z"
}
```

## Error Codes

### HTTP Status Codes

| Code | Description                                           |
| ---- | ----------------------------------------------------- |
| 200  | Success                                               |
| 201  | Created                                               |
| 400  | Bad Request - Invalid input parameters                |
| 401  | Unauthorized - Missing or invalid API key             |
| 403  | Forbidden - Insufficient permissions                  |
| 404  | Not Found - Resource not found                        |
| 422  | Unprocessable Entity - Validation error               |
| 429  | Too Many Requests - Rate limit exceeded               |
| 500  | Internal Server Error - Server error                  |
| 503  | Service Unavailable - Service temporarily unavailable |

### Error Types

| Error Code             | Description              |
| ---------------------- | ------------------------ |
| `VALIDATION_ERROR`     | Input validation failed  |
| `AUTHENTICATION_ERROR` | Authentication failed    |
| `AUTHORIZATION_ERROR`  | Insufficient permissions |
| `NOT_FOUND_ERROR`      | Resource not found       |
| `REPOSITORY_ERROR`     | Database/storage error   |
| `COLLECTION_ERROR`     | Data collection error    |
| `BACKTESTING_ERROR`    | Backtesting engine error |
| `STORAGE_ERROR`        | Object storage error     |
| `JOB_ERROR`            | Job management error     |
| `EXTERNAL_API_ERROR`   | External API error       |

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Public endpoints**: 100 requests per minute
- **Authenticated endpoints**: 1000 requests per minute
- **Admin endpoints**: 100 requests per minute

Rate limit headers are included in responses:

- `X-RateLimit-Limit`: Request limit per window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when the rate limit resets

## Pagination

Endpoints that return lists support pagination:

**Parameters:**

- `page` (integer): Page number (default: 1)
- `per_page` (integer): Items per page (default: 10, max: 100)

**Response Headers:**

- `X-Total-Count`: Total number of items
- `X-Page-Count`: Total number of pages
- `X-Current-Page`: Current page number

## Data Formats

### Date/Time Format

All dates and times are in ISO 8601 format with UTC timezone:

```bash
2024-01-01T12:00:00Z
```

### OHLCV Data Format

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "open": 45000.0,
  "high": 45100.0,
  "low": 44900.0,
  "close": 45050.0,
  "volume": 1000.5,
  "symbol": "BTCUSDT",
  "exchange": "binance",
  "timeframe": "1h"
}
```

### Performance Metrics Format

```json
{
  "total_return": 5.0,
  "annual_return": 60.0,
  "max_drawdown": 2.5,
  "sharpe_ratio": 1.2,
  "sortino_ratio": 1.5,
  "calmar_ratio": 2.4,
  "total_trades": 15,
  "winning_trades": 10,
  "losing_trades": 5,
  "win_rate": 0.67,
  "average_win": 2.0,
  "average_loss": -1.5,
  "profit_factor": 2.67,
  "volatility": 0.15,
  "var_95": -3.2
}
```

## Python Client Example

```python
import requests
import json
from datetime import datetime, timedelta

class MarketDatasetClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key} if api_key else {}

    def get_ohlcv(self, symbol="BTCUSDT", timeframe="1h",
                  start_date=None, end_date=None, limit=None):
        """Get OHLCV data for a symbol."""
        params = {
            "symbol": symbol,
            "timeframe": timeframe
        }

        if start_date:
            params["start_date"] = start_date.isoformat() + "Z"
        if end_date:
            params["end_date"] = end_date.isoformat() + "Z"
        if limit:
            params["limit"] = limit

        response = requests.get(
            f"{self.base_url}/market-data/ohlcv",
            params=params,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def run_backtest(self, symbol, strategy_type, start_date, end_date,
                    strategy_params=None, initial_capital=10000.0):
        """Run a backtest."""
        data = {
            "symbol": symbol,
            "timeframe": "1h",
            "exchange": "binance",
            "start_date": start_date.isoformat() + "Z",
            "end_date": end_date.isoformat() + "Z",
            "strategy_type": strategy_type,
            "strategy_params": strategy_params or {},
            "initial_capital": initial_capital,
            "commission": 0.001
        }

        response = requests.post(
            f"{self.base_url}/backtesting/run",
            json=data,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Usage example
client = MarketDatasetClient(api_key="your-api-key")

# Get OHLCV data
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
ohlcv_data = client.get_ohlcv("BTCUSDT", "1h", start_date, end_date)

# Run backtest
backtest_result = client.run_backtest(
    symbol="BTCUSDT",
    strategy_type="moving_average_crossover",
    start_date=start_date,
    end_date=end_date,
    strategy_params={"short_window": 10, "long_window": 30}
)

print(f"Backtest completed with {backtest_result['metrics']['total_return']}% return")
```

## WebSocket Support

For real-time data streaming, WebSocket endpoints are available:

```bash
ws://localhost:8000/ws/market-data/{symbol}
ws://localhost:8000/ws/backtest-progress/{backtest_id}
```

WebSocket messages follow the same JSON format as REST API responses.
