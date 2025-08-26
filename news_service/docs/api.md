# API Documentation

## Overview

The FinSight News Service provides both REST API and gRPC endpoints for news collection, storage, and retrieval. The API is built with FastAPI and supports comprehensive filtering, pagination, and real-time operations.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://api.finsight.com/news-service`

## Authentication

### API Key Authentication

For admin endpoints, use Bearer token authentication:

```bash
Authorization: Bearer YOUR_API_KEY
```

Set the API key in your environment:

```bash
SECRET_API_KEY=your-secret-api-key
```

## REST API Endpoints

### Health & Monitoring

#### GET /health

Service health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "service": "news-service",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "dependencies": {
    "mongodb": "healthy",
    "redis": "healthy",
    "rabbitmq": "healthy"
  },
  "uptime": 3600.5
}
```

#### GET /metrics

Service metrics and performance data.

**Response:**

```json
{
  "requests_total": 1500,
  "requests_success": 1450,
  "requests_error": 50,
  "average_response_time": 0.125,
  "active_connections": 25,
  "memory_usage_mb": 512.5,
  "cpu_usage_percent": 15.2
}
```

### News Operations

#### GET /news/

Search and retrieve news articles with flexible filtering.

**Query Parameters:**

- `source` (optional): Filter by news source (`coindesk`, `cointelegraph`)
- `keywords` (optional): Comma-separated keywords to search in title and description
- `tags` (optional): Comma-separated tags to filter by
- `start_date` (optional): Start date filter (ISO format)
- `end_date` (optional): End date filter (ISO format)
- `limit` (optional): Maximum items to return (1-1000, default: 100)
- `offset` (optional): Number of items to skip for pagination (default: 0)

**Example Request:**

```bash
curl "http://localhost:8000/news/?source=coindesk&keywords=bitcoin,ethereum&limit=20&offset=0"
```

**Response:**

```json
{
  "items": [
    {
      "source": "coindesk",
      "title": "Bitcoin Reaches New All-Time High",
      "url": "https://example.com/article1",
      "description": "Bitcoin has reached a new all-time high...",
      "published_at": "2024-01-15T10:00:00Z",
      "author": "John Doe",
      "tags": ["bitcoin", "cryptocurrency", "trading"],
      "image_url": "https://example.com/image1.jpg",
      "sentiment": "positive"
    }
  ],
  "total_count": 150,
  "limit": 20,
  "offset": 0,
  "has_more": true,
  "filters_applied": {
    "source": "coindesk",
    "keywords": ["bitcoin", "ethereum"],
    "limit": 20,
    "offset": 0
  }
}
```

#### GET /news/recent

Get recent news articles from the last specified hours.

**Query Parameters:**

- `hours` (optional): Hours to look back (1-168, default: 24)
- `source` (optional): Filter by news source
- `limit` (optional): Maximum items to return (1-1000, default: 100)

**Example Request:**

```bash
curl "http://localhost:8000/news/recent?hours=48&source=cointelegraph&limit=50"
```

#### POST /news/search/time-range

Search news articles within a specific time range.

**Request Body:**

```json
{
  "hours": 72,
  "source": "coindesk",
  "keywords": ["crypto", "regulation"],
  "limit": 100,
  "offset": 0
}
```

#### GET /news/keywords/{keywords}

Search news by specific keywords.

**Path Parameters:**

- `keywords`: Comma-separated keywords

**Query Parameters:**

- `source` (optional): Filter by news source
- `limit` (optional): Maximum items to return (1-1000, default: 100)
- `offset` (optional): Number of items to skip (default: 0)
- `hours` (optional): Hours to look back (1-8760)

**Example Request:**

```bash
curl "http://localhost:8000/news/keywords/bitcoin,ethereum?source=coindesk&limit=20"
```

#### GET /news/by-tag/{tags}

Get news articles by specific tags.

**Path Parameters:**

- `tags`: Comma-separated tags

**Query Parameters:**

- `source` (optional): Filter by news source
- `limit` (optional): Maximum items to return (1-1000, default: 100)
- `offset` (optional): Number of items to skip (default: 0)
- `hours` (optional): Hours to look back (1-8760)

**Example Request:**

```bash
curl "http://localhost:8000/news/by-tag/cryptocurrency,trading?limit=30"
```

#### GET /news/tags/available

Get available tags in the system.

**Query Parameters:**

- `source` (optional): Filter by news source
- `limit` (optional): Maximum tags to return (1-500, default: 100)

**Response:**

```json
{
  "tags": ["bitcoin", "ethereum", "cryptocurrency", "trading", "regulation"],
  "total_count": 45,
  "source_filter": "coindesk"
}
```

#### GET /news/health/check

News service specific health check.

**Response:**

```json
{
  "status": "healthy",
  "database_connection": "connected",
  "cache_connection": "connected",
  "message_queue": "connected",
  "last_collection": "2024-01-15T10:00:00Z",
  "total_articles": 15420
}
```

### Cache Management

#### GET /news/cache/stats

Get cache statistics (requires admin access).

**Headers:**

```bash
Authorization: Bearer YOUR_API_KEY
```

**Response:**

```json
{
  "cache_enabled": true,
  "total_keys": 1250,
  "memory_usage": "512MB",
  "hit_rate": 0.85,
  "miss_rate": 0.15,
  "evictions": 25,
  "endpoint_stats": {
    "search_news": {
      "hits": 450,
      "misses": 50,
      "ttl": 1800
    },
    "recent_news": {
      "hits": 300,
      "misses": 30,
      "ttl": 900
    }
  }
}
```

#### GET /news/cache/health

Check cache health status.

**Response:**

```json
{
  "status": "healthy",
  "redis_connection": "connected",
  "cache_enabled": true,
  "last_check": "2024-01-15T10:30:00Z"
}
```

#### POST /news/cache/invalidate

Invalidate all cache entries (requires admin access).

**Headers:**

```bash
Authorization: Bearer YOUR_API_KEY
```

**Response:**

```json
{
  "success": true,
  "message": "Cache invalidated successfully",
  "invalidated_keys": 1250,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Job Management (Admin Only)

#### GET /admin/jobs/status

Get job service status.

**Headers:**

```bash
Authorization: Bearer YOUR_API_KEY
```

**Response:**

```json
{
  "service": "news-crawler-job",
  "version": "1.0.0",
  "status": "running",
  "is_running": true,
  "pid": 12345,
  "pid_file": "/app/news_crawler_job.pid",
  "config_file": "/app/news_crawler_config.json",
  "log_file": "/app/logs/news_crawler_job.log",
  "scheduler_running": true,
  "next_run": "2024-01-15T12:00:00Z",
  "stats": {
    "total_jobs": 150,
    "successful_jobs": 145,
    "failed_jobs": 5,
    "last_run": "2024-01-15T06:00:00Z",
    "last_success": "2024-01-15T06:00:00Z",
    "last_error": null
  }
}
```

#### POST /admin/jobs/start

Start the job service.

**Headers:**

```bash
Authorization: Bearer YOUR_API_KEY
```

**Request Body:**

```json
{
  "config": {
    "sources": ["coindesk", "cointelegraph"],
    "max_items_per_source": 100,
    "schedule": "0 */6 * * *",
    "enable_fallback": true
  },
  "force_restart": false
}
```

**Response:**

```json
{
  "success": true,
  "message": "Job service started successfully",
  "status": "running",
  "timestamp": "2024-01-15T10:30:00Z",
  "details": {
    "pid": 12345,
    "config_applied": true
  }
}
```

#### POST /admin/jobs/stop

Stop the job service.

**Headers:**

```bash
Authorization: Bearer YOUR_API_KEY
```

**Request Body:**

```json
{
  "graceful": true
}
```

#### POST /admin/jobs/run

Run a manual job.

**Headers:**

```bash
Authorization: Bearer YOUR_API_KEY
```

**Request Body:**

```json
{
  "sources": ["coindesk"],
  "max_items_per_source": 50,
  "config_overrides": {
    "timeout": 30
  }
}
```

**Response:**

```json
{
  "status": "completed",
  "job_id": "job_20240115_103000",
  "sources": ["coindesk"],
  "max_items_per_source": 50,
  "start_time": "2024-01-15T10:30:00Z",
  "duration": 45.2,
  "results": {
    "coindesk": {
      "items_collected": 45,
      "items_stored": 42,
      "duplicates": 3
    }
  }
}
```

#### GET /admin/jobs/config

Get current job configuration.

**Headers:**

```bash
Authorization: Bearer YOUR_API_KEY
```

**Response:**

```json
{
  "sources": ["coindesk", "cointelegraph"],
  "collector_preferences": {
    "coindesk": "api_rest",
    "cointelegraph": "api_graphql"
  },
  "max_items_per_source": 100,
  "enable_fallback": true,
  "schedule": "0 */6 * * *",
  "config_overrides": {},
  "notification": {
    "enabled": false,
    "webhook_url": null,
    "email": null
  }
}
```

#### PUT /admin/jobs/config

Update job configuration.

**Headers:**

```bash
Authorization: Bearer YOUR_API_KEY
```

**Request Body:**

```json
{
  "sources": ["coindesk", "cointelegraph"],
  "max_items_per_source": 150,
  "schedule": "0 */4 * * *",
  "enable_fallback": true
}
```

#### GET /admin/jobs/stats

Get job statistics.

**Headers:**

```bash
Authorization: Bearer YOUR_API_KEY
```

**Response:**

```json
{
  "total_jobs": 150,
  "successful_jobs": 145,
  "failed_jobs": 5,
  "last_run": "2024-01-15T06:00:00Z",
  "last_success": "2024-01-15T06:00:00Z",
  "last_error": null,
  "average_duration": 45.2,
  "success_rate": 0.967
}
```

#### GET /admin/jobs/health

Job service health check.

**Headers:**

```bash
Authorization: Bearer YOUR_API_KEY
```

**Response:**

```json
{
  "status": "healthy",
  "job_service": "running",
  "scheduler": "active",
  "last_heartbeat": "2024-01-15T10:30:00Z",
  "dependencies": {
    "database": "connected",
    "message_queue": "connected"
  }
}
```

### Eureka Service Discovery

#### GET /eureka/status

Get Eureka client status.

**Response:**

```json
{
  "registered": true,
  "instance_id": "news-service-12345",
  "app_name": "news-service",
  "host_name": "news-service.local",
  "ip_address": "192.168.1.100",
  "port": 8000,
  "secure_port": 8443,
  "status": "UP",
  "last_heartbeat": "2024-01-15T10:30:00Z",
  "lease_renewal_interval": 30,
  "lease_expiration_duration": 90
}
```

#### POST /eureka/register

Register with Eureka (requires admin access).

**Headers:**

```bash
Authorization: Bearer YOUR_API_KEY
```

**Response:**

```json
{
  "success": true,
  "message": "Successfully registered with Eureka",
  "instance_id": "news-service-12345",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### POST /eureka/deregister

Deregister from Eureka (requires admin access).

**Headers:**

```bash
Authorization: Bearer YOUR_API_KEY
```

#### POST /eureka/re-register

Re-register with Eureka (requires admin access).

**Headers:**

```bash
Authorization: Bearer YOUR_API_KEY
```

### Search Operations

#### POST /search/

Search content using Tavily search engine.

**Request Body:**

```json
{
  "query": "bitcoin price analysis",
  "topic": "finance",
  "search_depth": "advanced",
  "time_range": "week",
  "include_answer": true,
  "max_results": 20,
  "chunks_per_source": 3
}
```

**Response:**

```json
{
  "query": "bitcoin price analysis",
  "total_results": 15,
  "results": [
    {
      "url": "https://example.com/article1",
      "title": "Bitcoin Price Analysis: Technical Indicators",
      "content": "Bitcoin has shown strong momentum...",
      "score": 0.95,
      "published_at": "2024-01-15T10:00:00Z",
      "source": "coindesk",
      "is_crawled": true,
      "metadata": {
        "author": "John Doe",
        "tags": ["bitcoin", "analysis"]
      }
    }
  ],
  "answer": "Based on recent analysis, Bitcoin shows strong technical indicators...",
  "follow_up_questions": [
    "What are the key support levels?",
    "How does this compare to previous cycles?"
  ],
  "response_time": 2.5,
  "search_depth": "advanced",
  "topic": "finance",
  "time_range": "week",
  "crawler_used": true
}
```

#### GET /search/financial-sentiment/{symbol}

Get financial sentiment for a specific symbol.

**Path Parameters:**

- `symbol`: Trading symbol (e.g., BTCUSDT, ETHUSDT)

**Query Parameters:**

- `days` (optional): Number of days to look back (1-30, default: 7)

**Example Request:**

```bash
curl "http://localhost:8000/search/financial-sentiment/BTCUSDT?days=14"
```

## gRPC API

The service also provides gRPC endpoints for high-performance communication. The gRPC server runs on port 50051 by default.

### gRPC Services

#### NewsService

- `SearchNews` - Search news articles
- `GetRecentNews` - Get recent news
- `GetNewsBySource` - Get news by source
- `SearchByKeywords` - Search by keywords
- `GetNewsByTags` - Get news by tags
- `GetNewsItem` - Get specific news item
- `GetAvailableTags` - Get available tags
- `GetNewsStatistics` - Get news statistics
- `DeleteNewsItem` - Delete news item

### gRPC Client Example

```python
import asyncio
import grpc
from grpc_generated import news_service_pb2, news_service_pb2_grpc

async def search_news():
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = news_service_pb2_grpc.NewsServiceStub(channel)

        request = news_service_pb2.SearchNewsRequest(
            keywords=["bitcoin", "ethereum"],
            sources=["coindesk"],
            limit=10,
            skip=0
        )

        response = await stub.SearchNews(request)
        print(f"Found {response.total_count} articles")

asyncio.run(search_news())
```

## Error Codes

### HTTP Status Codes

- `200 OK` - Request successful
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `422 Unprocessable Entity` - Validation error
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service temporarily unavailable

### Error Response Format

```json
{
  "error": "validation_error",
  "message": "Invalid request parameters",
  "details": {
    "field": "limit",
    "value": 1500,
    "constraint": "must be between 1 and 1000"
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_12345"
}
```

### Common Error Codes

| Error Code                  | Description                    | Solution                                       |
| --------------------------- | ------------------------------ | ---------------------------------------------- |
| `INVALID_SOURCE`            | Invalid news source specified  | Use valid sources: `coindesk`, `cointelegraph` |
| `INVALID_DATE_RANGE`        | Invalid date range parameters  | Ensure start_date < end_date                   |
| `RATE_LIMIT_EXCEEDED`       | API rate limit exceeded        | Implement exponential backoff                  |
| `DATABASE_CONNECTION_ERROR` | Database connection failed     | Check MongoDB connection                       |
| `CACHE_ERROR`               | Cache operation failed         | Check Redis connection                         |
| `MESSAGE_QUEUE_ERROR`       | Message queue operation failed | Check RabbitMQ connection                      |

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Public endpoints**: 100 requests per minute
- **Admin endpoints**: 50 requests per minute
- **Search endpoints**: 30 requests per minute

Rate limit headers are included in responses:

```bash
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248600
```

## Pagination

All list endpoints support pagination using `limit` and `offset` parameters:

- `limit`: Number of items per page (1-1000)
- `offset`: Number of items to skip

Pagination metadata is included in responses:

```json
{
  "items": [...],
  "total_count": 1500,
  "limit": 100,
  "offset": 0,
  "has_more": true
}
```

## Data Formats

### Date/Time Format

All dates are in ISO 8601 format: `YYYY-MM-DDTHH:MM:SSZ`

### URL Format

URLs must be valid HTTP/HTTPS URLs and are validated using Pydantic's HttpUrl type.

### Tags Format

Tags are case-sensitive strings, typically lowercase with hyphens for spaces.

## SDKs and Client Libraries

### Python Client

```python
from news_service_client import NewsServiceClient

client = NewsServiceClient("http://localhost:8000")
articles = await client.search_news(keywords=["bitcoin"], limit=10)
```

### JavaScript/TypeScript Client

```typescript
import { NewsServiceClient } from "@finsight/news-service-client";

const client = new NewsServiceClient("http://localhost:8000");
const articles = await client.searchNews({ keywords: ["bitcoin"], limit: 10 });
```

## Testing

### API Testing with curl

```bash
# Health check
curl http://localhost:8000/health

# Search news
curl "http://localhost:8000/news/?keywords=bitcoin&limit=5"

# Admin endpoint (with auth)
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8000/admin/jobs/status
```

### Load Testing

Use tools like Apache Bench or Artillery for load testing:

```bash
# Basic load test
ab -n 1000 -c 10 http://localhost:8000/health

# Search endpoint load test
ab -n 500 -c 5 "http://localhost:8000/news/?limit=10"
```

## Monitoring and Observability

### Health Checks

- Service health: `GET /health`
- Cache health: `GET /news/cache/health`
- Job health: `GET /admin/jobs/health`

### Metrics

- Service metrics: `GET /metrics`
- Cache statistics: `GET /news/cache/stats`
- Job statistics: `GET /admin/jobs/stats`

### Logging

The service uses structured logging with correlation IDs for request tracing. Log levels can be configured via environment variables.

## Support

For API support and questions:

1. Check the [Configuration Guide](configuration.md)
2. Review the [Architecture Guide](architecture.md)
3. Contact the development team
