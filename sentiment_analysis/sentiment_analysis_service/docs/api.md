# API Documentation - FinSight Sentiment Analysis Service

## Overview

The FinSight Sentiment Analysis Service provides a RESTful API for real-time sentiment analysis of financial news and market content. The service integrates with OpenAI's GPT models to deliver accurate sentiment classification with confidence scores and reasoning.

**Base URL**: `http://localhost:8002`  
**API Version**: `v1`  
**Content Type**: `application/json`

## Authentication

Currently, the service operates without authentication for development purposes. In production, consider implementing:

- API Key authentication
- JWT tokens
- Rate limiting per client

## Common Response Format

### Success Response

```json
{
  "status": "success",
  "data": { ... },
  "timestamp": "2025-01-19T21:00:00Z"
}
```

### Error Response

```json
{
  "error": "Error description",
  "detail": "Detailed error information",
  "status_code": 400,
  "timestamp": "2025-01-19T21:00:00Z"
}
```

## Common Error Codes

| Status Code | Description                             |
| ----------- | --------------------------------------- |
| 200         | Success                                 |
| 400         | Bad Request - Invalid input             |
| 404         | Not Found - Resource not found          |
| 422         | Validation Error - Invalid data format  |
| 500         | Internal Server Error                   |
| 503         | Service Unavailable - Service unhealthy |

## Endpoints

### 1. Root Endpoint

#### GET `/`

Returns basic service information.

**Response:**

```json
{
  "service": "sentiment-analysis-service",
  "status": "running",
  "version": "1.0.0",
  "description": "AI-powered sentiment analysis service",
  "timestamp": "2025-01-19T21:00:00Z"
}
```

### 2. Health Check

#### GET `/health`

Comprehensive health check for all service components.

**Response:**

```json
{
  "status": "healthy",
  "service": "sentiment-analysis-service",
  "timestamp": "2025-01-19T21:00:00Z",
  "components": {
    "sentiment_analyzer": "healthy",
    "database": "healthy",
    "message_broker": "healthy",
    "message_consumer": "running"
  }
}
```

**Possible Status Values:**

- `healthy`: All components operational
- `degraded`: Some components degraded but service functional
- `unhealthy`: Critical components failed

### 3. Service Metrics

#### GET `/metrics`

Returns service performance and operational metrics.

**Response:**

```json
{
  "service": "sentiment-analysis-service",
  "consumer_running": true,
  "uptime": "2h 15m 30s",
  "processed_messages": 150,
  "timestamp": "2025-01-19T21:00:00Z"
}
```

### 4. Sentiment Analysis API

#### GET `/api/v1/sentiment/health`

Health check specifically for the sentiment analysis component.

**Response:**

```json
{
  "status": "healthy",
  "service": "sentiment-analysis",
  "timestamp": "2025-01-19T21:00:00Z",
  "version": "1.0.0",
  "components": {
    "sentiment_analyzer": "available",
    "news_repository": "connected",
    "message_broker": "optional"
  }
}
```

#### POST `/api/v1/sentiment/test`

Test endpoint for manual sentiment analysis of text content.

**Request Body:**

```json
{
  "text": "Bitcoin reaches new all-time high as institutional adoption grows"
}
```

**Response:**

```json
{
  "status": "success",
  "input_text": "Bitcoin reaches new all-time high as institutional adoption grows",
  "sentiment_label": "positive",
  "confidence": 0.92,
  "scores": {
    "positive": 0.85,
    "negative": 0.08,
    "neutral": 0.07
  },
  "reasoning": "The text discusses Bitcoin reaching new highs and institutional adoption, which are generally positive indicators for cryptocurrency markets.",
  "analyzer_version": "openai-gpt-4o-mini"
}
```

## Message Queue Integration

The service integrates with RabbitMQ for asynchronous processing of news messages.

### Message Consumption

The service automatically consumes messages from the `news.sentiment_analysis` queue with routing key `news.sentiment.analyze`.

**Message Format (NewsMessageSchema):**

```json
{
  "id": "news_12345",
  "url": "https://example.com/news/article",
  "title": "Bitcoin Market Analysis",
  "description": "Comprehensive analysis of Bitcoin market trends...",
  "source": "crypto_news",
  "published_at": "2025-01-19T20:30:00Z",
  "author": "John Doe",
  "tags": ["bitcoin", "cryptocurrency", "market-analysis"],
  "fetched_at": "2025-01-19T20:35:00Z",
  "message_timestamp": "2025-01-19T20:35:00Z",
  "metadata": {
    "search_query": "bitcoin market",
    "source_credibility": 0.9
  }
}
```

### Result Publishing

After processing, the service publishes results to the `sentiment.results` queue with routing key `sentiment.results.processed`.

**Result Message Format (SentimentResultMessageSchema):**

```json
{
  "news_id": "news_12345",
  "url": "https://example.com/news/article",
  "title": "Bitcoin Market Analysis",
  "sentiment_label": "positive",
  "sentiment_scores": {
    "positive": 0.85,
    "negative": 0.08,
    "neutral": 0.07
  },
  "confidence": 0.92,
  "reasoning": "The text discusses Bitcoin reaching new highs and institutional adoption, which are generally positive indicators for cryptocurrency markets.",
  "processed_at": "2025-01-19T20:36:00Z",
  "processing_time_ms": 1250,
  "analyzer_version": "openai-gpt-4o-mini",
  "metadata": {
    "processing_attempts": 1,
    "cache_hit": false
  }
}
```

## Data Models

### SentimentAnalysisResult

```python
class SentimentAnalysisResult(BaseModel):
    label: SentimentLabel  # "positive", "negative", or "neutral"
    scores: SentimentScore
    confidence: float  # 0.0 to 1.0
    reasoning: Optional[str]
```

### SentimentScore

```python
class SentimentScore(BaseModel):
    positive: float  # 0.0 to 1.0
    negative: float  # 0.0 to 1.0
    neutral: float   # 0.0 to 1.0
```

### SentimentLabel

```python
class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
```

## Rate Limiting

The service implements rate limiting to prevent abuse:

- **Default Limit**: 50 requests per minute
- **Configurable**: Via `RATE_LIMIT_REQUESTS_PER_MINUTE` environment variable
- **Headers**: Rate limit information included in response headers

## Caching

Response caching is enabled by default:

- **TTL**: 3600 seconds (1 hour)
- **Configurable**: Via `CACHE_TTL_SECONDS` environment variable
- **Disable**: Set `ENABLE_CACHING=false`

## Error Handling

### OpenAI API Errors

```json
{
  "error": "OpenAI API Error",
  "detail": "Rate limit exceeded. Please try again later.",
  "status_code": 429,
  "timestamp": "2025-01-19T21:00:00Z"
}
```

### Validation Errors

```json
{
  "error": "Validation Error",
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ],
  "status_code": 422,
  "timestamp": "2025-01-19T21:00:00Z"
}
```

### Service Unavailable

```json
{
  "error": "Service Unavailable",
  "detail": "Sentiment analyzer is currently unavailable",
  "status_code": 503,
  "timestamp": "2025-01-19T21:00:00Z"
}
```

## Testing Examples

### cURL Examples

**Health Check:**

```bash
curl -X GET "http://localhost:8002/health" \
  -H "Accept: application/json"
```

**Test Sentiment Analysis:**

```bash
curl -X POST "http://localhost:8002/api/v1/sentiment/test" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "text": "Ethereum 2.0 upgrade shows promising results for scalability"
  }'
```

**Get Metrics:**

```bash
curl -X GET "http://localhost:8002/metrics" \
  -H "Accept: application/json"
```

### Python Examples

```python
import requests
import json

# Test sentiment analysis
def test_sentiment_analysis():
    url = "http://localhost:8002/api/v1/sentiment/test"
    payload = {
        "text": "Bitcoin adoption continues to grow worldwide"
    }

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        print(f"Sentiment: {result['sentiment_label']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Reasoning: {result['reasoning']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Health check
def check_health():
    url = "http://localhost:8002/health"
    response = requests.get(url)
    if response.status_code == 200:
        health = response.json()
        print(f"Service Status: {health['status']}")
        print(f"Components: {health['components']}")
    else:
        print(f"Health check failed: {response.status_code}")
```

### JavaScript Examples

```javascript
// Test sentiment analysis
async function testSentimentAnalysis() {
  const url = "http://localhost:8002/api/v1/sentiment/test";
  const payload = {
    text: "Bitcoin adoption continues to grow worldwide",
  };

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (response.ok) {
      const result = await response.json();
      console.log(`Sentiment: ${result.sentiment_label}`);
      console.log(`Confidence: ${result.confidence}`);
      console.log(`Reasoning: ${result.reasoning}`);
    } else {
      console.error(`Error: ${response.status} - ${response.statusText}`);
    }
  } catch (error) {
    console.error("Request failed:", error);
  }
}

// Health check
async function checkHealth() {
  const url = "http://localhost:8002/health";

  try {
    const response = await fetch(url);
    if (response.ok) {
      const health = await response.json();
      console.log(`Service Status: ${health.status}`);
      console.log(`Components:`, health.components);
    } else {
      console.error(`Health check failed: ${response.status}`);
    }
  } catch (error) {
    console.error("Health check failed:", error);
  }
}
```

## WebSocket Support

Currently, the service does not support WebSocket connections. All communication is handled via HTTP REST endpoints and RabbitMQ message queues.

## API Versioning

The service uses URL path versioning:

- Current version: `v1`
- Future versions: `v2`, `v3`, etc.
- Deprecation policy: 6 months notice for breaking changes

## Monitoring and Observability

### Health Check Endpoints

- `/health` - Overall service health
- `/api/v1/sentiment/health` - Sentiment analysis component health

### Metrics Endpoints

- `/metrics` - Service performance metrics

### Logging

- Structured JSON logging
- Correlation IDs for request tracking
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Security Considerations

### Input Validation

- All inputs are validated using Pydantic models
- Text length limits and content validation
- SQL injection protection through parameterized queries

### Rate Limiting

- Prevents API abuse
- Configurable limits per environment

### Error Handling

- No sensitive information in error messages
- Proper HTTP status codes
- Structured error responses

## Performance Considerations

### Async Processing

- All endpoints are asynchronous
- Non-blocking I/O operations
- Concurrent request handling

### Caching

- Response caching for repeated requests
- Configurable TTL settings
- Memory-efficient cache implementation

### Connection Pooling

- Database connection pooling
- Message broker connection reuse
- Optimized resource utilization
