# FinSight News Service

A high-performance, scalable news aggregation and processing service designed for financial markets. Built with FastAPI, MongoDB, and RabbitMQ, this service provides real-time news collection, storage, and integration with sentiment analysis pipelines.

## 🚀 Features

### Core Functionality

- **Multi-Source News Collection**: RSS and API-based collection from CoinDesk, CoinTelegraph, and other financial news sources
- **Intelligent Caching**: Redis-based caching with configurable TTL for optimal performance
- **Real-time Processing**: Asynchronous news processing with RabbitMQ message queues
- **Dual API Support**: REST API and gRPC endpoints for maximum flexibility
- **Service Discovery**: Eureka client integration for microservices architecture

### Advanced Capabilities

- **Sentiment Analysis Integration**: Automatic publishing to sentiment analysis service
- **Job Management**: Scheduled news collection with configurable cron jobs
- **Database Migration**: Seamless migration between local and cloud environments
- **Health Monitoring**: Comprehensive health checks and metrics
- **Rate Limiting**: Production-ready rate limiting with Redis backend for distributed, multi-instance protection

### Data Management

- **Duplicate Detection**: Smart duplicate detection using URL and GUID hashing
- **Flexible Search**: Advanced search with keywords, tags, date ranges, and source filtering
- **Pagination Support**: Efficient pagination for large result sets
- **Data Validation**: Comprehensive input validation using Pydantic models

## 🏗️ Architecture Overview

```mermaid
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   News Sources  │    │   External APIs │    │   RSS Feeds     │
│  (CoinDesk,     │    │  (Tavily, etc.) │    │                 │
│   CoinTelegraph)│    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    News Collectors        │
                    │  (RSS, API, GraphQL)      │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    News Service           │
                    │  (Business Logic)         │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌─────────▼─────────┐
│   MongoDB         │  │   Redis Cache     │  │   RabbitMQ        │
│  (News Storage)   │  │  (Performance)    │  │  (Message Queue)  │
└───────────────────┘  └───────────────────┘  └───────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    API Layer              │
                    │  (REST + gRPC)            │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    Rate Limiting          │
                    │  (Redis-backed)           │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    Sentiment Analysis     │
                    │    Service Integration    │
                    └───────────────────────────┘
```

## 🛠️ Technology Stack

- **Framework**: FastAPI (Python 3.12+)
- **Database**: MongoDB (Motor async driver)
- **Cache**: Redis
- **Message Queue**: RabbitMQ (aio-pika)
- **API**: REST + gRPC
- **Service Discovery**: Eureka Client
- **Rate Limiting**: slowapi with Redis backend
- **Containerization**: Docker & Docker Compose
- **Logging**: Structured logging with correlation IDs
- **Validation**: Pydantic v2 with comprehensive schemas

## 📋 Prerequisites

- Python 3.12+
- MongoDB 5.0+
- Redis 6.0+
- RabbitMQ 3.8+
- Docker & Docker Compose (for containerized deployment)

## 🚀 Quick Start

### Local Development

1. **Clone and Setup**

   ```bash
   cd news_service
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment Configuration**

   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

3. **Start Dependencies** (using Docker Compose)

   ```bash
   docker-compose up -d mongodb redis rabbitmq
   ```

4. **Run the Service**

   ```bash
   python -m src.main
   ```

### Docker Deployment

1. **Build and Run**

   ```bash
   docker-compose up --build
   ```

2. **Production Deployment**

   ```bash
   docker-compose -f docker-compose.yml up -d
   ```

## 🛡️ Rate Limiting

The service implements comprehensive rate limiting to protect against API abuse and ensure fair usage across all clients.

### Features

- **Distributed Rate Limiting**: Redis-backed storage for multi-instance deployments
- **Flexible Client Identification**: Support for API keys and IP-based identification
- **Proxy Support**: Proper handling of `X-Forwarded-For` headers
- **Per-Route Limits**: Different limits for different endpoint types
- **Exempt Endpoints**: Health checks and documentation endpoints bypass rate limiting
- **Response Headers**: Standard rate limit headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `Retry-After`)

### Configuration

```bash
# Enable/disable rate limiting
RATE_LIMIT_ENABLED=true

# Default rate limits (applied globally)
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_REQUESTS_PER_HOUR=1000
RATE_LIMIT_REQUESTS_PER_DAY=10000

# Redis backend configuration
RATE_LIMIT_STORAGE_URL=redis://localhost:6379/1
RATE_LIMIT_KEY_PREFIX=rate-limit:

# Client identification
RATE_LIMIT_BY_API_KEY=true
RATE_LIMIT_BY_IP=true
RATE_LIMIT_TRUST_PROXY=true

# Exempt endpoints
RATE_LIMIT_EXEMPT_ENDPOINTS=/health,/metrics,/docs,/redoc,/openapi.json

# Per-route limits
RATE_LIMIT_NEWS_SEARCH_PER_MINUTE=60
RATE_LIMIT_NEWS_SEARCH_PER_HOUR=500
RATE_LIMIT_ADMIN_PER_MINUTE=30
RATE_LIMIT_ADMIN_PER_HOUR=200
RATE_LIMIT_CACHE_PER_MINUTE=20
RATE_LIMIT_CACHE_PER_HOUR=100
```

### Rate Limit Tiers

| Endpoint Type   | Per Minute | Per Hour | Description                                 |
| --------------- | ---------- | -------- | ------------------------------------------- |
| **Default**     | 100        | 1,000    | Global default for all endpoints            |
| **News Search** | 60         | 500      | News search and retrieval endpoints         |
| **Admin**       | 30         | 200      | Job management and administrative endpoints |
| **Cache**       | 20         | 100      | Cache management endpoints                  |
| **Exempt**      | ∞          | ∞        | Health checks and documentation             |

### Client Identification

The service identifies clients using the following priority order:

1. **API Key** (from `Authorization: Bearer <key>` header)
2. **API Key** (from `X-API-Key` header)
3. **API Key** (from `api_key` query parameter)
4. **IP Address** (respecting proxy headers)

### Response Headers

When rate limiting is enabled, responses include:

```bash
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
Retry-After: 60
```

### Testing Rate Limits

```bash
# Test with API key
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "http://localhost:8000/news/"

# Test with IP-based identification
curl "http://localhost:8000/news/"

# Test exempt endpoint (no rate limiting)
curl "http://localhost:8000/health"

# Check rate limit headers
curl -I "http://localhost:8000/news/" | grep -i "x-ratelimit"
```

### Rate Limit Exceeded Response

When rate limits are exceeded, the service returns:

```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60,
  "limit": 100,
  "remaining": 0
}
```

## 📚 Documentation

- **[API Documentation](docs/api.md)** - Complete API reference with examples
- **[Configuration Guide](docs/configuration.md)** - Environment variables and settings
- **[Architecture Guide](docs/architecture.md)** - Detailed system architecture
- **[Deployment Guide](docs/deployment.md)** - Production deployment instructions

## 🧪 Testing

### API Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test news search
curl "http://localhost:8000/news/?limit=10&source=coindesk"

# Test with authentication
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "http://localhost:8000/admin/jobs/status"

# Test rate limiting
for i in {1..110}; do
  curl -s "http://localhost:8000/news/" > /dev/null
  echo "Request $i"
done
```

### Rate Limiting Tests

```bash
# Run rate limiting unit tests
pytest tests/test_rate_limiting.py -v

# Test specific rate limiting scenarios
pytest tests/test_rate_limiting.py::TestRateLimitUtils::test_get_client_identifier_with_api_key -v
```

### gRPC Testing

```bash
# Use the provided gRPC test client
python tests/grpc_test_client.py
```

## 📊 Monitoring

### Health Checks

- **Service Health**: `GET /health`
- **Cache Health**: `GET /news/cache/health`
- **Job Health**: `GET /admin/jobs/health`

### Metrics

- **Service Metrics**: `GET /metrics`
- **Cache Statistics**: `GET /news/cache/stats`
- **Job Statistics**: `GET /admin/jobs/stats`

### Rate Limiting Metrics

The service provides rate limiting status in the health endpoint:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "service": "news-service",
  "rate_limiting": {
    "enabled": true,
    "storage": "redis",
    "default_limits": "100/minute; 1000/hour; 10000/day",
    "exempt_endpoints": ["/health", "/metrics", "/docs"]
  }
}
```

## 🔧 Configuration

Key configuration areas:

- **Database**: MongoDB connection (local/cloud)
- **Cache**: Redis settings and TTL configuration
- **Message Queue**: RabbitMQ connection and routing
- **API Keys**: Tavily, admin access tokens
- **Service Discovery**: Eureka client settings
- **Rate Limiting**: Comprehensive rate limiting configuration

See [Configuration Guide](docs/configuration.md) for detailed settings.

## 🤝 Contributing

1. Follow the established code patterns and architecture
2. Use Pydantic models for data validation
3. Implement comprehensive error handling
4. Add appropriate logging and monitoring
5. Update documentation for new features
6. Include rate limiting considerations for new endpoints

## 📄 License

This project is part of the FinSight platform. See the main project license for details.

## 🆘 Support

For issues and questions:

1. Check the [documentation](docs/)
2. Review existing issues
3. Create a new issue with detailed information

---

## 🔒 Security Considerations

### Rate Limiting Security

- **API Key Protection**: Rate limits are enforced per API key for authenticated requests
- **IP-based Fallback**: Unauthenticated requests are limited by IP address
- **Proxy Handling**: Proper handling of `X-Forwarded-For` headers for proxy environments
- **Exempt Endpoints**: Critical endpoints (health, metrics) are exempt from rate limiting
- **Graceful Degradation**: Service continues to function even if Redis is unavailable

### Best Practices

1. **Use API Keys**: Always use API keys for production applications
2. **Monitor Usage**: Track rate limit headers to monitor API usage
3. **Implement Retry Logic**: Handle rate limit exceeded responses with exponential backoff
4. **Cache Responses**: Reduce API calls by caching responses when appropriate
5. **Use Appropriate Limits**: Configure rate limits based on your application needs

### Production Deployment

For production deployments:

1. **Redis Cluster**: Use Redis cluster for high availability
2. **Monitoring**: Monitor rate limiting metrics and adjust limits as needed
3. **Alerting**: Set up alerts for rate limit violations
4. **Documentation**: Document rate limits for API consumers
5. **Testing**: Test rate limiting behavior in staging environments
