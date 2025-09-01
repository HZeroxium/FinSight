# Rate Limiting Documentation

## Overview

The FinSight News Service implements comprehensive rate limiting to protect against API abuse, ensure fair usage across all clients, and maintain service stability. The rate limiting system is built on `slowapi` with Redis backend for distributed, multi-instance deployments.

## Features

### Core Capabilities

- **Distributed Rate Limiting**: Redis-backed storage ensures consistent limits across multiple service instances
- **Flexible Client Identification**: Support for API keys and IP-based identification with configurable priority
- **Proxy Support**: Proper handling of `X-Forwarded-For`, `X-Real-IP`, and `X-Client-IP` headers
- **Per-Route Limits**: Different rate limits for different endpoint types (search, admin, cache)
- **Exempt Endpoints**: Critical endpoints (health, metrics, documentation) bypass rate limiting
- **Standard Headers**: RFC-compliant rate limit headers in responses
- **Graceful Degradation**: Fallback to in-memory storage if Redis is unavailable

### Rate Limit Tiers

| Endpoint Category | Per Minute | Per Hour | Per Day | Description                                 |
| ----------------- | ---------- | -------- | ------- | ------------------------------------------- |
| **Default**       | 100        | 1,000    | 10,000  | Global default for all endpoints            |
| **News Search**   | 60         | 500      | 5,000   | News search and retrieval endpoints         |
| **Admin**         | 30         | 200      | 2,000   | Job management and administrative endpoints |
| **Cache**         | 20         | 100      | 1,000   | Cache management endpoints                  |
| **Exempt**        | ∞          | ∞        | ∞       | Health checks and documentation             |

## Configuration

### Environment Variables

All rate limiting configuration is managed through environment variables:

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

# Response headers configuration
RATE_LIMIT_INCLUDE_HEADERS=true
RATE_LIMIT_RETRY_AFTER_HEADER=true

# Client identification configuration
RATE_LIMIT_BY_API_KEY=true
RATE_LIMIT_BY_IP=true
RATE_LIMIT_TRUST_PROXY=true

# Exempt endpoints (comma-separated)
RATE_LIMIT_EXEMPT_ENDPOINTS=/health,/metrics,/docs,/redoc,/openapi.json

# Per-route rate limits
RATE_LIMIT_NEWS_SEARCH_PER_MINUTE=60
RATE_LIMIT_NEWS_SEARCH_PER_HOUR=500
RATE_LIMIT_ADMIN_PER_MINUTE=30
RATE_LIMIT_ADMIN_PER_HOUR=200
RATE_LIMIT_CACHE_PER_MINUTE=20
RATE_LIMIT_CACHE_PER_HOUR=100
```

### Configuration Validation

The system validates all rate limiting configuration at startup:

- Rate limit values must be positive integers
- Redis connection is validated (with fallback to memory)
- Exempt endpoints are parsed and validated
- Client identification settings are verified

## Client Identification

### Identification Methods

The service identifies clients using the following priority order:

1. **API Key from Authorization Header**

   ```text
   Authorization: Bearer your-api-key-here
   ```

2. **API Key from X-API-Key Header**

   ```text
   X-API-Key: your-api-key-here
   ```

3. **API Key from Query Parameter**

   ```text
   GET /news/?api_key=your-api-key-here
   ```

4. **IP Address** (with proxy header support)
   - Direct client IP
   - `X-Forwarded-For` header (first IP)
   - `X-Real-IP` header
   - `X-Client-IP` header

### Client Identifier Format

- **API Key**: `api_key:your-api-key-here`
- **IP Address**: `ip:192.168.1.100`
- **Unknown**: `ip:unknown`

### Proxy Configuration

When behind a proxy or load balancer:

```bash
# Trust proxy headers
RATE_LIMIT_TRUST_PROXY=true

# Example proxy setup
X-Forwarded-For: 203.0.113.1, 198.51.100.1
X-Real-IP: 203.0.113.1
X-Client-IP: 203.0.113.1
```

## Response Headers

When rate limiting is enabled, responses include standard rate limit headers:

```text
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
Retry-After: 60
```

### Header Descriptions

- **X-RateLimit-Limit**: Maximum requests allowed in the current time window
- **X-RateLimit-Remaining**: Number of requests remaining in the current time window
- **X-RateLimit-Reset**: Unix timestamp when the rate limit resets
- **Retry-After**: Number of seconds to wait before retrying (only when limit exceeded)

## Rate Limit Exceeded Response

When rate limits are exceeded, the service returns a 429 (Too Many Requests) response:

```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60,
  "limit": 100,
  "remaining": 0,
  "reset_time": "2024-01-01T12:01:00Z"
}
```

## Endpoint-Specific Limits

### News Search Endpoints

```python
# Applied to all news search endpoints
@limiter.limit("60/minute; 500/hour")
```

**Endpoints:**

- `GET /news/` - Search news
- `GET /news/recent` - Get recent news
- `POST /news/search/time-range` - Search by time range
- `GET /news/keywords/{keywords}` - Search by keywords
- `GET /news/by-tag/{tags}` - Get news by tags
- `GET /news/tags/available` - Get available tags

### Admin Endpoints

```python
# Applied to all admin endpoints
@limiter.limit("30/minute; 200/hour")
```

**Endpoints:**

- `GET /admin/jobs/status` - Get job status
- `POST /admin/jobs/start` - Start job
- `POST /admin/jobs/stop` - Stop job
- `POST /admin/jobs/run` - Run manual job
- `GET /admin/jobs/config` - Get job config
- `PUT /admin/jobs/config` - Update job config
- `GET /admin/jobs/stats` - Get job stats
- `GET /admin/jobs/health` - Job health check
- `GET /admin/jobs/` - Job service info
- `GET /admin/eureka/status` - Get Eureka status
- `POST /admin/eureka/register` - Register with Eureka
- `POST /admin/eureka/deregister` - Deregister from Eureka
- `POST /admin/eureka/re-register` - Re-register with Eureka
- `GET /admin/eureka/config` - Get Eureka config

### Cache Management Endpoints

```python
# Applied to all cache management endpoints
@limiter.limit("20/minute; 100/hour")
```

**Endpoints:**

- `GET /news/cache/stats` - Cache statistics
- `GET /news/cache/health` - Cache health check
- `POST /news/cache/invalidate` - Invalidate cache

### Exempt Endpoints

The following endpoints are exempt from rate limiting:

- `GET /health` - Service health check
- `GET /metrics` - Service metrics
- `GET /docs` - API documentation
- `GET /redoc` - Alternative API documentation
- `GET /openapi.json` - OpenAPI specification

## Usage Examples

### Basic API Usage

```bash
# Search news (60/minute limit)
curl "http://localhost:8000/news/?limit=10&source=coindesk"

# Check rate limit headers
curl -I "http://localhost:8000/news/" | grep -i "x-ratelimit"
```

### Authenticated API Usage

```bash
# Use API key for higher limits
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "http://localhost:8000/news/"

# Alternative API key header
curl -H "X-API-Key: YOUR_API_KEY" \
     "http://localhost:8000/news/"

# API key as query parameter
curl "http://localhost:8000/news/?api_key=YOUR_API_KEY"
```

### Admin Operations

```bash
# Check job status (30/minute limit)
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "http://localhost:8000/admin/jobs/status"

# Start a job
curl -X POST \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"job_type": "news_collection"}' \
     "http://localhost:8000/admin/jobs/start"
```

### Cache Management

```bash
# Check cache stats (20/minute limit)
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "http://localhost:8000/news/cache/stats"

# Invalidate cache
curl -X POST \
     -H "Authorization: Bearer YOUR_API_KEY" \
     "http://localhost:8000/news/cache/invalidate"
```

### Health Checks (No Rate Limiting)

```bash
# Health check (exempt from rate limiting)
curl "http://localhost:8000/health"

# Service metrics (exempt from rate limiting)
curl "http://localhost:8000/metrics"
```

## Testing Rate Limits

### Manual Testing

```bash
# Test rate limiting with a loop
for i in {1..110}; do
  response=$(curl -s -w "%{http_code}" "http://localhost:8000/news/")
  echo "Request $i: HTTP $response"

  if [[ $response == *"429"* ]]; then
    echo "Rate limit exceeded at request $i"
    break
  fi
done
```

### Automated Testing

```bash
# Run rate limiting unit tests
pytest tests/test_rate_limiting.py -v

# Test specific scenarios
pytest tests/test_rate_limiting.py::TestRateLimitUtils -v
pytest tests/test_rate_limiting.py::TestCreateLimiter -v
```

### Load Testing

```bash
# Use Apache Bench for load testing
ab -n 1000 -c 10 "http://localhost:8000/news/"

# Use wrk for more sophisticated load testing
wrk -t12 -c400 -d30s "http://localhost:8000/news/"
```

## Monitoring and Debugging

### Health Check Integration

The health endpoint includes rate limiting status:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "service": "news-service",
  "rate_limiting": {
    "enabled": true,
    "storage": "redis",
    "default_limits": "100/minute; 1000/hour; 10000/day",
    "exempt_endpoints": ["/health", "/metrics", "/docs"],
    "client_identification": {
      "by_api_key": true,
      "by_ip": true,
      "trust_proxy": true
    }
  }
}
```

### Logging

Rate limiting events are logged with appropriate levels:

```python
# Rate limit exceeded
logger.warning(f"Rate limit exceeded for client {client_id}")

# Redis connection issues
logger.error(f"Redis connection failed, falling back to memory storage")

# Configuration issues
logger.error(f"Invalid rate limit configuration: {error}")
```

### Metrics

Monitor rate limiting metrics:

- Rate limit violations per client
- Redis connection status
- Response times with rate limiting overhead
- Client identification method distribution

## Troubleshooting

### Common Issues

#### 1. Rate Limits Too Strict

**Symptoms:** Legitimate requests being rate limited

**Solutions:**

- Increase rate limit values in configuration
- Use API keys for authenticated requests
- Implement client-side caching
- Contact service administrators for limit adjustments

#### 2. Redis Connection Issues

**Symptoms:** Rate limiting not working or service startup failures

**Solutions:**

- Check Redis connection string
- Verify Redis server is running
- Check network connectivity
- Review Redis logs for errors

#### 3. Proxy Header Issues

**Symptoms:** All clients appearing to come from the same IP

**Solutions:**

- Verify `RATE_LIMIT_TRUST_PROXY=true`
- Check proxy configuration
- Ensure proper `X-Forwarded-For` headers
- Test with direct client connections

#### 4. API Key Not Recognized

**Symptoms:** Requests still rate limited by IP despite API key

**Solutions:**

- Verify API key format and placement
- Check header names and values
- Ensure API key is valid
- Review client identification logic

### Debug Mode

Enable debug logging for rate limiting:

```bash
# Set log level to DEBUG
LOG_LEVEL=DEBUG

# Check rate limiting logs
tail -f logs/news_service.log | grep -i "rate.limit"
```

### Configuration Validation

Validate rate limiting configuration:

```python
from src.core.config import Settings

settings = Settings()
print(settings.rate_limit_config)
```

## Best Practices

### For API Consumers

1. **Use API Keys**: Always use API keys for production applications
2. **Monitor Headers**: Track rate limit headers to monitor usage
3. **Implement Retry Logic**: Handle 429 responses with exponential backoff
4. **Cache Responses**: Reduce API calls by caching responses
5. **Batch Requests**: Combine multiple requests when possible

### For Service Administrators

1. **Monitor Usage**: Track rate limiting metrics and adjust limits
2. **Set Appropriate Limits**: Configure limits based on service capacity
3. **Use Redis Cluster**: Implement Redis cluster for high availability
4. **Document Limits**: Provide clear documentation of rate limits
5. **Test Thoroughly**: Test rate limiting in staging environments

### For Developers

1. **Add Rate Limiting**: Apply rate limiting to new endpoints
2. **Test Edge Cases**: Test rate limiting behavior thoroughly
3. **Handle Exceptions**: Implement proper error handling for rate limit exceeded
4. **Monitor Performance**: Track rate limiting overhead
5. **Update Documentation**: Keep rate limiting documentation current

## Security Considerations

### Rate Limiting Security

- **API Key Protection**: Rate limits are enforced per API key
- **IP-based Fallback**: Unauthenticated requests are limited by IP
- **Proxy Handling**: Proper handling of proxy headers
- **Exempt Endpoints**: Critical endpoints bypass rate limiting
- **Graceful Degradation**: Service continues without rate limiting if Redis fails

### Security Best Practices

1. **Use HTTPS**: Always use HTTPS in production
2. **Rotate API Keys**: Regularly rotate API keys
3. **Monitor Abuse**: Track unusual rate limiting patterns
4. **Implement Alerting**: Set up alerts for rate limit violations
5. **Review Logs**: Regularly review rate limiting logs

## Performance Considerations

### Rate Limiting Overhead

- **Redis Operations**: Each request requires Redis operations
- **Header Processing**: Additional header processing for client identification
- **Memory Usage**: In-memory fallback uses additional memory
- **Network Latency**: Redis network calls add latency

### Optimization Strategies

1. **Connection Pooling**: Use Redis connection pooling
2. **Caching**: Cache rate limit results when appropriate
3. **Batch Operations**: Batch Redis operations when possible
4. **Monitoring**: Monitor rate limiting performance impact
5. **Tuning**: Adjust Redis configuration for optimal performance

## Migration and Upgrades

### Upgrading Rate Limiting

1. **Backup Configuration**: Backup current rate limiting configuration
2. **Test in Staging**: Test new configuration in staging environment
3. **Gradual Rollout**: Roll out changes gradually
4. **Monitor Impact**: Monitor impact on service performance
5. **Update Documentation**: Update documentation and examples

### Configuration Migration

```bash
# Export current configuration
python -c "from src.core.config import Settings; import json; print(json.dumps(Settings().rate_limit_config, indent=2))"

# Import new configuration
# Update environment variables based on new requirements
```

## Support and Maintenance

### Getting Help

1. **Check Documentation**: Review this documentation thoroughly
2. **Test Configuration**: Validate your configuration
3. **Check Logs**: Review service logs for errors
4. **Monitor Metrics**: Check rate limiting metrics
5. **Contact Support**: Reach out to the development team

### Maintenance Tasks

1. **Regular Monitoring**: Monitor rate limiting metrics
2. **Configuration Review**: Regularly review rate limiting configuration
3. **Performance Tuning**: Optimize Redis configuration
4. **Security Updates**: Keep rate limiting components updated
5. **Documentation Updates**: Keep documentation current
