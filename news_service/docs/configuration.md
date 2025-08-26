# Configuration Guide

## Overview

The FinSight News Service uses environment variables for configuration management. All settings are validated using Pydantic v2 with comprehensive type checking and validation rules.

## Environment Variables

### Service Configuration

| Variable      | Type    | Default                | Required | Description                                         |
| ------------- | ------- | ---------------------- | -------- | --------------------------------------------------- |
| `APP_NAME`    | string  | `news-crawler-service` | No       | Application name for logging and identification     |
| `DEBUG`       | boolean | `false`                | No       | Enable debug mode for development                   |
| `ENVIRONMENT` | string  | `development`          | No       | Environment: `development`, `staging`, `production` |
| `HOST`        | string  | `0.0.0.0`              | No       | Host address to bind the service                    |
| `PORT`        | integer | `8000`                 | No       | Port for the REST API server                        |

### gRPC Configuration

| Variable                          | Type    | Default   | Required | Description                          |
| --------------------------------- | ------- | --------- | -------- | ------------------------------------ |
| `ENABLE_GRPC`                     | boolean | `true`    | No       | Enable gRPC server                   |
| `GRPC_HOST`                       | string  | `0.0.0.0` | No       | Host address for gRPC server         |
| `GRPC_PORT`                       | integer | `50051`   | No       | Port for gRPC server                 |
| `GRPC_MAX_WORKERS`                | integer | `10`      | No       | Maximum number of gRPC workers       |
| `GRPC_MAX_RECEIVE_MESSAGE_LENGTH` | integer | `4194304` | No       | Maximum receive message length (4MB) |
| `GRPC_MAX_SEND_MESSAGE_LENGTH`    | integer | `4194304` | No       | Maximum send message length (4MB)    |

### Eureka Service Discovery

| Variable                                                    | Type    | Default                 | Required | Description                              |
| ----------------------------------------------------------- | ------- | ----------------------- | -------- | ---------------------------------------- |
| `ENABLE_EUREKA_CLIENT`                                      | boolean | `true`                  | No       | Enable Eureka client registration        |
| `EUREKA_SERVER_URL`                                         | string  | `http://localhost:8761` | No       | Eureka server URL                        |
| `EUREKA_APP_NAME`                                           | string  | `news-service`          | No       | Application name for Eureka registration |
| `EUREKA_INSTANCE_ID`                                        | string  | `null`                  | No       | Instance ID for Eureka registration      |
| `EUREKA_HOST_NAME`                                          | string  | `null`                  | No       | Host name for Eureka registration        |
| `EUREKA_IP_ADDRESS`                                         | string  | `null`                  | No       | IP address for Eureka registration       |
| `EUREKA_PORT`                                               | integer | `8000`                  | No       | Port for Eureka registration             |
| `EUREKA_SECURE_PORT`                                        | integer | `8443`                  | No       | Secure port for Eureka                   |
| `EUREKA_SECURE_PORT_ENABLED`                                | boolean | `false`                 | No       | Enable secure port for Eureka            |
| `EUREKA_HOME_PAGE_URL`                                      | string  | `null`                  | No       | Home page URL for Eureka                 |
| `EUREKA_STATUS_PAGE_URL`                                    | string  | `null`                  | No       | Status page URL for Eureka               |
| `EUREKA_HEALTH_CHECK_URL`                                   | string  | `null`                  | No       | Health check URL for Eureka              |
| `EUREKA_VIP_ADDRESS`                                        | string  | `null`                  | No       | VIP address for Eureka                   |
| `EUREKA_SECURE_VIP_ADDRESS`                                 | string  | `null`                  | No       | Secure VIP address for Eureka            |
| `EUREKA_PREFER_IP_ADDRESS`                                  | boolean | `true`                  | No       | Prefer IP address over hostname          |
| `EUREKA_LEASE_RENEWAL_INTERVAL_IN_SECONDS`                  | integer | `30`                    | No       | Lease renewal interval (1-300)           |
| `EUREKA_LEASE_EXPIRATION_DURATION_IN_SECONDS`               | integer | `90`                    | No       | Lease expiration duration (30-900)       |
| `EUREKA_REGISTRY_FETCH_INTERVAL_SECONDS`                    | integer | `30`                    | No       | Registry fetch interval                  |
| `EUREKA_INSTANCE_INFO_REPLICATION_INTERVAL_SECONDS`         | integer | `30`                    | No       | Instance info replication interval       |
| `EUREKA_INITIAL_INSTANCE_INFO_REPLICATION_INTERVAL_SECONDS` | integer | `40`                    | No       | Initial replication interval             |
| `EUREKA_HEARTBEAT_INTERVAL_SECONDS`                         | integer | `30`                    | No       | Heartbeat interval                       |

#### Eureka Retry Configuration

| Variable                                  | Type    | Default | Required | Description                              |
| ----------------------------------------- | ------- | ------- | -------- | ---------------------------------------- |
| `EUREKA_REGISTRATION_RETRY_ATTEMPTS`      | integer | `3`     | No       | Registration retry attempts (1-10)       |
| `EUREKA_REGISTRATION_RETRY_DELAY_SECONDS` | integer | `5`     | No       | Initial registration retry delay         |
| `EUREKA_HEARTBEAT_RETRY_ATTEMPTS`         | integer | `3`     | No       | Heartbeat retry attempts (1-10)          |
| `EUREKA_HEARTBEAT_RETRY_DELAY_SECONDS`    | integer | `2`     | No       | Initial heartbeat retry delay            |
| `EUREKA_RETRY_BACKOFF_MULTIPLIER`         | float   | `2.0`   | No       | Exponential backoff multiplier (1.0-5.0) |
| `EUREKA_MAX_RETRY_DELAY_SECONDS`          | integer | `60`    | No       | Maximum retry delay                      |
| `EUREKA_ENABLE_AUTO_RE_REGISTRATION`      | boolean | `true`  | No       | Enable auto re-registration              |
| `EUREKA_RE_REGISTRATION_DELAY_SECONDS`    | integer | `10`    | No       | Re-registration delay                    |

### Database Configuration

#### Environment Selection

| Variable               | Type   | Default | Required | Description                              |
| ---------------------- | ------ | ------- | -------- | ---------------------------------------- |
| `DATABASE_ENVIRONMENT` | string | `local` | No       | Database environment: `local` or `cloud` |

#### MongoDB Local Configuration

| Variable                 | Type   | Default                     | Required | Description                     |
| ------------------------ | ------ | --------------------------- | -------- | ------------------------------- |
| `MONGODB_LOCAL_URL`      | string | `mongodb://localhost:27017` | No       | Local MongoDB connection string |
| `MONGODB_LOCAL_DATABASE` | string | `finsight_coindesk_news`    | No       | Local database name             |

#### MongoDB Cloud Configuration

| Variable                 | Type   | Default         | Required | Description                                                                    |
| ------------------------ | ------ | --------------- | -------- | ------------------------------------------------------------------------------ |
| `MONGODB_CLOUD_URL`      | string | `""`            | Yes\*    | MongoDB Atlas connection string (\*required when `DATABASE_ENVIRONMENT=cloud`) |
| `MONGODB_CLOUD_DATABASE` | string | `finsight_news` | No       | Cloud database name                                                            |

#### MongoDB Connection Options

| Variable                           | Type    | Default      | Required | Description                   |
| ---------------------------------- | ------- | ------------ | -------- | ----------------------------- |
| `MONGODB_COLLECTION_NEWS`          | string  | `news_items` | No       | News collection name          |
| `MONGODB_CONNECTION_TIMEOUT`       | integer | `10000`      | No       | Connection timeout (ms)       |
| `MONGODB_SERVER_SELECTION_TIMEOUT` | integer | `5000`       | No       | Server selection timeout (ms) |
| `MONGODB_MAX_POOL_SIZE`            | integer | `10`         | No       | Maximum connection pool size  |
| `MONGODB_MIN_POOL_SIZE`            | integer | `1`          | No       | Minimum connection pool size  |

### Redis Cache Configuration

| Variable                       | Type    | Default         | Required | Description                      |
| ------------------------------ | ------- | --------------- | -------- | -------------------------------- |
| `REDIS_HOST`                   | string  | `localhost`     | No       | Redis host address               |
| `REDIS_PORT`                   | integer | `6379`          | No       | Redis port                       |
| `REDIS_DB`                     | integer | `0`             | No       | Redis database number            |
| `REDIS_PASSWORD`               | string  | `null`          | No       | Redis password                   |
| `REDIS_KEY_PREFIX`             | string  | `news-service:` | No       | Redis key prefix                 |
| `REDIS_CONNECTION_TIMEOUT`     | integer | `5`             | No       | Connection timeout (seconds)     |
| `REDIS_SOCKET_TIMEOUT`         | integer | `5`             | No       | Socket timeout (seconds)         |
| `REDIS_SOCKET_CONNECT_TIMEOUT` | integer | `5`             | No       | Socket connect timeout (seconds) |
| `REDIS_SOCKET_KEEPALIVE`       | boolean | `true`          | No       | Enable socket keepalive          |
| `REDIS_RETRY_ON_TIMEOUT`       | boolean | `true`          | No       | Retry on timeout                 |
| `REDIS_MAX_CONNECTIONS`        | integer | `10`            | No       | Maximum connections (1-100)      |

### Cache TTL Configuration

| Variable                     | Type    | Default | Required | Description                                |
| ---------------------------- | ------- | ------- | -------- | ------------------------------------------ |
| `CACHE_TTL_SEARCH_NEWS`      | integer | `1800`  | No       | TTL for search news (30 minutes, 60-7200s) |
| `CACHE_TTL_RECENT_NEWS`      | integer | `900`   | No       | TTL for recent news (15 minutes, 60-3600s) |
| `CACHE_TTL_NEWS_BY_SOURCE`   | integer | `1800`  | No       | TTL for news by source (30 minutes)        |
| `CACHE_TTL_NEWS_BY_KEYWORDS` | integer | `1200`  | No       | TTL for news by keywords (20 minutes)      |
| `CACHE_TTL_NEWS_BY_TAGS`     | integer | `1800`  | No       | TTL for news by tags (30 minutes)          |
| `CACHE_TTL_AVAILABLE_TAGS`   | integer | `3600`  | No       | TTL for available tags (1 hour)            |
| `CACHE_TTL_REPOSITORY_STATS` | integer | `600`   | No       | TTL for repository stats (10 minutes)      |
| `CACHE_TTL_NEWS_ITEM`        | integer | `7200`  | No       | TTL for individual news item (2 hours)     |

### Cache Management

| Variable                           | Type    | Default          | Required | Description                     |
| ---------------------------------- | ------- | ---------------- | -------- | ------------------------------- |
| `ENABLE_CACHING`                   | boolean | `true`           | No       | Enable Redis caching            |
| `CACHE_TTL_SECONDS`                | integer | `300`            | No       | Default cache TTL (5 minutes)   |
| `CACHE_INVALIDATION_ENABLED`       | boolean | `true`           | No       | Enable cache invalidation       |
| `CACHE_INVALIDATION_PATTERN`       | string  | `news-service:*` | No       | Cache invalidation pattern      |
| `CACHE_INVALIDATION_DELAY_SECONDS` | integer | `5`              | No       | Delay before cache invalidation |

### RabbitMQ Configuration

| Variable                                 | Type   | Default                              | Required | Description                        |
| ---------------------------------------- | ------ | ------------------------------------ | -------- | ---------------------------------- |
| `RABBITMQ_URL`                           | string | `amqp://guest:guest@localhost:5672/` | No       | RabbitMQ connection URL            |
| `RABBITMQ_EXCHANGE`                      | string | `news.event`                         | No       | Main exchange name                 |
| `RABBITMQ_QUEUE_NEWS_TO_SENTIMENT`       | string | `news.sentiment_analysis`            | No       | Queue for sentiment analysis       |
| `RABBITMQ_QUEUE_SENTIMENT_RESULTS`       | string | `sentiment.results`                  | No       | Queue for sentiment results        |
| `RABBITMQ_ROUTING_KEY_NEWS_TO_SENTIMENT` | string | `news.sentiment.analyze`             | No       | Routing key for sentiment analysis |
| `RABBITMQ_ROUTING_KEY_SENTIMENT_RESULTS` | string | `sentiment.results.processed`        | No       | Routing key for sentiment results  |

### External API Configuration

| Variable         | Type   | Default | Required | Description                        |
| ---------------- | ------ | ------- | -------- | ---------------------------------- |
| `TAVILY_API_KEY` | string | `null`  | No       | Tavily search API key              |
| `SECRET_API_KEY` | string | `null`  | No       | Secret API key for admin endpoints |

### Crawler Configuration

| Variable                   | Type    | Default | Required | Description                       |
| -------------------------- | ------- | ------- | -------- | --------------------------------- |
| `ENABLE_ADVANCED_CRAWLING` | boolean | `true`  | No       | Enable advanced crawling features |
| `MAX_CONCURRENT_CRAWLS`    | integer | `10`    | No       | Maximum concurrent crawls (1-100) |
| `CRAWL_TIMEOUT`            | integer | `30`    | No       | Crawl timeout (seconds)           |
| `CRAWL_RETRY_ATTEMPTS`     | integer | `3`     | No       | Crawl retry attempts              |

### Job Management Configuration

| Variable                        | Type    | Default                         | Required | Description                   |
| ------------------------------- | ------- | ------------------------------- | -------- | ----------------------------- |
| `CRON_JOB_ENABLED`              | boolean | `true`                          | No       | Enable scheduled jobs         |
| `CRON_JOB_SCHEDULE`             | string  | `0 */6 * * *`                   | No       | Cron schedule (every 6 hours) |
| `CRON_JOB_MAX_ITEMS_PER_SOURCE` | integer | `100`                           | No       | Max items per source per job  |
| `CRON_JOB_SOURCES`              | list    | `["coindesk", "cointelegraph"]` | No       | Sources to crawl              |
| `CRON_JOB_CONFIG_FILE`          | string  | `news_crawler_config.json`      | No       | Job config file path          |
| `CRON_JOB_PID_FILE`             | string  | `news_crawler_job.pid`          | No       | PID file path                 |
| `CRON_JOB_LOG_FILE`             | string  | `logs/news_crawler_job.log`     | No       | Job log file path             |

### Logging Configuration

| Variable                    | Type    | Default | Required | Description                                                |
| --------------------------- | ------- | ------- | -------- | ---------------------------------------------------------- |
| `LOG_LEVEL`                 | string  | `INFO`  | No       | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `LOG_FILE_PATH`             | string  | `logs/` | No       | Log file directory                                         |
| `ENABLE_STRUCTURED_LOGGING` | boolean | `true`  | No       | Enable structured logging                                  |

### Rate Limiting

| Variable                         | Type    | Default | Required | Description                    |
| -------------------------------- | ------- | ------- | -------- | ------------------------------ |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | integer | `100`   | No       | Rate limit requests per minute |

## Configuration Examples

### Development Environment

```bash
# .env file for development
APP_NAME=news-service-dev
ENVIRONMENT=development
DEBUG=true
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_ENVIRONMENT=local
MONGODB_LOCAL_URL=mongodb://localhost:27017
MONGODB_LOCAL_DATABASE=finsight_news_dev

# Cache
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=1

# Message Queue
RABBITMQ_URL=amqp://guest:guest@localhost:5672/

# External APIs
TAVILY_API_KEY=your_tavily_api_key
SECRET_API_KEY=dev_secret_key

# Logging
LOG_LEVEL=DEBUG
ENABLE_STRUCTURED_LOGGING=true

# Job Management
CRON_JOB_ENABLED=false
CRON_JOB_SCHEDULE=0 */2 * * *
CRON_JOB_MAX_ITEMS_PER_SOURCE=50

# Eureka (disabled for development)
ENABLE_EUREKA_CLIENT=false
```

### Production Environment

```bash
# .env file for production
APP_NAME=news-service
ENVIRONMENT=production
DEBUG=false
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_ENVIRONMENT=cloud
MONGODB_CLOUD_URL=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_CLOUD_DATABASE=finsight_news_prod

# Cache
REDIS_HOST=redis-cluster.example.com
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
REDIS_MAX_CONNECTIONS=50

# Message Queue
RABBITMQ_URL=amqp://user:password@rabbitmq-cluster.example.com:5672/

# External APIs
TAVILY_API_KEY=your_production_tavily_key
SECRET_API_KEY=your_production_secret_key

# Logging
LOG_LEVEL=INFO
ENABLE_STRUCTURED_LOGGING=true

# Job Management
CRON_JOB_ENABLED=true
CRON_JOB_SCHEDULE=0 */6 * * *
CRON_JOB_MAX_ITEMS_PER_SOURCE=100

# Eureka
ENABLE_EUREKA_CLIENT=true
EUREKA_SERVER_URL=http://eureka-server.example.com:8761
EUREKA_APP_NAME=news-service
EUREKA_INSTANCE_ID=news-service-${HOSTNAME}

# Cache TTL (longer for production)
CACHE_TTL_SEARCH_NEWS=3600
CACHE_TTL_RECENT_NEWS=1800
CACHE_TTL_AVAILABLE_TAGS=7200

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=200
```

### Docker Compose Environment

```yaml
# docker-compose.yml environment section
environment:
  - APP_NAME=news-service
  - ENVIRONMENT=production
  - HOST=0.0.0.0
  - PORT=8000
  - GRPC_PORT=50051

  # Database
  - DATABASE_ENVIRONMENT=cloud
  - MONGODB_CLOUD_URL=${MONGODB_CLOUD_URL}
  - MONGODB_CLOUD_DATABASE=finsight_news

  # Cache
  - REDIS_HOST=${REDIS_HOST}
  - REDIS_PORT=6379
  - REDIS_PASSWORD=${REDIS_PASSWORD}

  # Message Queue
  - RABBITMQ_URL=${RABBITMQ_URL}

  # External APIs
  - TAVILY_API_KEY=${TAVILY_API_KEY}
  - SECRET_API_KEY=${SECRET_API_KEY}

  # Eureka
  - ENABLE_EUREKA_CLIENT=true
  - EUREKA_SERVER_URL=${EUREKA_SERVER_URL}
  - EUREKA_APP_NAME=news-service

  # Logging
  - LOG_LEVEL=INFO
  - ENABLE_STRUCTURED_LOGGING=true
```

## Configuration Validation

The service validates all configuration values at startup using Pydantic v2 validators:

### Database Environment Validation

- Must be either `local` or `cloud`
- Cloud URL required when environment is `cloud`

### Numeric Range Validation

- Port numbers: 1-65535
- Timeout values: 1-300 seconds
- Connection pool sizes: 1-100
- Rate limits: 1-1000 requests per minute
- Cache TTL: 60-7200 seconds

### Cron Schedule Validation

- Must be valid 5-field cron expression
- Format: `minute hour day month day_of_week`

### Log Level Validation

- Must be one of: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

## Configuration Management

### Environment-Specific Configs

The service supports different configuration files for different environments:

```bash
# Development
cp env.example .env.dev
# Edit .env.dev with development settings

# Production
cp env.example .env.prod
# Edit .env.prod with production settings

# Load specific environment
export ENV_FILE=.env.prod
```

### Configuration Overrides

You can override specific settings using environment variables:

```bash
# Override specific settings
export MONGODB_CLOUD_URL=mongodb+srv://new_user:new_pass@new_cluster.mongodb.net/
export REDIS_HOST=new-redis-host.example.com
export LOG_LEVEL=DEBUG
```

### Configuration Validation

The service validates configuration at startup:

```bash
# Check configuration without starting the service
python -c "from src.core.config import settings; print('Configuration valid')"
```

### Configuration Documentation

Generate configuration documentation:

```bash
# Export configuration schema
python -c "from src.core.config import Settings; print(Settings.model_json_schema())"
```

## Security Considerations

### Sensitive Data

- Store API keys and passwords in environment variables
- Never commit `.env` files to version control
- Use secret management services in production
- Rotate secrets regularly

### Network Security

- Use TLS/SSL for all external connections
- Implement proper firewall rules
- Use VPN for database connections
- Enable authentication for all services

### Access Control

- Use strong API keys for admin endpoints
- Implement proper rate limiting
- Monitor access logs
- Use least privilege principle

## Troubleshooting

### Common Configuration Issues

1. **Database Connection Failed**

   ```bash
   # Check MongoDB connection
   mongosh "your_connection_string"

   # Verify environment variable
   echo $MONGODB_CLOUD_URL
   ```

2. **Redis Connection Failed**

   ```bash
   # Check Redis connection
   redis-cli -h $REDIS_HOST -p $REDIS_PORT ping

   # Verify authentication
   redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD ping
   ```

3. **RabbitMQ Connection Failed**

   ```bash
   # Check RabbitMQ connection
   curl -u guest:guest http://localhost:15672/api/overview

   # Verify connection string
   echo $RABBITMQ_URL
   ```

4. **Eureka Registration Failed**

   ```bash
   # Check Eureka server
   curl http://localhost:8761/eureka/apps

   # Verify configuration
   echo $EUREKA_SERVER_URL
   echo $EUREKA_APP_NAME
   ```

### Configuration Debugging

Enable debug logging to troubleshoot configuration issues:

```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
```

Check the logs for configuration validation errors:

```bash
tail -f logs/news_crawler_main.log | grep -i config
```

## Best Practices

1. **Environment Separation**: Use different configurations for development, staging, and production
2. **Secret Management**: Use proper secret management tools (HashiCorp Vault, AWS Secrets Manager)
3. **Configuration Validation**: Always validate configuration before deployment
4. **Monitoring**: Monitor configuration changes and their impact
5. **Documentation**: Keep configuration documentation up to date
6. **Backup**: Backup configuration files and secrets
7. **Testing**: Test configuration changes in staging environment first
