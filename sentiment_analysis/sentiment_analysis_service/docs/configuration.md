# Configuration Guide - FinSight Sentiment Analysis Service

## Overview

The FinSight Sentiment Analysis Service uses a comprehensive configuration system based on Pydantic Settings, supporting environment variables, `.env` files, and configuration validation. This guide covers all configuration options, their default values, and usage examples.

## Configuration Structure

The service configuration is organized into logical groups:

- **Service Configuration**: Basic service settings
- **OpenAI Configuration**: AI model and API settings
- **Database Configuration**: MongoDB connection settings
- **Message Broker Configuration**: RabbitMQ settings
- **Processing Configuration**: Analysis and performance settings
- **Security Configuration**: Rate limiting and validation
- **Logging Configuration**: Log levels and output settings

## Environment Variables

### Service Configuration

| Variable      | Default                      | Required | Description                     |
| ------------- | ---------------------------- | -------- | ------------------------------- |
| `APP_NAME`    | `sentiment-analysis-service` | No       | Service name for identification |
| `DEBUG`       | `False`                      | No       | Enable debug mode               |
| `ENVIRONMENT` | `development`                | No       | Deployment environment          |

### OpenAI Configuration

| Variable             | Default       | Required | Description                             |
| -------------------- | ------------- | -------- | --------------------------------------- |
| `OPENAI_API_KEY`     | `None`        | **Yes**  | OpenAI API key for authentication       |
| `OPENAI_MODEL`       | `gpt-4o-mini` | No       | OpenAI model to use for analysis        |
| `OPENAI_TEMPERATURE` | `0.0`         | No       | Model temperature (0.0 = deterministic) |
| `OPENAI_MAX_TOKENS`  | `1000`        | No       | Maximum tokens for response             |

### MongoDB Configuration

| Variable                  | Default                     | Required | Description               |
| ------------------------- | --------------------------- | -------- | ------------------------- |
| `MONGODB_URL`             | `mongodb://localhost:27017` | No       | MongoDB connection string |
| `MONGODB_DATABASE`        | `finsight_news`             | No       | Database name             |
| `MONGODB_COLLECTION_NEWS` | `news_items`                | No       | News collection name      |

### RabbitMQ Configuration

| Variable                                 | Default                              | Required | Description                   |
| ---------------------------------------- | ------------------------------------ | -------- | ----------------------------- |
| `RABBITMQ_URL`                           | `amqp://guest:guest@localhost:5672/` | No       | RabbitMQ connection string    |
| `RABBITMQ_EXCHANGE`                      | `news.event`                         | No       | Exchange name for news events |
| `RABBITMQ_QUEUE_NEWS_TO_SENTIMENT`       | `news.sentiment_analysis`            | No       | Queue for incoming news       |
| `RABBITMQ_QUEUE_SENTIMENT_RESULTS`       | `sentiment.results`                  | No       | Queue for sentiment results   |
| `RABBITMQ_ROUTING_KEY_NEWS_TO_SENTIMENT` | `news.sentiment.analyze`             | No       | Routing key for news messages |
| `RABBITMQ_ROUTING_KEY_SENTIMENT_RESULTS` | `sentiment.results.processed`        | No       | Routing key for results       |
| `RABBITMQ_CONNECTION_TIMEOUT`            | `10`                                 | No       | Connection timeout in seconds |
| `RABBITMQ_RETRY_ATTEMPTS`                | `3`                                  | No       | Connection retry attempts     |

### Processing Configuration

| Variable                  | Default              | Required | Description                        |
| ------------------------- | -------------------- | -------- | ---------------------------------- |
| `ENABLE_BATCH_PROCESSING` | `True`               | No       | Enable batch processing mode       |
| `MAX_CONCURRENT_ANALYSIS` | `5`                  | No       | Maximum concurrent analysis tasks  |
| `ANALYSIS_TIMEOUT`        | `30`                 | No       | Analysis timeout in seconds        |
| `ANALYSIS_RETRY_ATTEMPTS` | `3`                  | No       | Retry attempts for failed analysis |
| `ANALYZER_VERSION`        | `openai-gpt-4o-mini` | No       | Analyzer version identifier        |
| `BATCH_SIZE`              | `10`                 | No       | Batch size for processing          |

### Cache Configuration

| Variable            | Default | Required | Description             |
| ------------------- | ------- | -------- | ----------------------- |
| `ENABLE_CACHING`    | `True`  | No       | Enable response caching |
| `CACHE_TTL_SECONDS` | `3600`  | No       | Cache TTL in seconds    |

### Rate Limiting

| Variable                         | Default | Required | Description           |
| -------------------------------- | ------- | -------- | --------------------- |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | `50`    | No       | Rate limit per minute |

### Logging Configuration

| Variable                    | Default | Required | Description                                           |
| --------------------------- | ------- | -------- | ----------------------------------------------------- |
| `LOG_LEVEL`                 | `INFO`  | No       | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `LOG_FILE_PATH`             | `logs/` | No       | Log file directory                                    |
| `ENABLE_STRUCTURED_LOGGING` | `True`  | No       | Enable structured JSON logging                        |

### Message Publishing Configuration

| Variable                         | Default | Required | Description                                |
| -------------------------------- | ------- | -------- | ------------------------------------------ |
| `ENABLE_MESSAGE_PUBLISHING`      | `True`  | No       | Enable result publishing to message broker |
| `ENABLE_ANALYZE_TEXT_PUBLISHING` | `True`  | No       | Enable publishing for analyze_text method  |

## Configuration Files

### .env File

Create a `.env` file in the service root directory:

```bash
# Service Configuration
APP_NAME=sentiment-analysis-service
DEBUG=false
ENVIRONMENT=production

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.0
OPENAI_MAX_TOKENS=1000

# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=finsight_news
MONGODB_COLLECTION_NEWS=news_items

# RabbitMQ Configuration
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
RABBITMQ_EXCHANGE=news.event
RABBITMQ_QUEUE_NEWS_TO_SENTIMENT=news.sentiment_analysis
RABBITMQ_QUEUE_SENTIMENT_RESULTS=sentiment.results
RABBITMQ_ROUTING_KEY_NEWS_TO_SENTIMENT=news.sentiment.analyze
RABBITMQ_ROUTING_KEY_SENTIMENT_RESULTS=sentiment.results.processed
RABBITMQ_CONNECTION_TIMEOUT=10
RABBITMQ_RETRY_ATTEMPTS=3

# Processing Configuration
ENABLE_BATCH_PROCESSING=true
MAX_CONCURRENT_ANALYSIS=5
ANALYSIS_TIMEOUT=30
ANALYSIS_RETRY_ATTEMPTS=3
ANALYZER_VERSION=openai-gpt-4o-mini
BATCH_SIZE=10

# Cache Configuration
ENABLE_CACHING=true
CACHE_TTL_SECONDS=3600

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=50

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/
ENABLE_STRUCTURED_LOGGING=true

# Message Publishing
ENABLE_MESSAGE_PUBLISHING=true
ENABLE_ANALYZE_TEXT_PUBLISHING=true
```

### YAML Configuration

For more complex configurations, you can use YAML files:

```yaml
# config.yaml
service:
  app_name: "sentiment-analysis-service"
  debug: false
  environment: "production"

openai:
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4o-mini"
  temperature: 0.0
  max_tokens: 1000

mongodb:
  url: "mongodb://localhost:27017"
  database: "finsight_news"
  collection_news: "news_items"

rabbitmq:
  url: "amqp://guest:guest@localhost:5672/"
  exchange: "news.event"
  queues:
    news_to_sentiment: "news.sentiment_analysis"
    sentiment_results: "sentiment.results"
  routing_keys:
    news_to_sentiment: "news.sentiment.analyze"
    sentiment_results: "sentiment.results.processed"
  connection_timeout: 10
  retry_attempts: 3

processing:
  enable_batch_processing: true
  max_concurrent_analysis: 5
  analysis_timeout: 30
  analysis_retry_attempts: 3
  analyzer_version: "openai-gpt-4o-mini"
  batch_size: 10

cache:
  enable_caching: true
  ttl_seconds: 3600

rate_limiting:
  requests_per_minute: 50

logging:
  level: "INFO"
  file_path: "logs/"
  enable_structured_logging: true

message_publishing:
  enable_message_publishing: true
  enable_analyze_text_publishing: true
```

## Environment-Specific Configurations

### Development Environment

```bash
# .env.development
DEBUG=true
ENVIRONMENT=development
LOG_LEVEL=DEBUG
ENABLE_CACHING=false
MAX_CONCURRENT_ANALYSIS=2
BATCH_SIZE=5
```

### Staging Environment

```bash
# .env.staging
DEBUG=false
ENVIRONMENT=staging
LOG_LEVEL=INFO
ENABLE_CACHING=true
MAX_CONCURRENT_ANALYSIS=3
BATCH_SIZE=8
```

### Production Environment

```bash
# .env.production
DEBUG=false
ENVIRONMENT=production
LOG_LEVEL=WARNING
ENABLE_CACHING=true
MAX_CONCURRENT_ANALYSIS=10
BATCH_SIZE=20
RATE_LIMIT_REQUESTS_PER_MINUTE=100
```

## Docker Configuration

### Docker Compose Environment Variables

```yaml
# docker-compose.yml
version: "3.8"
services:
  sentiment-analysis-service:
    image: finsight/sentiment-analysis-service:latest
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MONGODB_URL=mongodb://mongodb:27017
      - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
      - DEBUG=false
      - LOG_LEVEL=INFO
      - MAX_CONCURRENT_ANALYSIS=5
      - ENABLE_CACHING=true
    env_file:
      - .env
    depends_on:
      - mongodb
      - rabbitmq
```

### Docker Environment File

```bash
# .env.docker
OPENAI_API_KEY=your_openai_api_key_here
MONGODB_URL=mongodb://mongodb:27017
RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
DEBUG=false
ENVIRONMENT=docker
LOG_LEVEL=INFO
MAX_CONCURRENT_ANALYSIS=5
ENABLE_CACHING=true
```

## Kubernetes Configuration

### ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sentiment-analysis-config
  namespace: sentiment-analysis
data:
  APP_NAME: "sentiment-analysis-service"
  DEBUG: "false"
  ENVIRONMENT: "production"
  OPENAI_MODEL: "gpt-4o-mini"
  OPENAI_TEMPERATURE: "0.0"
  OPENAI_MAX_TOKENS: "1000"
  MONGODB_DATABASE: "finsight_news"
  MONGODB_COLLECTION_NEWS: "news_items"
  RABBITMQ_EXCHANGE: "news.event"
  RABBITMQ_QUEUE_NEWS_TO_SENTIMENT: "news.sentiment_analysis"
  RABBITMQ_QUEUE_SENTIMENT_RESULTS: "sentiment.results"
  RABBITMQ_ROUTING_KEY_NEWS_TO_SENTIMENT: "news.sentiment.analyze"
  RABBITMQ_ROUTING_KEY_SENTIMENT_RESULTS: "sentiment.results.processed"
  ENABLE_BATCH_PROCESSING: "true"
  MAX_CONCURRENT_ANALYSIS: "10"
  ANALYSIS_TIMEOUT: "30"
  ANALYSIS_RETRY_ATTEMPTS: "3"
  ENABLE_CACHING: "true"
  CACHE_TTL_SECONDS: "3600"
  RATE_LIMIT_REQUESTS_PER_MINUTE: "100"
  LOG_LEVEL: "INFO"
  ENABLE_STRUCTURED_LOGGING: "true"
  ENABLE_MESSAGE_PUBLISHING: "true"
  ENABLE_ANALYZE_TEXT_PUBLISHING: "true"
```

### Secret for Sensitive Data

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: sentiment-analysis-secrets
  namespace: sentiment-analysis
type: Opaque
data:
  OPENAI_API_KEY: <base64-encoded-api-key>
  MONGODB_URL: <base64-encoded-mongodb-url>
  RABBITMQ_URL: <base64-encoded-rabbitmq-url>
```

## Configuration Validation

### Pydantic Validation Rules

The service includes comprehensive validation for configuration values:

```python
# Validation examples from the configuration
@field_validator("openai_api_key", mode="before")
@classmethod
def validate_openai_api_key(cls, v):
    if v is None:
        v = os.getenv("OPENAI_API_KEY")
    if not v:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return v

@field_validator("log_level")
@classmethod
def validate_log_level(cls, v):
    levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if v.upper() not in levels:
        raise ValueError(f"log_level must be one of {sorted(levels)}")
    return v.upper()

@field_validator("max_concurrent_analysis")
@classmethod
def validate_max_concurrent_analysis(cls, v):
    if v < 1 or v > 50:
        raise ValueError("max_concurrent_analysis must be between 1 and 50")
    return v
```

### Configuration Validation at Startup

The service validates all configuration values during startup:

```python
# Configuration validation happens automatically
settings = Settings()  # This will raise ValidationError if invalid
```

## Configuration Best Practices

### Security

1. **Never commit sensitive data** to version control
2. **Use environment variables** for API keys and connection strings
3. **Implement secret management** in production (Kubernetes Secrets, HashiCorp Vault)
4. **Rotate credentials** regularly

### Performance

1. **Tune concurrency settings** based on your infrastructure
2. **Configure appropriate timeouts** for external services
3. **Enable caching** in production environments
4. **Monitor resource usage** and adjust limits accordingly

### Monitoring

1. **Set appropriate log levels** for each environment
2. **Enable structured logging** for better observability
3. **Configure health check endpoints** for monitoring
4. **Set up alerting** for configuration-related issues

### Environment Management

1. **Use separate configuration files** for different environments
2. **Implement configuration validation** in CI/CD pipelines
3. **Document configuration changes** and their impact
4. **Test configuration changes** in staging before production

## Troubleshooting Configuration Issues

### Common Issues

1. **Missing Required Variables**

   ```bash
   ValueError: OPENAI_API_KEY environment variable is required
   ```

   **Solution**: Ensure all required environment variables are set

2. **Invalid Configuration Values**

   ```bash
   ValueError: max_concurrent_analysis must be between 1 and 50
   ```

   **Solution**: Check configuration value ranges and constraints

3. **Connection Failures**

   ```bash
   ConnectionError: Failed to connect to MongoDB
   ```

   **Solution**: Verify connection strings and network connectivity

### Debug Configuration

Enable debug mode to see detailed configuration information:

```bash
DEBUG=true LOG_LEVEL=DEBUG python -m src.main
```

### Configuration Validation Script

Create a validation script to check configuration before deployment:

```python
# validate_config.py
from src.core.config import Settings

try:
    settings = Settings()
    print("✅ Configuration validation passed")
    print(f"Service: {settings.app_name}")
    print(f"Environment: {settings.environment}")
    print(f"OpenAI Model: {settings.openai_model}")
    print(f"MongoDB: {settings.mongodb_database}")
    print(f"RabbitMQ: {settings.rabbitmq_exchange}")
except Exception as e:
    print(f"❌ Configuration validation failed: {e}")
    exit(1)
```

## Configuration Migration

### Version Updates

When updating the service version, check for:

1. **New configuration options** that may need to be set
2. **Deprecated configuration** that should be removed
3. **Changed default values** that may affect behavior
4. **New validation rules** that may reject existing configurations

### Backward Compatibility

The service maintains backward compatibility for configuration changes:

1. **New optional fields** have sensible defaults
2. **Deprecated fields** are supported with warnings
3. **Breaking changes** are documented in release notes
4. **Migration guides** are provided for major changes
