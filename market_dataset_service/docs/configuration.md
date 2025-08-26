# Configuration Guide

## Overview

The FinSight Market Dataset Service uses environment variables for configuration management. This guide covers all available configuration options, their default values, and usage examples.

## Configuration Categories

### 1. Service Configuration

| Variable      | Type    | Default                  | Required | Description                                  |
| ------------- | ------- | ------------------------ | -------- | -------------------------------------------- |
| `APP_NAME`    | string  | `market-dataset-service` | No       | Application name                             |
| `DEBUG`       | boolean | `false`                  | No       | Enable debug mode                            |
| `ENVIRONMENT` | string  | `development`            | No       | Environment (development/staging/production) |
| `HOST`        | string  | `0.0.0.0`                | No       | Host address to bind to                      |
| `PORT`        | integer | `8000`                   | No       | Port to listen on                            |

### 2. Storage Configuration

| Variable                 | Type   | Default                         | Required | Description                                            |
| ------------------------ | ------ | ------------------------------- | -------- | ------------------------------------------------------ |
| `STORAGE_BASE_DIRECTORY` | string | `data/market_data`              | No       | Base directory for local storage                       |
| `STORAGE_PREFIX`         | string | `finsight/market_data/datasets` | No       | Object storage prefix                                  |
| `STORAGE_SEPARATOR`      | string | `/`                             | No       | Path separator for storage                             |
| `REPOSITORY_TYPE`        | string | `csv`                           | No       | Default repository type (csv/mongodb/influxdb/parquet) |

### 3. MongoDB Configuration

| Variable           | Type   | Default                      | Required | Description               |
| ------------------ | ------ | ---------------------------- | -------- | ------------------------- |
| `MONGODB_URL`      | string | `mongodb://localhost:27017/` | No       | MongoDB connection string |
| `MONGODB_DATABASE` | string | `finsight_market_data`       | No       | MongoDB database name     |

### 4. Object Storage Configuration

#### Storage Provider Selection

| Variable           | Type   | Default | Required | Description                               |
| ------------------ | ------ | ------- | -------- | ----------------------------------------- |
| `STORAGE_PROVIDER` | string | `minio` | No       | Storage provider (minio/digitalocean/aws) |

#### S3-Compatible Storage (MinIO)

| Variable                  | Type    | Default                 | Required | Description              |
| ------------------------- | ------- | ----------------------- | -------- | ------------------------ |
| `S3_ENDPOINT_URL`         | string  | `http://localhost:9000` | No       | S3 endpoint URL          |
| `S3_ACCESS_KEY`           | string  | `minioadmin`            | No       | S3 access key            |
| `S3_SECRET_KEY`           | string  | `minioadmin`            | No       | S3 secret key            |
| `S3_REGION_NAME`          | string  | `us-east-1`             | No       | S3 region name           |
| `S3_BUCKET_NAME`          | string  | `market-data`           | No       | S3 bucket name           |
| `S3_USE_SSL`              | boolean | `false`                 | No       | Use SSL for S3           |
| `S3_VERIFY_SSL`           | boolean | `true`                  | No       | Verify SSL certificates  |
| `S3_SIGNATURE_VERSION`    | string  | `s3v4`                  | No       | S3 signature version     |
| `S3_MAX_POOL_CONNECTIONS` | integer | `50`                    | No       | Max connection pool size |

#### DigitalOcean Spaces

| Variable              | Type   | Default                               | Required | Description         |
| --------------------- | ------ | ------------------------------------- | -------- | ------------------- |
| `SPACES_ENDPOINT_URL` | string | `https://nyc3.digitaloceanspaces.com` | No       | Spaces endpoint URL |
| `SPACES_ACCESS_KEY`   | string | ``                                    | Yes\*    | Spaces access key   |
| `SPACES_SECRET_KEY`   | string | ``                                    | Yes\*    | Spaces secret key   |
| `SPACES_REGION_NAME`  | string | `nyc3`                                | No       | Spaces region name  |
| `SPACES_BUCKET_NAME`  | string | `finsight-market-data`                | No       | Spaces bucket name  |

#### AWS S3

| Variable                | Type   | Default                | Required | Description           |
| ----------------------- | ------ | ---------------------- | -------- | --------------------- |
| `AWS_ACCESS_KEY_ID`     | string | ``                     | Yes\*    | AWS access key ID     |
| `AWS_SECRET_ACCESS_KEY` | string | ``                     | Yes\*    | AWS secret access key |
| `AWS_REGION_NAME`       | string | `us-east-1`            | No       | AWS region name       |
| `AWS_BUCKET_NAME`       | string | `finsight-market-data` | No       | AWS bucket name       |

### 5. Data Collection Configuration

| Variable              | Type    | Default                   | Required | Description                           |
| --------------------- | ------- | ------------------------- | -------- | ------------------------------------- |
| `DEFAULT_SYMBOLS`     | string  | `BTCUSDT,ETHUSDT,BNBUSDT` | No       | Comma-separated default symbols       |
| `DEFAULT_TIMEFRAMES`  | string  | `1h,4h,1d`                | No       | Comma-separated default timeframes    |
| `MAX_OHLCV_LIMIT`     | integer | `1000`                    | No       | Maximum OHLCV records per request     |
| `MAX_TRADES_LIMIT`    | integer | `1000`                    | No       | Maximum trades records per request    |
| `MAX_ORDERBOOK_LIMIT` | integer | `100`                     | No       | Maximum orderbook records per request |

### 6. Exchange Configuration

#### Binance API

| Variable                      | Type    | Default  | Required | Description                |
| ----------------------------- | ------- | -------- | -------- | -------------------------- |
| `BINANCE_API_KEY`             | string  | ``       | No       | Binance API key            |
| `BINANCE_SECRET_KEY`          | string  | ``       | No       | Binance secret key         |
| `BINANCE_REQUESTS_PER_MINUTE` | integer | `1200`   | No       | Rate limit for Binance API |
| `BINANCE_ORDERS_PER_SECOND`   | integer | `10`     | No       | Orders per second limit    |
| `BINANCE_ORDERS_PER_DAY`      | integer | `200000` | No       | Orders per day limit       |

### 7. Cross-Repository Configuration

| Variable                     | Type    | Default        | Required | Description                       |
| ---------------------------- | ------- | -------------- | -------- | --------------------------------- |
| `SOURCE_REPOSITORY_TYPE`     | string  | `csv`          | No       | Source repository type            |
| `SOURCE_TIMEFRAME`           | string  | `1h`           | No       | Source timeframe                  |
| `TARGET_REPOSITORY_TYPE`     | string  | `csv`          | No       | Target repository type            |
| `TARGET_TIMEFRAMES`          | string  | `2h,4h,12h,1d` | No       | Comma-separated target timeframes |
| `ENABLE_PARALLEL_CONVERSION` | boolean | `true`         | No       | Enable parallel conversion        |
| `MAX_CONCURRENT_CONVERSIONS` | integer | `3`            | No       | Max concurrent conversions        |
| `CONVERSION_BATCH_SIZE`      | integer | `1000`         | No       | Batch size for conversions        |

### 8. Logging Configuration

| Variable                    | Type    | Default | Required | Description                                       |
| --------------------------- | ------- | ------- | -------- | ------------------------------------------------- |
| `LOG_LEVEL`                 | string  | `INFO`  | No       | Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL) |
| `LOG_FILE_PATH`             | string  | `logs/` | No       | Log file directory                                |
| `ENABLE_STRUCTURED_LOGGING` | boolean | `true`  | No       | Enable structured JSON logging                    |

### 9. Cache Configuration

| Variable            | Type    | Default | Required | Description           |
| ------------------- | ------- | ------- | -------- | --------------------- |
| `ENABLE_CACHING`    | boolean | `true`  | No       | Enable caching        |
| `CACHE_TTL_SECONDS` | integer | `300`   | No       | Cache TTL in seconds  |
| `CACHE_MAX_SIZE`    | integer | `1000`  | No       | Maximum cache entries |

### 10. Admin API Configuration

| Variable  | Type   | Default                                  | Required | Description   |
| --------- | ------ | ---------------------------------------- | -------- | ------------- |
| `API_KEY` | string | `admin-default-key-change-in-production` | No       | Admin API key |

### 11. Cron Job Configuration

| Variable                       | Type    | Default                    | Required | Description                   |
| ------------------------------ | ------- | -------------------------- | -------- | ----------------------------- |
| `CRON_JOB_ENABLED`             | boolean | `false`                    | No       | Enable cron job               |
| `CRON_JOB_SCHEDULE`            | string  | `0 */4 * * *`              | No       | Cron schedule (every 4 hours) |
| `CRON_JOB_MAX_SYMBOLS_PER_RUN` | integer | `10`                       | No       | Max symbols per job run       |
| `CRON_JOB_LOG_FILE`            | string  | `logs/market_data_job.log` | No       | Job log file path             |
| `CRON_JOB_PID_FILE`            | string  | `market_data_job.pid`      | No       | Job PID file path             |

### 12. Demo Configuration

| Variable           | Type    | Default | Required | Description             |
| ------------------ | ------- | ------- | -------- | ----------------------- |
| `DEMO_MAX_SYMBOLS` | integer | `5`     | No       | Max symbols for demo    |
| `DEMO_DAYS_BACK`   | integer | `7`     | No       | Days back for demo data |

### 13. Eureka Service Discovery Configuration

| Variable                                                    | Type    | Default                  | Required | Description                        |
| ----------------------------------------------------------- | ------- | ------------------------ | -------- | ---------------------------------- |
| `ENABLE_EUREKA_CLIENT`                                      | boolean | `true`                   | No       | Enable Eureka client               |
| `EUREKA_SERVER_URL`                                         | string  | `http://localhost:8761`  | No       | Eureka server URL                  |
| `EUREKA_APP_NAME`                                           | string  | `market-dataset-service` | No       | Application name for Eureka        |
| `EUREKA_INSTANCE_ID`                                        | string  | `None`                   | No       | Instance ID for Eureka             |
| `EUREKA_HOST_NAME`                                          | string  | `None`                   | No       | Host name for Eureka               |
| `EUREKA_IP_ADDRESS`                                         | string  | `None`                   | No       | IP address for Eureka              |
| `EUREKA_PORT`                                               | integer | `8000`                   | No       | Port for Eureka registration       |
| `EUREKA_SECURE_PORT`                                        | integer | `8443`                   | No       | Secure port for Eureka             |
| `EUREKA_SECURE_PORT_ENABLED`                                | boolean | `false`                  | No       | Enable secure port                 |
| `EUREKA_HOME_PAGE_URL`                                      | string  | `None`                   | No       | Home page URL                      |
| `EUREKA_STATUS_PAGE_URL`                                    | string  | `None`                   | No       | Status page URL                    |
| `EUREKA_HEALTH_CHECK_URL`                                   | string  | `None`                   | No       | Health check URL                   |
| `EUREKA_VIP_ADDRESS`                                        | string  | `None`                   | No       | VIP address                        |
| `EUREKA_SECURE_VIP_ADDRESS`                                 | string  | `None`                   | No       | Secure VIP address                 |
| `EUREKA_PREFER_IP_ADDRESS`                                  | boolean | `true`                   | No       | Prefer IP over hostname            |
| `EUREKA_LEASE_RENEWAL_INTERVAL_IN_SECONDS`                  | integer | `30`                     | No       | Lease renewal interval             |
| `EUREKA_LEASE_EXPIRATION_DURATION_IN_SECONDS`               | integer | `90`                     | No       | Lease expiration duration          |
| `EUREKA_REGISTRY_FETCH_INTERVAL_SECONDS`                    | integer | `30`                     | No       | Registry fetch interval            |
| `EUREKA_INSTANCE_INFO_REPLICATION_INTERVAL_SECONDS`         | integer | `30`                     | No       | Instance info replication interval |
| `EUREKA_INITIAL_INSTANCE_INFO_REPLICATION_INTERVAL_SECONDS` | integer | `40`                     | No       | Initial replication interval       |
| `EUREKA_HEARTBEAT_INTERVAL_SECONDS`                         | integer | `30`                     | No       | Heartbeat interval                 |

#### Eureka Retry Configuration

| Variable                                  | Type    | Default | Required | Description                 |
| ----------------------------------------- | ------- | ------- | -------- | --------------------------- |
| `EUREKA_REGISTRATION_RETRY_ATTEMPTS`      | integer | `3`     | No       | Registration retry attempts |
| `EUREKA_REGISTRATION_RETRY_DELAY_SECONDS` | integer | `5`     | No       | Registration retry delay    |
| `EUREKA_HEARTBEAT_RETRY_ATTEMPTS`         | integer | `3`     | No       | Heartbeat retry attempts    |
| `EUREKA_HEARTBEAT_RETRY_DELAY_SECONDS`    | integer | `2`     | No       | Heartbeat retry delay       |
| `EUREKA_RETRY_BACKOFF_MULTIPLIER`         | float   | `2.0`   | No       | Retry backoff multiplier    |
| `EUREKA_MAX_RETRY_DELAY_SECONDS`          | integer | `60`    | No       | Maximum retry delay         |
| `EUREKA_ENABLE_AUTO_RE_REGISTRATION`      | boolean | `true`  | No       | Enable auto re-registration |
| `EUREKA_RE_REGISTRATION_DELAY_SECONDS`    | integer | `10`    | No       | Re-registration delay       |

## Configuration Examples

### Development Environment

```bash
# .env file for development
APP_NAME=market-dataset-service
DEBUG=true
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8000

# Storage configuration
STORAGE_BASE_DIRECTORY=data/market_data
REPOSITORY_TYPE=csv
STORAGE_PROVIDER=minio

# MinIO configuration
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET_NAME=market-data

# Data collection
DEFAULT_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT
DEFAULT_TIMEFRAMES=1h,4h,1d
MAX_OHLCV_LIMIT=1000

# Logging
LOG_LEVEL=DEBUG
ENABLE_STRUCTURED_LOGGING=true

# Admin API
API_KEY=dev-admin-key

# Eureka (optional)
ENABLE_EUREKA_CLIENT=false
```

### Production Environment

```bash
# .env file for production
APP_NAME=market-dataset-service
DEBUG=false
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000

# Storage configuration
STORAGE_BASE_DIRECTORY=/app/data
REPOSITORY_TYPE=mongodb
STORAGE_PROVIDER=digitalocean

# MongoDB configuration
MONGODB_URL=mongodb://user:password@mongodb:27017/
MONGODB_DATABASE=finsight_market_data

# DigitalOcean Spaces configuration
SPACES_ENDPOINT_URL=https://nyc3.digitaloceanspaces.com
SPACES_ACCESS_KEY=your_spaces_access_key
SPACES_SECRET_KEY=your_spaces_secret_key
SPACES_REGION_NAME=nyc3
SPACES_BUCKET_NAME=finsight-market-data

# Data collection
DEFAULT_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,SOLUSDT
DEFAULT_TIMEFRAMES=1m,5m,15m,1h,4h,1d
MAX_OHLCV_LIMIT=10000

# Exchange configuration
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key

# Cross-repository configuration
SOURCE_REPOSITORY_TYPE=csv
TARGET_REPOSITORY_TYPE=parquet
TARGET_TIMEFRAMES=2h,4h,12h,1d
ENABLE_PARALLEL_CONVERSION=true
MAX_CONCURRENT_CONVERSIONS=5

# Logging
LOG_LEVEL=INFO
LOG_FILE_PATH=/app/logs
ENABLE_STRUCTURED_LOGGING=true

# Cache
ENABLE_CACHING=true
CACHE_TTL_SECONDS=600
CACHE_MAX_SIZE=2000

# Admin API
API_KEY=your_secure_admin_api_key

# Cron job
CRON_JOB_ENABLED=true
CRON_JOB_SCHEDULE=0 */4 * * *
CRON_JOB_MAX_SYMBOLS_PER_RUN=20

# Eureka service discovery
ENABLE_EUREKA_CLIENT=true
EUREKA_SERVER_URL=http://eureka-server:8761
EUREKA_APP_NAME=market-dataset-service
EUREKA_PREFER_IP_ADDRESS=true
```

### Docker Compose Environment

```yaml
# docker-compose.yml environment section
environment:
  # Application settings
  - APP_ENV=production
  - DEBUG=false
  - API_HOST=0.0.0.0
  - API_PORT=8000

  # Storage configuration
  - STORAGE_BASE_DIRECTORY=/app/data
  - REPOSITORY_TYPE=csv

  # MongoDB configuration
  - MONGODB_URL=mongodb://mongodb:27017/
  - MONGODB_DATABASE=finsight_market_data

  # S3-compatible storage (MinIO)
  - S3_ENDPOINT_URL=http://minio:9000
  - S3_ACCESS_KEY=${S3_ACCESS_KEY:-minioadmin}
  - S3_SECRET_KEY=${S3_SECRET_KEY:-minioadmin}
  - S3_BUCKET_NAME=${S3_BUCKET_NAME:-market-data}
  - S3_USE_SSL=false

  # DigitalOcean Spaces (alternative)
  - SPACES_ENDPOINT_URL=${SPACES_ENDPOINT_URL:-}
  - SPACES_REGION_NAME=${SPACES_REGION_NAME:-}
  - SPACES_ACCESS_KEY=${SPACES_ACCESS_KEY:-}
  - SPACES_SECRET_KEY=${SPACES_SECRET_KEY:-}
  - SPACES_BUCKET_NAME=${SPACES_BUCKET_NAME:-}
  - STORAGE_PROVIDER=${STORAGE_PROVIDER:-minio}

  # Binance API (optional)
  - BINANCE_API_KEY=${BINANCE_API_KEY:-}
  - BINANCE_SECRET_KEY=${BINANCE_SECRET_KEY:-}

  # Admin API
  - API_KEY=${API_KEY:-admin-default-key-change-in-production}

  # Eureka service discovery
  - ENABLE_EUREKA_CLIENT=${ENABLE_EUREKA_CLIENT:-true}
  - EUREKA_SERVER_URL=${EUREKA_SERVER_URL:-http://eureka-server:8761}
```

## Configuration Validation Details

The service validates configuration at startup using Pydantic validators:

### Environment Validation

```python
@field_validator("environment")
@classmethod
def validate_environment(cls, v):
    allowed_envs = {"development", "staging", "production", "testing"}
    if v.lower() not in allowed_envs:
        raise ValueError(f"environment must be one of {sorted(allowed_envs)}")
    return v.lower()
```

### Log Level Validation

```python
@field_validator("log_level")
@classmethod
def validate_log_level(cls, v):
    levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if v.upper() not in levels:
        raise ValueError(f"log_level must be one of {sorted(levels)}")
    return v.upper()
```

### Storage Provider Validation

```python
@field_validator("storage_provider")
@classmethod
def validate_storage_provider(cls, v):
    allowed_providers = {"minio", "digitalocean", "aws", "s3"}
    if v.lower() not in allowed_providers:
        raise ValueError(f"storage_provider must be one of {sorted(allowed_providers)}")
    return v.lower()
```

## Configuration Management

### Environment Variable Parsing

The service supports parsing comma-separated values from environment variables:

```python
@field_validator("default_symbols", mode="before")
@classmethod
def parse_symbols(cls, v):
    """Parse comma-separated symbols from environment variable"""
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return v
```

### Storage Configuration Builder

The service provides a method to build storage configuration based on the selected provider:

```python
def get_storage_config(self) -> Dict[str, Any]:
    """Get storage configuration based on the selected provider."""
    if self.storage_provider == "minio":
        return {
            "endpoint_url": self.s3_endpoint_url,
            "access_key": self.s3_access_key,
            "secret_key": self.s3_secret_key,
            "region_name": self.s3_region_name,
            "bucket_name": self.s3_bucket_name,
            "use_ssl": self.s3_use_ssl,
            "verify_ssl": self.s3_verify_ssl,
            "signature_version": self.s3_signature_version,
            "max_pool_connections": self.s3_max_pool_connections,
        }
    # ... other providers
```

## Security Considerations

### Sensitive Data

- Store API keys and secrets in environment variables
- Never commit sensitive data to version control
- Use `.env.example` for documentation without real values
- Rotate API keys regularly

### Production Security

```bash
# Production security checklist
- Use strong, unique API keys
- Enable SSL/TLS for all external communications
- Use secure storage providers (AWS S3, DigitalOcean Spaces)
- Implement proper access controls
- Monitor and log all access attempts
- Regular security audits and updates
```

## Troubleshooting

### Common Configuration Issues

#### 1. Storage Connection Issues

```bash
# Check storage configuration
curl -H "X-API-Key: your-key" http://localhost:8000/admin/health

# Verify MinIO connection
curl -f http://localhost:9000/minio/health/live
```

#### 2. Database Connection Issues

```bash
# Check MongoDB connection
mongosh "mongodb://localhost:27017/finsight_market_data"

# Verify database exists
show dbs
```

#### 3. Eureka Registration Issues

```bash
# Check Eureka status
curl -H "X-API-Key: your-key" http://localhost:8000/eureka/status

# Verify Eureka server
curl http://localhost:8761/eureka/apps
```

#### 4. API Key Issues

```bash
# Test API key
curl -H "X-API-Key: your-key" http://localhost:8000/admin/stats

# Check authentication error
curl http://localhost:8000/admin/stats
# Should return 401 Unauthorized
```

### Configuration Validation

The service provides configuration validation endpoints:

```bash
# Get current configuration (masked)
curl -H "X-API-Key: your-key" http://localhost:8000/admin/config

# Validate configuration
curl -H "X-API-Key: your-key" http://localhost:8000/admin/health
```

## Best Practices

### 1. Environment-Specific Configuration

- Use different configuration files for different environments
- Use environment variables for sensitive data
- Validate configuration at startup

### 2. Security

- Use strong, unique API keys
- Enable SSL/TLS in production
- Implement proper access controls
- Regular security audits

### 3. Performance

- Configure appropriate cache settings
- Use connection pooling for databases
- Optimize storage settings for your use case
- Monitor resource usage

### 4. Monitoring

- Enable structured logging
- Configure health checks
- Set up monitoring and alerting
- Track configuration changes

### 5. Backup and Recovery

- Regular configuration backups
- Document configuration changes
- Test configuration in staging
- Have rollback procedures
