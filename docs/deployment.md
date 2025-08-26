# FinSight Deployment Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Development Environment](#development-environment)
3. [Production Deployment](#production-deployment)
4. [Configuration Management](#configuration-management)
5. [Monitoring & Health Checks](#monitoring--health-checks)
6. [Troubleshooting](#troubleshooting)
7. [Backup & Recovery](#backup--recovery)
8. [Security Hardening](#security-hardening)
9. [Performance Tuning](#performance-tuning)
10. [Maintenance Procedures](#maintenance-procedures)

## Quick Start

### Prerequisites

- **Docker & Docker Compose**: Version 20.10+ for containerization
- **Python**: Version 3.9+ for local development
- **Git**: For version control
- **Make**: For build automation (optional)
- **kubectl**: For production deployment (optional)
- **helm**: For Kubernetes package management (optional)

### Local Development Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-org/finsight.git
   cd finsight
   ```

2. **Start infrastructure services:**

   ```bash
   docker-compose up -d rabbitmq redis eureka-server postgres minio mlflow
   ```

3. **Set up environment variables:**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start individual services:**

   ```bash
   # Market Dataset Service
   cd market_dataset_service
   python -m src.main

   # News Service
   cd ../news_service
   python -m src.main

   # Sentiment Analysis Service
   cd ../sentiment_analysis
   python -m sentiment_analysis_service.src.main

   # Prediction Service
   cd ../prediction_service
   python -m src.main
   ```

### Docker Quick Start

**All-in-one deployment:**

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

**Individual service deployment:**

```bash
# Deploy specific services
docker-compose up -d market-dataset-service
docker-compose up -d news-service
docker-compose up -d sentiment-analysis-service
docker-compose up -d prediction-service
```

## Development Environment

### Local Development Workflow

#### 1. Environment Setup

**Virtual Environment:**

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Service-Specific Setup:**

```bash
# Market Dataset Service
cd market_dataset_service
pip install -r requirements.txt
python -m src.main

# News Service
cd ../news_service
pip install -r requirements.txt
python -m src.main
```

#### 2. Database Setup

**MongoDB (for Market Dataset & News Services):**

```bash
# Start MongoDB
docker run -d --name mongodb \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  mongo:6.0

# Create databases and users
docker exec -it mongodb mongosh
```

**PostgreSQL (for MLflow):**

```bash
# Start PostgreSQL
docker run -d --name postgres \
  -p 5432:5432 \
  -e POSTGRES_DB=mlflow \
  -e POSTGRES_USER=mlflow \
  -e POSTGRES_PASSWORD=mlflow \
  postgres:15
```

#### 3. External API Setup

**Binance API:**

```bash
# Set environment variables
export BINANCE_API_KEY="your-binance-api-key"
export BINANCE_SECRET_KEY="your-binance-secret-key"
```

**Tavily API:**

```bash
export TAVILY_API_KEY="your-tavily-api-key"
```

**OpenAI API:**

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

#### 4. Development Tools

**Code Quality:**

```bash
# Install development dependencies
pip install black isort mypy flake8 pytest

# Format code
black .
isort .

# Type checking
mypy .

# Linting
flake8 .

# Run tests
pytest
```

**API Testing:**

```bash
# Test Market Dataset Service
curl -X GET "http://localhost:8000/health"

# Test News Service
curl -X GET "http://localhost:8001/health"

# Test Sentiment Analysis Service
curl -X GET "http://localhost:8002/health"

# Test Prediction Service
curl -X GET "http://localhost:8003/health"
```

### Development Configuration

**Environment Variables Template:**

```bash
# Copy and customize for each service
cp market_dataset_service/env.example market_dataset_service/.env
cp news_service/env.example news_service/.env
cp sentiment_analysis/sentiment_analysis_service/env.example sentiment_analysis/sentiment_analysis_service/.env
cp prediction_service/env.example prediction_service/.env
```

**Service Configuration:**

```python
# Example configuration structure
class Config:
    # Database
    MONGODB_URL: str = "mongodb://localhost:27017"
    POSTGRES_URL: str = "postgresql://mlflow:mlflow@localhost:5432/mlflow"

    # External APIs
    BINANCE_API_KEY: str = ""
    BINANCE_SECRET_KEY: str = ""
    TAVILY_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # Service settings
    LOG_LEVEL: str = "INFO"
    API_KEY: str = "your-api-key"
    ADMIN_API_KEY: str = "your-admin-api-key"
```

## Production Deployment

### Kubernetes Deployment

#### 1. Cluster Setup

**Prerequisites:**

- Kubernetes cluster (1.24+)
- Helm (3.10+)
- kubectl configured
- Istio service mesh (optional)

**Cluster Requirements:**

- **CPU**: Minimum 8 cores
- **Memory**: Minimum 16GB RAM
- **Storage**: 100GB+ persistent storage
- **Nodes**: At least 3 worker nodes

#### 2. Helm Charts

**Create Helm charts structure:**

```bash
helm/
├── charts/
│   ├── market-dataset-service/
│   ├── news-service/
│   ├── sentiment-analysis-service/
│   ├── prediction-service/
│   └── infrastructure/
├── values.yaml
└── Chart.yaml
```

**Install infrastructure:**

```bash
# Add required repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

# Install infrastructure components
helm install rabbitmq bitnami/rabbitmq \
  --set auth.username=admin \
  --set auth.password=password \
  --namespace finsight

helm install redis bitnami/redis \
  --set auth.password=password \
  --namespace finsight

helm install mongodb bitnami/mongodb \
  --set auth.rootPassword=password \
  --namespace finsight

helm install postgresql bitnami/postgresql \
  --set auth.postgresPassword=password \
  --namespace finsight
```

**Deploy services:**

```bash
# Deploy all services
helm install finsight ./helm/charts \
  --namespace finsight \
  --values ./helm/values-production.yaml

# Deploy individual services
helm install market-dataset ./helm/charts/market-dataset-service \
  --namespace finsight \
  --values ./helm/charts/market-dataset-service/values-production.yaml
```

#### 3. Service Configuration

**Kubernetes ConfigMaps:**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: finsight-config
  namespace: finsight
data:
  log_level: "INFO"
  environment: "production"
  api_version: "v1"
  cors_origins: "https://app.finsight.com"
```

**Kubernetes Secrets:**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: finsight-secrets
  namespace: finsight
type: Opaque
data:
  binance-api-key: <base64-encoded>
  binance-secret-key: <base64-encoded>
  tavily-api-key: <base64-encoded>
  openai-api-key: <base64-encoded>
  api-key: <base64-encoded>
  admin-api-key: <base64-encoded>
```

**Service Deployments:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: market-dataset-service
  namespace: finsight
spec:
  replicas: 3
  selector:
    matchLabels:
      app: market-dataset-service
  template:
    metadata:
      labels:
        app: market-dataset-service
    spec:
      containers:
        - name: market-dataset-service
          image: finsight/market-dataset-service:latest
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: finsight-config
            - secretRef:
                name: finsight-secrets
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
```

### Docker Swarm Deployment

**Initialize Swarm:**

```bash
# Initialize swarm mode
docker swarm init

# Create overlay network
docker network create -d overlay finsight-network
```

**Deploy Stack:**

```bash
# Deploy entire stack
docker stack deploy -c docker-compose.prod.yml finsight

# Check stack status
docker stack services finsight

# Scale services
docker service scale finsight_market-dataset-service=3
```

### Cloud Deployment

#### AWS ECS Deployment

**ECS Task Definition:**

```json
{
  "family": "finsight-market-dataset",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123456789012:role/finsight-task-role",
  "containerDefinitions": [
    {
      "name": "market-dataset-service",
      "image": "finsight/market-dataset-service:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "secrets": [
        {
          "name": "BINANCE_API_KEY",
          "valueFrom": "arn:aws:ssm:us-east-1:123456789012:parameter/finsight/binance-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/finsight",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "market-dataset"
        }
      }
    }
  ]
}
```

**Load Balancer Configuration:**

```yaml
# Application Load Balancer
Listener:
  - Protocol: HTTPS
    Port: 443
    DefaultActions:
      - Type: forward
        TargetGroupArn: !Ref TargetGroup

TargetGroup:
  Type: AWS::ElasticLoadBalancingV2::TargetGroup
  Properties:
    Name: finsight-target-group
    Port: 8000
    Protocol: HTTP
    VpcId: !Ref VPC
    TargetType: ip
    HealthCheckPath: /health
    HealthCheckIntervalSeconds: 30
```

#### Google Cloud Run

**Service Configuration:**

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: market-dataset-service
spec:
  template:
    spec:
      containers:
        - image: gcr.io/finsight/market-dataset-service:latest
          ports:
            - containerPort: 8000
          env:
            - name: LOG_LEVEL
              value: "INFO"
            - name: BINANCE_API_KEY
              valueFrom:
                secretKeyRef:
                  name: binance-api-key
                  key: api-key
          resources:
            limits:
              cpu: "1"
              memory: "2Gi"
```

## Configuration Management

### Environment-Specific Configuration

**Development Configuration:**

```yaml
# config/development.yaml
environment: development
debug: true
log_level: DEBUG
database:
  mongodb_url: "mongodb://localhost:27017"
  postgres_url: "postgresql://mlflow:mlflow@localhost:5432/mlflow"
caching:
  redis_url: "redis://localhost:6379"
message_queue:
  rabbitmq_url: "amqp://guest:guest@localhost:5672/"
api:
  rate_limit: 1000
  cors_origins: ["http://localhost:3000"]
```

**Production Configuration:**

```yaml
# config/production.yaml
environment: production
debug: false
log_level: INFO
database:
  mongodb_url: "mongodb://mongodb:27017"
  postgres_url: "postgresql://mlflow:mlflow@postgresql:5432/mlflow"
caching:
  redis_url: "redis://redis:6379"
message_queue:
  rabbitmq_url: "amqp://admin:password@rabbitmq:5672/"
api:
  rate_limit: 100
  cors_origins: ["https://app.finsight.com"]
monitoring:
  prometheus_enabled: true
  jaeger_enabled: true
```

### Configuration Validation

**Configuration Schema:**

```python
from pydantic import BaseSettings, Field

class DatabaseConfig(BaseSettings):
    mongodb_url: str = Field(..., description="MongoDB connection URL")
    postgres_url: str = Field(..., description="PostgreSQL connection URL")

    class Config:
        env_prefix = "DB_"

class APIConfig(BaseSettings):
    rate_limit: int = Field(default=1000, description="API rate limit")
    cors_origins: list[str] = Field(default=[], description="CORS origins")

    class Config:
        env_prefix = "API_"

class Config(BaseSettings):
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Log level")
    database: DatabaseConfig
    api: APIConfig

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

### Secrets Management

**Kubernetes Secrets:**

```bash
# Create secrets
kubectl create secret generic finsight-secrets \
  --from-literal=binance-api-key="your-api-key" \
  --from-literal=binance-secret-key="your-secret-key" \
  --from-literal=tavily-api-key="your-tavily-key" \
  --namespace finsight

# Update secrets
kubectl patch secret finsight-secrets \
  --patch '{"data":{"binance-api-key":"new-base64-encoded-value"}}' \
  --namespace finsight
```

**AWS Secrets Manager:**

```bash
# Store secrets
aws secretsmanager create-secret \
  --name "finsight/binance-api-key" \
  --description "Binance API key for FinSight" \
  --secret-string '{"api-key":"your-api-key","secret-key":"your-secret-key"}'

# Retrieve secrets
aws secretsmanager get-secret-value --secret-id "finsight/binance-api-key"
```

## Monitoring & Health Checks

### Health Check Endpoints

**Service Health Checks:**

```bash
# Basic health check
curl -X GET "http://localhost:8000/health"

# Detailed health check (admin only)
curl -H "X-API-Key: admin-key" \
     -X GET "http://localhost:8000/admin/health"

# Health check with dependencies
curl -X GET "http://localhost:8000/health/detailed"
```

**Expected Responses:**

```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "service": "market-dataset-service",
    "version": "1.0.0",
    "timestamp": "2024-01-15T10:30:00Z",
    "dependencies": {
      "mongodb": "connected",
      "redis": "connected",
      "rabbitmq": "connected"
    }
  }
}
```

### Monitoring Stack

**Prometheus Configuration:**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "finsight-services"
    static_configs:
      - targets:
          [
            "market-dataset:8000",
            "news-service:8001",
            "sentiment-service:8002",
            "prediction-service:8003",
          ]
    metrics_path: "/metrics"
    scrape_interval: 5s

  - job_name: "finsight-infrastructure"
    static_configs:
      - targets: ["rabbitmq:15692", "redis:6379", "mongodb:9216"]
```

**Grafana Dashboards:**

```json
{
  "dashboard": {
    "title": "FinSight Platform Dashboard",
    "panels": [
      {
        "title": "Service Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"finsight-services\"}",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "http_request_duration_seconds{job=\"finsight-services\"}",
            "legendFormat": "{{instance}} - {{endpoint}}"
          }
        ]
      }
    ]
  }
}
```

### Alerting Rules

**Prometheus Alerting:**

```yaml
# alerts.yml
groups:
  - name: finsight-alerts
    rules:
      - alert: ServiceDown
        expr: up{job="finsight-services"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.instance }} is down"
          description: "Service {{ $labels.instance }} has been down for more than 1 minute"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} requests per second"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, http_request_duration_seconds) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }} seconds"
```

## Troubleshooting

### Common Issues

#### 1. Service Startup Issues

**Problem**: Service fails to start

```bash
# Check logs
docker-compose logs service-name

# Check environment variables
env | grep SERVICE_NAME

# Check port conflicts
netstat -tulpn | grep :8000
```

**Solution**: Verify configuration and dependencies

#### 2. Database Connection Issues

**Problem**: Cannot connect to database

```bash
# Test database connectivity
docker exec -it mongodb mongosh --eval "db.runCommand('ping')"

# Check network connectivity
docker network ls
docker network inspect finsight_default
```

**Solution**: Verify database is running and network configuration

#### 3. API Key Issues

**Problem**: Authentication failures

```bash
# Verify API key format
echo $API_KEY | wc -c

# Check API key in headers
curl -v -H "X-API-Key: your-key" http://localhost:8000/health
```

**Solution**: Ensure API keys are properly set and formatted

#### 4. Performance Issues

**Problem**: Slow response times

```bash
# Check resource usage
docker stats

# Monitor database performance
docker exec -it mongodb mongosh --eval "db.currentOp()"

# Check Redis memory usage
docker exec -it redis redis-cli info memory
```

**Solution**: Scale resources or optimize queries

### Debug Commands

**Service Debugging:**

```bash
# Get service logs
docker-compose logs -f service-name

# Access service shell
docker-compose exec service-name bash

# Check service health
curl -X GET "http://localhost:8000/health"

# Test API endpoints
curl -H "X-API-Key: your-key" \
     -X POST "http://localhost:8000/market-data/ohlcv" \
     -H "Content-Type: application/json" \
     -d '{"symbol":"BTCUSDT","timeframe":"1h"}'
```

**Database Debugging:**

```bash
# MongoDB
docker exec -it mongodb mongosh
use finsight
db.collection_name.find().limit(5)

# PostgreSQL
docker exec -it postgres psql -U mlflow -d mlflow
SELECT * FROM experiments LIMIT 5;

# Redis
docker exec -it redis redis-cli
KEYS *
MONITOR
```

## Backup & Recovery

### Database Backup

**MongoDB Backup:**

```bash
# Create backup
docker exec mongodb mongodump \
  --db finsight \
  --out /backup/$(date +%Y%m%d_%H%M%S)

# Restore backup
docker exec mongodb mongorestore \
  --db finsight \
  /backup/20240115_103000/finsight/
```

**PostgreSQL Backup:**

```bash
# Create backup
docker exec postgres pg_dump \
  -U mlflow mlflow \
  > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore backup
docker exec -i postgres psql \
  -U mlflow mlflow \
  < backup_20240115_103000.sql
```

**Automated Backup Script:**

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# MongoDB backup
docker exec mongodb mongodump \
  --db finsight \
  --out $BACKUP_DIR/mongodb

# PostgreSQL backup
docker exec postgres pg_dump \
  -U mlflow mlflow \
  > $BACKUP_DIR/postgresql.sql

# Redis backup
docker exec redis redis-cli SAVE
docker cp redis:/data/dump.rdb $BACKUP_DIR/redis.rdb

# Compress backup
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR

# Upload to cloud storage (optional)
aws s3 cp $BACKUP_DIR.tar.gz s3://finsight-backups/
```

### Disaster Recovery

**Recovery Procedures:**

```bash
# 1. Stop all services
docker-compose down

# 2. Restore databases
docker exec mongodb mongorestore --db finsight /backup/mongodb/finsight/
docker exec -i postgres psql -U mlflow mlflow < /backup/postgresql.sql
docker cp /backup/redis.rdb redis:/data/dump.rdb
docker exec redis redis-cli BGREWRITEAOF

# 3. Restart services
docker-compose up -d

# 4. Verify recovery
curl -X GET "http://localhost:8000/health"
```

## Security Hardening

### Network Security

**Firewall Configuration:**

```bash
# Allow only necessary ports
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw allow 8000/tcp  # Market Dataset Service
ufw allow 8001/tcp  # News Service
ufw allow 8002/tcp  # Sentiment Analysis Service
ufw allow 8003/tcp  # Prediction Service
ufw deny 27017/tcp  # MongoDB (internal only)
ufw deny 6379/tcp   # Redis (internal only)

# Enable firewall
ufw enable
```

**TLS/SSL Configuration:**

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name api.finsight.com;

    ssl_certificate /etc/ssl/certs/finsight.crt;
    ssl_certificate_key /etc/ssl/private/finsight.key;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Application Security

**API Security Headers:**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.finsight.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Trusted hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["api.finsight.com", "localhost"]
)

# Security headers
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

**Rate Limiting:**

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/v1/predict")
@limiter.limit("10/minute")  # 10 requests per minute
async def predict(request: Request):
    # Implementation
    pass
```

## Performance Tuning

### Database Optimization

**MongoDB Optimization:**

```javascript
// Create indexes
db.ohlcv_data.createIndex({ symbol: 1, timestamp: -1 });
db.ohlcv_data.createIndex({ exchange: 1, timeframe: 1 });
db.backtest_results.createIndex({ backtest_id: 1 });

// Optimize queries
db.ohlcv_data
  .find({
    symbol: "BTCUSDT",
    timestamp: {
      $gte: new Date("2024-01-01"),
      $lte: new Date("2024-01-31"),
    },
  })
  .hint({ symbol: 1, timestamp: -1 });
```

**PostgreSQL Optimization:**

```sql
-- Create indexes
CREATE INDEX idx_experiments_name ON experiments(name);
CREATE INDEX idx_metrics_run_id ON metrics(run_id);
CREATE INDEX idx_models_name_version ON models(name, version);

-- Optimize queries
EXPLAIN ANALYZE SELECT * FROM experiments WHERE name = 'patchedtst_btcusdt';
```

### Caching Strategy

**Redis Caching:**

```python
import redis
from functools import wraps

redis_client = redis.Redis(host='redis', port=6379, db=0)

def cache_result(ttl=300):  # 5 minutes default
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            redis_client.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator

@cache_result(ttl=600)  # 10 minutes
async def get_ohlcv_data(symbol: str, timeframe: str):
    # Implementation
    pass
```

### Load Balancing

**Nginx Load Balancer:**

```nginx
upstream finsight_backend {
    least_conn;  # Least connections algorithm

    server market-dataset-1:8000 weight=3 max_fails=3 fail_timeout=30s;
    server market-dataset-2:8000 weight=3 max_fails=3 fail_timeout=30s;
    server market-dataset-3:8000 weight=3 max_fails=3 fail_timeout=30s;

    keepalive 32;  # Keep-alive connections
}

server {
    listen 80;
    server_name api.finsight.com;

    location / {
        proxy_pass http://finsight_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;

        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
}
```

## Maintenance Procedures

### Regular Maintenance

**Daily Tasks:**

```bash
# Check service health
curl -X GET "http://localhost:8000/health"
curl -X GET "http://localhost:8001/health"
curl -X GET "http://localhost:8002/health"
curl -X GET "http://localhost:8003/health"

# Check disk usage
df -h

# Check memory usage
free -h

# Check log file sizes
du -sh /var/log/finsight/*
```

**Weekly Tasks:**

```bash
# Database maintenance
docker exec mongodb mongosh --eval "db.runCommand('compact')"
docker exec postgres psql -U mlflow -d mlflow -c "VACUUM ANALYZE;"

# Log rotation
logrotate -f /etc/logrotate.d/finsight

# Backup verification
./verify_backup.sh
```

**Monthly Tasks:**

```bash
# Security updates
apt update && apt upgrade

# Performance analysis
./performance_analysis.sh

# Capacity planning
./capacity_analysis.sh
```

### Update Procedures

**Service Updates:**

```bash
# 1. Create backup
./backup.sh

# 2. Update images
docker-compose pull

# 3. Deploy updates
docker-compose up -d --no-deps

# 4. Verify deployment
./health_check.sh

# 5. Rollback if needed
docker-compose up -d --no-deps
```

**Rollback Procedures:**

```bash
# Rollback to previous version
docker-compose down
docker tag finsight/market-dataset-service:previous finsight/market-dataset-service:latest
docker-compose up -d

# Verify rollback
curl -X GET "http://localhost:8000/health"
```

---

_This deployment guide provides comprehensive instructions for deploying and maintaining the FinSight platform in various environments._
