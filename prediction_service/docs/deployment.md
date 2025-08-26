# FinSight Prediction Service - Deployment Guide

> **Complete Deployment Guide for Production and Development Environments**

## 🌐 Overview

The FinSight Prediction Service supports multiple deployment strategies from local development to production-scale Kubernetes clusters.

### Deployment Options

| Environment     | Complexity | Infrastructure                     | Use Case                   |
| --------------- | ---------- | ---------------------------------- | -------------------------- |
| **Development** | Low        | Docker Compose                     | Local development, testing |
| **Staging**     | Medium     | Docker Compose + External Services | Pre-production testing     |
| **Production**  | High       | Kubernetes + Cloud Services        | Production workloads       |

## 🔧 Prerequisites

### System Requirements

- **CPU**: 4+ cores (8+ recommended)
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 50GB+ available space
- **Network**: Stable internet connection

### Software Dependencies

- **Docker**: 20.10+ with Docker Compose 2.0+
- **Python**: 3.12+ (for local development)
- **Git**: Latest version
- **kubectl**: Kubernetes CLI (for K8s deployment)

### External Services

- **Eureka Server**: Service discovery
- **Redis**: Job queue and caching
- **Cloud Storage**: S3-compatible storage

## 🚀 Development Deployment

### Local Development Setup

```bash
# Clone and setup
git clone <repository-url>
cd FinSight/prediction_service

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env with your settings

# Start dependencies
docker-compose up -d redis minio

# Run service
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Compose Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f prediction_service

# Stop services
docker-compose down
```

## 🏭 Production Deployment

### Production Configuration

```bash
# Environment variables
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export ENABLE_METRICS=true
export EUREKA_SERVER_URL=https://eureka.yourdomain.com
export REDIS_URL=redis://redis-cluster.yourdomain.com:6379
export S3_ENDPOINT=https://s3.yourdomain.com
```

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: "3.8"
services:
  prediction_service:
    image: finsight/prediction_service:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 🐳 Docker Deployment

### Docker Image Building

```dockerfile
# Dockerfile
FROM python:3.12-slim

# Install dependencies
RUN apt-get update && apt-get install -y curl

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create application user
RUN useradd --create-home --shell /bin/bash app
USER app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=app:app src/ ./src/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Deploy

```bash
# Build image
docker build -t finsight/prediction_service:latest .

# Push to registry
docker push finsight/prediction_service:latest

# Deploy with Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

## ☸️ Kubernetes Deployment

### Basic Kubernetes Manifests

#### Namespace

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: finsight
  labels:
    name: finsight
    environment: production
```

#### ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prediction-service-config
  namespace: finsight
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  ENABLE_METRICS: "true"
```

#### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prediction-service
  namespace: finsight
spec:
  replicas: 3
  selector:
    matchLabels:
      app: prediction-service
  template:
    metadata:
      labels:
        app: prediction-service
    spec:
      containers:
        - name: prediction-service
          image: finsight/prediction_service:latest
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: prediction-service-config
          resources:
            requests:
              memory: "2Gi"
              cpu: "1000m"
            limits:
              memory: "4Gi"
              cpu: "2000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 40
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
```

#### Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: prediction-service
  namespace: finsight
spec:
  type: ClusterIP
  ports:
    - port: 80
      targetPort: 8000
      protocol: TCP
  selector:
    app: prediction-service
```

### Deploy to Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check deployment status
kubectl get pods -n finsight
kubectl get services -n finsight

# View logs
kubectl logs -f deployment/prediction-service -n finsight
```

## 🔄 CI/CD Pipeline

### GitHub Actions

#### CI Pipeline

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml
```

#### CD Pipeline

```yaml
# .github/workflows/cd.yml
name: CD Pipeline

on:
  push:
    tags:
      - "v*"

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: finsight/prediction_service:latest
      - name: Deploy to production
        run: |
          kubectl set image deployment/prediction_service \
            prediction_service=finsight/prediction_service:latest \
            -n finsight
```

### Jenkins Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps { checkout scm }
        }
        stage('Test') {
            steps {
                sh 'python -m pip install -r requirements.txt'
                sh 'python -m pytest tests/ --cov=src'
            }
        }
        stage('Build') {
            steps {
                script {
                    docker.build("finsight/prediction_service:${env.BUILD_NUMBER}")
                }
            }
        }
        stage('Deploy') {
            steps {
                script {
                    if (env.BRANCH_NAME == 'main') {
                        sh "kubectl set image deployment/prediction_service \
                            prediction_service=finsight/prediction_service:${env.BUILD_NUMBER} \
                            -n finsight"
                    }
                }
            }
        }
    }
}
```

## 📊 Monitoring & Health Checks

### Health Check Endpoints

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health/detailed")
async def detailed_health_check():
    health_status = {
        "status": "healthy",
        "checks": {}
    }

    # Check Redis connection
    try:
        redis_client.ping()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"

    return health_status
```

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(duration)

    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Monitoring Stack

```yaml
# docker-compose.monitoring.yml
version: "3.8"
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  grafana_data:
```

## 🐛 Troubleshooting

### Common Issues

```bash
# Service won't start
docker-compose logs prediction_service
docker-compose exec prediction_service env | grep -E "(EUREKA|REDIS|S3)"

# Model loading issues
ls -la models/
df -h
free -h

# External service connections
docker-compose exec prediction_service redis-cli -h redis ping
curl http://localhost:8761/eureka/apps
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
docker-compose restart prediction_service

# Debug endpoint
curl http://localhost:8000/debug/info
```

## ⚡ Performance Tuning

### Application Tuning

```python
# Worker configuration
import multiprocessing

if __name__ == "__main__":
    workers = multiprocessing.cpu_count() * 2 + 1
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        workers=workers,
        loop="uvloop",
        http="httptools"
    )
```

### Kubernetes Tuning

```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: prediction-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: prediction_service
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

---

**For more information, see the [Architecture Documentation](architecture.md) and [Configuration Guide](configuration.md).**
