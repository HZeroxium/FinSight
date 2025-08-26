# FinSight Sentiment Analysis Platform - Deployment Guide

> **Complete Deployment Guide for Production and Development Environments**

## 🌐 Overview

The FinSight Sentiment Analysis platform supports multiple deployment strategies from local development to production-scale Kubernetes clusters.

### **Deployment Options**

| Environment     | Complexity | Infrastructure                     | Use Case                   |
| --------------- | ---------- | ---------------------------------- | -------------------------- |
| **Development** | Low        | Docker Compose                     | Local development, testing |
| **Staging**     | Medium     | Docker Compose + External Services | Pre-production testing     |
| **Production**  | High       | Kubernetes + Cloud Services        | Production workloads       |

## 🔧 Prerequisites

### **System Requirements**

- **CPU**: 4+ cores (8+ recommended)
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 50GB+ available space
- **Network**: Stable internet connection

### **Software Dependencies**

- **Docker**: 20.10+ with Docker Compose 2.0+
- **Python**: 3.11+ (for local development)
- **Git**: Latest version
- **kubectl**: Kubernetes CLI (for K8s deployment)

### **External Services**

- **MLflow Server**: Experiment tracking and model registry
- **MinIO/S3**: Object storage for artifacts
- **NVIDIA GPU**: CUDA support (optional but recommended)

## 🚀 Development Deployment

### **Local Development Setup**

```bash
# Clone and setup
git clone <repository-url>
cd FinSight/sentiment_analysis

# Model Builder Service
cd sentiment_analysis_model_builder
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
pip install -r requirements.txt
cp env.example .env
# Edit .env with your settings

# Inference Engine
cd ../sentiment_analysis_inference_engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env

# Start dependencies
cd ../sentiment_analysis_model_builder
docker-compose up -d minio mlflow

# Run services
# Terminal 1: Model Builder
cd sentiment_analysis_model_builder
python -m src.cli train --data data/news_dataset_sample.json

# Terminal 2: Inference Engine
cd sentiment_analysis_inference_engine
python main.py
```

### **Docker Compose Development**

```bash
# Start all services
cd sentiment_analysis_model_builder
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🏭 Production Deployment

### **Production Configuration**

```bash
# Environment variables
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export ENABLE_METRICS=true
export MLFLOW_TRACKING_URI=https://mlflow.yourdomain.com
export S3_ENDPOINT=https://s3.yourdomain.com
export TRITON_HOST=triton.yourdomain.com
```

### **Production Docker Compose**

```yaml
# docker-compose.prod.yml
version: "3.8"
services:
  model_builder:
    image: finsight/sentiment-model-builder:latest
    deploy:
      replicas: 2
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

  inference_engine:
    image: finsight/sentiment-inference-engine:latest
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
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 🐳 Docker Deployment

### **Docker Image Building**

```dockerfile
# Model Builder Dockerfile
FROM python:3.11-slim

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
CMD ["python", "-m", "src.cli", "serve"]
```

### **Build and Deploy**

```bash
# Build images
docker build -t finsight/sentiment-model-builder:latest sentiment_analysis_model_builder/
docker build -t finsight/sentiment-inference-engine:latest sentiment_analysis_inference_engine/

# Push to registry
docker push finsight/sentiment-model-builder:latest
docker push finsight/sentiment-inference-engine:latest

# Deploy with Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

## ☸️ Kubernetes Deployment

### **Basic Kubernetes Manifests**

#### **Namespace**

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: sentiment-analysis
  labels:
    name: sentiment-analysis
    environment: production
```

#### **ConfigMap**

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sentiment-analysis-config
  namespace: sentiment-analysis
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  ENABLE_METRICS: "true"
```

#### **Model Builder Deployment**

```yaml
# k8s/model-builder-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-builder
  namespace: sentiment-analysis
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-builder
  template:
    metadata:
      labels:
        app: model-builder
    spec:
      containers:
        - name: model-builder
          image: finsight/sentiment-model-builder:latest
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: sentiment-analysis-config
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

#### **Inference Engine Deployment**

```yaml
# k8s/inference-engine-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-engine
  namespace: sentiment-analysis
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference-engine
  template:
    metadata:
      labels:
        app: inference-engine
    spec:
      containers:
        - name: inference-engine
          image: finsight/sentiment-inference-engine:latest
          ports:
            - containerPort: 8080
          envFrom:
            - configMapRef:
                name: sentiment-analysis-config
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
              port: 8080
            initialDelaySeconds: 40
            periodSeconds: 30
```

#### **Services**

```yaml
# k8s/services.yaml
apiVersion: v1
kind: Service
metadata:
  name: model-builder-service
  namespace: sentiment-analysis
spec:
  type: ClusterIP
  ports:
    - port: 80
      targetPort: 8000
      protocol: TCP
  selector:
    app: model-builder
---
apiVersion: v1
kind: Service
metadata:
  name: inference-engine-service
  namespace: sentiment-analysis
spec:
  type: ClusterIP
  ports:
    - port: 80
      targetPort: 8080
      protocol: TCP
  selector:
    app: inference-engine
```

### **Deploy to Kubernetes**

```bash
# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/model-builder-deployment.yaml
kubectl apply -f k8s/inference-engine-deployment.yaml
kubectl apply -f k8s/services.yaml

# Check deployment status
kubectl get pods -n sentiment-analysis
kubectl get services -n sentiment-analysis

# View logs
kubectl logs -f deployment/model-builder -n sentiment-analysis
kubectl logs -f deployment/inference-engine -n sentiment-analysis
```

## 🔄 CI/CD Pipeline

### **GitHub Actions**

#### **CI Pipeline**

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
          python-version: "3.11"
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml
```

#### **CD Pipeline**

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
      - name: Build and push Docker images
        uses: docker/build-push-action@v4
        with:
          context: ./sentiment_analysis_model_builder
          push: true
          tags: finsight/sentiment-model-builder:latest
      - name: Deploy to production
        run: |
          kubectl set image deployment/model-builder \
            model-builder=finsight/sentiment-model-builder:latest \
            -n sentiment-analysis
```

### **Jenkins Pipeline**

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
                    docker.build("finsight/sentiment-model-builder:${env.BUILD_NUMBER}")
                    docker.build("finsight/sentiment-inference-engine:${env.BUILD_NUMBER}")
                }
            }
        }
        stage('Deploy') {
            steps {
                script {
                    if (env.BRANCH_NAME == 'main') {
                        sh "kubectl set image deployment/model-builder \
                            model-builder=finsight/sentiment-model-builder:${env.BUILD_NUMBER} \
                            -n sentiment-analysis"
                    }
                }
            }
        }
    }
}
```

## 📊 Monitoring & Health Checks

### **Health Check Endpoints**

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

    # Check MLflow connection
    try:
        mlflow_client.ping()
        health_status["checks"]["mlflow"] = "healthy"
    except Exception as e:
        health_status["checks"]["mlflow"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"

    return health_status
```

### **Prometheus Metrics**

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

### **Monitoring Stack**

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

### **Common Issues**

```bash
# Service won't start
docker-compose logs model_builder
docker-compose logs inference_engine

# Model loading issues
ls -la models/
df -h
free -h

# External service connections
curl http://localhost:5000/health  # MLflow
curl http://localhost:9000/minio/health/live  # MinIO
```

### **Debug Mode**

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
docker-compose restart model_builder

# Debug endpoints
curl http://localhost:8000/debug/info
curl http://localhost:8080/debug/info
```

## ⚡ Performance Tuning

### **Application Tuning**

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

### **Kubernetes Tuning**

```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-engine-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference_engine
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
