# FinSight Sentiment Analysis Model Builder - Deployment Guide

> **Essential Deployment Guide for Local Development and Production Environments**  
> Streamlined instructions for deploying the Model Builder service

## 🔧 Prerequisites

### **System Requirements**

| Component   | Minimum             | Recommended     | Production       |
| ----------- | ------------------- | --------------- | ---------------- |
| **CPU**     | 4 cores             | 8 cores         | 16+ cores        |
| **RAM**     | 8GB                 | 16GB            | 32GB+            |
| **Storage** | 50GB                | 100GB           | 500GB+           |
| **GPU**     | None                | NVIDIA RTX 3080 | NVIDIA A100/V100 |
| **OS**      | Linux/macOS/Windows | Linux           | Linux            |

### **Required Software**

- **Python**: 3.11+ (3.12 recommended)
- **Docker**: 20.10+ (for containerized deployment)
- **Docker Compose**: 2.0+ (for multi-container deployment)
- **Git**: 2.30+ (for version control)

### **External Dependencies**

- **MLflow Server**: For experiment tracking and model registry
- **Object Storage**: S3/MinIO for model artifacts
- **GPU Resources**: NVIDIA GPU with CUDA support (optional but recommended)

## 🏠 Local Development

### **Quick Start**

```bash
# Clone and setup
git clone <repository-url>
cd sentiment_analysis_model_builder

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Configuration
cp env.example .env
# Edit .env with your settings

# Verify installation
sentiment-health
```

### **Basic Configuration**

```bash
# .env
LOG_LEVEL=DEBUG
API_HOST=localhost
API_PORT=8000

# Data configuration
DATA_INPUT_PATH=data/news_dataset_sample.json
DATA_INPUT_FORMAT=json
DATA_TEXT_COLUMN=text
DATA_LABEL_COLUMN=label

# Training configuration
TRAINING_BACKBONE=ProsusAI/finbert
TRAINING_BATCH_SIZE=8
TRAINING_LEARNING_RATE=2e-5
TRAINING_NUM_EPOCHS=1

# Registry configuration
REGISTRY_TRACKING_URI=sqlite:///mlruns.db
REGISTRY_MODEL_NAME=crypto-news-sentiment
REGISTRY_MODEL_STAGE=Staging

# Security
SECURITY_API_KEY=dev-api-key-123
```

### **Running the Service**

```bash
# CLI mode
sentiment-train \
    --data data/news_dataset_sample.json \
    --output outputs/training_run_$(date +%Y%m%d_%H%M%S) \
    --experiment crypto-sentiment-v1

# API mode
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

## 🐳 Docker Deployment

### **Basic Docker Setup**

```bash
# Build image
docker build -t finsight/sentiment-model-builder:latest .

# Run container
docker run -p 8000:8000 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    finsight/sentiment-model-builder:latest
```

### **Docker Compose**

```yaml
# docker-compose.yml
version: "3.8"

services:
  sentiment-model-builder:
    image: finsight/sentiment-model-builder:latest
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - DATA_INPUT_PATH=/app/data/news_dataset.json
      - TRAINING_BATCH_SIZE=16
      - REGISTRY_TRACKING_URI=http://mlflow:5000
      - REGISTRY_ARTIFACT_LOCATION=s3://models
      - REGISTRY_AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - REGISTRY_AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - REGISTRY_S3_BUCKET=${S3_BUCKET}
      - SECURITY_API_KEY=${API_KEY}
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    depends_on:
      - mlflow
    restart: unless-stopped

  mlflow:
    image: python:3.11-slim
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_TRACKING_URI=sqlite:///mlruns.db
    volumes:
      - mlflow_data:/mlflow
    command: >
      sh -c "pip install mlflow &&
             mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns.db"

volumes:
  mlflow_data:
```

```bash
# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## ☸️ Kubernetes Deployment

### **Basic Kubernetes Setup**

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: sentiment-analysis
```

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sentiment-model-builder-config
  namespace: sentiment-analysis
data:
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  TRAINING_BATCH_SIZE: "32"
  TRAINING_FP16: "true"
  REGISTRY_TRACKING_URI: "http://mlflow-service:5000"
  REGISTRY_ARTIFACT_LOCATION: "s3://prod-models"
```

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: sentiment-model-builder-secret
  namespace: sentiment-analysis
type: Opaque
data:
  api-key: <base64-encoded-api-key>
  aws-access-key-id: <base64-encoded-aws-access-key>
  aws-secret-access-key: <base64-encoded-aws-secret-key>
```

```yaml
# k8s/deployment.yaml
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
                name: sentiment-model-builder-config
          env:
            - name: SECURITY_API_KEY
              valueFrom:
                secretKeyRef:
                  name: sentiment-model-builder-secret
                  key: api-key
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

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: model-builder-service
  namespace: sentiment-analysis
spec:
  selector:
    app: model-builder
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP
```

### **Apply Kubernetes Manifests**

```bash
# Create namespace and apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check status
kubectl get pods -n sentiment-analysis
kubectl get services -n sentiment-analysis
```

### **Helm Deployment**

```bash
# Install with Helm
helm install sentiment-model-builder ./helm-charts/sentiment-model-builder \
    --namespace sentiment-analysis \
    --create-namespace \
    --values values-prod.yaml

# Upgrade deployment
helm upgrade sentiment-model-builder ./helm-charts/sentiment-model-builder \
    --namespace sentiment-analysis \
    --values values-prod.yaml
```

## 🔄 CI/CD Pipelines

### **GitHub Actions**

```yaml
# .github/workflows/ci.yml
name: CI/CD

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
          pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest tests/ --cov=src
      - name: Run linting
        run: |
          black --check src/
          isort --check-only src/
          flake8 src/

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: |
          docker build -t finsight/sentiment-model-builder:latest .
          docker push finsight/sentiment-model-builder:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Kubernetes
        run: |
          kubectl apply -f k8s/namespace.yaml
          kubectl apply -f k8s/configmap.yaml
          kubectl apply -f k8s/secret.yaml
          kubectl apply -f k8s/deployment.yaml
          kubectl apply -f k8s/service.yaml
          kubectl rollout status deployment/model-builder -n sentiment-analysis
```

## 📊 Monitoring

### **Health Checks**

```python
@app.get("/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with dependencies."""
    checks = {
        "mlflow": check_mlflow_connection(),
        "gpu": check_gpu_availability(),
        "storage": check_storage_access()
    }

    overall_status = "healthy" if all(checks.values()) else "unhealthy"

    return {
        "status": overall_status,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }
```

### **Prometheus Metrics**

```python
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
TRAINING_JOBS = Gauge('training_jobs_active', 'Number of active training jobs')
GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'GPU utilization percentage')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_DURATION.observe(duration)

    return response
```

### **Grafana Dashboard**

```json
{
  "dashboard": {
    "title": "Sentiment Model Builder",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Active Training Jobs",
        "type": "stat",
        "targets": [
          {
            "expr": "training_jobs_active"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "gpu_utilization_percent"
          }
        ]
      }
    ]
  }
}
```

## 🔧 Troubleshooting

### **Common Issues**

#### **1. CUDA Out of Memory**

```bash
# Reduce batch size
export TRAINING_BATCH_SIZE=8
export TRAINING_EVAL_BATCH_SIZE=16

# Enable gradient accumulation
export TRAINING_GRADIENT_ACCUMULATION_STEPS=2

# Enable gradient checkpointing
export TRAINING_GRADIENT_CHECKPOINTING=true

# Use mixed precision
export TRAINING_FP16=true
```

#### **2. MLflow Connection Issues**

```bash
# Check MLflow server
curl http://mlflow:5000/health

# Test connection
python -c "
import mlflow
mlflow.set_tracking_uri('http://mlflow:5000')
print('MLflow connection successful')
"
```

#### **3. Data Loading Errors**

```bash
# Validate data format
sentiment-validate-data data/news_dataset.json \
    --format json \
    --text-column text \
    --label-column label

# Check file permissions
ls -la data/news_dataset.json
```

### **Debug Mode**

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
sentiment-train --log-level DEBUG

# Check system resources
htop
nvidia-smi
df -h
```

### **Log Analysis**

```bash
# View recent logs
docker-compose logs -f sentiment-model-builder

# Filter logs by level
docker-compose logs sentiment-model-builder | grep ERROR

# Export logs
docker-compose logs sentiment-model-builder > logs.txt
```

### **Performance Tuning**

```bash
# Enable mixed precision training
export TRAINING_FP16=true

# Enable gradient checkpointing
export TRAINING_GRADIENT_CHECKPOINTING=true

# Optimize data loading
export TRAINING_DATALOADER_NUM_WORKERS=4
export TRAINING_PIN_MEMORY=true

# Use gradient accumulation
export TRAINING_GRADIENT_ACCUMULATION_STEPS=2
```

---

**For more information, see the [API Documentation](api.md), [Configuration Guide](configuration.md), and [Architecture Documentation](architecture.md).**
