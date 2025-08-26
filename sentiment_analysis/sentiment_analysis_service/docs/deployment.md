# Deployment Guide - FinSight Sentiment Analysis Service

## Overview

This guide covers deployment strategies for the FinSight Sentiment Analysis Service across different environments, from local development to production Kubernetes clusters. The service is designed to be deployed as a containerized microservice with comprehensive monitoring and health checks.

## Prerequisites

### Software Requirements

- **Python 3.9+** with pip
- **Docker 20.10+** and Docker Compose 2.0+
- **MongoDB 5.0+** (local or remote)
- **RabbitMQ 3.8+** (local or remote)
- **OpenAI API Key** for sentiment analysis
- **Git** for version control

### System Requirements

- **CPU**: Minimum 2 cores, recommended 4+ cores
- **Memory**: Minimum 2GB RAM, recommended 4GB+ RAM
- **Storage**: Minimum 10GB available disk space
- **Network**: Internet access for OpenAI API calls

### External Dependencies

- **OpenAI API**: Active API key with sufficient credits
- **MongoDB**: Accessible database instance
- **RabbitMQ**: Message broker instance
- **Network**: Proper firewall rules for service communication

## Local Development Deployment

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd sentiment_analysis/sentiment_analysis_service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration Setup

```bash
# Copy environment template
cp env.example .env

# Edit .env with your configuration
nano .env
```

**Required .env configuration:**

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=finsight_news

# RabbitMQ Configuration
RABBITMQ_URL=amqp://guest:guest@localhost:5672/

# Service Configuration
DEBUG=true
LOG_LEVEL=DEBUG
ENABLE_CACHING=false
```

### 3. Start Dependencies

```bash
# Start MongoDB (if not running)
mongod --dbpath /path/to/data/db

# Start RabbitMQ (if not running)
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management

# Or use Docker Compose for dependencies
docker-compose -f docker-compose.deps.yml up -d
```

### 4. Run the Service

```bash
# Run with Python directly
python -m src.main

# Or with uvicorn
uvicorn src.main:app --host 0.0.0.0 --port 8002 --reload

# Or with the built-in runner
python src/main.py
```

### 5. Verify Deployment

```bash
# Health check
curl http://localhost:8002/health

# Test sentiment analysis
curl -X POST http://localhost:8002/api/v1/sentiment/test \
  -H "Content-Type: application/json" \
  -d '{"text": "Bitcoin adoption continues to grow worldwide"}'

# Check API documentation
open http://localhost:8002/docs
```

## Docker Deployment

### 1. Basic Docker Setup

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY common/ ./common/

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the application
CMD ["python", "-m", "src.main"]
```

### 2. Build and Run

```bash
# Build the image
docker build -t finsight-sentiment-service .

# Run the container
docker run -d \
  --name sentiment-service \
  -p 8002:8002 \
  --env-file .env \
  finsight-sentiment-service

# Check container status
docker ps
docker logs sentiment-service
```

### 3. Docker Compose Setup

```yaml
# docker-compose.yml
version: "3.8"

services:
  sentiment-analysis-service:
    build: .
    ports:
      - "8002:8002"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MONGODB_URL=mongodb://mongodb:27017
      - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
      - DEBUG=false
      - LOG_LEVEL=INFO
    env_file:
      - .env
    depends_on:
      - mongodb
      - rabbitmq
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  mongodb:
    image: mongo:6.0
    ports:
      - "27017:27017"
    environment:
      - MONGO_INITDB_DATABASE=finsight_news
    volumes:
      - mongodb_data:/data/db
    restart: unless-stopped

  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      - RABBITMQ_DEFAULT_USER=guest
      - RABBITMQ_DEFAULT_PASS=guest
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    restart: unless-stopped

volumes:
  mongodb_data:
  rabbitmq_data:
```

### 4. Docker Compose Deployment

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f sentiment-analysis-service

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Kubernetes Deployment

### 1. Namespace Setup

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: sentiment-analysis
  labels:
    name: sentiment-analysis
    app: finsight
```

### 2. Configuration Management

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

### 3. Secrets Management

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

### 4. Service Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-analysis-service
  namespace: sentiment-analysis
  labels:
    app: sentiment-analysis-service
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentiment-analysis-service
  template:
    metadata:
      labels:
        app: sentiment-analysis-service
        version: v1
    spec:
      containers:
        - name: sentiment-analysis-service
          image: finsight/sentiment-analysis-service:latest
          imagePullPolicy: Always
          ports:
            - containerPort: 8002
              name: http
          envFrom:
            - configMapRef:
                name: sentiment-analysis-config
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: sentiment-analysis-secrets
                  key: OPENAI_API_KEY
            - name: MONGODB_URL
              valueFrom:
                secretKeyRef:
                  name: sentiment-analysis-secrets
                  key: MONGODB_URL
            - name: RABBITMQ_URL
              valueFrom:
                secretKeyRef:
                  name: sentiment-analysis-secrets
                  key: RABBITMQ_URL
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8002
            initialDelaySeconds: 60
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health
              port: 8002
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          volumeMounts:
            - name: logs
              mountPath: /app/logs
      volumes:
        - name: logs
          emptyDir: {}
      restartPolicy: Always
```

### 5. Service Definition

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: sentiment-analysis-service
  namespace: sentiment-analysis
  labels:
    app: sentiment-analysis-service
spec:
  type: ClusterIP
  ports:
    - port: 80
      targetPort: 8002
      protocol: TCP
      name: http
  selector:
    app: sentiment-analysis-service
```

### 6. Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sentiment-analysis-ingress
  namespace: sentiment-analysis
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  rules:
    - host: sentiment-api.finsight.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: sentiment-analysis-service
                port:
                  number: 80
  tls:
    - hosts:
        - sentiment-api.finsight.com
      secretName: sentiment-analysis-tls
```

### 7. Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sentiment-analysis-hpa
  namespace: sentiment-analysis
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sentiment-analysis-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
```

### 8. Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Apply configuration
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# Deploy service
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Apply ingress (if using)
kubectl apply -f k8s/ingress.yaml

# Apply HPA
kubectl apply -f k8s/hpa.yaml

# Check deployment status
kubectl get pods -n sentiment-analysis
kubectl get services -n sentiment-analysis
kubectl get ingress -n sentiment-analysis
```

## CI/CD Pipeline Deployment

### 1. GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy Sentiment Analysis Service

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/sentiment-analysis-service

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
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov

      - name: Run tests
        run: |
          pytest --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'

    steps:
      - uses: actions/checkout@v3

      - name: Deploy to staging
        run: |
          echo "Deploying to staging environment"
          # Add staging deployment logic

  deploy-production:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v3

      - name: Deploy to production
        run: |
          echo "Deploying to production environment"
          # Add production deployment logic
```

### 2. Jenkins Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'finsight/sentiment-analysis-service'
        DOCKER_TAG = "${env.BUILD_NUMBER}"
        KUBECONFIG = credentials('kubeconfig')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Test') {
            steps {
                sh 'python -m pip install --upgrade pip'
                sh 'pip install -r requirements.txt'
                sh 'pip install pytest pytest-asyncio pytest-cov'
                sh 'pytest --cov=src --cov-report=xml'
            }
            post {
                always {
                    publishCoverage adapters: [coberturaAdapter('coverage.xml')]
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}")
                    docker.build("${DOCKER_IMAGE}:latest")
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                script {
                    docker.withRegistry('https://registry.example.com', 'registry-credentials') {
                        docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").push()
                        docker.image("${DOCKER_IMAGE}:latest").push()
                    }
                }
            }
        }

        stage('Deploy to Staging') {
            when {
                branch 'develop'
            }
            steps {
                sh 'kubectl apply -f k8s/namespace.yaml'
                sh 'kubectl apply -f k8s/configmap.yaml'
                sh 'kubectl apply -f k8s/secret.yaml'
                sh 'kubectl apply -f k8s/deployment.yaml'
                sh 'kubectl apply -f k8s/service.yaml'
            }
        }

        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                sh 'kubectl apply -f k8s/namespace.yaml'
                sh 'kubectl apply -f k8s/configmap.yaml'
                sh 'kubectl apply -f k8s/secret.yaml'
                sh 'kubectl apply -f k8s/deployment.yaml'
                sh 'kubectl apply -f k8s/service.yaml'
                sh 'kubectl apply -f k8s/ingress.yaml'
                sh 'kubectl apply -f k8s/hpa.yaml'
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}
```

## Monitoring and Health Checks

### 1. Health Check Endpoints

```bash
# Service health
curl http://localhost:8002/health

# Sentiment service health
curl http://localhost:8002/api/v1/sentiment/health

# Service metrics
curl http://localhost:8002/metrics
```

### 2. Prometheus Metrics

```yaml
# k8s/prometheus-service-monitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: sentiment-analysis-monitor
  namespace: sentiment-analysis
spec:
  selector:
    matchLabels:
      app: sentiment-analysis-service
  endpoints:
    - port: http
      interval: 30s
      path: /metrics
```

### 3. Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Sentiment Analysis Service",
    "panels": [
      {
        "title": "Service Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"sentiment-analysis-service\"}",
            "legendFormat": "Service Status"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"sentiment-analysis-service\"}[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      }
    ]
  }
}
```

## Troubleshooting

### 1. Common Issues

**Service won't start:**

```bash
# Check logs
docker logs sentiment-service
kubectl logs -n sentiment-analysis deployment/sentiment-analysis-service

# Check configuration
kubectl describe configmap sentiment-analysis-config -n sentiment-analysis
kubectl describe secret sentiment-analysis-secrets -n sentiment-analysis
```

**Health check failures:**

```bash
# Check service endpoints
curl -v http://localhost:8002/health

# Check dependencies
kubectl get pods -n sentiment-analysis
kubectl describe pod <pod-name> -n sentiment-analysis
```

**OpenAI API errors:**

```bash
# Verify API key
echo $OPENAI_API_KEY

# Check API quota
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

### 2. Performance Tuning

**Increase concurrency:**

```bash
# Update deployment
kubectl patch deployment sentiment-analysis-service \
  -n sentiment-analysis \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"sentiment-analysis-service","env":[{"name":"MAX_CONCURRENT_ANALYSIS","value":"20"}]}]}}}}'
```

**Scale horizontally:**

```bash
# Scale deployment
kubectl scale deployment sentiment-analysis-service \
  -n sentiment-analysis \
  --replicas=5
```

**Resource limits:**

```bash
# Update resource limits
kubectl patch deployment sentiment-analysis-service \
  -n sentiment-analysis \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"sentiment-analysis-service","resources":{"requests":{"memory":"1Gi","cpu":"500m"},"limits":{"memory":"2Gi","cpu":"1000m"}}}]}}}}'
```

## Security Considerations

### 1. Network Security

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: sentiment-analysis-network-policy
  namespace: sentiment-analysis
spec:
  podSelector:
    matchLabels:
      app: sentiment-analysis-service
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8002
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: mongodb
      ports:
        - protocol: TCP
          port: 27017
    - to:
        - namespaceSelector:
            matchLabels:
              name: rabbitmq
      ports:
        - protocol: TCP
          port: 5672
    - to: []
      ports:
        - protocol: TCP
          port: 443
```

### 2. Pod Security

```yaml
# k8s/pod-security.yaml
apiVersion: v1
kind: Pod
metadata:
  name: sentiment-analysis-service
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
  containers:
    - name: sentiment-analysis-service
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop:
            - ALL
```

## Backup and Recovery

### 1. Data Backup

```bash
# MongoDB backup
mongodump --uri="mongodb://localhost:27017/finsight_news" --out=/backup

# Configuration backup
kubectl get configmap sentiment-analysis-config \
  -n sentiment-analysis \
  -o yaml > config-backup.yaml

kubectl get secret sentiment-analysis-secrets \
  -n sentiment-analysis \
  -o yaml > secrets-backup.yaml
```

### 2. Service Recovery

```bash
# Rollback deployment
kubectl rollout undo deployment/sentiment-analysis-service \
  -n sentiment-analysis

# Restore from backup
kubectl apply -f config-backup.yaml
kubectl apply -f secrets-backup.yaml
```

## Performance Optimization

### 1. Resource Optimization

```yaml
# k8s/resource-optimization.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-analysis-service
spec:
  template:
    spec:
      containers:
        - name: sentiment-analysis-service
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          env:
            - name: MAX_CONCURRENT_ANALYSIS
              value: "5"
            - name: BATCH_SIZE
              value: "5"
            - name: ENABLE_CACHING
              value: "true"
            - name: CACHE_TTL_SECONDS
              value: "1800"
```

### 2. Scaling Policies

```yaml
# k8s/scaling-policy.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sentiment-analysis-hpa
spec:
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 20
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
```

This deployment guide provides comprehensive coverage of all deployment scenarios for the FinSight Sentiment Analysis Service, from local development to production Kubernetes clusters with proper monitoring, security, and performance optimization.
