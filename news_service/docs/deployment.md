# Deployment Guide

## Overview

This guide covers deployment strategies for the FinSight News Service across different environments, from local development to production Kubernetes clusters.

## Prerequisites

### System Requirements

- **CPU**: 2+ cores (4+ for production)
- **Memory**: 4GB+ RAM (8GB+ for production)
- **Storage**: 20GB+ available disk space
- **Network**: Stable internet connection for external APIs

### Software Dependencies

- **Docker**: 20.10+ (for containerized deployment)
- **Docker Compose**: 2.0+ (for local orchestration)
- **Kubernetes**: 1.24+ (for production deployment)
- **kubectl**: Latest version
- **Helm**: 3.8+ (for Kubernetes package management)

### External Dependencies

- **MongoDB**: 5.0+ (local or cloud)
- **Redis**: 6.0+ (local or cloud)
- **RabbitMQ**: 3.8+ (local or cloud)
- **Eureka Server**: 2.0+ (for service discovery)

## Local Development Deployment

### 1. Docker Compose Setup

#### Basic Development Environment

```yaml
# docker-compose.dev.yml
version: "3.8"

services:
  # News Service
  news-service:
    build:
      context: .
      dockerfile: Dockerfile
      target: development
    ports:
      - "8000:8000"
      - "50051:50051"
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
      - LOG_LEVEL=DEBUG
      - DATABASE_ENVIRONMENT=local
      - MONGODB_LOCAL_URL=mongodb://mongodb:27017
      - REDIS_HOST=redis
      - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
      - ENABLE_EUREKA_CLIENT=false
    volumes:
      - ./src:/app/src
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      - mongodb
      - redis
      - rabbitmq
    networks:
      - news-service-network

  # MongoDB
  mongodb:
    image: mongo:6.0
    ports:
      - "27017:27017"
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=password
    volumes:
      - mongodb_data:/data/db
    networks:
      - news-service-network

  # Redis
  redis:
    image: redis:7.0-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - news-service-network

  # RabbitMQ
  rabbitmq:
    image: rabbitmq:3.11-management
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      - RABBITMQ_DEFAULT_USER=guest
      - RABBITMQ_DEFAULT_PASS=guest
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    networks:
      - news-service-network

volumes:
  mongodb_data:
  redis_data:
  rabbitmq_data:

networks:
  news-service-network:
    driver: bridge
```

#### Start Development Environment

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f news-service

# Stop services
docker-compose -f docker-compose.dev.yml down
```

### 2. Local Python Development

#### Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

#### Environment Configuration

```bash
# Copy environment template
cp env.example .env

# Edit configuration
nano .env
```

#### Start Dependencies

```bash
# Start MongoDB, Redis, RabbitMQ
docker-compose -f docker-compose.dev.yml up -d mongodb redis rabbitmq

# Or use individual containers
docker run -d --name mongodb -p 27017:27017 mongo:6.0
docker run -d --name redis -p 6379:6379 redis:7.0-alpine
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3.11-management
```

#### Run Service

```bash
# Run with Python
python -m src.main

# Run with uvicorn
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Run with gunicorn (production-like)
gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Docker Deployment

### 1. Production Docker Image

#### Multi-stage Dockerfile

```dockerfile
# Multi-stage Dockerfile for FinSight News Service
FROM python:3.12-slim AS builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION="1.0.0"
ARG VCS_REF

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy and install common module first
COPY common_module /opt/common
WORKDIR /opt/common
RUN pip install --no-deps -e .

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Production stage
FROM python:3.12-slim AS production

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_ENV=production \
    HOST=0.0.0.0 \
    PORT=8000 \
    GRPC_PORT=50051

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy common module source to maintain editable installation
COPY --from=builder /opt/common /opt/common

# Create application directory and set permissions
WORKDIR /app
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Copy application code with proper ownership
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser src/*.py /app/
COPY --chown=appuser:appuser requirements.txt /app/
COPY --chown=appuser:appuser scripts/docker-entrypoint.sh /app/docker-entrypoint.sh

# Create required directories and set permissions
RUN mkdir -p /app/logs /app/data && \
    chmod +x /app/docker-entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose ports
EXPOSE ${PORT} ${GRPC_PORT}

# Default commands
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["server"]
```

#### Build and Push Image

```bash
# Build image
docker build -t finsight/news-service:latest .

# Tag for registry
docker tag finsight/news-service:latest your-registry.com/finsight/news-service:v1.0.0

# Push to registry
docker push your-registry.com/finsight/news-service:v1.0.0
```

### 2. Production Docker Compose

#### Production Environment

```yaml
# docker-compose.prod.yml
version: "3.8"

services:
  news-service:
    image: your-registry.com/finsight/news-service:v1.0.0
    container_name: news-service
    ports:
      - "8000:8000"
      - "50051:50051"
    env_file:
      - .env.prod
    environment:
      - APP_ENV=production
      - HOST=0.0.0.0
      - PORT=8000
      - GRPC_PORT=50051
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped
    command: ["server"]
    networks:
      - news-service-network

  # Optional: Local MongoDB for testing
  mongodb:
    image: mongo:6.0
    container_name: mongodb
    ports:
      - "27017:27017"
    environment:
      - MONGO_INITDB_ROOT_USERNAME=${MONGODB_USERNAME}
      - MONGO_INITDB_ROOT_PASSWORD=${MONGODB_PASSWORD}
    volumes:
      - mongodb_data:/data/db
    networks:
      - news-service-network
    profiles:
      - local-db

  # Optional: Local Redis for testing
  redis:
    image: redis:7.0-alpine
    container_name: redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - news-service-network
    profiles:
      - local-cache

  # Optional: Local RabbitMQ for testing
  rabbitmq:
    image: rabbitmq:3.11-management
    container_name: rabbitmq
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      - RABBITMQ_DEFAULT_USER=${RABBITMQ_USERNAME}
      - RABBITMQ_DEFAULT_PASS=${RABBITMQ_PASSWORD}
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    networks:
      - news-service-network
    profiles:
      - local-mq

networks:
  news-service-network:
    driver: bridge
    name: news-service-network

volumes:
  mongodb_data:
    driver: local
  redis_data:
    driver: local
  rabbitmq_data:
    driver: local
```

#### Deploy Production

```bash
# Deploy with external dependencies
docker-compose -f docker-compose.prod.yml up -d

# Deploy with local dependencies
docker-compose -f docker-compose.prod.yml --profile local-db --profile local-cache --profile local-mq up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f news-service

# Scale service
docker-compose -f docker-compose.prod.yml up -d --scale news-service=3
```

## Kubernetes Deployment

### 1. Helm Chart Structure

#### Chart Directory Structure

```bash
helm/
├── Chart.yaml
├── values.yaml
├── values-dev.yaml
├── values-prod.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── hpa.yaml
│   ├── pdb.yaml
│   └── _helpers.tpl
└── charts/
```

#### Chart.yaml

```yaml
apiVersion: v2
name: news-service
description: FinSight News Service Helm Chart
type: application
version: 1.0.0
appVersion: "1.0.0"
keywords:
  - news
  - fintech
  - api
home: https://github.com/finsight/news-service
sources:
  - https://github.com/finsight/news-service
maintainers:
  - name: FinSight Team
    email: team@finsight.com
```

#### values.yaml

```yaml
# Default values for news-service
replicaCount: 3

image:
  repository: your-registry.com/finsight/news-service
  tag: "1.0.0"
  pullPolicy: IfNotPresent

imagePullSecrets:
  - name: registry-secret

nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podAnnotations: {}

podSecurityContext:
  fsGroup: 1000
  runAsUser: 1000
  runAsGroup: 1000

securityContext:
  capabilities:
    drop:
      - ALL
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: 1000

service:
  type: ClusterIP
  port: 8000
  grpcPort: 50051

ingress:
  enabled: true
  className: "nginx"
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: news-service.finsight.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: news-service-tls
      hosts:
        - news-service.finsight.com

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}

tolerations: []

affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchExpressions:
              - key: app.kubernetes.io/name
                operator: In
                values:
                  - news-service
          topologyKey: kubernetes.io/hostname

env:
  APP_NAME: "news-service"
  ENVIRONMENT: "production"
  HOST: "0.0.0.0"
  PORT: "8000"
  GRPC_PORT: "50051"
  LOG_LEVEL: "INFO"
  ENABLE_GRPC: "true"
  ENABLE_EUREKA_CLIENT: "true"
  DATABASE_ENVIRONMENT: "cloud"
  ENABLE_CACHING: "true"
  CRON_JOB_ENABLED: "true"

secrets:
  mongodb_cloud_url: ""
  redis_password: ""
  rabbitmq_url: ""
  tavily_api_key: ""
  secret_api_key: ""
  eureka_server_url: ""

configMap:
  mongodb_cloud_database: "finsight_news"
  redis_host: "redis-cluster.example.com"
  redis_port: "6379"
  eureka_app_name: "news-service"

persistence:
  enabled: false
  storageClass: ""
  accessMode: ReadWriteOnce
  size: 10Gi

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s
  prometheusRule:
    enabled: true
```

### 2. Kubernetes Manifests

#### Deployment

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "news-service.fullname" . }}
  labels:
    {{- include "news-service.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "news-service.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "news-service.selectorLabels" . | nindent 8 }}
      {{- with .Values.podAnnotations }}
      annotations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "news-service.serviceAccountName" . }}
      securityContext:
        {{- toYaml .Values.podSecurityContext | nindent 8 }}
      containers:
        - name: {{ .Chart.Name }}
          securityContext:
            {{- toYaml .Values.securityContext | nindent 12 }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: {{ .Values.service.port }}
              protocol: TCP
            - name: grpc
              containerPort: {{ .Values.service.grpcPort }}
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 60
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          env:
            {{- range $key, $value := .Values.env }}
            - name: {{ $key }}
              value: {{ $value | quote }}
            {{- end }}
            {{- range $key, $value := .Values.secrets }}
            - name: {{ $key }}
              valueFrom:
                secretKeyRef:
                  name: {{ include "news-service.fullname" $ }}-secrets
                  key: {{ $key }}
            {{- end }}
            {{- range $key, $value := .Values.configMap }}
            - name: {{ $key }}
              valueFrom:
                configMapKeyRef:
                  name: {{ include "news-service.fullname" $ }}-config
                  key: {{ $key }}
            {{- end }}
          {{- if .Values.persistence.enabled }}
          volumeMounts:
            - name: data
              mountPath: /app/data
            - name: logs
              mountPath: /app/logs
          {{- end }}
      {{- if .Values.persistence.enabled }}
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: {{ include "news-service.fullname" . }}-data
        - name: logs
          persistentVolumeClaim:
            claimName: {{ include "news-service.fullname" . }}-logs
      {{- end }}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
```

#### Service

```yaml
# templates/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: { { include "news-service.fullname" . } }
  labels: { { - include "news-service.labels" . | nindent 4 } }
spec:
  type: { { .Values.service.type } }
  ports:
    - port: { { .Values.service.port } }
      targetPort: http
      protocol: TCP
      name: http
    - port: { { .Values.service.grpcPort } }
      targetPort: grpc
      protocol: TCP
      name: grpc
  selector: { { - include "news-service.selectorLabels" . | nindent 4 } }
```

#### Horizontal Pod Autoscaler

```yaml
# templates/hpa.yaml
{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "news-service.fullname" . }}
  labels:
    {{- include "news-service.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "news-service.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    {{- if .Values.autoscaling.targetCPUUtilizationPercentage }}
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
    {{- end }}
    {{- if .Values.autoscaling.targetMemoryUtilizationPercentage }}
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetMemoryUtilizationPercentage }}
    {{- end }}
{{- end }}
```

### 3. Deploy to Kubernetes

#### Install Helm Chart

```bash
# Add Helm repository (if using external repo)
helm repo add finsight https://charts.finsight.com
helm repo update

# Install chart
helm install news-service ./helm \
  --namespace finsight \
  --create-namespace \
  --values values-prod.yaml

# Upgrade existing deployment
helm upgrade news-service ./helm \
  --namespace finsight \
  --values values-prod.yaml

# Uninstall
helm uninstall news-service --namespace finsight
```

#### Environment-Specific Values

##### Development Values (values-dev.yaml)

```yaml
replicaCount: 1

image:
  tag: "latest"

resources:
  limits:
    cpu: 500m
    memory: 1Gi
  requests:
    cpu: 250m
    memory: 512Mi

autoscaling:
  enabled: false

env:
  ENVIRONMENT: "development"
  DEBUG: "true"
  LOG_LEVEL: "DEBUG"
  ENABLE_EUREKA_CLIENT: "false"

ingress:
  enabled: false
```

##### Production Values (values-prod.yaml)

```yaml
replicaCount: 3

image:
  tag: "1.0.0"

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10

env:
  ENVIRONMENT: "production"
  DEBUG: "false"
  LOG_LEVEL: "INFO"
  ENABLE_EUREKA_CLIENT: "true"

ingress:
  enabled: true
```

## CI/CD Pipeline

### 1. GitHub Actions Workflow

#### .github/workflows/deploy.yml

```yaml
name: Deploy News Service

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/news-service

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run tests
        run: |
          pytest tests/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-dev:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: development

    steps:
      - uses: actions/checkout@v4

      - name: Set up kubectl
        uses: azure/setup-kubectl@v3
        with:
          version: "latest"

      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBE_CONFIG_DEV }}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig

      - name: Deploy to development
        run: |
          helm upgrade --install news-service ./helm \
            --namespace finsight-dev \
            --create-namespace \
            --values values-dev.yaml \
            --set image.tag=${{ github.sha }}

  deploy-prod:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production

    steps:
      - uses: actions/checkout@v4

      - name: Set up kubectl
        uses: azure/setup-kubectl@v3
        with:
          version: "latest"

      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBE_CONFIG_PROD }}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig

      - name: Deploy to production
        run: |
          helm upgrade --install news-service ./helm \
            --namespace finsight \
            --create-namespace \
            --values values-prod.yaml \
            --set image.tag=${{ github.sha }}

      - name: Verify deployment
        run: |
          kubectl rollout status deployment/news-service -n finsight --timeout=300s
          kubectl get pods -n finsight -l app.kubernetes.io/name=news-service
```

### 2. ArgoCD Configuration

#### Application Manifest

```yaml
# argocd-app.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: news-service
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/finsight/news-service
    targetRevision: main
    path: helm
    helm:
      valueFiles:
        - values-prod.yaml
      parameters:
        - name: image.tag
          value: "1.0.0"
  destination:
    server: https://kubernetes.default.svc
    namespace: finsight
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

## Production Deployment Checklist

### Pre-deployment

- [ ] **Environment Configuration**

  - [ ] All environment variables configured
  - [ ] Secrets properly stored in Kubernetes
  - [ ] ConfigMaps created and validated
  - [ ] Database connection strings verified

- [ ] **Infrastructure Readiness**

  - [ ] Kubernetes cluster provisioned
  - [ ] Load balancer configured
  - [ ] SSL certificates installed
  - [ ] Monitoring stack deployed
  - [ ] Logging infrastructure ready

- [ ] **Dependencies**
  - [ ] MongoDB cluster accessible
  - [ ] Redis cluster accessible
  - [ ] RabbitMQ cluster accessible
  - [ ] Eureka server running
  - [ ] External APIs configured

### Deployment Checklist

- [ ] **Application Deployment**

  - [ ] Docker image built and pushed
  - [ ] Helm chart deployed
  - [ ] Services exposed correctly
  - [ ] Ingress configured
  - [ ] Health checks passing

- [ ] **Verification**
  - [ ] Pods running successfully
  - [ ] Services responding
  - [ ] API endpoints accessible
  - [ ] gRPC endpoints working
  - [ ] Database connections established

### Post-deployment

- [ ] **Monitoring**

  - [ ] Metrics collection working
  - [ ] Alerts configured
  - [ ] Dashboards accessible
  - [ ] Log aggregation functional

- [ ] **Testing**
  - [ ] Health endpoints responding
  - [ ] API functionality verified
  - [ ] Load testing completed
  - [ ] Integration tests passing

## Monitoring & Observability

### 1. Prometheus Configuration

#### ServiceMonitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: news-service
  namespace: finsight
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: news-service
  endpoints:
    - port: http
      path: /metrics
      interval: 30s
      scrapeTimeout: 10s
```

#### PrometheusRule

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: news-service-alerts
  namespace: finsight
spec:
  groups:
    - name: news-service
      rules:
        - alert: NewsServiceDown
          expr: up{app="news-service"} == 0
          for: 1m
          labels:
            severity: critical
          annotations:
            summary: "News Service is down"
            description: "News Service has been down for more than 1 minute"

        - alert: HighErrorRate
          expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
          for: 2m
          labels:
            severity: warning
          annotations:
            summary: "High error rate detected"
            description: "Error rate is above 10% for the last 5 minutes"

        - alert: HighResponseTime
          expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "High response time detected"
            description: "95th percentile response time is above 1 second"
```

### 2. Grafana Dashboard

#### Dashboard Configuration

```json
{
  "dashboard": {
    "title": "News Service Dashboard",
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
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          }
        ]
      }
    ]
  }
}
```

## Troubleshooting

### Common Issues

#### 1. Pod Startup Issues

```bash
# Check pod status
kubectl get pods -n finsight -l app.kubernetes.io/name=news-service

# Check pod logs
kubectl logs -n finsight deployment/news-service

# Check pod events
kubectl describe pod -n finsight <pod-name>
```

#### 2. Service Connectivity

```bash
# Check service endpoints
kubectl get endpoints -n finsight news-service

# Test service connectivity
kubectl run test-pod --image=curlimages/curl -i --rm --restart=Never -- \
  curl http://news-service:8000/health
```

#### 3. Database Connection Issues

```bash
# Check database connectivity from pod
kubectl exec -n finsight deployment/news-service -- \
  python -c "from src.core.config import settings; print(settings.mongodb_url)"
```

#### 4. Resource Issues

```bash
# Check resource usage
kubectl top pods -n finsight

# Check resource limits
kubectl describe pod -n finsight <pod-name> | grep -A 10 "Limits:"
```

### Debug Commands

#### Health Check

```bash
# Check service health
curl -f http://localhost:8000/health

# Check detailed health
curl http://localhost:8000/health | jq .
```

#### Metrics

```bash
# Get service metrics
curl http://localhost:8000/metrics

# Check specific metrics
curl http://localhost:8000/metrics | grep http_requests_total
```

#### Logs

```bash
# Follow logs
kubectl logs -n finsight -f deployment/news-service

# Get logs with timestamps
kubectl logs -n finsight deployment/news-service --timestamps

# Get logs for specific time range
kubectl logs -n finsight deployment/news-service --since=1h
```

## Security Considerations

### 1. Network Security

- Use network policies to restrict pod communication
- Implement TLS for all external communications
- Use service mesh for advanced traffic management
- Configure proper firewall rules

### 2. Access Control

- Use RBAC for Kubernetes access control
- Implement proper service accounts
- Use secrets for sensitive data
- Enable audit logging

### 3. Container Security

- Run containers as non-root users
- Use read-only root filesystems
- Implement security contexts
- Scan images for vulnerabilities

### 4. Data Protection

- Encrypt data at rest and in transit
- Implement proper backup strategies
- Use secure communication protocols
- Monitor for data breaches

## Backup & Recovery

### 1. Database Backup

```bash
# MongoDB backup
mongodump --uri="mongodb://username:password@host:port/database" \
  --out=/backup/$(date +%Y%m%d_%H%M%S)

# Redis backup
redis-cli -h host -p port -a password BGSAVE
```

### 2. Configuration Backup

```bash
# Backup Kubernetes resources
kubectl get all -n finsight -o yaml > backup-$(date +%Y%m%d).yaml

# Backup secrets
kubectl get secrets -n finsight -o yaml > secrets-backup-$(date +%Y%m%d).yaml
```

### 3. Disaster Recovery

- Maintain multiple replicas across availability zones
- Use persistent volumes for data storage
- Implement automated backup procedures
- Test recovery procedures regularly
