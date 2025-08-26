# Deployment Guide

## Overview

This guide covers deployment strategies for the FinSight Market Dataset Service from local development to production Kubernetes clusters.

## Prerequisites

### System Requirements

- **CPU**: 2+ cores (4+ for production)
- **Memory**: 4GB+ RAM (8GB+ for production)
- **Storage**: 20GB+ available disk space
- **Network**: Stable internet connection

### Software Dependencies

- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Kubernetes**: 1.24+
- **kubectl**: Latest version
- **Helm**: 3.8+

### External Dependencies

- **MongoDB**: 5.0+ (local or cloud)
- **InfluxDB**: 2.0+ (optional, for time-series data)
- **Redis**: 6.0+ (optional, for caching)
- **MinIO/S3**: Object storage
- **Eureka Server**: 2.0+ (optional, for service discovery)

## Local Development Deployment

### 1. Docker Compose Setup

#### Development Environment

```yaml
# docker-compose.yml
version: "3.8"

services:
  market-dataset-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=true
      - ENVIRONMENT=development
      - REPOSITORY_TYPE=csv
      - STORAGE_PROVIDER=minio
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - mongodb
      - minio
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  mongodb:
    image: mongo:5.0
    ports:
      - "27017:27017"
    environment:
      - MONGO_INITDB_DATABASE=finsight_market_data
    volumes:
      - mongodb_data:/data/db

  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

volumes:
  mongodb_data:
  minio_data:
```

#### Production Environment

```yaml
# docker-compose.prod.yml
version: "3.8"

services:
  market-dataset-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=false
      - ENVIRONMENT=production
      - REPOSITORY_TYPE=mongodb
      - STORAGE_PROVIDER=minio
      - MONGODB_URL=mongodb://mongodb:27017/
      - S3_ENDPOINT_URL=http://minio:9000
      - S3_ACCESS_KEY=${S3_ACCESS_KEY}
      - S3_SECRET_KEY=${S3_SECRET_KEY}
      - API_KEY=${API_KEY}
    depends_on:
      - mongodb
      - minio
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "1.0"
        reservations:
          memory: 1G
          cpus: "0.5"
```

### 2. Environment Configuration

#### Development (.env)

```bash
# Service Configuration
APP_NAME=market-dataset-service
DEBUG=true
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8000

# Storage Configuration
STORAGE_BASE_DIRECTORY=data/market_data
REPOSITORY_TYPE=csv
STORAGE_PROVIDER=minio

# MinIO Configuration
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET_NAME=market-data

# Data Collection
DEFAULT_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT
DEFAULT_TIMEFRAMES=1h,4h,1d
MAX_OHLCV_LIMIT=1000

# Admin API
API_KEY=dev-admin-key

# Eureka (optional)
ENABLE_EUREKA_CLIENT=false
```

#### Production (.env)

```bash
# Service Configuration
APP_NAME=market-dataset-service
DEBUG=false
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000

# Storage Configuration
STORAGE_BASE_DIRECTORY=/app/data
REPOSITORY_TYPE=mongodb
STORAGE_PROVIDER=digitalocean

# MongoDB Configuration
MONGODB_URL=mongodb://user:password@mongodb:27017/
MONGODB_DATABASE=finsight_market_data

# DigitalOcean Spaces Configuration
SPACES_ENDPOINT_URL=https://nyc3.digitaloceanspaces.com
SPACES_ACCESS_KEY=your_spaces_access_key
SPACES_SECRET_KEY=your_spaces_secret_key
SPACES_REGION_NAME=nyc3
SPACES_BUCKET_NAME=finsight-market-data

# Exchange Configuration
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key

# Admin API
API_KEY=your_secure_admin_api_key

# Cron Job
CRON_JOB_ENABLED=true
CRON_JOB_SCHEDULE=0 */4 * * *

# Eureka Service Discovery
ENABLE_EUREKA_CLIENT=true
EUREKA_SERVER_URL=http://eureka-server:8761
```

### 3. Build and Run

#### Development

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f market-dataset-service

# Access services
# API: http://localhost:8000
# MinIO Console: http://localhost:9001
# MongoDB: localhost:27017
```

#### Production

```bash
# Build and start with production config
docker-compose -f docker-compose.prod.yml up -d

# Scale service
docker-compose -f docker-compose.prod.yml up -d --scale market-dataset-service=3
```

## Kubernetes Deployment

### 1. Namespace Setup

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: finsight
  labels:
    name: finsight
```

### 2. ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: market-dataset-config
  namespace: finsight
data:
  APP_NAME: "market-dataset-service"
  ENVIRONMENT: "production"
  REPOSITORY_TYPE: "mongodb"
  STORAGE_PROVIDER: "digitalocean"
  DEFAULT_SYMBOLS: "BTCUSDT,ETHUSDT,BNBUSDT"
  DEFAULT_TIMEFRAMES: "1h,4h,1d"
  MAX_OHLCV_LIMIT: "10000"
  CRON_JOB_ENABLED: "true"
  CRON_JOB_SCHEDULE: "0 */4 * * *"
  ENABLE_EUREKA_CLIENT: "true"
```

### 3. Secret

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: market-dataset-secrets
  namespace: finsight
type: Opaque
data:
  API_KEY: <base64-encoded-api-key>
  BINANCE_API_KEY: <base64-encoded-binance-key>
  BINANCE_SECRET_KEY: <base64-encoded-binance-secret>
  SPACES_ACCESS_KEY: <base64-encoded-spaces-key>
  SPACES_SECRET_KEY: <base64-encoded-spaces-secret>
  MONGODB_URL: <base64-encoded-mongodb-url>
```

### 4. Deployment

```yaml
# deployment.yaml
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
                name: market-dataset-config
            - secretRef:
                name: market-dataset-secrets
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "2Gi"
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
          volumeMounts:
            - name: data-volume
              mountPath: /app/data
            - name: logs-volume
              mountPath: /app/logs
      volumes:
        - name: data-volume
          persistentVolumeClaim:
            claimName: market-data-pvc
        - name: logs-volume
          emptyDir: {}
```

### 5. Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: market-dataset-service
  namespace: finsight
spec:
  selector:
    app: market-dataset-service
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP
```

### 6. Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: market-dataset-ingress
  namespace: finsight
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - api.finsight.com
      secretName: finsight-tls
  rules:
    - host: api.finsight.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: market-dataset-service
                port:
                  number: 80
```

### 7. Persistent Volume

```yaml
# pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: market-data-pv
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteMany
  hostPath:
    path: /data/market-dataset
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: market-data-pvc
  namespace: finsight
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
```

### 8. Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f namespace.yaml

# Apply configurations
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f pv.yaml

# Deploy application
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Check deployment
kubectl get pods -n finsight
kubectl get services -n finsight
kubectl get ingress -n finsight
```

## Helm Chart Deployment

### 1. Helm Chart Structure

```bash
market-dataset-service/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   └── _helpers.tpl
└── charts/
```

### 2. Values Configuration

```yaml
# values.yaml
replicaCount: 3

image:
  repository: finsight/market-dataset-service
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.finsight.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: finsight-tls
      hosts:
        - api.finsight.com

resources:
  requests:
    memory: 1Gi
    cpu: 500m
  limits:
    memory: 2Gi
    cpu: 1000m

config:
  environment: production
  repository_type: mongodb
  storage_provider: digitalocean
  default_symbols: "BTCUSDT,ETHUSDT,BNBUSDT"
  default_timeframes: "1h,4h,1d"
  cron_job_enabled: true
  enable_eureka_client: true

persistence:
  enabled: true
  size: 100Gi
  storageClass: ""
```

### 3. Deploy with Helm

```bash
# Add Helm repository
helm repo add finsight https://charts.finsight.com
helm repo update

# Install chart
helm install market-dataset-service finsight/market-dataset-service \
  --namespace finsight \
  --create-namespace \
  --values values.yaml

# Upgrade deployment
helm upgrade market-dataset-service finsight/market-dataset-service \
  --namespace finsight \
  --values values.yaml

# Uninstall
helm uninstall market-dataset-service -n finsight
```

## CI/CD Pipeline

### 1. GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy Market Dataset Service

on:
  push:
    branches: [main]
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
          pip install -r requirements-dev.txt

      - name: Run tests
        run: |
          pytest tests/ -v --cov=src

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

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            finsight/market-dataset-service:latest
            finsight/market-dataset-service:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v3

      - name: Set up kubectl
        uses: azure/setup-kubectl@v3
        with:
          version: "latest"

      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig

      - name: Deploy to Kubernetes
        run: |
          kubectl apply -f k8s/namespace.yaml
          kubectl apply -f k8s/configmap.yaml
          kubectl apply -f k8s/secret.yaml
          kubectl apply -f k8s/deployment.yaml
          kubectl apply -f k8s/service.yaml
          kubectl apply -f k8s/ingress.yaml

      - name: Verify deployment
        run: |
          kubectl rollout status deployment/market-dataset-service -n finsight
          kubectl get pods -n finsight
```

### 2. Jenkins Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'finsight/market-dataset-service'
        DOCKER_TAG = "${env.BUILD_NUMBER}"
        KUBE_NAMESPACE = 'finsight'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Test') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pip install -r requirements-dev.txt'
                sh 'pytest tests/ -v --cov=src'
                sh 'black --check src/'
                sh 'isort --check-only src/'
                sh 'flake8 src/'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}")
                    docker.withRegistry('https://registry.hub.docker.com', 'docker-hub-credentials') {
                        docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").push()
                        docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").push('latest')
                    }
                }
            }
        }

        stage('Deploy to Staging') {
            when {
                branch 'develop'
            }
            steps {
                sh "kubectl set image deployment/market-dataset-service market-dataset-service=${DOCKER_IMAGE}:${DOCKER_TAG} -n ${KUBE_NAMESPACE}"
                sh "kubectl rollout status deployment/market-dataset-service -n ${KUBE_NAMESPACE}"
            }
        }

        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Deploy to production?'
                sh "kubectl set image deployment/market-dataset-service market-dataset-service=${DOCKER_IMAGE}:${DOCKER_TAG} -n ${KUBE_NAMESPACE}"
                sh "kubectl rollout status deployment/market-dataset-service -n ${KUBE_NAMESPACE}"
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}
```

## Monitoring and Observability

### 1. Health Checks

```bash
# Service health
curl http://localhost:8000/health

# Detailed health
curl -H "X-API-Key: your-key" http://localhost:8000/admin/health

# Job health
curl -H "X-API-Key: your-key" http://localhost:8000/jobs/status
```

### 2. Prometheus Metrics

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s

    scrape_configs:
    - job_name: 'market-dataset-service'
      static_configs:
      - targets: ['market-dataset-service.finsight.svc.cluster.local:8000']
      metrics_path: '/metrics'
      scrape_interval: 30s
```

### 3. Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Market Dataset Service",
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
            "expr": "rate(http_requests_total{status=~\"4..|5..\"}[5m])",
            "legendFormat": "Error rate"
          }
        ]
      },
      {
        "title": "Data Collection Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(data_collection_total[5m])",
            "legendFormat": "Records/second"
          }
        ]
      }
    ]
  }
}
```

### 4. Log Aggregation

```yaml
# fluentd-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
  namespace: logging
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/market-dataset-service-*.log
      pos_file /var/log/market-dataset-service.log.pos
      tag kubernetes.*
      read_from_head true
      <parse>
        @type json
        time_format %Y-%m-%dT%H:%M:%S.%NZ
      </parse>
    </source>

    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch-master.logging.svc.cluster.local
      port 9200
      logstash_format true
      logstash_prefix market-dataset-service
      <buffer>
        @type file
        path /var/log/fluentd-buffers/kubernetes.system.buffer
        flush_mode interval
        retry_type exponential_backoff
        flush_interval 5s
        retry_forever false
        retry_max_interval 30
        chunk_limit_size 2M
        queue_limit_length 8
        overflow_action block
      </buffer>
    </match>
```

## Troubleshooting Common Issues

### 1. Service Startup Issues

#### Service Won't Start

```bash
# Check container logs
docker-compose logs market-dataset-service

# Check service status
docker-compose ps

# Verify environment variables
docker-compose exec market-dataset-service env | grep -E "(DEBUG|ENVIRONMENT|REPOSITORY_TYPE)"

# Check port conflicts
netstat -tulpn | grep :8000
```

#### Configuration Errors

```bash
# Validate configuration
curl -H "X-API-Key: your-key" http://localhost:8000/admin/health

# Check configuration endpoint
curl -H "X-API-Key: your-key" http://localhost:8000/admin/config

# Verify environment file
cat .env | grep -v "^#" | grep -v "^$"
```

### 2. Database Connection Issues

#### MongoDB Connection Problems

```bash
# Test MongoDB connectivity
docker-compose exec market-dataset-service mongosh "mongodb://mongodb:27017/finsight_market_data"

# Check MongoDB logs
docker-compose logs mongodb

# Verify MongoDB status
docker-compose exec mongodb mongosh --eval "db.serverStatus()"

# Check network connectivity
docker-compose exec market-dataset-service ping mongodb
```

#### InfluxDB Connection Issues

```bash
# Test InfluxDB connectivity
curl -G "http://localhost:8086/query" --data-urlencode "q=SHOW DATABASES"

# Check InfluxDB logs
docker-compose logs influxdb

# Verify bucket exists
curl -G "http://localhost:8086/query" --data-urlencode "q=SHOW BUCKETS"
```

### 3. Storage Issues

#### MinIO/S3 Problems

```bash
# Test MinIO connectivity
curl -f http://localhost:9000/minio/health/live

# Check MinIO logs
docker-compose logs minio

# Verify bucket exists
docker-compose exec minio mc ls /data/market-data

# Test S3 operations
aws --endpoint-url=http://localhost:9000 s3 ls s3://market-data
```

#### Storage Permission Issues

```bash
# Check file permissions
ls -la data/market_data/

# Fix permissions if needed
chmod -R 755 data/
chown -R 1000:1000 data/

# Check container user
docker-compose exec market-dataset-service id
```

### 4. Performance Issues

#### High Memory Usage

```bash
# Check memory usage
docker stats market-dataset-service

# Monitor memory in container
docker-compose exec market-dataset-service top

# Check for memory leaks
docker-compose exec market-dataset-service ps aux --sort=-%mem

# Analyze memory usage
docker-compose exec market-dataset-service python -c "
import psutil
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available / 1024**3:.2f} GB')
"
```

#### Slow Response Times

```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/health"

# Monitor database performance
docker-compose exec mongodb mongosh --eval "
db.currentOp()
db.getProfilingStatus()
"

# Check storage performance
docker-compose exec market-dataset-service iostat -x 1 5
```

### 5. Scaling Issues

#### Horizontal Scaling Problems

```bash
# Check service discovery
curl -H "X-API-Key: your-key" http://localhost:8000/eureka/status

# Verify load balancer
curl -H "Host: api.finsight.com" http://localhost/80/health

# Check service instances
kubectl get pods -n finsight -l app=market-dataset-service

# Monitor service endpoints
kubectl get endpoints -n finsight
```

#### Auto-scaling Issues

```bash
# Check HPA status
kubectl get hpa -n finsight

# Describe HPA for details
kubectl describe hpa market-dataset-service -n finsight

# Check metrics server
kubectl top pods -n finsight

# Verify custom metrics
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/finsight/pods/*/http_requests_per_second"
```

## Security Best Practices

### 1. Network Security

#### Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: market-dataset-network-policy
  namespace: finsight
spec:
  podSelector:
    matchLabels:
      app: market-dataset-service
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
          port: 8000
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: finsight
      ports:
        - protocol: TCP
          port: 27017
    - to:
        - namespaceSelector:
            matchLabels:
              name: finsight
      ports:
        - protocol: TCP
          port: 9000
    - to: []
      ports:
        - protocol: TCP
          port: 443
        - protocol: TCP
          port: 80
```

#### TLS Configuration

```yaml
# tls-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: finsight-tls
  namespace: finsight
type: kubernetes.io/tls
data:
  tls.crt: <base64-encoded-certificate>
  tls.key: <base64-encoded-private-key>
```

### 2. Access Control

#### RBAC Configuration

```yaml
# rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: market-dataset-service
  namespace: finsight
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: market-dataset-role
  namespace: finsight
rules:
  - apiGroups: [""]
    resources: ["pods", "services", "endpoints"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["configmaps", "secrets"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: market-dataset-rolebinding
  namespace: finsight
subjects:
  - kind: ServiceAccount
    name: market-dataset-service
    namespace: finsight
roleRef:
  kind: Role
  name: market-dataset-role
  apiGroup: rbac.authorization.k8s.io
```

#### API Key Management

```bash
# Generate secure API key
openssl rand -hex 32

# Rotate API key
kubectl patch secret market-dataset-secrets -n finsight \
  --type='json' -p='[{"op": "replace", "path": "/data/API_KEY", "value": "<new-base64-key>"}]'

# Update deployment
kubectl rollout restart deployment/market-dataset-service -n finsight
```

### 3. Container Security

#### Security Context

```yaml
# security-context.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: market-dataset-service
  namespace: finsight
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
      containers:
        - name: market-dataset-service
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop:
                - ALL
          volumeMounts:
            - name: tmp
              mountPath: /tmp
            - name: varlog
              mountPath: /var/log
      volumes:
        - name: tmp
          emptyDir: {}
        - name: varlog
          emptyDir: {}
```

#### Pod Security Standards

```yaml
# psp.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: market-dataset-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - "configMap"
    - "emptyDir"
    - "projected"
    - "secret"
    - "downwardAPI"
    - "persistentVolumeClaim"
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: "MustRunAsNonRoot"
  seLinux:
    rule: "RunAsAny"
  supplementalGroups:
    rule: "MustRunAs"
    ranges:
      - min: 1
        max: 65535
  fsGroup:
    rule: "MustRunAs"
    ranges:
      - min: 1
        max: 65535
  readOnlyRootFilesystem: true
```

## Backup and Recovery

### 1. Data Backup Strategy

#### MongoDB Backup

```bash
# Create backup script
#!/bin/bash
BACKUP_DIR="/backups/mongodb"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="finsight_market_data_$DATE"

# Create backup
docker-compose exec mongodb mongodump \
  --db finsight_market_data \
  --out /backups/$BACKUP_NAME

# Compress backup
tar -czf $BACKUP_DIR/$BACKUP_NAME.tar.gz -C /backups $BACKUP_NAME

# Clean up temporary files
rm -rf /backups/$BACKUP_NAME

# Upload to cloud storage
aws s3 cp $BACKUP_DIR/$BACKUP_NAME.tar.gz s3://finsight-backups/mongodb/

echo "Backup completed: $BACKUP_NAME.tar.gz"
```

#### Storage Backup

```bash
# Backup MinIO data
#!/bin/bash
BACKUP_DIR="/backups/minio"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="minio_market_data_$DATE"

# Create backup
docker-compose exec minio mc mirror /data/market-data /backups/$BACKUP_NAME

# Compress backup
tar -czf $BACKUP_DIR/$BACKUP_NAME.tar.gz -C /backups $BACKUP_NAME

# Upload to cloud storage
aws s3 cp $BACKUP_DIR/$BACKUP_NAME.tar.gz s3://finsight-backups/minio/

echo "MinIO backup completed: $BACKUP_NAME.tar.gz"
```

### 2. Recovery Procedures

#### MongoDB Recovery

```bash
# Restore MongoDB from backup
#!/bin/bash
BACKUP_FILE="$1"
RESTORE_DIR="/tmp/restore"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

# Extract backup
tar -xzf $BACKUP_FILE -C /tmp

# Stop service
docker-compose stop market-dataset-service

# Drop existing database
docker-compose exec mongodb mongosh --eval "use finsight_market_data; db.dropDatabase()"

# Restore from backup
docker-compose exec -T mongodb mongorestore --db finsight_market_data < /tmp/restore/finsight_market_data.bson

# Start service
docker-compose start market-dataset-service

# Clean up
rm -rf /tmp/restore

echo "MongoDB recovery completed"
```

#### Storage Recovery

```bash
# Restore MinIO from backup
#!/bin/bash
BACKUP_FILE="$1"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

# Stop service
docker-compose stop market-dataset-service

# Extract backup
tar -xzf $BACKUP_FILE -C /tmp

# Clear existing data
docker-compose exec minio mc rm --recursive --force /data/market-data

# Restore from backup
docker-compose exec minio mc mirror /tmp/restore /data/market-data

# Start service
docker-compose start market-dataset-service

# Clean up
rm -rf /tmp/restore

echo "MinIO recovery completed"
```

### 3. Automated Backup

#### Cron Job for Backups

```bash
# /etc/cron.daily/finsight-backup
#!/bin/bash

# Set environment
export PATH=/usr/local/bin:/usr/bin:/bin
export BACKUP_DIR="/backups"
export DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR/{mongodb,minio,logs}

# MongoDB backup
docker-compose exec -T mongodb mongodump \
  --db finsight_market_data \
  --archive > $BACKUP_DIR/mongodb/finsight_market_data_$DATE.archive

# MinIO backup
docker-compose exec minio mc mirror /data/market-data $BACKUP_DIR/minio/minio_$DATE

# Compress backups
tar -czf $BACKUP_DIR/mongodb/finsight_market_data_$DATE.tar.gz \
  -C $BACKUP_DIR/mongodb finsight_market_data_$DATE.archive

tar -czf $BACKUP_DIR/minio/minio_$DATE.tar.gz \
  -C $BACKUP_DIR/minio minio_$DATE

# Upload to cloud storage
aws s3 cp $BACKUP_DIR/mongodb/finsight_market_data_$DATE.tar.gz \
  s3://finsight-backups/mongodb/

aws s3 cp $BACKUP_DIR/minio/minio_$DATE.tar.gz \
  s3://finsight-backups/minio/

# Clean up old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
find $BACKUP_DIR -name "*.archive" -mtime +30 -delete

# Log backup completion
echo "$(date): Backup completed successfully" >> $BACKUP_DIR/logs/backup.log
```

## Support and Maintenance

### 1. Regular Maintenance Tasks

#### Daily Tasks

```bash
# Check service health
curl -f http://localhost:8000/health || echo "Service unhealthy"

# Monitor resource usage
docker stats --no-stream market-dataset-service

# Check log file sizes
du -sh logs/*.log

# Verify data freshness
curl -H "X-API-Key: your-key" http://localhost:8000/admin/health
```

#### Weekly Tasks

```bash
# Database maintenance
docker-compose exec mongodb mongosh --eval "
use finsight_market_data;
db.runCommand({compact: 'ohlcv'});
db.runCommand({compact: 'backtest_results'});
"

# Storage cleanup
docker-compose exec minio mc admin info /data

# Log rotation
logrotate -f /etc/logrotate.d/finsight

# Performance analysis
curl -H "X-API-Key: your-key" http://localhost:8000/admin/stats
```

#### Monthly Tasks

```bash
# Security updates
docker-compose pull
docker-compose up -d --force-recreate

# Backup verification
# Test restore procedures in staging environment

# Performance review
# Analyze metrics and identify optimization opportunities

# Capacity planning
# Review storage usage and plan for growth
```

### 2. Monitoring and Alerting

#### Health Check Script

```bash
#!/bin/bash
# health-check.sh

SERVICE_URL="http://localhost:8000"
API_KEY="your-api-key"
ALERT_EMAIL="admin@finsight.com"

# Check service health
HEALTH_RESPONSE=$(curl -s -w "%{http_code}" -H "X-API-Key: $API_KEY" \
  "$SERVICE_URL/admin/health" -o /tmp/health_response)

if [ "$HEALTH_RESPONSE" != "200" ]; then
    echo "Service health check failed: HTTP $HEALTH_RESPONSE" | \
    mail -s "FinSight Service Alert" $ALERT_EMAIL
    exit 1
fi

# Check data freshness
DATA_FRESH=$(curl -s -H "X-API-Key: $API_KEY" \
  "$SERVICE_URL/admin/health" | jq -r '.data_fresh')

if [ "$DATA_FRESH" != "true" ]; then
    echo "Data freshness check failed" | \
    mail -s "FinSight Data Alert" $ALERT_EMAIL
    exit 1
fi

echo "Health check passed at $(date)"
```

#### Performance Monitoring

```bash
#!/bin/bash
# performance-monitor.sh

SERVICE_URL="http://localhost:8000"
API_KEY="your-api-key"
LOG_FILE="/var/log/finsight-performance.log"

# Measure response time
RESPONSE_TIME=$(curl -w "%{time_total}" -s -o /dev/null \
  -H "X-API-Key: $API_KEY" "$SERVICE_URL/health")

# Check memory usage
MEMORY_USAGE=$(docker stats --no-stream --format "table {{.MemPerc}}" \
  market-dataset-service | tail -n +2 | sed 's/%//')

# Log performance metrics
echo "$(date),$RESPONSE_TIME,$MEMORY_USAGE" >> $LOG_FILE

# Alert if performance degrades
if (( $(echo "$RESPONSE_TIME > 1.0" | bc -l) )); then
    echo "High response time: ${RESPONSE_TIME}s" | \
    mail -s "FinSight Performance Alert" admin@finsight.com
fi

if (( $(echo "$MEMORY_USAGE > 80" | bc -l) )); then
    echo "High memory usage: ${MEMORY_USAGE}%" | \
    mail -s "FinSight Performance Alert" admin@finsight.com
fi
```

### 3. Troubleshooting Guide

#### Common Error Codes

| Error Code | Description                | Resolution                                        |
| ---------- | -------------------------- | ------------------------------------------------- |
| `E001`     | Database connection failed | Check MongoDB status and network connectivity     |
| `E002`     | Storage connection failed  | Verify MinIO/S3 configuration and credentials     |
| `E003`     | API key invalid            | Validate API key and check authentication headers |
| `E004`     | Rate limit exceeded        | Implement backoff strategy or increase limits     |
| `E005`     | Data validation failed     | Check input data format and validation rules      |

#### Performance Tuning

```bash
# Database optimization
docker-compose exec mongodb mongosh --eval "
use finsight_market_data;
db.ohlcv.createIndex({exchange: 1, symbol: 1, timeframe: 1, timestamp: 1});
db.ohlcv.createIndex({timestamp: 1}, {expireAfterSeconds: 7776000}); // 90 days TTL
"

# Storage optimization
docker-compose exec minio mc admin config set /data/ cache.enable=on
docker-compose exec minio mc admin config set /data/ cache.drives=auto
docker-compose exec minio mc admin config set /data/ cache.expiry=168h

# Service optimization
docker-compose exec market-dataset-service python -c "
import gc
gc.collect()
print('Garbage collection completed')
"
```

---

**Next Steps:**

1. **Test Deployment**: Deploy to staging environment first
2. **Load Testing**: Validate performance under expected load
3. **Security Audit**: Review security configurations
4. **Monitoring Setup**: Configure comprehensive monitoring
5. **Backup Testing**: Verify backup and recovery procedures
6. **Documentation**: Update team documentation and runbooks

For additional support, refer to the [Architecture Documentation](architecture.md) and [Configuration Guide](configuration.md).
