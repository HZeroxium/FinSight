# FinSight Platform Documentation

Welcome to the comprehensive documentation for the FinSight AI-Powered Financial Analysis Platform. This documentation covers all aspects of the platform from architecture to deployment to API usage.

## 📚 Documentation Index

### 🏗️ Architecture & Design

- **[Platform Architecture](architecture.md)** - High-level platform overview and system design
- **[Technical Architecture](architecture/technical-architecture.md)** - Detailed technical architecture, patterns, and component interactions
- **[Common Module Documentation](../common/README.md)** - Shared utilities and services documentation

### 🔌 API Documentation

- **[API Reference](api.md)** - Complete REST API documentation for all services
- **[Service-Specific APIs](../market_dataset_service/docs/)** - Individual service API documentation
- **[OpenAPI Specifications](../news_service/misc/openapi.json)** - Machine-readable API specifications

### 🚀 Deployment & Operations

- **[Deployment Guide](deployment.md)** - Complete deployment instructions for development and production
- **[Docker Configuration](../docker-compose.yml)** - Container orchestration setup
- **[Production Setup](deployment.md#production-deployment)** - Production deployment procedures

### 🔧 Development

- **[Quick Start Guide](../README.md#quick-start)** - Get up and running quickly
- **[Development Environment](deployment.md#development-environment)** - Local development setup
- **[Testing Guidelines](../README.md#testing)** - Testing procedures and best practices

### 📊 Monitoring & Maintenance

- **[Health Checks](deployment.md#monitoring--health-checks)** - Service health monitoring
- **[Troubleshooting](deployment.md#troubleshooting)** - Common issues and solutions
- **[Performance Tuning](deployment.md#performance-tuning)** - Optimization strategies

## 🎯 Quick Navigation

### For New Users

1. Start with the **[Platform Architecture](architecture.md)** to understand the system
2. Follow the **[Quick Start Guide](../README.md#quick-start)** to get running
3. Explore the **[API Reference](api.md)** to learn how to use the services

### For Developers

1. Set up your **[Development Environment](deployment.md#development-environment)**
2. Review the **[Technical Architecture](architecture/technical-architecture.md)**
3. Check the **[Common Module](../common/README.md)** for shared utilities
4. Use the **[API Reference](api.md)** for integration

### For DevOps/Operations

1. Review the **[Deployment Guide](deployment.md)** for production setup
2. Configure **[Monitoring & Health Checks](deployment.md#monitoring--health-checks)**
3. Set up **[Backup & Recovery](deployment.md#backup--recovery)** procedures
4. Implement **[Security Hardening](deployment.md#security-hardening)**

### For System Administrators

1. Understand the **[Infrastructure Architecture](architecture/technical-architecture.md#infrastructure-architecture)**
2. Configure **[Monitoring & Observability](architecture/technical-architecture.md#monitoring--observability)**
3. Set up **[Performance & Scalability](architecture/technical-architecture.md#performance--scalability)**
4. Implement **[Security Architecture](architecture/technical-architecture.md#security-architecture)**

## 📋 Service Documentation

### Core Services

#### Market Dataset Service

- **Purpose**: Financial data collection, storage, and backtesting
- **Port**: 8000
- **Documentation**: [Service Details](../market_dataset_service/README.md)
- **Key Features**:
  - Real-time Binance data collection
  - Multiple storage backends (CSV, Parquet, MongoDB, InfluxDB)
  - Advanced backtesting strategies
  - Data validation and quality checks

#### News Service

- **Purpose**: News aggregation, processing, and search integration
- **Port**: 8001
- **Documentation**: [Service Details](../news_service/docs/)
- **Key Features**:
  - Multi-source news collection (Coindesk, Cointelegraph)
  - Tavily search integration
  - Parallel processing with configurable workers
  - Caching and rate limiting

#### Sentiment Analysis Service

- **Purpose**: Financial sentiment analysis using AI models
- **Port**: 8002
- **Documentation**: [Service Details](../sentiment_analysis/README.md)
- **Key Features**:
  - Multiple AI models (FinBERT, BERT)
  - Batch processing capabilities
  - Model versioning and management
  - GPU acceleration support

#### Prediction Service

- **Purpose**: AI-powered time series forecasting
- **Port**: 8003
- **Documentation**: [Service Details](../prediction_service/README.md)
- **Key Features**:
  - Advanced AI models (PatchTST, PatchTSMixer)
  - Intelligent fallback strategies
  - MLflow integration
  - Multiple serving backends

## 🔧 Configuration Reference

### Environment Variables

#### Common Configuration

```bash
# Service Configuration
LOG_LEVEL=INFO
ENVIRONMENT=production
API_VERSION=v1

# Database Configuration
MONGODB_URL=mongodb://mongodb:27017
POSTGRES_URL=postgresql://mlflow:mlflow@postgresql:5432/mlflow
REDIS_URL=redis://redis:6379

# Message Queue Configuration
RABBITMQ_URL=amqp://admin:password@rabbitmq:5672/

# API Keys (External Services)
BINANCE_API_KEY=your-binance-api-key
BINANCE_SECRET_KEY=your-binance-secret-key
TAVILY_API_KEY=your-tavily-api-key
OPENAI_API_KEY=your-openai-api-key

# Authentication
API_KEY=your-api-key
ADMIN_API_KEY=your-admin-api-key
```

#### Service-Specific Configuration

```bash
# Market Dataset Service
MARKET_DATA_COLLECTION_INTERVAL=300
BACKTEST_STRATEGIES_PATH=/app/strategies

# News Service
NEWS_SOURCES=coindesk,cointelegraph
NEWS_COLLECTION_INTERVAL=600

# Sentiment Analysis Service
SENTIMENT_MODEL_PATH=/app/models/finbert
GPU_ENABLED=true

# Prediction Service
PREDICTION_MODEL_PATH=/app/models/patchedtst
MLFLOW_TRACKING_URI=http://mlflow:5000
```

### Docker Configuration

#### Development Environment

```yaml
# docker-compose.yml
version: "3.8"
services:
  market-dataset-service:
    image: finsight/market-dataset-service:latest
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=DEBUG
    volumes:
      - ./market_dataset_service:/app

  news-service:
    image: finsight/news-service:latest
    ports:
      - "8001:8001"
    environment:
      - LOG_LEVEL=DEBUG
    volumes:
      - ./news_service:/app
```

#### Production Environment

```yaml
# docker-compose.prod.yml
version: "3.8"
services:
  market-dataset-service:
    image: finsight/market-dataset-service:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: "1.0"
          memory: 1G
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
```

## 📊 Monitoring & Health Checks

### Health Check Endpoints

```bash
# Basic health checks
curl http://localhost:8000/health  # Market Dataset Service
curl http://localhost:8001/health  # News Service
curl http://localhost:8002/health  # Sentiment Analysis Service
curl http://localhost:8003/health  # Prediction Service

# Detailed health checks (admin only)
curl -H "X-API-Key: admin-key" http://localhost:8000/admin/health
```

### Key Metrics

- **Service Availability**: All services should return 200 OK
- **Response Time**: < 500ms for most endpoints
- **Error Rate**: < 1% for all endpoints
- **Database Connectivity**: All databases should be connected
- **External API Status**: Binance, Tavily, OpenAI APIs should be accessible

## 🔒 Security

### Authentication

- **API Key Authentication**: Required for all business endpoints
- **Admin API Keys**: Separate keys for administrative operations
- **Rate Limiting**: Configurable per service and endpoint

### Network Security

- **TLS/SSL**: HTTPS for all external communications
- **Internal Networks**: Services communicate over internal Docker networks
- **Firewall Rules**: Minimal external port exposure

### Data Security

- **Encryption at Rest**: Database and storage encryption
- **Encryption in Transit**: TLS for all communications
- **Access Control**: Role-based access control (RBAC)

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/finsight.git
cd finsight
```

### 2. Start Infrastructure

```bash
docker-compose up -d rabbitmq redis eureka-server postgres minio mlflow
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 4. Start Services

```bash
# Start all services
docker-compose up -d

# Or start individually
docker-compose up -d market-dataset-service
docker-compose up -d news-service
docker-compose up -d sentiment-analysis-service
docker-compose up -d prediction-service
```

### 5. Verify Deployment

```bash
# Check service health
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

## 📞 Support

### Documentation Issues

If you find issues with the documentation:

1. Check the [GitHub Issues](https://github.com/your-org/finsight/issues)
2. Create a new issue with the `documentation` label
3. Provide specific details about the problem

### Technical Support

For technical issues:

1. Check the [Troubleshooting Guide](deployment.md#troubleshooting)
2. Review the [Common Issues](#common-issues) section
3. Create a GitHub issue with logs and error details

### Feature Requests

For new features or improvements:

1. Check existing [GitHub Issues](https://github.com/your-org/finsight/issues)
2. Create a new issue with the `enhancement` label
3. Provide detailed requirements and use cases

## 📈 Contributing

### Documentation Contributions

We welcome contributions to improve the documentation:

1. Fork the repository
2. Make your changes in a feature branch
3. Submit a pull request with a clear description
4. Ensure all links work and formatting is correct

### Code Contributions

For code contributions:

1. Review the [Contributing Guidelines](../CONTRIBUTING.md)
2. Follow the coding standards and patterns
3. Include tests for new functionality
4. Update documentation for new features

---

_This documentation is continuously updated. For the latest version, always refer to the main repository._
