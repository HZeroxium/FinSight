# FinSight AI-Powered Financial Analysis Platform

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.13-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/architecture-Hexagonal-9cf.svg)](https://alistair.cockburn.us/hexagonal-architecture/)

> **Enterprise-Grade AI-Powered Financial Analysis Platform**  
> Comprehensive microservices platform for cryptocurrency trading, news analysis, sentiment analysis, and time series prediction

## 🚀 Overview

FinSight is a production-ready, enterprise-grade AI-powered financial analysis platform designed for cryptocurrency markets. Built with modern microservices architecture, it provides comprehensive capabilities for market data collection, news analysis, sentiment analysis, and AI-powered time series forecasting.

The platform leverages state-of-the-art AI/ML technologies including transformer models (PatchTST, PatchTSMixer, FinBERT), advanced data engineering pipelines, and robust cloud-native architecture to deliver high-performance financial analysis services.

## 🏗️ Platform Architecture

```mermaid
graph TB
    subgraph "Frontend & APIs"
        FE[Frontend Applications]
        API[API Gateways]
        CLI[CLI Tools]
    end

    subgraph "Core Services"
        MD[Market Dataset Service<br/>Port: 8000]
        NS[News Service<br/>Port: 8001]
        SA[Sentiment Analysis Service<br/>Port: 8002]
        PS[Prediction Service<br/>Port: 8003]
    end

    subgraph "Shared Infrastructure"
        EU[Eureka Server<br/>Port: 8761]
        RQ[RabbitMQ<br/>Port: 5672]
        RD[Redis<br/>Port: 6379]
        ML[MLflow<br/>Port: 5000]
        MI[MinIO/S3<br/>Port: 9000]
    end

    subgraph "Data Layer"
        MG[MongoDB]
        IN[InfluxDB]
        PG[PostgreSQL]
        FS[File Storage]
    end

    subgraph "External Integrations"
        BN[Binance API]
        TV[Tavily API]
        OA[OpenAI API]
        GA[Google AI]
    end

    FE --> API
    API --> MD
    API --> NS
    API --> SA
    API --> PS

    MD --> EU
    NS --> EU
    SA --> EU
    PS --> EU

    MD --> RQ
    NS --> RQ
    SA --> RQ
    PS --> RQ

    MD --> RD
    NS --> RD
    SA --> RD
    PS --> RD

    PS --> ML
    SA --> ML

    MD --> MI
    PS --> MI

    MD --> MG
    MD --> IN
    NS --> MG
    ML --> PG
    MD --> FS

    MD --> BN
    NS --> TV
    SA --> OA
    SA --> GA

    style MD fill:#e3f2fd
    style NS fill:#f3e5f5
    style SA fill:#e8f5e8
    style PS fill:#fff3e0
    style EU fill:#ffebee
    style RQ fill:#f1f8e9
    style RD fill:#fff8e1
```

## 🎯 Core Services

### **Market Dataset Service** 📊

**Port: 8000** | [Service Documentation](market_dataset_service/README.md)

Comprehensive market data management and backtesting service featuring:

- **Multi-Exchange Support**: Binance, Coinbase, Kraken integration
- **Multi-Storage Backends**: CSV, MongoDB, InfluxDB, Parquet, S3-compatible storage
- **Advanced Backtesting**: Multiple engines (Backtrader, TA4J, ZOOT) with strategy library
- **Data Quality Assurance**: Automated validation, gap detection, and integrity checks
- **Real-time Collection**: Scheduled data collection with configurable intervals

### **News Service** 📰

**Port: 8001** | [Service Documentation](news_service/README.md)

High-performance news aggregation and processing service featuring:

- **Multi-Source Collection**: RSS and API-based collection from CoinDesk, CoinTelegraph
- **Intelligent Caching**: Redis-based caching with configurable TTL
- **Dual API Support**: REST API and gRPC endpoints
- **Sentiment Integration**: Automatic publishing to sentiment analysis service
- **Job Management**: Scheduled news collection with configurable cron jobs

### **Sentiment Analysis Service** 🧠

**Port: 8002** | [Service Documentation](sentiment_analysis/README.md)

AI-powered financial sentiment analysis platform featuring:

- **Model Building**: Fine-tune FinBERT and transformer models for crypto news
- **Inference Engine**: GPU-accelerated inference with Triton Server
- **Multiple Formats**: Support for JSON, JSONL, CSV, Parquet
- **Production Ready**: ONNX/TorchScript export, MLflow integration
- **Batch Processing**: Efficient handling of large text volumes

### **Prediction Service** 🔮

**Port: 8003** | [Service Documentation](prediction_service/README.md)

Advanced AI-powered time series forecasting service featuring:

- **State-of-the-Art Models**: PatchTST, PatchTSMixer, and custom transformers
- **Intelligent Fallback**: Smart model selection with timeframe and symbol strategies
- **Multiple Serving Backends**: Simple, TorchScript, TorchServe, Triton Inference Server
- **Cloud-Native**: Built-in support for AWS S3, DigitalOcean Spaces, MinIO
- **Experiment Tracking**: MLflow integration for model lifecycle management

## 🏛️ Architecture Principles

### **Hexagonal Architecture (Ports & Adapters)**

Each service follows clean architecture principles with clear separation of concerns:

- **API Layer**: FastAPI routers and controllers
- **Business Layer**: Core business logic and services
- **Adapter Layer**: External integrations and data access
- **Infrastructure Layer**: Storage, messaging, and external services

### **Microservices Design**

- **Service Discovery**: Eureka client integration for dynamic service registration
- **Message Queues**: RabbitMQ for asynchronous inter-service communication
- **Distributed Caching**: Redis for performance optimization across services
- **Container Orchestration**: Docker and Docker Compose for deployment

### **AI/ML First Design**

- **Model Versioning**: MLflow for experiment tracking and model lifecycle
- **GPU Acceleration**: NVIDIA Triton Server for high-performance inference
- **Scalable Training**: Distributed training capabilities with cloud storage
- **Intelligent Fallbacks**: Robust error handling and model selection strategies

## 🚀 Quick Start

### **Prerequisites**

- **Python**: 3.12+ (recommended)
- **Docker**: 20.10+ with Docker Compose 2.0+
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 50GB+ available space
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### **1. Clone Repository**

```bash
git clone <repository-url>
cd FinSight
```

### **2. Start Infrastructure Services**

```bash
# Start core infrastructure (RabbitMQ, Redis, Eureka, PostgreSQL, MinIO, MLflow)
docker-compose up -d rabbitmq redis eureka-server postgres minio mlflow

# Verify infrastructure is running
docker-compose ps
```

### **3. Configure Environment**

```bash
# Set up environment variables for each service
cp common/.env.example common/.env
cp market_dataset_service/env.example market_dataset_service/.env
cp news_service/env.example news_service/.env
cp sentiment_analysis/sentiment_analysis_model_builder/env.example sentiment_analysis/sentiment_analysis_model_builder/.env
cp prediction_service/env.example prediction_service/.env

# Edit configuration files with your API keys and settings
```

### **4. Install Dependencies**

```bash
# Install shared utilities
pip install -e ./common

# Install service dependencies
cd market_dataset_service && pip install -r requirements.txt && cd ..
cd news_service && pip install -r requirements.txt && cd ..
cd sentiment_analysis/sentiment_analysis_model_builder && pip install -r requirements.txt && cd ../..
cd prediction_service && pip install -r requirements.txt && cd ..
```

### **5. Start Services**

```bash
# Option 1: Start all services with Docker Compose
docker-compose up -d

# Option 2: Start services individually
cd market_dataset_service && python -m src.main &
cd news_service && python -m src.main &
cd sentiment_analysis/sentiment_analysis_inference_engine && python main.py &
cd prediction_service && python -m src.main &
```

### **6. Verify Deployment**

```bash
# Check service health
curl http://localhost:8000/health  # Market Dataset Service
curl http://localhost:8001/health  # News Service
curl http://localhost:8002/health  # Sentiment Analysis Service
curl http://localhost:8003/health  # Prediction Service

# Check infrastructure
curl http://localhost:8761        # Eureka Server
curl http://localhost:5000/health # MLflow
```

## 📚 Documentation

### **Platform Documentation**

- **[API Documentation](docs/api.md)** - Complete REST API reference with examples
- **[Technical Architecture](docs/architecture/technical-architecture.md)** - Detailed system design and patterns
- **[Deployment Guide](docs/deployment.md)** - Production deployment and configuration
- **[Documentation Index](docs/README.md)** - Complete documentation navigation

### **Service-Specific Documentation**

- **[Common Module](common/README.md)** - Shared utilities and services
- **[Market Dataset Service](market_dataset_service/README.md)** - Data collection and backtesting
- **[News Service](news_service/README.md)** - News aggregation and processing
- **[Sentiment Analysis](sentiment_analysis/README.md)** - AI sentiment analysis platform
- **[Prediction Service](prediction_service/README.md)** - Time series forecasting

## 🧪 Testing

### **Quick API Testing**

```bash
# Market data collection
curl -X POST "http://localhost:8000/market-data/collect" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT", "timeframe": "1h"}'

# News search
curl "http://localhost:8001/news/?limit=10&source=coindesk"

# Sentiment analysis
curl -X POST "http://localhost:8002/sentiment/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Bitcoin shows strong bullish signals today."}'

# Time series prediction
curl -X POST "http://localhost:8003/prediction/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT", "timeframe": "1d", "n_steps": 1}'
```

### **Comprehensive Testing**

```bash
# Run all tests
pytest tests/ -v --cov=src

# Service-specific tests
cd market_dataset_service && pytest tests/ -v && cd ..
cd news_service && pytest tests/ -v && cd ..
cd sentiment_analysis && pytest tests/ -v && cd ..
cd prediction_service && pytest tests/ -v && cd ..
```

## 🔧 Configuration

### **Key Environment Variables**

```bash
# Core Configuration
LOG_LEVEL=INFO
ENVIRONMENT=production
API_VERSION=v1

# Database Configuration
MONGODB_URL=mongodb://mongodb:27017
POSTGRES_URL=postgresql://mlflow:mlflow@postgresql:5432/mlflow
REDIS_URL=redis://redis:6379

# Message Queue
RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/

# External APIs
BINANCE_API_KEY=your-binance-api-key
BINANCE_SECRET_KEY=your-binance-secret-key
TAVILY_API_KEY=your-tavily-api-key
OPENAI_API_KEY=your-openai-api-key

# Authentication
API_KEY=your-api-key
ADMIN_API_KEY=your-admin-api-key
```

### **Service Ports**

- **Market Dataset Service**: 8000
- **News Service**: 8001
- **Sentiment Analysis Service**: 8002
- **Prediction Service**: 8003
- **Eureka Server**: 8761
- **MLflow**: 5000
- **MinIO Console**: 9001
- **RabbitMQ Management**: 15672

## 📊 Performance & Monitoring

### **Health Checks**

```bash
# Basic health checks
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health

# Detailed health checks (admin only)
curl -H "X-API-Key: admin-key" http://localhost:8000/admin/health
```

### **Key Metrics**

- **Response Time**: < 500ms for most endpoints
- **Error Rate**: < 1% for all endpoints
- **Availability**: 99.9% uptime target
- **Throughput**: Configurable based on deployment scale

### **Monitoring Stack**

- **Prometheus**: Metrics collection
- **Grafana**: Dashboard visualization
- **Jaeger**: Distributed tracing
- **ELK Stack**: Log aggregation and analysis

## 🔒 Security

### **Authentication & Authorization**

- **API Key Authentication**: Required for all business endpoints
- **Admin API Keys**: Separate keys for administrative operations
- **Rate Limiting**: Configurable per service and endpoint
- **Role-Based Access**: Different permission levels for operations

### **Data Protection**

- **Encryption at Rest**: Database and storage encryption
- **Encryption in Transit**: TLS for all communications
- **Environment Variables**: Sensitive data in environment variables
- **Input Validation**: Comprehensive Pydantic schema validation

### **Network Security**

- **Internal Networks**: Services communicate over Docker networks
- **Firewall Rules**: Minimal external port exposure
- **TLS/SSL**: HTTPS for all external communications

## 🚀 Production Deployment

### **Docker Deployment**

```bash
# Production deployment with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale market-dataset-service=3
docker-compose up -d --scale prediction-service=2
```

### **Kubernetes Deployment**

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check deployment status
kubectl get pods -n finsight
```

### **Cloud Deployment**

- **AWS**: ECS, EKS, or EC2 deployment
- **Google Cloud**: GKE or Compute Engine
- **Azure**: AKS or Container Instances
- **DigitalOcean**: Kubernetes or App Platform

## 🤝 Contributing

### **Development Setup**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow coding standards (Black, isort, mypy)
4. Add comprehensive tests (minimum 80% coverage)
5. Update documentation
6. Submit a pull request

### **Code Standards**

- **Python**: PEP 8, Black, isort, flake8
- **Type Hints**: Full type annotations for all functions
- **Documentation**: Google-style docstrings
- **Testing**: pytest with coverage requirements
- **Architecture**: Hexagonal design principles

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### **Getting Help**

- **Documentation**: [docs/](docs/) directory
- **Issues**: [GitHub Issues](https://github.com/your-org/finsight/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/finsight/discussions)

### **Community**

- **Slack**: [FinSight Community](https://finsight-community.slack.com)
- **Discord**: [FinSight Discord](https://discord.gg/finsight)
- **Email**: support@finsight.ai

### **Troubleshooting**

- **Common Issues**: [Troubleshooting Guide](docs/deployment.md#troubleshooting)
- **Performance Tuning**: [Performance Guide](docs/deployment.md#performance-tuning)
- **Security Hardening**: [Security Guide](docs/deployment.md#security-hardening)

## 🔄 Roadmap

### **Upcoming Features**

- **Real-time Streaming**: WebSocket support for live data
- **Advanced Analytics**: Machine learning integration across services
- **Multi-Asset Support**: Stocks, forex, and commodities
- **Mobile API**: Optimized endpoints for mobile applications
- **Advanced Backtesting**: More sophisticated strategy testing

### **Platform Enhancements**

- **Auto-scaling**: Kubernetes HPA integration
- **Multi-region**: Global deployment support
- **Advanced Monitoring**: Custom dashboards and alerting
- **API Gateway**: Centralized API management

---

**Built with ❤️ by the FinSight Team**

_For detailed information about each service, see the [service-specific documentation](docs/README.md#service-documentation) and [platform documentation](docs/README.md)._
