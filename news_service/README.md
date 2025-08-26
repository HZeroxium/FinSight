# FinSight News Service

A high-performance, scalable news aggregation and processing service designed for financial markets. Built with FastAPI, MongoDB, and RabbitMQ, this service provides real-time news collection, storage, and integration with sentiment analysis pipelines.

## 🚀 Features

### Core Functionality

- **Multi-Source News Collection**: RSS and API-based collection from CoinDesk, CoinTelegraph, and other financial news sources
- **Intelligent Caching**: Redis-based caching with configurable TTL for optimal performance
- **Real-time Processing**: Asynchronous news processing with RabbitMQ message queues
- **Dual API Support**: REST API and gRPC endpoints for maximum flexibility
- **Service Discovery**: Eureka client integration for microservices architecture

### Advanced Capabilities

- **Sentiment Analysis Integration**: Automatic publishing to sentiment analysis service
- **Job Management**: Scheduled news collection with configurable cron jobs
- **Database Migration**: Seamless migration between local and cloud environments
- **Health Monitoring**: Comprehensive health checks and metrics
- **Rate Limiting**: Intelligent rate limiting to respect API quotas

### Data Management

- **Duplicate Detection**: Smart duplicate detection using URL and GUID hashing
- **Flexible Search**: Advanced search with keywords, tags, date ranges, and source filtering
- **Pagination Support**: Efficient pagination for large result sets
- **Data Validation**: Comprehensive input validation using Pydantic models

## 🏗️ Architecture Overview

```mermaid
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   News Sources  │    │   External APIs │    │   RSS Feeds     │
│  (CoinDesk,     │    │  (Tavily, etc.) │    │                 │
│   CoinTelegraph)│    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    News Collectors        │
                    │  (RSS, API, GraphQL)      │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    News Service           │
                    │  (Business Logic)         │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌─────────▼─────────┐
│   MongoDB         │  │   Redis Cache     │  │   RabbitMQ        │
│  (News Storage)   │  │  (Performance)    │  │  (Message Queue)  │
└───────────────────┘  └───────────────────┘  └───────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    API Layer              │
                    │  (REST + gRPC)            │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    Sentiment Analysis     │
                    │    Service Integration    │
                    └───────────────────────────┘
```

## 🛠️ Technology Stack

- **Framework**: FastAPI (Python 3.12+)
- **Database**: MongoDB (Motor async driver)
- **Cache**: Redis
- **Message Queue**: RabbitMQ (aio-pika)
- **API**: REST + gRPC
- **Service Discovery**: Eureka Client
- **Containerization**: Docker & Docker Compose
- **Logging**: Structured logging with correlation IDs
- **Validation**: Pydantic v2 with comprehensive schemas

## 📋 Prerequisites

- Python 3.12+
- MongoDB 5.0+
- Redis 6.0+
- RabbitMQ 3.8+
- Docker & Docker Compose (for containerized deployment)

## 🚀 Quick Start

### Local Development

1. **Clone and Setup**

   ```bash
   cd news_service
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment Configuration**

   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

3. **Start Dependencies** (using Docker Compose)

   ```bash
   docker-compose up -d mongodb redis rabbitmq
   ```

4. **Run the Service**

   ```bash
   python -m src.main
   ```

### Docker Deployment

1. **Build and Run**

   ```bash
   docker-compose up --build
   ```

2. **Production Deployment**

   ```bash
   docker-compose -f docker-compose.yml up -d
   ```

## 📚 Documentation

- **[API Documentation](docs/api.md)** - Complete API reference with examples
- **[Configuration Guide](docs/configuration.md)** - Environment variables and settings
- **[Architecture Guide](docs/architecture.md)** - Detailed system architecture
- **[Deployment Guide](docs/deployment.md)** - Production deployment instructions

## 🧪 Testing

### API Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test news search
curl "http://localhost:8000/news/?limit=10&source=coindesk"

# Test with authentication
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "http://localhost:8000/admin/jobs/status"
```

### gRPC Testing

```bash
# Use the provided gRPC test client
python tests/grpc_test_client.py
```

## 📊 Monitoring

### Health Checks

- **Service Health**: `GET /health`
- **Cache Health**: `GET /news/cache/health`
- **Job Health**: `GET /admin/jobs/health`

### Metrics

- **Service Metrics**: `GET /metrics`
- **Cache Statistics**: `GET /news/cache/stats`
- **Job Statistics**: `GET /admin/jobs/stats`

## 🔧 Configuration

Key configuration areas:

- **Database**: MongoDB connection (local/cloud)
- **Cache**: Redis settings and TTL configuration
- **Message Queue**: RabbitMQ connection and routing
- **API Keys**: Tavily, admin access tokens
- **Service Discovery**: Eureka client settings

See [Configuration Guide](docs/configuration.md) for detailed settings.

## 🤝 Contributing

1. Follow the established code patterns and architecture
2. Use Pydantic models for data validation
3. Implement comprehensive error handling
4. Add appropriate logging and monitoring
5. Update documentation for new features

## 📄 License

This project is part of the FinSight platform. See the main project license for details.

## 🆘 Support

For issues and questions:

1. Check the [documentation](docs/)
2. Review existing issues
3. Create a new issue with detailed information

---
