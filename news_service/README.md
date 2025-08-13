# News Crawler Service

A modern, production-ready news crawler service built with FastAPI that provides both REST API endpoints and background cron job functionality for automated news collection from cryptocurrency sources.

## Features

- 🚀 **FastAPI REST API** - Modern async web framework
- 🕐 **Cron Job Scheduler** - Automated news collection using APScheduler
- 📰 **Multi-source Support** - CoinDesk, CoinTelegraph, and more
- 🔄 **Background Processing** - Asynchronous news collection and processing
- 📊 **MongoDB Storage** - Persistent storage for news articles
- 🐰 **RabbitMQ Integration** - Message queue for sentiment analysis
- 🔍 **Tavily Search** - AI-powered search engine integration
- 📈 **Health Monitoring** - Comprehensive health checks and metrics
- 🎯 **Dependency Injection** - Clean architecture with DI pattern
- 🛡️ **Error Handling** - Robust error handling and logging
- 🔧 **Configuration Management** - Environment-based configuration

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd news_crawler

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Required: TAVILY_API_KEY
```

### 3. Start the FastAPI Server

```bash
# Method 1: Using Python module
python -m src.main

# Method 2: Using batch file (Windows)
start_server.bat

# Method 3: Using uvicorn directly
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start the Cron Job Service

```bash
# Start the background job service
python -m src.news_crawler_job start

# Or using the management script (Windows)
manage_job.bat start
```

## API Endpoints

### Core Endpoints

- `GET /` - Service information
- `GET /health` - Health check with component status
- `GET /metrics` - Service metrics and statistics

### Search Endpoints

- `POST /api/v1/search/` - Search news articles
- `GET /api/v1/search/financial-sentiment/{symbol}` - Get financial sentiment for symbol
- `GET /api/v1/search/trending` - Get trending news topics

### News Endpoints

- `GET /api/v1/news/` - Get news articles with filters
- `GET /api/v1/news/stats` - Get news statistics
- `GET /api/v1/news/sources` - Get available news sources

### API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

## Cron Job Management

### Commands

```bash
# Start the cron job service
python -m src.news_crawler_job start

# Stop the cron job service
python -m src.news_crawler_job stop

# Check service status
python -m src.news_crawler_job status

# Run a manual crawl
python -m src.news_crawler_job run

# View/update configuration
python -m src.news_crawler_job config
```

### Windows Management Script

```batch
# Start service
manage_job.bat start

# Stop service
manage_job.bat stop

# Check status
manage_job.bat status

# Run manual crawl
manage_job.bat run

# View configuration
manage_job.bat config
```

## Configuration

### Environment Variables

| Variable            | Description               | Default                              |
| ------------------- | ------------------------- | ------------------------------------ |
| `TAVILY_API_KEY`    | Tavily API key (required) | -                                    |
| `MONGODB_URL`       | MongoDB connection URL    | `mongodb://localhost:27017`          |
| `MONGODB_DATABASE`  | Database name             | `news_crawler`                       |
| `RABBITMQ_URL`      | RabbitMQ connection URL   | `amqp://guest:guest@localhost:5672/` |
| `DEBUG`             | Debug mode                | `false`                              |
| `HOST`              | Server host               | `0.0.0.0`                            |
| `PORT`              | Server port               | `8000`                               |
| `CRON_JOB_SCHEDULE` | Cron schedule expression  | `0 */1 * * *`                        |
| `LOG_LEVEL`         | Logging level             | `INFO`                               |

### Cron Job Configuration

Edit `news_crawler_config.json`:

```json
{
  "sources": ["coindesk", "cointelegraph"],
  "collector_preferences": {
    "coindesk": "api_rest",
    "cointelegraph": "api_graphql"
  },
  "max_items_per_source": 100,
  "enable_fallback": true,
  "schedule": "0 */1 * * *",
  "config_overrides": {
    "timeout": 30,
    "retry_attempts": 3
  },
  "notification": {
    "enabled": false,
    "webhook_url": null,
    "email": null
  }
}
```

### Cron Schedule Examples

- `0 */1 * * *` - Every hour
- `0 */2 * * *` - Every 2 hours
- `0 0 * * *` - Daily at midnight
- `0 0 */3 * *` - Every 3 days
- `0 9 * * 1-5` - Weekdays at 9 AM

## Architecture

### Core Components

1. **FastAPI Application** (`src/main.py`)

   - REST API endpoints
   - Health monitoring
   - Dependency injection
   - Error handling

2. **Cron Job Service** (`src/news_crawler_job.py`)

   - APScheduler-based job scheduling
   - Process management
   - Configuration management
   - Status monitoring

3. **News Collector Service** (`src/services/news_collector_service.py`)

   - Multi-source news collection
   - Adapter pattern for different sources
   - Error handling and fallback

4. **Data Layer** (`src/repositories/`)

   - MongoDB repository
   - Data persistence
   - Query optimization

5. **Message Broker** (`src/adapters/rabbitmq_broker.py`)
   - RabbitMQ integration
   - Async message publishing
   - Error handling

### Design Patterns

- **Dependency Injection**: Clean separation of concerns
- **Repository Pattern**: Data access abstraction
- **Adapter Pattern**: Multiple news source support
- **Factory Pattern**: Service instantiation
- **Observer Pattern**: Event-driven architecture

## Development

### Project Structure

```plaintext
news_crawler/
├── src/
│   ├── main.py              # FastAPI application
│   ├── news_crawler_job.py  # Cron job service
│   ├── adapters/            # External service adapters
│   ├── core/                # Core configuration
│   ├── repositories/        # Data access layer
│   ├── services/            # Business logic
│   ├── routers/             # API route handlers
│   ├── schemas/             # Pydantic models
│   └── utils/               # Utility functions
├── logs/                    # Log files
├── requirements.txt         # Dependencies
├── .env.example            # Environment template
└── news_crawler_config.json # Job configuration
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_news_service.py
```

### Code Quality

```bash
# Format code
black src/

# Sort imports
isort src/

# Type checking
mypy src/

# Linting
flake8 src/
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY .env .env

# Start FastAPI server
CMD ["python", "-m", "src.main"]

# Or start cron job service
# CMD ["python", "-m", "src.news_crawler_job", "start"]
```

### Production Considerations

1. **Environment Variables**: Use secure secret management
2. **Database**: Use MongoDB replica set for high availability
3. **Message Queue**: Use RabbitMQ cluster
4. **Monitoring**: Add Prometheus/Grafana monitoring
5. **Logging**: Use centralized logging (ELK stack)
6. **Security**: Enable authentication and rate limiting

## Troubleshooting

### Common Issues

1. **Service won't start**

   - Check environment variables
   - Verify MongoDB connection
   - Check port availability

2. **Cron job not running**

   - Check PID file permissions
   - Verify cron schedule syntax
   - Check logs for errors

3. **API errors**
   - Verify Tavily API key
   - Check MongoDB connection
   - Review error logs

### Log Files

- `logs/news_crawler_main.log` - FastAPI server logs
- `logs/news_crawler_job.log` - Cron job logs
- `logs/news_service.log` - News service logs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
