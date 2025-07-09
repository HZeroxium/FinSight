# FinSight Common Module

A shared utilities library for the FinSight AI-powered financial analysis platform. This module provides common functionality that can be reused across all FinSight services.

## Features

- **Logging**: Structured logging with multiple backends (console, file)
- **Caching**: Multi-backend caching system (in-memory, Redis, file-based)
- **LLM Integration**: Unified interface for different LLM providers (OpenAI, Google AI)
- **Dependency Injection**: Factory patterns for easy testing and configuration

## Installation

### For Development (Editable Install)

From the root FinSight directory:

```bash
# Install in editable mode so changes are reflected immediately
pip install -e ./common
```

### For Production

```bash
pip install ./common
```

## Usage

### Logging

```python
from common.logger import LoggerFactory, LoggerType

# Create a logger
logger = LoggerFactory.create_logger(
    name="my_service",
    logger_type=LoggerType.STANDARD,
    log_file="logs/my_service.log"
)

logger.info("Service started successfully")
logger.error("An error occurred", extra={"error_code": 500})
```

### Caching

```python
from common.cache import CacheFactory, CacheType

# Create a cache instance
cache = CacheFactory.create_cache(
    cache_type=CacheType.REDIS,
    redis_url="redis://localhost:6379"
)

# Use the cache
await cache.set("key", "value", ttl=3600)
value = await cache.get("key")
```

### LLM Integration

```python
from common.llm import LLMFacade

# Create LLM facade
llm = LLMFacade()

# Generate text
response = await llm.generate_text(
    prompt="Analyze this financial news...",
    model="gpt-4"
)
```

## Architecture

The common module follows these design patterns:

- **Factory Pattern**: For creating instances of loggers, caches, and LLM clients
- **Interface Segregation**: Clear interfaces for different components
- **Dependency Injection**: Easy to mock and test components
- **Configuration Management**: Environment-based configuration with Pydantic

## Development

### Running Tests

```bash
# Install with development dependencies
pip install -e "./common[dev]"

# Run tests
pytest common/tests/
```

### Code Quality

```bash
# Format code
black common/

# Sort imports
isort common/

# Type checking
mypy common/
```

## Services Integration

This module is designed to be used across all FinSight services:

- `ai_prediction/`: AI model building and prediction
- `backtesting/`: Market data collection and backtesting
- `news_crawler/`: News collection and processing
- `sentiment_analysis/`: Sentiment analysis of financial news

## Requirements

- Python 3.10+
- See `pyproject.toml` for full dependency list

## License

MIT License - see LICENSE file for details.
