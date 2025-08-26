# FinSight Sentiment Analysis Inference Engine

Automated Triton Inference Server deployment and REST API for FinBERT sentiment analysis.

## Overview

This inference engine provides a complete automated solution for deploying fine-tuned FinBERT models using NVIDIA Triton Inference Server. It eliminates manual Docker commands and provides simple REST API endpoints for sentiment analysis.

## Features

- **Automated Triton Server Management**: Automatic Docker container lifecycle management
- **Simple REST API**: Easy-to-use endpoints for single and batch sentiment analysis
- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **High Performance**: GPU-accelerated inference with dynamic batching
- **Health Monitoring**: Built-in health checks and metrics collection
- **Async Support**: Full async/await implementation for optimal performance

## Quick Start

### Prerequisites

- Docker with GPU support (NVIDIA Docker)
- Python 3.8+
- NVIDIA GPU (optional but recommended)
- Fine-tuned FinBERT model from `sentiment_analysis_model_builder`

### Installation

1. **Clone and setup environment**:

```bash
cd sentiment_analysis_inference_engine
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment**:

```bash
cp .env.example .env
# Edit .env file with your configuration
```

3. **Ensure model is available**:
   Make sure you have run the model builder and have the Triton model repository at:

```
../sentiment_analysis_model_builder/models/triton_model_repository/
```

### Running the Service

**Start the inference engine**:

```bash
python main.py
```

The service will:

1. Load configuration
2. Start Triton Docker container automatically
3. Initialize sentiment analysis service
4. Start FastAPI server on port 8080

## API Usage

### Single Text Analysis

```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "The market is showing strong positive signals today."}'
```

Response:

```json
{
  "text": "The market is showing strong positive signals today.",
  "label": "POSITIVE",
  "confidence": 0.8245,
  "scores": {
    "negative": 0.0856,
    "neutral": 0.0899,
    "positive": 0.8245
  },
  "processing_time_ms": 45.2
}
```

### Batch Analysis

```bash
curl -X POST "http://localhost:8080/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "texts": [
         "The market is bullish today.",
         "Economic indicators show decline.",
         "Neutral market conditions observed."
       ]
     }'
```

### Health Check

```bash
curl "http://localhost:8080/health"
```

### Service Metrics

```bash
curl "http://localhost:8080/metrics"
```

## Configuration

### Environment Variables

| Variable                   | Default                                                            | Description              |
| -------------------------- | ------------------------------------------------------------------ | ------------------------ |
| `TRITON_HOST`              | localhost                                                          | Triton server host       |
| `TRITON_HTTP_PORT`         | 8000                                                               | Triton HTTP port         |
| `TRITON_MODEL_REPOSITORY`  | ../sentiment_analysis_model_builder/models/triton_model_repository | Path to model repository |
| `SENTIMENT_MAX_BATCH_SIZE` | 32                                                                 | Maximum batch size       |
| `API_PORT`                 | 8080                                                               | FastAPI server port      |
| `API_LOG_LEVEL`            | INFO                                                               | Logging level            |

### Configuration Files

- `.env`: Environment variables
- `src/core/config.py`: Pydantic configuration models

## Architecture

### Components

1. **TritonServerManager** (`src/services/triton_manager.py`)

   - Docker container lifecycle management
   - Health monitoring and status checks
   - Automatic startup and shutdown

2. **SentimentAnalysisService** (`src/services/sentiment_service.py`)

   - Text preprocessing and tokenization
   - Triton client integration
   - Batch processing optimization

3. **FastAPI Application** (`main.py`)
   - REST API endpoints
   - Error handling and middleware
   - Async request processing

### Data Flow

```
Client Request → FastAPI → SentimentAnalysisService → Triton Server → ONNX Model
             ←          ←                        ←              ←
```

## API Endpoints

### Core Endpoints

- `GET /` - Server information
- `GET /health` - Health check
- `POST /predict` - Single text sentiment analysis
- `POST /predict/batch` - Batch sentiment analysis

### Monitoring Endpoints

- `GET /metrics` - Service metrics
- `GET /model/info` - Model information

### Admin Endpoints

- `POST /admin/restart` - Restart Triton server
- `POST /admin/reset-metrics` - Reset metrics

## Error Handling

The service provides comprehensive error handling:

- **TritonServerError**: Triton server issues (503 Service Unavailable)
- **SentimentAnalysisError**: Analysis failures (400 Bad Request)
- **ValidationError**: Input validation errors (422 Unprocessable Entity)
- **Global Exception Handler**: Unexpected errors (500 Internal Server Error)

## Logging

Structured logging with:

- Console output with colors
- File logging (`logs/inference_engine.log`)
- Request tracing and performance metrics
- Error tracking and debugging info

## Performance

### Optimization Features

- **GPU Acceleration**: CUDA support for model inference
- **Dynamic Batching**: Automatic request batching in Triton
- **Connection Pooling**: Efficient HTTP client usage
- **Async Processing**: Non-blocking request handling

### Monitoring

- Request count and success rate
- Average processing time
- Server uptime
- Memory and GPU utilization (via Triton metrics)

## Troubleshooting

### Common Issues

1. **Docker not found**

   ```
   Error: Failed to connect to Docker
   Solution: Ensure Docker is installed and running
   ```

2. **Model not found**

   ```
   Error: Model repository not found
   Solution: Run sentiment_analysis_model_builder first
   ```

3. **GPU not available**

   ```
   Warning: CUDA not available, using CPU
   Solution: Install NVIDIA Docker and GPU drivers
   ```

4. **Port already in use**
   ```
   Error: Port 8000 already in use
   Solution: Stop existing Triton container or change ports
   ```

### Debug Mode

Enable debug logging:

```bash
export DEBUG=true
export API_LOG_LEVEL=DEBUG
python main.py
```

### Manual Triton Management

If automatic management fails, you can run Triton manually:

```bash
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $(pwd)/../sentiment_analysis_model_builder/models/triton_model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/
isort src/
```

### Adding New Features

1. Add new endpoints in `main.py`
2. Extend service classes in `src/services/`
3. Update schemas in `src/models/schemas.py`
4. Add configuration in `src/core/config.py`

## Production Deployment

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

Build and run:

```bash
docker build -t finbert-inference .
docker run -p 8080:8080 finbert-inference
```

### Load Balancing

For high availability, deploy multiple instances behind a load balancer:

```bash
# Instance 1
API_PORT=8080 python main.py &

# Instance 2
API_PORT=8081 python main.py &
```

### Monitoring Integration

Integrate with monitoring systems:

- Prometheus metrics: `/metrics` endpoint
- Health checks: `/health` endpoint
- Structured logging for log aggregation

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure code passes formatting checks
5. Submit pull request

## License

This project is part of the FinSight financial analysis platform.

## Support

For issues and questions:

1. Check troubleshooting section
2. Review logs in `logs/` directory
3. Create GitHub issue with relevant details
