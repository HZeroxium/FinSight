# Crypto News Sentiment Analysis Model Builder

A production-ready Python project for training and packaging sentiment analysis models for cryptocurrency news using Hugging Face Transformers and MLflow.

## 🚀 Features

- **Fine-tune FinBERT** and other transformer models for crypto news sentiment analysis
- **Multi-format data support**: JSON, JSONL, CSV, Parquet
- **Reproducible training** with deterministic seeds and comprehensive logging
- **Model export** to ONNX and TorchScript formats with validation
- **MLflow integration** for experiment tracking and model versioning
- **S3/MinIO support** for artifact storage
- **Clean CLI interface** using Typer
- **Production-ready** with Docker support and comprehensive testing

## 📋 Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA support (optional, for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for large datasets)

## 🏗️ Architecture

The project follows a modular, layered architecture:

```mermaid
src/
├── core/           # Configuration management
├── data/           # Data loading and preprocessing
├── models/         # Training and export logic
├── registry/       # MLflow model registry integration
└── cli.py         # Command-line interface
```

### Key Design Principles

- **Hexagonal Architecture**: Clear separation between core business logic and external dependencies
- **Dependency Injection**: Configuration-driven service instantiation
- **Interface Segregation**: Focused, cohesive interfaces for each component
- **SOLID Principles**: Single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd sentiment_analysis_model_builder

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 2. Configuration

```bash
# Copy example configuration
cp env.example .env

# Edit configuration (see Configuration section below)
nano .env
```

### 3. Train a Model

```bash
# Train using the sample dataset
sentiment-train \
    --data data/news_dataset_sample.json \
    --output outputs/training_run_$(date +%Y%m%d_%H%M%S) \
    --experiment crypto-sentiment-v1
```

### 4. Export Model

```bash
# Export to ONNX format
sentiment-export \
    outputs/training_run_*/model \
    --output models/exported \
    --format onnx
```

### 5. Register Model

```bash
# Register in MLflow (use run_id from training output)
sentiment-register \
    outputs/training_run_*/model \
    --run-id <mlflow_run_id> \
    --stage Staging
```

## ⚙️ Configuration

The project uses Pydantic v2 with environment variable support. All configuration can be set via:

- Environment variables
- `.env` file
- CLI arguments (override defaults)

### Key Configuration Sections

#### Data Configuration

```bash
DATA_INPUT_PATH=data/news_dataset_sample.json
DATA_INPUT_FORMAT=json
DATA_TEXT_COLUMN=text
DATA_LABEL_COLUMN=label
```

#### Training Configuration

```bash
TRAINING_BACKBONE=ProsusAI/finbert
TRAINING_BATCH_SIZE=16
TRAINING_LEARNING_RATE=2e-5
TRAINING_NUM_EPOCHS=3
TRAINING_RANDOM_SEED=42
```

#### Export Configuration

```bash
EXPORT_FORMAT=onnx
EXPORT_ONNX_OPSET_VERSION=17
EXPORT_VALIDATE_EXPORT=true
```

#### Registry Configuration

```bash
REGISTRY_TRACKING_URI=sqlite:///mlruns.db
REGISTRY_MODEL_NAME=crypto-news-sentiment
REGISTRY_MODEL_STAGE=Staging
```

## 📊 Data Format

The system supports multiple input formats with flexible field mapping:

### Required Fields

- `text`: Main text content for sentiment analysis
- `label`: Sentiment label (NEGATIVE, NEUTRAL, POSITIVE)

### Optional Fields

- `id`: Article identifier
- `title`: Article title
- `published_at`: Publication date
- `tickers`: Related cryptocurrency symbols
- `split`: Data split assignment (train/val/test)

### Example JSON Format

```json
{
  "text": "Bitcoin price surges to new all-time high...",
  "label": "POSITIVE",
  "title": "Bitcoin Reaches New High",
  "published_at": "2025-01-15T10:00:00Z",
  "tickers": ["BTC", "ETH"]
}
```

## 🔧 CLI Commands

### Training

```bash
sentiment-train [OPTIONS]
  --data PATH              Input data file path
  --output PATH            Output directory
  --experiment TEXT        MLflow experiment name
  --config PATH            Configuration file
  --log-level TEXT         Logging level
```

### Export

```bash
sentiment-export MODEL_PATH [OPTIONS]
  --output PATH            Output directory
  --format TEXT            Export format (onnx/torchscript/both)
  --validate/--no-validate Validate exported models
```

### Registration

```bash
sentiment-register MODEL_PATH [OPTIONS]
  --run-id TEXT            MLflow run ID
  --description TEXT       Model description
  --stage TEXT             Initial stage (Staging/Production/Archived)
```

### Utilities

```bash
sentiment-list-models      # List models in registry
sentiment-info             # Show configuration info
```

## 🐳 Docker Support

### Build Images

```bash
# Development image
make docker-build-dev

# Production image
make docker-build-prod

# Latest image
make docker-build
```

### Run Containers

```bash
# Development container
make docker-run-dev

# Production container
make docker-run
```

### Docker Compose

```bash
# Start development environment
make docker-compose-dev

# Start production environment
make docker-compose-prod
```

## 🧪 Testing

### Run Tests

```bash
# All tests
make test

# With coverage
pytest --cov=sentiment_analysis_model_builder --cov-report=html

# Specific test file
pytest tests/test_data_loader.py -v
```

### Code Quality

```bash
# Linting
make lint

# Formatting
make format

# All quality checks
make quality
```

### Pre-commit Hooks

```bash
# Install hooks
make install-dev

# Run manually
make pre-commit
```

## 📈 MLflow Integration

### Local Development

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns.db

# View experiments
open http://localhost:5000
```

### Remote Tracking

```bash
# Set remote tracking URI
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000

# Set artifact storage
export MLFLOW_ARTIFACT_STORE=s3://your-bucket/path
```

### Model Registry

```bash
# List models
sentiment-list-models

# Get model info
mlflow models describe --name crypto-news-sentiment

# Transition stages
mlflow models transition-stage --name crypto-news-sentiment --version 1 --stage Production
```

## 🚀 Production Deployment

### Environment Variables

```bash
# Production configuration
export LOG_LEVEL=WARNING
export REGISTRY_TRACKING_URI=http://mlflow-prod:5000
export REGISTRY_ARTIFACT_LOCATION=s3://prod-models
export REGISTRY_AWS_ACCESS_KEY_ID=your-key
export REGISTRY_AWS_SECRET_ACCESS_KEY=your-secret
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-analysis-builder
spec:
  replicas: 1
  selector:
    matchLabels:
      app: sentiment-analysis-builder
  template:
    metadata:
      labels:
        app: sentiment-analysis-builder
    spec:
      containers:
        - name: sentiment-analysis-builder
          image: sentiment-analysis-builder:prod
          env:
            - name: REGISTRY_TRACKING_URI
              value: "http://mlflow-service:5000"
          volumeMounts:
            - name: data-volume
              mountPath: /app/data
            - name: outputs-volume
              mountPath: /app/outputs
      volumes:
        - name: data-volume
          persistentVolumeClaim:
            claimName: data-pvc
        - name: outputs-volume
          persistentVolumeClaim:
            claimName: outputs-pvc
```

## 🔍 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size
export TRAINING_BATCH_SIZE=8
export TRAINING_EVAL_BATCH_SIZE=16

# Use gradient accumulation
export TRAINING_GRADIENT_ACCUMULATION_STEPS=2
```

#### 2. MLflow Connection Issues

```bash
# Check tracking URI
sentiment-info

# Test connection
python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlruns.db'); print('OK')"
```

#### 3. Data Loading Errors

```bash
# Validate data format
python -c "from sentiment_analysis_model_builder.data.data_loader import DataLoader; print('OK')"

# Check file permissions
ls -la data/news_dataset_sample.json
```

#### 4. Model Export Failures

```bash
# Check ONNX opset compatibility
export EXPORT_ONNX_OPSET_VERSION=16

# Disable validation for debugging
export EXPORT_VALIDATE_EXPORT=false
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
sentiment-train --log-level DEBUG
```

### Performance Optimization

```bash
# Use mixed precision training
export TRAINING_FP16=true

# Enable gradient checkpointing
export TRAINING_GRADIENT_CHECKPOINTING=true

# Optimize data loading
export TRAINING_DATALOADER_NUM_WORKERS=4
```

## 📚 API Reference

### Core Classes

#### Config

Main configuration class combining all sub-configurations.

```python
from sentiment_analysis_model_builder.core.config import Config

config = Config()
print(config.training.backbone)
print(config.preprocessing.label_mapping)
```

#### DataLoader

Handles data loading and preprocessing.

```python
from sentiment_analysis_model_builder.data.data_loader import DataLoader

loader = DataLoader(config.data, config.preprocessing)
articles = loader.load_data()
```

#### SentimentTrainer

Manages model training and evaluation.

```python
from sentiment_analysis_model_builder.models.trainer import SentimentTrainer

trainer = SentimentTrainer(config.training, dataset_preparator)
model, tokenizer, metrics = trainer.train(datasets, output_dir)
```

#### ModelExporter

Handles model export to various formats.

```python
from sentiment_analysis_model_builder.models.exporter import ModelExporter

exporter = ModelExporter(config.export)
export_paths = exporter.export_model(model_path, output_dir)
```

#### MLflowRegistry

Manages model registration and versioning.

```python
from sentiment_analysis_model_builder.registry.mlflow_registry import MLflowRegistry

registry = MLflowRegistry(config.registry)
model_uri = registry.register_model(model_path, run_id)
```

## 🤝 Contributing

### Development Setup

```bash
# Clone and setup
git clone <repository-url>
cd sentiment_analysis_model_builder
make dev-setup

# Run quality checks
make quality

# Run tests
make test
```

### Code Style

- Follow PEP 8 with 88-character line length
- Use type hints for all functions
- Include comprehensive docstrings
- Run pre-commit hooks before committing

### Testing Guidelines

- Maintain 80%+ code coverage
- Include unit and integration tests
- Test error conditions and edge cases
- Use descriptive test names

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the transformer models
- [MLflow](https://mlflow.org/) for experiment tracking and model management
- [ProsusAI FinBERT](https://huggingface.co/ProsusAI/finbert) for the base model
- [PyTorch](https://pytorch.org/) for the deep learning framework
