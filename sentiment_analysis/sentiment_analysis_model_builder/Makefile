# Makefile for Sentiment Analysis Model Builder

.PHONY: help install install-dev test lint format clean build docker-build docker-run train export register

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests with coverage"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build the package"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  train        - Train a model (example)"
	@echo "  export       - Export a model (example)"
	@echo "  register     - Register a model (example)"

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

# Run tests
test:
	pytest tests/ -v --cov=sentiment_analysis_model_builder --cov-report=html --cov-report=term-missing

# Run linting
lint:
	ruff check .
	mypy sentiment_analysis_model_builder/

# Format code
format:
	black .
	isort .

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Build the package
build: clean
	python -m build

# Build Docker image
docker-build:
	docker build -t sentiment-analysis-builder:latest .

# Build Docker image (development)
docker-build-dev:
	docker build --target development -t sentiment-analysis-builder:dev .

# Build Docker image (production)
docker-build-prod:
	docker build --target production -t sentiment-analysis-builder:prod .

# Run Docker container
docker-run:
	docker run -it --rm -v $(PWD)/data:/app/data -v $(PWD)/outputs:/app/outputs sentiment-analysis-builder:latest

# Run Docker container (development)
docker-run-dev:
	docker run -it --rm -v $(PWD):/app -v $(PWD)/data:/app/data -v $(PWD)/outputs:/app/outputs sentiment-analysis-builder:dev

# Example: Train a model
train:
	sentiment-train --data data/news_dataset_sample.json --output outputs/training_run_$(shell date +%Y%m%d_%H%M%S)

# Example: Export a model
export:
	sentiment-export outputs/training_run_*/model --output models/exported --format onnx

# Example: Register a model (requires run_id from training)
register:
	@echo "Usage: make register RUN_ID=<mlflow_run_id>"
	@echo "Example: make register RUN_ID=abc123def456"
ifdef RUN_ID
	sentiment-register outputs/training_run_*/model --run-id $(RUN_ID)
else
	@echo "Please provide RUN_ID parameter"
	@exit 1
endif

# List available models
list-models:
	sentiment-list-models

# Show configuration info
info:
	sentiment-info

# Quick start: train, export, and register
quick-start: train export
	@echo "Training and export completed. Use 'make register RUN_ID=<run_id>' to register the model."

# Development workflow
dev-setup: install-dev
	@echo "Development environment setup complete!"

# Pre-commit hooks
pre-commit:
	pre-commit run --all-files

# Check code quality
quality: lint format test
	@echo "Code quality checks completed!"

# Docker Compose for development
docker-compose-dev:
	docker-compose -f docker-compose.dev.yml up -d

# Docker Compose for production
docker-compose-prod:
	docker-compose -f docker-compose.prod.yml up -d

# Stop Docker Compose services
docker-compose-stop:
	docker-compose down

# Show logs
logs:
	docker-compose logs -f

# Backup outputs
backup:
	tar -czf outputs_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz outputs/

# Clean outputs
clean-outputs:
	rm -rf outputs/*
	rm -rf models/exported/*

# Show system info
system-info:
	@echo "Python version:"
	python --version
	@echo "\nPyTorch version:"
	python -c "import torch; print(torch.__version__)"
	@echo "\nCUDA available:"
	python -c "import torch; print(torch.cuda.is_available())"
	@echo "\nMLflow version:"
	python -c "import mlflow; print(mlflow.__version__)"
