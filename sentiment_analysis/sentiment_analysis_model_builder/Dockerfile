# Multi-stage Dockerfile for Sentiment Analysis Model Builder
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  git \
  && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Change ownership to app user
RUN chown -R app:app /app
USER app

# Create necessary directories
RUN mkdir -p outputs models/exported .cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import sentiment_analysis_model_builder; print('OK')" || exit 1

# Default command
CMD ["sentiment-train", "--help"]


# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov black isort ruff mypy

# Install pre-commit hooks
RUN pip install --no-cache-dir pre-commit

# Copy pre-commit configuration
COPY .pre-commit-config.yaml .

# Install pre-commit hooks
RUN pre-commit install

# Development command
CMD ["bash"]


# Production stage
FROM base as production

# Remove development tools
RUN pip uninstall -y pytest pytest-cov black isort ruff mypy pre-commit

# Optimize for production
ENV PYTHONOPTIMIZE=1

# Production command
CMD ["sentiment-train", "--help"]
