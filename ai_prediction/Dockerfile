# Dockerfile

# Multi-stage Dockerfile for FinSight AI Prediction Service
FROM python:3.12-slim AS builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION="1.0.0"
ARG VCS_REF

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  curl \
  git \
  && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy and install common module first
COPY common /opt/common
WORKDIR /opt/common
RUN pip install -e .

# Copy requirements and install Python dependencies
COPY ai_prediction/requirements-docker-minimal.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt


# Production stage
FROM python:3.12-slim AS production

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PATH="/opt/venv/bin:$PATH" \
  APP_ENV=production \
  HOST=0.0.0.0 \
  PORT=8000 \
  API_HOST=0.0.0.0 \
  API_PORT=8000 \
  DEBUG=false

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
  useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy common module source to maintain editable installation
COPY --from=builder /opt/common /opt/common

# Create application directory and set permissions
WORKDIR /app
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Copy application code with proper ownership
COPY --chown=appuser:appuser ai_prediction/src/ /app/src/
COPY --chown=appuser:appuser ai_prediction/requirements.txt /app/
COPY --chown=appuser:appuser ai_prediction/scripts/docker-entrypoint.sh /app/docker-entrypoint.sh

# Create required directories and set permissions
RUN mkdir -p /app/logs /app/data /app/models && \
  chmod +x /app/docker-entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Expose port
EXPOSE ${API_PORT}

# Default commands
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["server"]