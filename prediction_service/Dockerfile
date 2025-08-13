# Multi-stage Dockerfile for FinSight AI Prediction Service
# Optimized for minimal size and fast builds

# Build stage for dependencies
FROM python:3.12-slim AS builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION="1.0.0"
ARG VCS_REF

# Add image metadata
LABEL maintainer="FinSight Team" \
  version="${VERSION}" \
  description="FinSight AI Prediction Service" \
  build-date="${BUILD_DATE}" \
  vcs-ref="${VCS_REF}"

# Set environment variables for build optimization
ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_DEFAULT_TIMEOUT=100 \
  DEBIAN_FRONTEND=noninteractive \
  # PyTorch specific optimizations
  TORCH_CUDA_ARCH_LIST="" \
  FORCE_CUDA=0

# Install only essential system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  curl \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean \
  && rm -rf /var/cache/apt/archives/*

# Create virtual environment with optimizations
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip with cache optimization
RUN pip install --upgrade pip==24.2 setuptools==70.3.0 wheel==0.44.0

# Copy and install common module first (leverage Docker layer caching)
# Note: Assumes common module is copied to build context as 'common_module'
COPY common_module /opt/common
WORKDIR /opt/common
RUN pip install --no-deps -e .

# Install Python dependencies with CPU-only PyTorch
COPY requirements-docker-minimal.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r /tmp/requirements.txt


# Production stage - minimal runtime image
FROM python:3.12-slim AS production

# Set optimized runtime environment variables
ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PATH="/opt/venv/bin:$PATH" \
  # Application settings
  APP_ENV=production \
  API_HOST=0.0.0.0 \
  API_PORT=8000 \
  DEBUG=false \
  # Performance optimizations
  FORCE_CPU=true \
  MALLOC_TRIM_THRESHOLD_=100000 \
  MALLOC_MMAP_THRESHOLD_=100000 \
  # Security
  PYTHONHASHSEED=random \
  # Disable telemetry
  PYTHONDONTWARNMETRYDISTANCE=1

# Create non-root user for security (use numeric ID for better compatibility)
RUN groupadd --gid 1001 --system appuser && \
  useradd --uid 1001 --gid 1001 --system --shell /bin/bash --create-home appuser

# Install minimal runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  ca-certificates \
  tini \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean \
  && rm -rf /var/cache/apt/archives/* /tmp/* /var/tmp/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy common module source to maintain editable installation
COPY --from=builder /opt/common /opt/common

# Create application directory with proper permissions
WORKDIR /app
RUN mkdir -p /app/logs /app/data /app/models /app/tmp \
  && chown -R appuser:appuser /app

# Copy application code and entrypoint script
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser scripts/docker-entrypoint.py /app/docker-entrypoint.py

# Switch to non-root user
USER appuser

# Health check with better error handling
HEALTHCHECK --interval=30s --timeout=15s --start-period=45s --retries=3 \
  CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Expose port
EXPOSE ${API_PORT}

# Use tini for proper signal handling and PID 1
ENTRYPOINT ["tini", "--", "python", "/app/docker-entrypoint.py"]
CMD ["server"]