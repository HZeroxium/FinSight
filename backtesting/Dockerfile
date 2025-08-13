# Multi-stage Dockerfile for FinSight Backtesting Service
# Optimized for minimal size and fast builds

###########
# Builder #
###########
FROM python:3.12-slim AS builder

ARG BUILD_DATE
ARG VERSION="1.0.0"
ARG VCS_REF

LABEL maintainer="FinSight Team" \
  version="${VERSION}" \
  description="FinSight Backtesting Service" \
  build-date="${BUILD_DATE}" \
  vcs-ref="${VCS_REF}"

ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_DEFAULT_TIMEOUT=100 \
  DEBIAN_FRONTEND=noninteractive

# System deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  curl \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Isolated venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Tooling
RUN pip install --upgrade pip==24.2 setuptools==70.3.0 wheel==0.44.0

# Install common module first (for better layer caching)
COPY common_module /opt/common
WORKDIR /opt/common
RUN pip install --no-deps -e . || echo "Warning: Common module installation failed, continuing..."

# Dependencies (use BuildKit cache mount to speed up rebuilds)
# Tip: Ensure DOCKER_BUILDKIT=1 (Docker Desktop uses it by default)
COPY requirements-docker-minimal.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
  PIP_NO_CACHE_DIR=0 pip install -r /tmp/requirements.txt \
  && rm -f /tmp/requirements.txt

##############
# Production #
##############
FROM python:3.12-slim AS production

ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PATH="/opt/venv/bin:$PATH" \
  # App settings
  APP_ENV=production \
  API_HOST=0.0.0.0 \
  API_PORT=8000 \
  DEBUG=false \
  # glibc malloc tuning (optional; keep only if measured helpful)
  MALLOC_TRIM_THRESHOLD_=100000 \
  MALLOC_MMAP_THRESHOLD_=100000 \
  # Security
  PYTHONHASHSEED=random

# Non-root user
RUN groupadd --gid 1001 --system appuser && \
  useradd  --uid 1001 --gid 1001 --system --shell /bin/bash --create-home appuser

# Minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  ca-certificates \
  tini \
  && rm -rf /var/lib/apt/lists/*

# Bring venv + editable source directory
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/common /opt/common

# App files
WORKDIR /app
RUN mkdir -p /app/logs /app/data /app/models /app/cache /app/temp \
  && chown -R appuser:appuser /app

COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser scripts/docker-entrypoint.py /app/docker-entrypoint.py
COPY --chown=appuser:appuser market_data_job_config.json /app/market_data_job_config.json

USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=15s --start-period=45s --retries=3 \
  CMD curl -fsS "http://localhost:${API_PORT}/health" || exit 1

# Default config safeguard
RUN if [ ! -s /app/market_data_job_config.json ]; then \
  echo '{"priority_symbols": ["BTCUSDT", "ETHUSDT"], "max_lookback_days": 30}' > /app/market_data_job_config.json; \
  fi

EXPOSE ${API_PORT}

# Proper PID 1 handling
ENTRYPOINT ["tini", "--"]
CMD ["python", "/app/docker-entrypoint.py", "server"]
