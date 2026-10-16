# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

LABEL maintainer="Bookeeper Contributors" \
      description="Bookeeper: Idea-based knowledge graph search engine and research workbench"

# Install essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    p7zip-full \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Environment configurations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    VIRTUAL_ENV="/app/.venv" \
    PATH="/app/.venv/bin:$PATH" \
    BOOKEEPER_HOST="0.0.0.0" \
    BOOKEEPER_PORT="8000"

WORKDIR /app

# Copy dependency manifests first to leverage Docker layer caching
COPY pyproject.toml uv.lock README.md ./
COPY archive_downloader/pyproject.toml archive_downloader/README.md ./archive_downloader/

# Synchronize dependencies into the virtual environment without installing project code yet
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source code
COPY src ./src
COPY archive_downloader/src ./archive_downloader/src
COPY config.yaml ./config.yaml.default

# Install the workspace packages into the virtual environment
RUN uv sync --frozen --no-dev

# Create data directories and setup non-root user
RUN mkdir -p /app/output /app/archive_downloader/downloads && \
    useradd -u 1000 -U -d /app -s /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Healthcheck to verify the web UI responds
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

ENTRYPOINT ["bookeeper"]
CMD ["ui", "--host", "0.0.0.0", "--port", "8000"]
