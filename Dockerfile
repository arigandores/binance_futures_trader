# Dockerfile for Binance Sector Shot Detector
# Multi-stage build with Poetry for Python 3.12
# Production-ready with non-root user and security best practices

# =============================================================================
# Stage 1: Builder - Install dependencies with Poetry
# =============================================================================
FROM python:3.12-slim as builder

# Environment variables for Poetry and Python optimization
ENV POETRY_VERSION=1.8.2 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install Poetry
RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

# Set working directory
WORKDIR /app

# Copy dependency files first (for cache optimization)
COPY pyproject.toml poetry.lock ./

# Install dependencies only (not the project itself yet)
# --only=main: Install only main dependencies (exclude dev)
# --no-root: Don't install the project package yet
# --no-directory: Don't look for project files (they don't exist yet)
RUN poetry install --only=main --no-root --no-directory

# Copy application code
COPY detector/ ./detector/

# Install the project itself (now that code is present)
RUN poetry install --only-root

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM python:3.12-slim as runtime

# Python optimization environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user for security
# UID/GID 1000 for compatibility with most systems
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application code from builder stage
COPY --from=builder --chown=appuser:appuser /app/detector /app/detector

# Activate virtual environment by adding to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Create data directory for SQLite database
RUN mkdir -p /app/data && chown appuser:appuser /app/data

# Health check to verify SQLite database accessibility
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sqlite3; conn=sqlite3.connect('/app/data/market.db'); conn.close()" || exit 1

# Switch to non-root user
USER appuser

# Default command: run detector with config.yaml
CMD ["python", "-m", "detector", "run", "--config", "config.yaml"]
