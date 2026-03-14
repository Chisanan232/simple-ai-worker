# Build stage
FROM python:3.13-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy only requirements files first to leverage Docker cache
COPY pyproject.toml uv.lock LICENSE README.md ./

# Copy application source so uv can build the editable install correctly
# (this populates _simple_ai_worker.pth with /app so `src` is importable)
COPY src/ ./src/

# Create virtual environment and install dependencies
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv sync --locked --all-extras

# Final stage
FROM python:3.13-slim

# Set environment variables
# For *PYTHONPATH="/app"*
    # Ensure the project root is on PYTHONPATH so 'src' is always importable
    # when running the installed entry-point scripts directly (without uv run).
# For *UV_NO_CACHE=1* and *UV_CACHE_DIR=/tmp/uv-cache*
    # Prevent uv from writing a cache inside the container's writable layer at
    # runtime.  Without this, `uv run` tries to create /app/.cache/uv and
    # immediately hits "No space left on device" because the root overlay
    # filesystem is full.  UV_NO_CACHE=1 disables all cache I/O; UV_CACHE_DIR
    # is a belt-and-suspenders fallback pointing at /tmp which has its own
    # tmpfs budget and is always writable.
# For *XDG_DATA_HOME=/tmp/crewai-data*
    # Redirect crewai's appdirs storage (ChromaDB, SQLite) away from
    # /app/.local (overlay layer, may be read-only at runtime) to /tmp
    # which is always a writable tmpfs mount in every Docker runtime.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    SERVER_PORT=8000 \
    UV_NO_CACHE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    XDG_DATA_HOME=/tmp/crewai-data

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# uv plants its own binary inside the venv at .venv/bin/uv (and uvx) during
# `uv sync`.  That binary is compiled for the builder stage's architecture and
# will fail with "Exec format error" at runtime when the final image runs on a
# different arch (e.g. amd64 builder → arm64 runtime, or cross-build on
# Apple Silicon).  The correct uv binary for this stage is already installed
# at /bin/uv via the COPY --from=ghcr.io/astral-sh/uv line above, so the
# venv copies must be removed to prevent PATH (/app/.venv/bin takes priority)
# from resolving to the wrong binary.
RUN rm -f /app/.venv/bin/uv /app/.venv/bin/uvx

# Copy application code
COPY . .

# Create a non-root user, pre-create writable directories needed at runtime,
# and set permissions.
#
# /app/.local/share is created here so that crewai's TokenManager and
# db_storage_path() (both of which call mkdir on ~/.local/share/...) succeed
# without hitting "No space left on device" on the Docker overlay layer.
# The directory is owned by appuser so no root privileges are needed at runtime.
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app appuser && \
    mkdir -p /app/.local/share/crewai/credentials && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port from environment variable
EXPOSE ${SERVER_PORT}

# Set the entry point
CMD ["bash", "./scripts/docker/run-server.sh"]
