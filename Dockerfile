# Use the official uv image with Debian slim
FROM ghcr.io/astral-sh/uv:debian-slim

# Set working directory
WORKDIR /app

# Enable bytecode compilation for performance
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files first for Docker layer caching
COPY pyproject.toml uv.lock ./

# Explicitly install Python 3.14 as strictly required by pyproject.toml
RUN uv python install 3.14

# Install dependencies (this caches the layer)
RUN uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application code
COPY . /app

# Sync the project execution entry points
RUN uv sync --frozen --no-dev

# Expose default port
EXPOSE 8082

# Start the Proxy.
# Render automatically injects the $PORT env var which pydantic seamlessly overrides 'port' with.
CMD [".venv/bin/python", "server.py"]
