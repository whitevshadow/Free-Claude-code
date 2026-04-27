# Stage 1: Build Open WebUI frontend
FROM node:22-alpine AS webui-frontend
WORKDIR /app
COPY open-webui/package*.json ./
RUN npm ci --force
COPY open-webui/ .
RUN npm run build

# Stage 2: Final runtime
FROM python:3.14-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    USE_SLIM_DOCKER=true \
    PORT=8082 \
    PYTHONPATH=/app/backend

# Install system dependencies for Open WebUI and proxy
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    python3-dev \
    libmariadb-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    zstd \
    netcat-openbsd \
    jq \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir --upgrade pip uv

WORKDIR /app

# Install Open WebUI Python dependencies
COPY open-webui/backend/requirements.txt /tmp/open-webui-requirements.txt
RUN uv pip install --system -r /tmp/open-webui-requirements.txt

# Ensure gunicorn is available for Open WebUI
RUN uv pip install --system gunicorn

# Copy built frontend from the builder stage
COPY --from=webui-frontend /app/build ./build
COPY --from=webui-frontend /app/CHANGELOG.md ./CHANGELOG.md
COPY --from=webui-frontend /app/package.json ./package.json

# Copy Open WebUI source tree (includes backend, frontend source, etc.)
COPY open-webui/ ./

# Copy Claude Code Proxy sources and install proxy dependencies
COPY pyproject.toml uv.lock server.py /app/
COPY api/ /app/api/
COPY cli/ /app/cli/
COPY config/ /app/config/
COPY messaging/ /app/messaging/
COPY providers/ /app/providers/
COPY trees/ /app/trees/
COPY models_config.json /app/models_config.json
COPY nvidia_nim_models.json /app/nvidia_nim_models.json
RUN uv pip install --system .

# Copy and set the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose both common ports (Render will override PORT)
EXPOSE 8082 3000

# Use the unified entrypoint
CMD ["/entrypoint.sh"]
