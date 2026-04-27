#!/bin/bash
set -e

SERVICE_MODE="${SERVICE_MODE:-}"
if [ -z "$SERVICE_MODE" ]; then
  echo "ERROR: SERVICE_MODE environment variable is required (proxy or webui)" >&2
  exit 1
fi

PORT="${PORT:-10000}"
echo "Starting in $SERVICE_MODE mode on port $PORT"

if [ "$SERVICE_MODE" = "webui" ]; then
  echo "Starting Open WebUI"
  cd /app
  export PORT
  mkdir -p /app/backend/data
  chmod -R 755 /app/backend/data || true
  exec gunicorn open_webui.main:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 300
else
  echo "Starting Claude Code Proxy"
  cd /app
  export PORT
  exec python server.py
fi
