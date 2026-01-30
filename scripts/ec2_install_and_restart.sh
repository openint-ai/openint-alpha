#!/usr/bin/env bash
# Run on EC2 after rsync. Installs Python deps and (re)starts the merged app.
# Usage: from repo root on EC2, run: ./scripts/ec2_install_and_restart.sh

set -e
cd "$(dirname "$0")/.."
APP_ROOT="$(pwd)"

if [ ! -d "openint-ui/dist" ]; then
  echo "openint-ui/dist not found. Run build.sh locally and deploy_to_ec2.sh first."
  exit 1
fi

# Single venv at repo root
VENV="$APP_ROOT/venv"
if [ ! -d "$VENV" ]; then
  echo "Creating venv at $VENV..."
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# Install all required packages (backend pulls in most; add agents/vectordb/graph for path resolution)
pip install -q -r openint-backend/requirements.txt
pip install -q -r openint-agents/requirements.txt
pip install -q -r openint-vectordb/requirements.txt
pip install -q -r openint-graph/requirements.txt

# Optional: gunicorn for production (install if not present)
pip install -q gunicorn 2>/dev/null || true

# Load .env if present (REDIS_HOST, MILVUS_HOST, NEO4J_URI, etc.)
if [ -f "$APP_ROOT/.env" ]; then
  set -a
  source "$APP_ROOT/.env"
  set +a
fi
export PORT="${PORT:-3001}"
export SERVE_UI=1
export FLASK_ENV=production
export PYTHONPATH="$APP_ROOT/openint-agents:$APP_ROOT/openint-vectordb/milvus:$APP_ROOT/openint-graph:$APP_ROOT"

# Restart: kill existing process on PORT if any, then start
if command -v lsof >/dev/null 2>&1; then
  OLD_PID=$(lsof -ti:$PORT 2>/dev/null || true)
  if [ -n "$OLD_PID" ]; then
    echo "Stopping existing process on port $PORT (PID $OLD_PID)..."
    kill $OLD_PID 2>/dev/null || true
    sleep 2
  fi
fi

cd "$APP_ROOT"
# Run from repo root so main.py resolves _repo_root = parent of openint-backend
if command -v gunicorn >/dev/null 2>&1; then
  echo "Starting OpenInt with gunicorn on 0.0.0.0:$PORT..."
  exec gunicorn -w 1 -b 0.0.0.0:$PORT --chdir openint-backend "main:app"
else
  echo "Starting OpenInt with python on 0.0.0.0:$PORT..."
  exec python3 openint-backend/main.py
fi
