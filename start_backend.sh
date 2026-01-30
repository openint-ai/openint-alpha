#!/bin/bash

# Driver script to start OpenInt Backend
# Starts the Flask backend API server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env from repo root so HF_TOKEN (and others) are available for backend
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 Starting OpenInt Backend${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi

# Check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0
    else
        return 1
    fi
}

# Navigate to backend directory
cd openint-backend

# Create virtual environment if needed, or recreate if interpreter path is broken (e.g. moved project)
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
else
    if ! venv/bin/python3 -c "import sys" 2>/dev/null; then
        echo -e "${YELLOW}Removing broken venv (bad interpreter path)...${NC}"
        rm -rf venv requirements_installed.flag
        echo -e "${YELLOW}Creating fresh virtual environment...${NC}"
        python3 -m venv venv
    fi
fi

# Use venv binaries explicitly (avoids system pip / PEP 668)
VENV_PYTHON="venv/bin/python3"
VENV_PIP="venv/bin/pip"

# Install dependencies if needed
if [ ! -f "requirements_installed.flag" ]; then
    echo -e "${YELLOW}Installing dependencies into venv...${NC}"
    "$VENV_PIP" install -q -r requirements.txt
    touch requirements_installed.flag
else
    # Verify critical dependencies are installed (flask_cors, redis for model registry + chat cache)
    if ! "$VENV_PYTHON" -c "import flask_cors" 2>/dev/null || ! "$VENV_PYTHON" -c "import redis" 2>/dev/null; then
        echo -e "${YELLOW}Missing dependencies detected, reinstalling...${NC}"
        "$VENV_PIP" install -q -r requirements.txt
        touch requirements_installed.flag
    fi
fi

# Double-check flask_cors is available
if ! "$VENV_PYTHON" -c "import flask_cors" 2>/dev/null; then
    echo -e "${RED}❌ flask_cors still not available after installation${NC}"
    echo -e "${YELLOW}   Trying to install directly into venv...${NC}"
    "$VENV_PIP" install -q flask-cors
fi

# Ensure redis is available (model registry + chat cache)
if ! "$VENV_PYTHON" -c "import redis" 2>/dev/null; then
    echo -e "${YELLOW}Installing redis into venv (required for model registry and chat cache)...${NC}"
    "$VENV_PIP" install -q "redis>=5.0.0"
fi

# Check if backend is already running and kill it
PORT=${PORT:-3001}
if check_port $PORT; then
    echo -e "${YELLOW}⚠️  Port $PORT already in use, killing existing process...${NC}"
    PID=$(lsof -ti:$PORT 2>/dev/null | head -1)
    if [ ! -z "$PID" ]; then
        PROCESS=$(ps -p $PID -o comm= 2>/dev/null || echo "unknown")
        echo -e "${YELLOW}   Killing process $PID ($PROCESS)${NC}"
        kill $PID 2>/dev/null || kill -9 $PID 2>/dev/null || true
        sleep 1
        # Double-check and force kill if still running
        if check_port $PORT; then
            lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
            sleep 1
        fi
        echo -e "${GREEN}✅ Port $PORT freed${NC}"
    else
        lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
fi

# Start backend
echo -e "${GREEN}Starting backend on port $PORT...${NC}"
echo -e "${BLUE}Backend URL: http://localhost:$PORT${NC}"
echo ""

"$VENV_PYTHON" main.py
