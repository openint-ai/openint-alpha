#!/bin/bash

# Driver script to start OpenInt UI
# Starts the React frontend development server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🎨 Starting OpenInt UI${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js not found${NC}"
    echo -e "${YELLOW}   Please install Node.js 18+ from https://nodejs.org/${NC}"
    exit 1
fi

# Check npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm not found${NC}"
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

# Navigate to UI directory
cd openint-ui

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    npm install
fi

# Check if UI is already running and kill it
# Vite defaults to port 5173, but we'll use 3000 for consistency
FRONTEND_PORT=${FRONTEND_PORT:-3000}
if check_port $FRONTEND_PORT; then
    echo -e "${YELLOW}⚠️  Port $FRONTEND_PORT already in use, killing existing process...${NC}"
    PID=$(lsof -ti:$FRONTEND_PORT 2>/dev/null | head -1)
    if [ ! -z "$PID" ]; then
        PROCESS=$(ps -p $PID -o comm= 2>/dev/null || echo "unknown")
        echo -e "${YELLOW}   Killing process $PID ($PROCESS)${NC}"
        kill $PID 2>/dev/null || kill -9 $PID 2>/dev/null || true
        sleep 1
        # Double-check and force kill if still running
        if check_port $FRONTEND_PORT; then
            lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
            sleep 1
        fi
        echo -e "${GREEN}✅ Port $FRONTEND_PORT freed${NC}"
    else
        lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
fi

# Set backend port (for proxy configuration)
BACKEND_PORT=${BACKEND_PORT:-3001}
export BACKEND_PORT=$BACKEND_PORT

# Warn if backend is not reachable (avoids ECONNREFUSED / "Failed to fetch" in browser)
backend_reachable=0
if command -v curl &>/dev/null; then
    curl -s -o /dev/null -w "%{http_code}" --connect-timeout 1 "http://127.0.0.1:$BACKEND_PORT/api/health" 2>/dev/null | grep -q 200 && backend_reachable=1
elif command -v nc &>/dev/null; then
    nc -z 127.0.0.1 $BACKEND_PORT 2>/dev/null && backend_reachable=1
fi
if [ "$backend_reachable" = "0" ]; then
    echo -e "${RED}⚠️  Backend not detected on port $BACKEND_PORT${NC}"
    echo -e "${YELLOW}   Start it in another terminal: ./start_backend.sh${NC}"
    echo -e "${YELLOW}   Otherwise you will see 'Failed to fetch' or proxy errors when using Chat.${NC}"
    echo ""
fi

# Set API URL if not set (for API client)
if [ -z "$VITE_API_URL" ]; then
    export VITE_API_URL="http://localhost:$BACKEND_PORT"
    echo -e "${YELLOW}💡 Using default API URL: $VITE_API_URL${NC}"
    echo -e "${YELLOW}   Set VITE_API_URL environment variable to change${NC}"
    echo ""
fi

# Start UI
echo -e "${GREEN}Starting UI on port $FRONTEND_PORT...${NC}"
echo -e "${BLUE}UI URL: http://localhost:$FRONTEND_PORT${NC}"
echo -e "${BLUE}Backend API: http://localhost:$BACKEND_PORT${NC}"
echo -e "${YELLOW}Note: Vite proxy configured in vite.config.ts${NC}"
echo ""

# Export port for vite.config.ts
export FRONTEND_PORT=$FRONTEND_PORT
# Log UI dev server output to ui.log in openint-ui (do not commit); also show in terminal
npm run dev -- --port $FRONTEND_PORT --host 0.0.0.0 2>&1 | tee -a ui.log
