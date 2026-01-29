#!/bin/bash

# Startup script for OpenInt System Services
# Starts agent system and backend (UI runs separately)

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
echo -e "${BLUE}🏦 Starting OpenInt System Services${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0
    else
        return 1
    fi
}

# Start Agent System
echo -e "${BLUE}🚀 Starting Agent System...${NC}"
cd openint-agents

if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate

if [ ! -f "requirements_installed.flag" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -q -r requirements.txt
    touch requirements_installed.flag
fi

# Check if agent system is already running
if check_port 8001; then
    echo -e "${YELLOW}⚠️  Port 8001 already in use (agent system may already be running)${NC}"
else
    echo -e "${GREEN}Starting agent system on port 8001...${NC}"
    python3 main.py &
    AGENT_PID=$!
    echo $AGENT_PID > ../.agent_system.pid
    echo -e "${GREEN}✅ Agent system started (PID: $AGENT_PID)${NC}"
fi

cd ..

# Wait a moment for agent system to initialize
sleep 2

# Start Backend
echo ""
echo -e "${BLUE}🚀 Starting Backend...${NC}"
cd openint-backend

if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate

if [ ! -f "requirements_installed.flag" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -q -r requirements.txt
    touch requirements_installed.flag
fi

# Check if backend is already running and kill it
PORT=${PORT:-3001}
if check_port $PORT; then
    echo -e "${YELLOW}⚠️  Port $PORT already in use, killing existing process...${NC}"
    PID=$(lsof -ti:$PORT 2>/dev/null | head -1)
    if [ ! -z "$PID" ]; then
        kill $PID 2>/dev/null || kill -9 $PID 2>/dev/null || true
        sleep 1
        lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
    echo -e "${GREEN}✅ Port $PORT freed${NC}"
fi

echo -e "${GREEN}Starting backend on port $PORT...${NC}"
PORT=$PORT python3 main.py &
BACKEND_PID=$!
echo $BACKEND_PID > ../.backend.pid
echo -e "${GREEN}✅ Backend started (PID: $BACKEND_PID)${NC}"

cd ..

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ Services Started${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Services:${NC}"
echo -e "  • Agent System: ${BLUE}http://localhost:8001${NC}"
echo -e "  • Backend:       ${BLUE}http://localhost:3001${NC}"
echo ""
echo -e "${YELLOW}To start UI separately:${NC}"
echo -e "  cd openint-ui && npm install && npm run dev"
echo ""
echo -e "${YELLOW}To stop services:${NC}"
echo -e "  ./stop_services.sh"
echo ""
echo -e "${YELLOW}To view logs:${NC}"
echo -e "  tail -f .agent_system.log"
echo -e "  tail -f .backend.log"
echo ""
