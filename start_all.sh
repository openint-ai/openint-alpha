#!/bin/bash

# Driver script to start all OpenInt services
# Starts agents, backend, and optionally UI

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
echo -e "${BLUE}🏦 Starting All OpenInt Services${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Parse arguments
START_UI=false
if [[ "$1" == "--with-ui" ]] || [[ "$1" == "-u" ]]; then
    START_UI=true
fi

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

# Check if agent system is already running and kill it
if check_port 8001; then
    echo -e "${YELLOW}⚠️  Port 8001 already in use, killing existing process...${NC}"
    PID=$(lsof -ti:8001 2>/dev/null | head -1)
    if [ ! -z "$PID" ]; then
        kill $PID 2>/dev/null || kill -9 $PID 2>/dev/null || true
        sleep 1
    fi
    lsof -ti:8001 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

echo -e "${GREEN}Starting agent system...${NC}"
python3 main.py > ../.agent_system.log 2>&1 &
AGENT_PID=$!
echo $AGENT_PID > ../.agent_system.pid
echo -e "${GREEN}✅ Agent system started (PID: $AGENT_PID)${NC}"

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

echo -e "${GREEN}Starting backend on port $PORT...${NC}"
PORT=$PORT python3 main.py > ../.backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > ../.backend.pid
echo -e "${GREEN}✅ Backend started (PID: $BACKEND_PID)${NC}"

cd ..

# Start UI if requested
if [ "$START_UI" = true ]; then
    echo ""
    echo -e "${BLUE}🎨 Starting UI...${NC}"
    
    if ! command -v npm &> /dev/null; then
        echo -e "${YELLOW}⚠️  npm not found, skipping UI${NC}"
        echo -e "${YELLOW}   Install Node.js to start UI${NC}"
    else
        cd openint-ui
        
        if [ ! -d "node_modules" ]; then
            echo -e "${YELLOW}Installing UI dependencies...${NC}"
            npm install
        fi
        
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
        
        export VITE_API_URL="http://localhost:$PORT"
        echo -e "${GREEN}Starting UI on port $FRONTEND_PORT...${NC}"
        npm run dev -- --port $FRONTEND_PORT > ../.ui.log 2>&1 &
        UI_PID=$!
        echo $UI_PID > ../.ui.pid
        echo -e "${GREEN}✅ UI started (PID: $UI_PID)${NC}"
        
        cd ..
    fi
fi

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ Services Started${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Services:${NC}"
echo -e "  • Agent System: ${BLUE}http://localhost:8001${NC} (internal)"
echo -e "  • Backend:       ${BLUE}http://localhost:$PORT${NC}"

if [ "$START_UI" = true ]; then
    echo -e "  • UI:            ${BLUE}http://localhost:${FRONTEND_PORT:-3000}${NC}"
fi

echo ""
echo -e "${YELLOW}To stop services:${NC}"
echo -e "  ./stop_services.sh"
echo ""
echo -e "${YELLOW}To view logs:${NC}"
echo -e "  tail -f .agent_system.log"
echo -e "  tail -f .backend.log"
if [ "$START_UI" = true ]; then
    echo -e "  tail -f .ui.log"
fi
echo ""
echo -e "${YELLOW}To start UI separately:${NC}"
echo -e "  ./start_ui.sh"
echo ""
