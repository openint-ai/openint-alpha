#!/bin/bash

# Stop script for OpenInt System Services

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Stopping OpenInt System Services...${NC}"

# Stop Agent System
if [ -f ".agent_system.pid" ]; then
    PID=$(cat .agent_system.pid)
    if ps -p $PID > /dev/null 2>&1; then
        kill $PID
        echo -e "${GREEN}✅ Stopped agent system (PID: $PID)${NC}"
    else
        echo -e "${YELLOW}⚠️  Agent system process not found${NC}"
    fi
    rm -f .agent_system.pid
else
    echo -e "${YELLOW}⚠️  No agent system PID file found${NC}"
fi

# Stop Backend
if [ -f ".backend.pid" ]; then
    PID=$(cat .backend.pid)
    if ps -p $PID > /dev/null 2>&1; then
        kill $PID
        echo -e "${GREEN}✅ Stopped backend (PID: $PID)${NC}"
    else
        echo -e "${YELLOW}⚠️  Backend process not found${NC}"
    fi
    rm -f .backend.pid
else
    echo -e "${YELLOW}⚠️  No backend PID file found${NC}"
fi

# Also check for old UI backend PID file
if [ -f ".ui_backend.pid" ]; then
    PID=$(cat .ui_backend.pid)
    if ps -p $PID > /dev/null 2>&1; then
        kill $PID
        echo -e "${GREEN}✅ Stopped old UI backend (PID: $PID)${NC}"
    fi
    rm -f .ui_backend.pid
fi

# Stop UI if running
if [ -f ".ui.pid" ]; then
    PID=$(cat .ui.pid)
    if ps -p $PID > /dev/null 2>&1; then
        kill $PID
        echo -e "${GREEN}✅ Stopped UI (PID: $PID)${NC}"
    else
        echo -e "${YELLOW}⚠️  UI process not found${NC}"
    fi
    rm -f .ui.pid
else
    echo -e "${YELLOW}⚠️  No UI PID file found${NC}"
fi

# Also try to kill by port (fallback)
if command -v lsof &> /dev/null; then
    # Kill processes on port 8001 (agents)
    lsof -ti:8001 | xargs kill -9 2>/dev/null || true
    # Kill processes on port 3001 (backend)
    lsof -ti:3001 | xargs kill -9 2>/dev/null || true
    # Also check old port 5000
    lsof -ti:5000 | xargs kill -9 2>/dev/null || true
    # Kill processes on port 3000 (UI)
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
fi

echo -e "${GREEN}✅ All services stopped${NC}"
