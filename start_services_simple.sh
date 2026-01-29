#!/bin/bash

# Simple startup script that shows output

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "🏦 Starting OpenInt System Services"
echo "=========================================="
echo ""

# Start Agent System in background with output
echo "🚀 Starting Agent System..."
cd openint-agents
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt
python3 main.py > ../.agent_system.log 2>&1 &
AGENT_PID=$!
echo "✅ Agent system started (PID: $AGENT_PID)"
echo "   Logs: .agent_system.log"
cd ..

sleep 2

# Start UI Backend
echo ""
echo "🚀 Starting UI Backend..."
cd openint-backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt
PORT=5001 python3 main.py > ../.backend.log 2>&1 &
BACKEND_PID=$!
echo "✅ UI backend started (PID: $BACKEND_PID)"
echo "   URL: http://localhost:5001"
echo "   Logs: .backend.log"
cd ..

echo ""
echo "=========================================="
echo "✅ Services Started"
echo "=========================================="
echo ""
echo "Agent System: http://localhost:8001 (check logs)"
echo "UI Backend:   http://localhost:5001"
echo ""
echo "View logs:"
echo "  tail -f .agent_system.log"
echo "  tail -f .ui_backend.log"
echo ""
echo "Stop services:"
echo "  kill $AGENT_PID $BACKEND_PID"
echo ""
