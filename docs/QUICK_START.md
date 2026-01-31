# Quick Start Guide

Get OpenInt services running quickly.

## Prerequisites

- Python 3.8+
- Node.js 18+
- Milvus running (Docker or standalone)

## Quick Start

### Option 1: Start Everything at Once

```bash
# Start agents + backend + UI
./start_all.sh --with-ui
```

### Option 2: Start Services Separately

**Terminal 1 - Backend:**
```bash
./start_backend.sh
```

**Terminal 2 - UI:**
```bash
./start_ui.sh
```

**Terminal 3 - Agents (if needed separately):**
```bash
cd openint-agents
source venv/bin/activate
python3 main.py
```

## First Time Setup

### 1. Fix UI (if needed)
```bash
./fix_ui.sh
```

### 2. Install Backend Dependencies
```bash
cd openint-backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Install UI Dependencies
```bash
cd openint-ui
npm install
```

## Verify Services

### Check Backend
```bash
curl http://localhost:3001/api/health
```

### Check UI
Open browser: http://localhost:3000

### Check Agents
```bash
curl http://localhost:3001/api/agents
```

## Common Issues

### Port Already in Use
```bash
# Kill process on port
lsof -ti:3001 | xargs kill -9
lsof -ti:3000 | xargs kill -9
```

### UI Not Starting
```bash
./fix_ui.sh
```

### Connection Refused
1. Check if services are running: `ps aux | grep -E "(main.py|vite)"`
2. Check ports: `lsof -i :3000 -i :3001`
3. Check logs: `tail -f .backend.log` or `tail -f .ui.log`

## Stop Services

```bash
./stop_services.sh
```

## Service URLs

- **Backend API**: http://localhost:3001
- **UI**: http://localhost:3000
- **API Health**: http://localhost:3001/api/health
- **List Agents**: http://localhost:3001/api/agents

## Next Steps

1. Generate data: `cd openint-testdata && python generators/generate_openint_test_data.py --quick` (output: openint-testdata/data/)
2. Load data: `python loaders/load_openint_data_to_milvus.py`
3. Start using the UI!
