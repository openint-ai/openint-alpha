# Troubleshooting Guide

Common issues and solutions for OpenInt services.

## Connection Refused Errors

### Issue: "ERR_CONNECTION_REFUSED" when accessing localhost

**Possible Causes:**
1. Service not actually running
2. Port conflicts
3. Corrupted node_modules (for UI)
4. Wrong port configuration

**Solutions:**

#### Check if services are running:
```bash
# Check ports
lsof -i :3000 -i :3001 -i :5173

# Check processes
ps aux | grep -E "(main.py|vite|npm)"
```

#### Fix UI node_modules:
```bash
./fix_ui.sh
# Or manually:
cd openint-ui
rm -rf node_modules package-lock.json
npm install
```

#### Fix port conflicts:
```bash
# Kill process on port 3001
lsof -ti:3001 | xargs kill -9

# Or use different port
PORT=5001 ./start_backend.sh
FRONTEND_PORT=3001 ./start_ui.sh
```

## Port Already in Use

### Backend Port 3001
```bash
# Option 1: Kill existing process
lsof -ti:3001 | xargs kill -9

# Option 2: Use different port
PORT=5001 ./start_backend.sh
```

### UI Port 3000
```bash
# Option 1: Kill existing process
lsof -ti:3000 | xargs kill -9

# Option 2: Use different port
FRONTEND_PORT=3001 ./start_ui.sh
```

## UI Not Starting

### Corrupted node_modules
```bash
./fix_ui.sh
```

### Missing dependencies
```bash
cd openint-ui
npm install
```

### Vite errors
Check `.ui.log` for specific errors:
```bash
tail -f .ui.log
```

## Backend Not Starting

### Missing dependencies
```bash
cd openint-backend
pip install -r requirements.txt
```

### Agent system not available
The backend needs the agent system. Start it first:
```bash
./start_all.sh
```

### Check logs
```bash
tail -f .backend.log
```

## Service URLs

After starting services, access:

- **Backend API**: http://localhost:3001
- **UI**: http://localhost:3000
- **Health Check**: http://localhost:3001/api/health

## Common Commands

```bash
# Stop all services
./stop_services.sh

# Check what's using ports
lsof -i :3000 -i :3001

# View logs
tail -f .backend.log
tail -f .ui.log
tail -f .agent_system.log

# Restart services
./stop_services.sh
./start_all.sh --with-ui
```

## Milvus: Collection Empty / No Search Results

If the Milvus collection exists but search returns no data:

### 1. Run the diagnostic script
```bash
python openint-vectordb/check_milvus.py
```
This checks connection, entity count, and search.

### 2. Verify loader found testdata
The loader looks for `openint-testdata/testdata/`. Generate data first:
```bash
cd openint-testdata
python generators/generate_openint_test_data.py
python loaders/load_openint_data_to_milvus.py
```

### 3. Ensure .env is used
Both loader and backend use `MILVUS_COLLECTION` from `.env` (repo root). If they use different collections, data won't appear. Run the loader from repo root or openint-testdata so it finds the same `.env`:
```bash
# From repo root
python openint-testdata/loaders/load_openint_data_to_milvus.py
```

### 4. Check embedding model
The loader needs `sentence-transformers` for embeddings. If it fails, batches are skipped:
```bash
pip install sentence-transformers
```

---

## Environment Variables

Set these if needed:

```bash
# Backend port
export PORT=5001

# UI port
export FRONTEND_PORT=3001

# Backend API URL (for UI)
export VITE_API_URL=http://localhost:5001
```
