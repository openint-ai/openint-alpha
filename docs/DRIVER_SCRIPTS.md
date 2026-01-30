# Driver Scripts Guide

Driver scripts to start and manage OpenInt services.

## Available Scripts

### 1. `start_backend.sh`
Starts the backend API server.

```bash
./start_backend.sh
```

**Options:**
- `PORT=5001 ./start_backend.sh` - Use custom port

**Features:**
- Creates virtual environment if needed
- Installs dependencies automatically
- Checks if port is already in use
- Runs on port 3001 by default

### 2. `start_ui.sh`
Starts the React frontend development server.

```bash
./start_ui.sh
```

**Options:**
- `FRONTEND_PORT=3001 ./start_ui.sh` - Use custom port
- `VITE_API_URL=http://localhost:5001 ./start_ui.sh` - Set backend API URL

**Features:**
- Checks for Node.js and npm
- Installs dependencies if needed
- Configures API URL automatically
- Runs on port 3000 by default

### 3. `start_all.sh`
Starts all services (agents + backend, optionally UI).

```bash
# Start agents and backend
./start_all.sh

# Start agents, backend, and UI
./start_all.sh --with-ui
# or
./start_all.sh -u
```

**Features:**
- Starts agent system
- Starts backend API
- Optionally starts UI
- Creates all necessary virtual environments
- Installs dependencies automatically

### 4. `start_services.sh`
Original script - starts agents and backend (legacy, use `start_all.sh` instead).

### 5. `stop_services.sh`
Stops all running services.

```bash
./stop_services.sh
```

**Features:**
- Stops agent system
- Stops backend
- Stops UI (if running)
- Cleans up PID files
- Kills processes by port as fallback

## Usage Examples

### Start Backend Only
```bash
./start_backend.sh
```

### Start UI Only
```bash
./start_ui.sh
```

### Start All Services (Backend + Agents)
```bash
./start_all.sh
```

### Start Everything (Including UI)
```bash
./start_all.sh --with-ui
```

### Custom Ports
```bash
# Backend on port 5001
PORT=5001 ./start_backend.sh

# UI on port 3001, connecting to backend on 5001
FRONTEND_PORT=3001 VITE_API_URL=http://localhost:5001 ./start_ui.sh
```

### Stop All Services
```bash
./stop_services.sh
```

## Service URLs

After starting services:

- **Backend API**: http://localhost:3001
- **UI**: http://localhost:3000
- **Agent System**: Internal (port 8001)

## Logs

View service logs:

```bash
# Agent system logs
tail -f .agent_system.log

# Backend logs
tail -f .backend.log

# UI logs
tail -f .ui.log
```

## Troubleshooting

### Port Already in Use
If a port is already in use:
1. Check what's using it: `lsof -i :3001`
2. Stop the service: `./stop_services.sh`
3. Or use a different port: `PORT=5001 ./start_backend.sh`

### Dependencies Not Installing
- Ensure Python 3.8+ is installed
- Ensure Node.js 18+ is installed (for UI)
- Check internet connection

### Services Not Starting
- Check logs in `.agent_system.log`, `.backend.log`, `.ui.log`
- Verify Milvus is running (for agents)
- Check environment variables in `.env`
