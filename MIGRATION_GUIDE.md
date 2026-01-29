# Migration Guide: Monolith to Multi-Project Architecture

This guide helps migrate from the monolithic structure to the new multi-project architecture.

## Overview

The project has been split into three main projects:

1. **openInt-ui-backend**: UI and API backend
2. **openInt-testdata**: Test data generation and loading
3. **openInt-agents**: AI agents with multi-agent communication

## Migration Steps

### 1. Move UI Components

**From**: `web_ui.py`, `frontend/`, `templates/`, `static/`
**To**: `openInt-ui-backend/`

```bash
# Move frontend
mv frontend openInt-ui-backend/

# Move backend files
mv web_ui.py openInt-ui-backend/backend/main.py
mv templates openInt-ui-backend/backend/
mv static openInt-ui-backend/backend/
```

### 2. Move Test Data Components

**From**: `generate_openInt_test_data.py`, `load_openInt_data_to_milvus.py`
**To**: `openInt-testdata/`

```bash
# Already done - files are in openInt-testdata/generators/ and loaders/
```

### 3. Move Agent Components

**From**: `agent.py`, `milvus_client.py`, `tools.py`
**To**: `openInt-agents/`

```bash
# Copy milvus_client to agents (or create wrapper)
cp milvus_client.py openInt-agents/tools/
cp agent.py openInt-agents/agents/base_agent.py  # Already refactored
```

### 4. Update Imports

Update import paths in all files:

**Old**:
```python
from milvus_client import MilvusClient
```

**New**:
```python
import sys
sys.path.insert(0, '../openInt-agents')
from tools.milvus_client import MilvusClient
```

Or use proper package management (recommended).

### 5. Update Configuration

Create `.env` files in each project:

**openInt-ui-backend/.env**:
```
AGENT_API_URL=http://localhost:8001
PORT=5000
```

**openInt-agents/.env**:
```
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=openInt_data
```

**openInt-testdata/.env**:
```
MILVUS_HOST=localhost
MILVUS_PORT=19530
DATA_DIR=./data
```

### 6. Update Dependencies

Each project has its own `requirements.txt`:
- `openInt-ui-backend/backend/requirements.txt`
- `openInt-agents/requirements.txt`
- `openInt-testdata/requirements.txt`

Install dependencies for each project separately.

## Running the New Architecture

### Start Agent System

```bash
cd openInt-agents
pip install -r requirements.txt
python main.py
```

### Start UI Backend

```bash
cd openInt-ui-backend/backend
pip install -r requirements.txt
python main.py
```

### Generate Test Data

```bash
cd openInt-testdata
pip install -r requirements.txt
python generators/generate_openInt_test_data.py
python loaders/load_openInt_data_to_milvus.py
```

## Benefits

1. **Separation of Concerns**: Each project has a single responsibility
2. **Independent Deployment**: Deploy services independently
3. **Scalability**: Scale each service independently
4. **Maintainability**: Easier to maintain and test
5. **Team Collaboration**: Different teams can work on different projects

## Next Steps

1. Set up proper package management (pip install -e .)
2. Add Docker containers for each service
3. Set up CI/CD pipelines
4. Add API documentation (OpenAPI/Swagger)
5. Implement proper logging and monitoring
