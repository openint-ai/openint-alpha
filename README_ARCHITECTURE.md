# openInt System - Multi-Project Architecture

This repository contains a scalable multi-project architecture for a openInt system with AI agents.

## 🏗️ Architecture Overview

```
openint-alpha/
├── openInt-ui-backend/      # UI & API Backend
├── openInt-testdata/        # Test Data Generation & Loading  
├── openInt-agents/          # AI Agents System (Multi-Agent)
└── shared/               # Shared Utilities & Contracts
```

## 📦 Projects

### 1. openInt-ui-backend
**Purpose**: User interface and API gateway

- React frontend for chat interface
- Flask/FastAPI backend
- REST API endpoints
- WebSocket for real-time communication
- API gateway to agent system

**Quick Start**:
```bash
cd openInt-ui-backend/backend
pip install -r requirements.txt
python main.py
```

### 2. openInt-testdata
**Purpose**: Test data generation and loading

- Generate openInt test data (customers, transactions)
- Load data into Milvus vector database
- Data validation and quality checks

**Quick Start**:
```bash
cd openInt-testdata
pip install -r requirements.txt
python generators/generate_openInt_test_data.py --quick
python loaders/load_openInt_data_to_milvus.py
```

### 3. openInt-agents
**Purpose**: AI agent system with multi-agent collaboration

- Multiple specialized agents (Search, Analysis, Recommendation, etc.)
- Agent communication framework (Message Bus)
- Agent registry and orchestration
- Milvus integration for vector search
- Multi-agent collaboration

**Quick Start**:
```bash
cd openInt-agents
pip install -r requirements.txt
python main.py
```

### 4. shared
**Purpose**: Shared utilities and contracts

- Common configuration
- API schemas
- Shared utilities

## 🤖 Agent System

### Agent Types

1. **Search Agent**: Semantic search in Milvus
2. **Analysis Agent**: Data analysis and insights
3. **Recommendation Agent**: Personalized recommendations
4. **Transaction Agent**: Transaction-specific queries
5. **Customer Agent**: Customer profile and history

### Agent Communication

Agents communicate through:
- **Message Bus**: Pub/sub messaging system
- **Agent Registry**: Service discovery
- **Orchestrator**: Coordinates multi-agent workflows

### Example Usage

```python
from openInt_agents.communication.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator()

# Process query with multi-agent collaboration
response = orchestrator.process_query(
    "Find customers with high credit scores in California"
)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 18+ (for frontend)
- Milvus running (Docker or standalone)
- Redis (optional, for production message bus)

### Setup

1. **Clone and navigate**:
```bash
cd openint-alpha
```

2. **Set up each project**:
```bash
# Test Data
cd openInt-testdata
pip install -r requirements.txt

# Agents
cd ../openInt-agents
pip install -r requirements.txt

# UI Backend
cd ../openInt-ui-backend/backend
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
# Copy .env.example files and configure
cp .env.example .env  # In each project
```

4. **Generate test data**:
```bash
cd openInt-testdata
python generators/generate_openInt_test_data.py --quick
python loaders/load_openInt_data_to_milvus.py
```

5. **Start services**:
```bash
# Terminal 1: Start agent system
cd openInt-agents
python main.py

# Terminal 2: Start UI backend
cd openInt-ui-backend/backend
python main.py

# Terminal 3: Start frontend (if separate)
cd openInt-ui-backend/frontend
npm install
npm run dev
```

## 📡 API Endpoints

### UI Backend (`http://localhost:3001`)

- `POST /api/chat` - Send chat message
- `GET /api/agents` - List available agents
- `GET /api/query/<query_id>` - Get query result
- `GET /api/health` - Health check

### Example API Call

```bash
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find customers with high credit scores",
    "session_id": "session123"
  }'
```

## 🔄 Communication Flow

```
User Query
    ↓
UI Backend (openInt-ui-backend)
    ↓
API Gateway
    ↓
Agent Orchestrator (openInt-agents)
    ↓
┌─────────────┬─────────────┬─────────────┐
│   Search    │  Analysis   │ Recommendation│
│   Agent     │   Agent     │    Agent      │
└─────────────┴─────────────┴─────────────┘
    ↓              ↓              ↓
Milvus Vector DB ← ─ ─ ─ ─ ─ ─ ─ ┘
    ↓
Results Aggregation
    ↓
Response to User
```

## 🛠️ Development

### Adding a New Agent

1. Create agent class in `openInt-agents/agents/`:
```python
from agents.base_agent import BaseAgent, AgentResponse, AgentCapability

class MyAgent(BaseAgent):
    def __init__(self):
        capabilities = [
            AgentCapability(name="my_capability", description="...")
        ]
        super().__init__(
            name="my_agent",
            description="...",
            capabilities=capabilities
        )
    
    def process_query(self, query: str, context: Dict = None):
        # Implement your logic
        return AgentResponse(success=True, results=[...], message="...")
```

2. Register in `openInt-agents/main.py`:
```python
from agents.my_agent import MyAgent

def initialize_agents(orchestrator):
    my_agent = MyAgent()
    return [my_agent, ...]
```

### Adding API Endpoints

Add routes in `openInt-ui-backend/backend/main.py`:
```python
@app.route('/api/my-endpoint', methods=['POST'])
def my_endpoint():
    # Your logic
    return jsonify({"result": "..."})
```

## 📚 Documentation

- [Architecture Overview](ARCHITECTURE.md)
- [Migration Guide](MIGRATION_GUIDE.md)
- [Agent System README](openInt-agents/README.md)
- [Test Data README](openInt-testdata/README.md)
- [UI Backend README](openInt-ui-backend/README.md)

## 🔧 Configuration

See `shared/config/config.py` for shared configuration. Each project can override with its own `.env` file.

## 🐳 Docker (Future)

Each project can be containerized:
- `openInt-ui-backend/Dockerfile`
- `openInt-agents/Dockerfile`
- `openInt-testdata/Dockerfile`

Use `docker-compose.yml` to orchestrate all services.

## 📝 License

[Your License Here]
