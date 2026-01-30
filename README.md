# OpenInt Alpha

## 🎬 Product Experience

How the product looks, feels, and works:

| Feature | Description |
|--------|---------------|
| **Chat** | Ask questions in natural language; get answers with semantic search and source citations. |
| **Compare** | Enter a sentence and see how 3 embedding models annotate it (green = agreement, amber = disagreement). |
| **A2A** | Agent-to-Agent: sg-agent generates sentences → modelmgmt-agent annotates them (Google A2A protocol). |

### Chat
![Chat](docs/gifs/chat.gif)
*Chat: natural-language queries, semantic search, and citations.*

### Compare
![Compare](docs/gifs/compare.gif)
*Compare: multi-model semantic annotation (3 models, agreement/difference highlighting).*

### A2A
![A2A](docs/gifs/a2a.gif)
*A2A: sentence generation (sg-agent) → semantic annotation (modelmgmt-agent) with wedge flow.*

> **To capture or update GIFs:** See [docs/gifs/README.md](docs/gifs/README.md) for steps to record each flow.

---

A scalable multi-project architecture for building agentic AI systems with multi-agent collaboration.

## 🏗️ Architecture

This project uses a modular architecture with separate projects for different concerns:

```
openint-alpha/
├── docs/                 # Documentation (setup, architecture, guides)
├── openint-agents/       # AI Agents System (Multi-Agent)
├── openint-backend/      # Backend API (Flask)
├── openint-ui/           # Frontend (React)
├── openint-testdata/     # Test Data Generation & Loading
├── openint-datahub/      # DataHub Metadata Integration
├── openint-vectordb/     # Vector DB (Milvus client)
├── openint-graph/        # Graph DB (Neo4j client)
├── openint-mcp/          # MCP Server
├── shared/               # Shared Utilities & Contracts
├── samples/              # Sample files
└── testdata/             # Generated Test Data (gitignored; generate via openint-testdata)
```

## 🎯 Features

- **Multi-Agent System**: Agents that communicate and collaborate
- **Agent Communication**: Message bus and orchestration framework
- **Vector Database**: Milvus integration for semantic search
- **REST API**: Backend API for frontend integration
- **Modern UI**: React frontend with TypeScript
- **Test Data**: Comprehensive test data generation tools
- **DataHub Integration**: Metadata catalog and governance (see [DATAHUB_SETUP.md](docs/DATAHUB_SETUP.md))

## 🚀 Quick Start

### 1. Start Agent System
```bash
cd openint-agents
pip install -r requirements.txt
python main.py
```

### 2. Start Backend
```bash
cd openint-backend
pip install -r requirements.txt
python main.py
```

### 3. Start Frontend
```bash
cd openint-ui
npm install
npm run dev
```

### Or use the startup script:
```bash
./start_services.sh  # Starts agents and backend
# Then start UI separately:
cd openint-ui && npm run dev
```

### 4. Start DataHub (Optional - for metadata catalog)
```bash
# Start DataHub services
docker-compose -f docker-compose.datahub.yml up -d

# Ingest testdata metadata
cd openint-datahub
python ingest_metadata.py

# See docs/DATAHUB_SETUP.md for detailed instructions
```

## 📦 Projects

### openint-agents
Multi-agent AI system with:
- Agent communication framework
- Message bus for inter-agent communication
- Agent registry for service discovery
- Orchestrator for coordinating workflows
- **modelmgmt-agent**: Model registry (Hugging Face + Redis) and sentence semantic annotation; used by the backend for `/api/semantic/*` when agents are loaded

### openint-backend
Flask API backend providing:
- REST API endpoints
- Gateway to agent system
- CORS support for frontend

### openint-ui
React frontend with:
- TypeScript
- Vite build system
- Chat interface for agent interaction

### openint-testdata
Test data generation and loading:
- Generate test data (customers, transactions)
- Load data into Milvus
- Data validation tools

## 📚 Documentation

Setup and reference docs live in [docs/](docs/) (DataHub, Quick Start, Troubleshooting, etc.).

- [Architecture Overview](docs/README_ARCHITECTURE.md) - Detailed architecture documentation
- [Migration Guide](docs/MIGRATION_GUIDE.md) - Guide for migrating from old structure
- [Agent System](openint-agents/README.md) - Agent system documentation
- [Backend API](openint-backend/README.md) - Backend API documentation
- [Frontend](openint-ui/README.md) - Frontend documentation
- [Test Data](openint-testdata/README.md) - Test data generation guide

## 🔧 Configuration

Set environment variables in `.env`:
- `MILVUS_HOST` - Milvus host (default: localhost)
- `MILVUS_PORT` - Milvus port (default: 19530)
- `PORT` - Backend port (default: 5000)

## 🛠️ Development

Each project has its own dependencies and can be developed independently:
- `openint-agents/requirements.txt`
- `openint-backend/requirements.txt`
- `openint-testdata/requirements.txt`
- `openint-ui/package.json`

## 📝 License

[Your License Here]
