# OpenInt Alpha

## 🎬 Product Experience

How the product looks, feels, and works:

| Feature | Description |
|--------|---------------|
| **Chat** | Ask questions in natural language; get answers with semantic search and source citations. |
| **Compare** | Enter a sentence and see how 3 embedding models annotate it (green = agreement, amber = disagreement). |
| **A2A** | Agent-to-Agent: all agent communication uses the A2A protocol; LangGraph orchestrates select_agents → run_agents → aggregate; sg-agent and modelmgmt-agent run the sentence → annotation flow. |

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
└── data/                 # Generated Data (gitignored; generate via openint-testdata)
```

## 🎯 Features

- **Multi-Agent System**: Agents that communicate and collaborate
- **Agent Communication**: All agent communication over **A2A (Agent-to-Agent)** protocol; LangGraph orchestration (select_agents → run_agents → aggregate)
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
- **A2A protocol**: All agent communication (search_agent, graph_agent, sg-agent, modelmgmt-agent) goes over Google A2A (Agent Card + message/send)
- **LangGraph orchestration**: select_agents → run_agents → aggregate; backend passes an A2A runner so orchestrator invokes agents via A2A
- Agent registry for service discovery
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

## Deploy to AWS EC2 (merged backend+UI)

Build the UI and deploy to an EC2 instance in one flow:

```bash
./build.sh
export OPENINT_EC2_KEY=~/.ssh/openint.pem
export OPENINT_EC2_HOST=ec2-user@ec2-3-148-183-18.us-east-2.compute.amazonaws.com
./deploy_to_ec2.sh
```

See [docs/DEPLOY_EC2.md](docs/DEPLOY_EC2.md) for env vars, systemd, and Nginx.

## 📚 Documentation

Setup and reference docs live in [docs/](docs/) (DataHub, Quick Start, Troubleshooting, etc.).

- [Deploy to EC2](docs/DEPLOY_EC2.md) - Build merged backend+UI and deploy to AWS EC2
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
