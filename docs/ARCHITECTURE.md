# openInt System Architecture

This document describes the multi-project architecture for the openInt system.

## Project Structure

```
openint-alpha/
├── openInt-ui-backend/          # UI & API Backend
│   ├── frontend/            # React UI
│   ├── backend/             # Flask/FastAPI backend
│   └── README.md
│
├── openInt-testdata/           # Test Data Generation & Loading
│   ├── generators/          # Data generation scripts
│   ├── loaders/             # Data loading scripts
│   └── README.md
│
├── openInt-agents/             # AI Agents System
│   ├── agents/              # Individual agent implementations
│   ├── communication/        # Agent communication framework
│   ├── tools/               # Agent tools
│   └── README.md
│
└── shared/                   # Shared utilities and contracts
    ├── config/               # Shared configuration
    ├── schemas/             # API schemas and contracts
    └── utils/               # Shared utilities
```

## Project Responsibilities

### openInt-ui-backend
- **Purpose**: User interface and API backend
- **Responsibilities**:
  - React frontend for chat interface
  - REST/GraphQL API endpoints
  - WebSocket for real-time chat
  - User session management
  - API gateway to agent system

### openInt-testdata
- **Purpose**: Test data generation and loading
- **Responsibilities**:
  - Generate openInt test data (customers, transactions)
  - Load data into Milvus/vector databases
  - Data validation and quality checks
  - Data migration scripts

### openInt-agents
- **Purpose**: AI agent system with multi-agent collaboration
- **Responsibilities**:
  - Agent implementations (search, graph, sg-agent, modelmgmt-agent)
  - **modelmgmt-agent**: Model registry (Hugging Face + Redis), sentence semantic annotation; backend uses it for semantic API when agents are loaded
  - Agent communication framework
  - Milvus integration for vector search
  - Agent orchestration and coordination
  - Tool system for agent capabilities

## Agent Communication Framework

All agent communication uses the **A2A (Agent-to-Agent)** protocol (Google A2A spec: Agent Card + message/send). The backend exposes each agent as an A2A endpoint and the LangGraph orchestrator invokes agents via A2A.

1. **A2A Protocol**: Every agent (search_agent, graph_agent, sg-agent, modelmgmt-agent) is exposed as an A2A server; the orchestrator calls agents through `invoke_agent_via_a2a` so all traffic goes over A2A.
2. **LangGraph**: Orchestration flow select_agents → run_agents → aggregate; the backend passes an A2A runner into the orchestrator.
3. **Agent Registry**: Discovery and routing of agents.
4. **Message Bus**: Fallback for standalone agent runs (no backend).
5. **API Gateway**: External interface to the agent system.

## Communication Flow

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
Milvus Vector DB ← ─ ─ ─ ─ ─ ─ ─ ─ ┘
    ↓
Results Aggregation
    ↓
Response to User
```

## Technology Stack

- **UI Backend**: React + Flask/FastAPI
- **Agents**: Python with LangChain/autogen
- **Vector DB**: Milvus
- **Communication**: Redis/RabbitMQ for message bus
- **API**: REST + WebSocket
