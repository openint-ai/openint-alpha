# OpenInt Agents System

Multi-agent AI system for querying data, analytics, insights, and customer interactions. This repo contains the **agent runtime** (orchestrator, search/graph/schema agents). Developers typically connect via the **OpenInt backend API** or the **OpenInt MCP server**—not by running this package alone.

---

## For developers: connect via API or MCP

### Option 1 — REST API (recommended)

The **OpenInt backend** (`openint-backend`) runs the agent system in-process and exposes HTTP endpoints. Use it for chat, listing agents, and semantic interpretation.

**Base URL:** `http://localhost:3001` (set `PORT` to change)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Liveness/health check |
| `/api/ready` | GET | Whether the agent system is ready (may take 30–90s on first load) |
| `/api/chat` | POST | Send a message; agents process it and return an aggregated answer |
| `/api/agents` | GET | List available agents (name, description, capabilities) |
| `/api/query/<query_id>` | GET | Poll or fetch result for a query by ID |
| `/api/semantic/interpret` | GET | Semantic interpretation of a sentence (query params: `sentence`, `model`) |
| `/api/semantic/interpret-all` | GET/POST | Interpret with all supported models |
| `/api/semantic/models` | GET | List available semantic model IDs |

**Start the backend (from repo root):**

```bash
cd openint-backend
pip install -r requirements.txt
python main.py
# Listens on http://localhost:3001
```

**Example: chat**

```bash
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me transactions for customer 1001 in California"}'
```

**Example: list agents**

```bash
curl http://localhost:3001/api/agents
```

**Example: semantic interpret (API)**

```bash
curl "http://localhost:3001/api/semantic/interpret?sentence=Show%20me%20transactions%20in%20California&model=mukaj/fin-mpnet-base"
```

**Example: check readiness before chat**

```bash
curl http://localhost:3001/api/ready
# 200 => agent system ready; 503 => wait and retry
```

---

### Option 2 — MCP (Model Context Protocol)

Use the **OpenInt MCP server** (`openint-mcp`) so MCP-capable clients (Cursor, Claude Desktop, custom assistants) can call semantic tools. The MCP server talks to the OpenInt backend; it does not run agents itself.

**Prerequisites:** OpenInt backend running (e.g. `http://localhost:3001`).

**Start MCP server:**

```bash
cd openint-mcp
pip install -r requirements.txt
export OPENINT_BACKEND_URL=http://localhost:3001
python server.py
```

**MCP tools:**

| Tool | Purpose |
|------|---------|
| `semantic_interpret` | Interpret a sentence with a given model (`sentence`, optional `model`) |
| `semantic_interpret_all` | Interpret with all supported models |
| `semantic_list_models` | List available model IDs |
| `semantic_list_models_with_meta` | List models with metadata (display name, author, url) |

**Cursor:** In Settings → MCP, add a server with command `python`, args `server.py` (full path to `openint-mcp/server.py`), and env `OPENINT_BACKEND_URL=http://localhost:3001`.

**Claude Desktop:** Configure the same command/args/env in your MCP config (e.g. `claude_desktop_config.json`). See [openint-mcp/README.md](../openint-mcp/README.md) for details.

---

## Architecture (agent runtime)

### Agent types

- **Search Agent** — Semantic search in Milvus vector DB  
- **Graph Agent** — Graph/relationship queries  
- **Schema Generator Agent (sg-agent)** — DataHub-backed schema and example sentences  
- **Model Management Agent (modelmgmt-agent)** — Hugging Face + Redis model registry; annotates sentences with semantic tags  

### Communication

- **Message bus** — Pub/sub (in-process; Redis optional for scaling)  
- **Agent registry** — Service discovery and capabilities  
- **Orchestrator** — Routes queries to agents and aggregates responses  

---

## Running the agent system standalone

Use this for **development** or **programmatic use** (e.g. scripting) when you don’t need the HTTP API.

```bash
cd openint-agents
pip install -r requirements.txt
cp .env.example .env   # optional: set Milvus and API keys
python main.py
```

**Use the orchestrator in Python:**

```python
import sys
sys.path.insert(0, "/path/to/openint-agents")
from communication.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator()
response = orchestrator.process_query("Find customers with high credit scores in California")
# response has query_id, agent_responses, aggregated answer, etc.
```

In production, the **backend** runs the orchestrator and agents; clients use the **API** or **MCP**, not this entry point.

---

## Environment

- **openint-agents:** Optional `.env` for Milvus, Ollama (sg-agent), etc.  
- **openint-backend:** Uses same env; ensure backend can resolve `openint-agents` (e.g. repo layout or `PYTHONPATH`).  
- **openint-mcp:** `OPENINT_BACKEND_URL` (default `http://localhost:3001`).

---

## Adding or changing agents

See `agents/README.md` (if present) and the `agents/`, `sg_agent/`, and `modelmgmt_agent/` packages for implementing new agents and registering them with the orchestrator in `main.py`. **modelmgmt-agent** provides the model registry and semantic annotation used by the backend for `/api/semantic/*`.
