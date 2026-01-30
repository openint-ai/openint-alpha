# OpenInt MCP Server — Semantic Layer for Agents

This MCP (Model Context Protocol) server exposes the **OpenInt semantic layer** so any MCP-capable agent (Cursor, Claude Desktop, custom assistants) can interpret natural language sentences and list available models via tools.

## Prerequisites

- Python 3.10+
- **OpenInt backend** running and reachable (e.g. `http://localhost:3001`). The MCP server calls the backend HTTP API; it does not run embedding models itself. The backend uses **modelmgmt-agent** for semantic interpretation when the agent system is loaded.

## Install

```bash
cd openint-mcp
pip install -r requirements.txt
# or: uv pip install -r requirements.txt
```

## Configure

| Variable | Description | Default |
|----------|-------------|--------|
| `OPENINT_BACKEND_URL` | Base URL of the OpenInt backend API | `http://localhost:3001` |

Example:

```bash
export OPENINT_BACKEND_URL=http://localhost:3001
```

## Run (stdio)

For Cursor, Claude Desktop, and other clients that use stdio:

```bash
python server.py
# or: uv run server.py
```

The server reads/writes JSON-RPC on stdin/stdout.

## MCP Tools

| Tool | Purpose |
|------|---------|
| `semantic_interpret` | Interpret a sentence with a given model. Args: `sentence` (required), `model` (optional, default `mukaj/fin-mpnet-base`). Returns tags, highlighted_segments, token_semantics, embedding_stats. |
| `semantic_interpret_all` | Interpret a sentence with **all** supported models (same as UI dropdown). Arg: `sentence`. Returns `{ success, query, models: { model_id: result } }`. |
| `semantic_list_models` | List available model IDs for semantic interpretation. No args. |
| `semantic_list_models_with_meta` | List supported models with metadata (id, display_name, author, description, details, url). No args. |

## Cursor

1. Start the OpenInt backend (e.g. `./start_backend.sh` or `python openint-backend/main.py`).
2. In Cursor: **Settings → MCP → Add server** (or edit your MCP config).
3. Add a server that runs this MCP, for example:

   **Command:** `python`  
   **Args:** `server.py` (use the full path to `openint-mcp/server.py` if needed)  
   **Env:** `OPENINT_BACKEND_URL=http://localhost:3001`

4. The agent can then call `semantic_interpret` and `semantic_list_models` when it needs sentence interpretation or the model list.

## Claude Desktop

Configure the MCP server in Claude Desktop’s config (e.g. `claude_desktop_config.json`) with the same command, args, and env. See [MCP documentation](https://modelcontextprotocol.io) for your client’s exact format.

## Example agent flow

1. User: “What does ‘Show me transactions for customer 1001 in California’ mean semantically?”
2. Agent calls `semantic_interpret(sentence="Show me transactions for customer 1001 in California", model="mukaj/fin-mpnet-base")`.
3. MCP server calls `GET /api/semantic/interpret?sentence=...&model=...` on the backend and returns the JSON to the agent.
4. Agent summarizes tags (customer_id, state, intent, etc.) and optionally token_semantics for the user.

## REST API

The same capabilities are available directly via HTTP: see the [API docs](/api-docs) in the OpenInt UI (GET `/api/semantic/interpret`, POST `/api/semantic/preview`, GET `/api/semantic/models`).

## Plan

See [MCP_SEMANTIC_LAYER_PLAN.md](../MCP_SEMANTIC_LAYER_PLAN.md) in the repo root for the full plan (architecture, tools, transport, integration).
