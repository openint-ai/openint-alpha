# MCP Server Plan: Semantic Layer for Agents

## Goal

Expose the **semantic layer** (sentence interpretation) via an **MCP (Model Context Protocol) server** so any MCP-capable agent (Cursor, Claude Desktop, custom assistants) can call tools to:

- **Interpret a sentence** with a chosen embedding model → get tags, highlighted segments, token semantics.
- **List available models** → get model IDs to pass to the interpret tool.

Agents then use the semantic layer without calling the REST API directly; they use MCP tools instead.

---

## Architecture

```
Agent (Cursor / Claude / custom)
    │
    │  MCP (stdio or HTTP)
    ▼
openint-mcp server
    │
    │  HTTP (GET /api/semantic/interpret, GET /api/semantic/models)
    ▼
openint-backend (existing Flask API)
```

- **MCP server**: Lightweight; no embedding models. It forwards tool calls to the backend API.
- **Backend**: Uses **modelmgmt-agent** (openint-agents) for semantic logic when agents are loaded; otherwise falls back to local multi_model_semantic and model_registry. No change to existing APIs.

---

## MCP Tools

| Tool | Purpose | Arguments | Returns |
|------|---------|-----------|--------|
| `semantic_interpret` | Interpret a sentence with a given model | `sentence` (str, required), `model` (str, optional, default `mukaj/fin-mpnet-base`) | Same as `GET /api/semantic/interpret`: `success`, `query`, `model`, `tags`, `highlighted_segments`, `token_semantics`, `embedding_stats` |
| `semantic_list_models` | List model IDs available for semantic interpretation | None | Same as `GET /api/semantic/models`: `success`, `models`, `count` |

Tool descriptions (docstrings) should be clear so the agent knows when to call them (e.g. “Get semantic interpretation of a natural language sentence for a given embedding model.”).

---

## Implementation Choices

### 1. MCP server calls backend HTTP API (recommended)

- **Pros**: No duplicate logic; no heavy deps (sentence-transformers) in the MCP process; backend stays single source of truth.
- **Cons**: Backend must be running and reachable (e.g. `OPENINT_BACKEND_URL=http://localhost:3001`).
- **How**: Use `requests` or `urllib` in the MCP server to call `GET /api/semantic/interpret?sentence=...&model=...` and `GET /api/semantic/models`.

### 2. MCP server embeds semantic logic (alternative)

- **Pros**: Single process; works without a separate backend.
- **Cons**: Duplicates logic; MCP process would need sentence-transformers and semantic analyzer (modelmgmt-agent or multi_model_semantic); heavier and slower to start.
- **When**: Use only if you need the MCP server to run standalone (e.g. no network to backend).

**Recommendation**: Implement (1) first; document (2) as a future option.

---

## Package Layout

```
openint-mcp/
├── README.md           # How to run, configure, use with Cursor/Claude
├── requirements.txt    # mcp, requests (or httpx)
├── server.py           # FastMCP app, tool definitions, HTTP client to backend
└── MCP_SEMANTIC_LAYER_PLAN.md  # This plan (or link to repo root)
```

- **Entry point**: `python -m openint_mcp.server` or `uv run server.py` (stdio).
- **Config**: Env var `OPENINT_BACKEND_URL` (default `http://localhost:3001`). No secrets required for local use.

---

## Transport

- **stdio** (default): For Cursor, Claude Desktop, and CLI. Run the server as a subprocess; it reads/writes JSON-RPC on stdin/stdout.
- **Streamable HTTP** (optional): For server-to-server or remote agents. Same tools; different transport (e.g. `mcp.run(transport="streamable-http")`).

Plan for **stdio first**; add HTTP later if needed.

---

## Agent Integration

### Cursor

- In Cursor MCP settings, add a server that runs the openint-mcp command, e.g.:
  - **Command**: `python` (or `uv run`)
  - **Args**: `-m openint_mcp.server` (or path to `server.py`)
  - **Env**: `OPENINT_BACKEND_URL=http://localhost:3001`
- The agent can then call `semantic_interpret` and `semantic_list_models` when it needs sentence interpretation or model list.

### Claude Desktop / other MCP clients

- Configure the MCP server the same way (command + args + env). Use the client’s docs for “add MCP server” or “custom tools”.

### Example agent flow

1. User: “What does the sentence ‘Show me transactions for customer 1001 in California’ mean semantically?”
2. Agent decides to call `semantic_interpret(sentence="Show me transactions for customer 1001 in California", model="mukaj/fin-mpnet-base")`.
3. MCP server forwards to backend `GET /api/semantic/interpret?sentence=...&model=...`, returns JSON to agent.
4. Agent summarizes tags (customer_id, state, intent, etc.) and optionally token_semantics for the user.

---

## Docs and UX

- **README in openint-mcp**: Prerequisites (Python 3.10+, backend running), install, env vars, run with stdio, example tool calls, optional HTTP, link to main API docs (`/api-docs`) for full response shape.
- **Main repo**: In README or ARCHITECTURE, add one line: “Semantic layer is also exposed to agents via the openint-mcp MCP server (see openint-mcp/README.md).”
- **API docs (UI)**: Optional short “For AI agents” section linking to MCP and `semantic_interpret` / `semantic_list_models`.

---

## Summary

| Item | Decision |
|------|----------|
| **Purpose** | Expose semantic interpretation + list models to any MCP-capable agent |
| **Tools** | `semantic_interpret(sentence, model?)`, `semantic_list_models()` |
| **Implementation** | MCP server in `openint-mcp/` calling existing backend HTTP API |
| **Config** | `OPENINT_BACKEND_URL` (default localhost:3001) |
| **Transport** | stdio first; optional streamable HTTP later |
| **Dependencies** | MCP Python SDK (`mcp`), `requests` |

## Implementation

- **Location**: `openint-mcp/`
- **Entry**: `python server.py` (or `uv run server.py`) — stdio transport.
- **Details**: See `openint-mcp/README.md` for install, config, Cursor/Claude setup, and example flows.
