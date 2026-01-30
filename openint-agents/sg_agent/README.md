# Schema Generator Agent (sg-agent)

The **Schema Generator Agent** connects to DataHub (running locally), reads dataset schemas, and uses the best available generative model to produce example sentences that:

- **Analysts** would ask (e.g. "Top 10 customers by transaction count", "Which ZIP codes have the most failed transactions?")
- **Customer care / support** would ask (e.g. "Show me transactions for customer CUST00000001", "List disputes for this account")
- **Business analysts in a bank** would ask (e.g. "Compare ACH vs wire transaction volumes", "States with highest share of international wires")

## How it works

1. **Schema source**
   - Tries to connect to DataHub at `DATAHUB_GMS_URL` (default `http://localhost:8080`) and read dataset schemas.
   - If DataHub is unavailable, falls back to `openint-datahub/schemas.py` (the same schema definitions used to update DataHub).

2. **Sentence generation (Ollama, open-source LLM)**
   - Uses **Ollama** to generate banking, data, analytics, and regulatory example questions. Context comes from the **DataHub catalog** schema (datasets and fields). No template fallback.
   - Requires Ollama running locally. Model: `OLLAMA_MODEL` (default `llama3.2`). Run `ollama serve` and `ollama pull llama3.2` (or another model).

## Configuration

| Variable | Description | Default |
|---------|-------------|---------|
| `DATAHUB_GMS_URL` | DataHub GMS URL | `http://localhost:8080` |
| `DATAHUB_TOKEN` | DataHub API token (or `.datahub_token` in openint-datahub) | — |
| `OLLAMA_HOST` | Ollama API base URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model name for sentence generation | `llama3.2` |
| `REDIS_HOST` / `REDIS_PORT` | Redis for schema cache (survives restarts) | `127.0.0.1` / `6379` |

## Capability

- **Name:** `suggestions`
- **Description:** Generate example questions that analysts and customer care would ask. Context from DataHub catalog.
- **Input:** Optional `query` (hint: "analyst", "customer care", "business analyst"), optional `count`.
- **Output:** List of `{ "query": "<sentence>", "category": "Analyst" | "Customer Care" | "Business Analyst" | "Regulatory", "source": "ollama" }`.

## Dependencies

- **Required:** None beyond the openint-agents base (schema is loaded from `openint-datahub/schemas.py` when DataHub is not used).
- **Sentence generation:** Ollama running locally (e.g. `ollama serve` and `ollama pull llama3.2`). No external API key required.
- **Optional:** `acryl-datahub` – for reading schema from DataHub (install from `openint-datahub/requirements.txt` if you use DataHub).

## Usage

The agent is registered with the orchestrator. Queries containing words like "suggest", "example question", "what can I ask", "generate sentence", "sample query", "analyst question", or "business analyst" are routed to sg-agent.

Example (via backend chat):

- "Give me example questions an analyst would ask"
- "Suggest sample queries for customer care"
- "What can I ask as a business analyst?"
