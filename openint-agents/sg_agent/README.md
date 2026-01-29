# Schema Generator Agent (sg-agent)

The **Schema Generator Agent** connects to DataHub (running locally), reads dataset schemas, and uses the best available generative model to produce example sentences that:

- **Analysts** would ask (e.g. "Top 10 customers by transaction count", "Which ZIP codes have the most failed transactions?")
- **Customer care / support** would ask (e.g. "Show me transactions for customer CUST00000001", "List disputes for this account")
- **Business analysts in a bank** would ask (e.g. "Compare ACH vs wire transaction volumes", "States with highest share of international wires")

## How it works

1. **Schema source**
   - Tries to connect to DataHub at `DATAHUB_GMS_URL` (default `http://localhost:8080`) and read dataset schemas.
   - If DataHub is unavailable, falls back to `openint-datahub/schemas.py` (the same schema definitions used to update DataHub).

2. **Sentence generation**
   - If `OPENAI_API_KEY` is set: uses OpenAI (best available model; default `gpt-4o-mini`, override with `OPENAI_CHAT_MODEL`) to generate natural-language example questions from the schema.
   - Otherwise: uses template-based generation from the schema so it works without any API key.

## Configuration

| Variable | Description | Default |
|---------|-------------|---------|
| `DATAHUB_GMS_URL` | DataHub GMS URL | `http://localhost:8080` |
| `DATAHUB_TOKEN` | DataHub API token (or `.datahub_token` in openint-datahub) | — |
| `OPENAI_API_KEY` | OpenAI API key for LLM-generated sentences | — |
| `OPENAI_CHAT_MODEL` | OpenAI model name | `gpt-4o-mini` |

## Capability

- **Name:** `suggestions`
- **Description:** Generate example questions that analysts and customer care would ask.
- **Input:** Optional `query` (hint: "analyst", "customer care", "business analyst"), optional `count`.
- **Output:** List of `{ "query": "<sentence>", "category": "Analyst" | "Customer Care" | "Business Analyst", "source": "openai" | "template" }`.

## Dependencies

- **Required:** None beyond the openint-agents base (schema is loaded from `openint-datahub/schemas.py` when DataHub is not used).
- **Optional:**
  - `openai` – for LLM-generated sentences when `OPENAI_API_KEY` is set.
  - `acryl-datahub` – for reading schema from DataHub (install from `openint-datahub/requirements.txt` if you use DataHub).

## Usage

The agent is registered with the orchestrator. Queries containing words like "suggest", "example question", "what can I ask", "generate sentence", "sample query", "analyst question", or "business analyst" are routed to sg-agent.

Example (via backend chat):

- "Give me example questions an analyst would ask"
- "Suggest sample queries for customer care"
- "What can I ask as a business analyst?"
