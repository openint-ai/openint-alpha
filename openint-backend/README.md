# OpenInt Backend

API backend for OpenInt system - provides REST API and gateway to agent system.

## Structure

```
openint-backend/
├── api/            # API routes (future)
├── models/        # Data models (future)
└── main.py         # Flask app entry point
```

## Setup

```bash
cd openint-backend
pip install -r requirements.txt
python main.py
```

## API Endpoints

- `POST /api/chat` - Send chat message to agent system
- `GET /api/agents` - List available agents
- `GET /api/query/<query_id>` - Get query result
- `GET /api/health` - Health check
- `GET /api/ready` - Readiness: returns 200 when chat can be served, 503 otherwise

### Startup and readiness

The server is **ready for chat** when the agent system (orchestrator + agents) has initialized. That usually happens within **a few seconds to ~30 seconds** after you run `./start_backend.sh` (or `python main.py`). You’ll see either:

- `Agent orchestrator initialized` and `Backend ready for chat requests` → chat will work.
- `Backend listening but agent system not initialized` → chat will return 503 until the process is restarted after fixing any import/agent errors (check logs).

Check readiness with: `curl -s http://localhost:3001/api/ready`. If you get 503, wait a bit and retry, or check backend logs for errors.

### Semantic layer (sentence interpretation)

- `GET /api/semantic/interpret?sentence=...&model=...` - Interpret a sentence with the given model (query params; easy for curl and browser).
- `POST /api/semantic/preview` - Same interpretation; body: `{"query": "sentence", "model": "mukaj/fin-mpnet-base"}`.
- `GET /api/semantic/interpret-all?sentence=...` - Interpret a sentence with **all** supported models (same as UI dropdown). Returns `{ success, query, models: { model_id: result } }`.
- `POST /api/semantic/interpret-all` - Same; body: `{"query": "sentence"}`.
- `GET /api/semantic/models` - List available model IDs and dimensions.
- `GET /api/semantic/models-with-meta` - List supported models with metadata (id, display_name, author, description, details, url).

## Redis-backed Model Registry (fast scaling)

The backend treats Redis as the **authoritative artifact store for ML models** so new replicas can hydrate from the internal cache in **O(1)** time instead of waiting on HuggingFace downloads.

- **Boot**: When loading an embedding model (e.g. `mukaj/fin-mpnet-base`), the backend first checks Redis. If the model blob is present, it is unpacked and loaded locally (fast). If not, one replica acquires a Redis lock, downloads from HuggingFace, saves a zip to Redis, and releases the lock.
- **Thundering-herd protection**: Only the pod that holds the lock downloads and writes to Redis; other pods wait (poll) for the blob to appear, then load from Redis. This avoids multiple pods saturating the network by writing simultaneously.
- **Configuration**: Same Redis as chat cache (`REDIS_HOST`, `REDIS_PORT`). Model blobs use a separate Redis client with `decode_responses=False`. Keys: `model_registry:{model}:blob`, `model_registry:{model}:lock`, `model_registry:{model}:ready`.

Used automatically by multi-model semantic analysis and by the Milvus client when the registry is on the path.

## Configuration

Set environment variables in `.env`:
- `AGENT_API_URL`: URL to agent system (default: http://localhost:8001)
- `PORT`: Backend port (default: 5000)
- `CORS_ORIGINS`: Allowed CORS origins (default: http://localhost:3000)

**Neo4j (GraphAgent):** For graph/relationship queries, set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, and optionally `NEO4J_DATABASE` (e.g. `graph.db` when using DataHub’s Neo4j). See `openint-graph/README.md`.

**Hugging Face:** Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) for higher rate limits and faster model downloads; it also suppresses the "unauthenticated requests" warning. Create a token at https://huggingface.co/settings/tokens and add `HF_TOKEN=your_token` to `.env` or your environment.

**Redis (model cache + chat cache):** The dropdown embedding/semantic models are downloaded once from Hugging Face and stored in Redis so subsequent loads are fast. Same Redis is used for chat response cache. Default: `REDIS_HOST=127.0.0.1`, `REDIS_PORT=6379`.

- **Start Redis:** From repo root: `docker compose -f docker-compose.redis.yml up -d`
- **Check connectivity:** `cd openint-backend && python scripts/check_redis_registry.py` — pings Redis and lists cached model keys
- **Pre-warm a model:** `python scripts/check_redis_registry.py --warm mukaj/fin-mpnet-base` (or `--warm-all` for all dropdown models)
- **Health:** `GET /api/health` returns `redis_cache.connected` and `redis_model_registry.connected`; if Redis is down, model registry fails fast and falls back to HuggingFace.

## Observability (OpenTelemetry + structured logging)

The backend uses **structured JSON logging** and optional **OpenTelemetry tracing** for observability-friendly output.

- **Logging**: One JSON object per line (`timestamp`, `level`, `logger`, `message`, optional `trace_id`/`span_id`, and `extra` fields). Use `LOG_JSON=1` to force JSON; when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, JSON is the default.
- **Trace correlation**: When OpenTelemetry is available, `trace_id` and `span_id` are added to log records so logs can be correlated with traces.
- **Flask**: HTTP requests are auto-instrumented (spans for each request).

Environment variables:
- `OTEL_SERVICE_NAME`: Service name (default: `openint-backend`)
- `OTEL_EXPORTER_OTLP_ENDPOINT`: OTLP HTTP endpoint for traces (e.g. `http://localhost:4318/v1/traces`). If unset, traces are printed to console.
- `LOG_LEVEL`: `WARNING` (default – log only issues), `INFO`, or `DEBUG` for verbose.
- `LOG_JSON`: Set to `1` for JSON logs (default: human-readable when no OTLP endpoint)
- `OTEL_PYTHON_LOG_CORRELATION`: Set to `true` to add trace/span IDs to log records (default: enabled when OpenTelemetry is used)

## Development

The backend connects to the OpenInt agents system and provides API endpoints for the frontend.
