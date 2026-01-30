# modelmgmt-agent: Model Management Agent

modelmgmt-agent downloads embedding models from **Hugging Face**, stores them in **Redis** for in-memory lookup, and **annotates sentences** with semantic tags (entities, intents, schema fields). It works with **sg-agent**: sg-agent generates example sentences from DataHub schema + LLM; modelmgmt-agent annotates those (and any) sentences.

## Responsibilities

- **Model registry**: Download models from Hugging Face; store in Redis so replicas hydrate in O(1). Thundering-herd protection: one writer, others wait.
- **Agent state (Redis)**: When `context.session_id` is set, session/task state (last_models_used, task_id, last_query) is persisted in Redis so agent memory survives restarts and multi-day tasks.
- **Sentence annotation**: Run one or all configured models on a sentence; return tags, highlighted segments, and schema-based matches (DataHub schema from sg-agent / openint-datahub).

## Capabilities

- **semantic_annotate**: Annotate a sentence with one or all embedding models. Input: sentence; optional `model` (single ID) or use all. Output: tags, highlighted_segments, best_model, schema_assets.

## Environment

- **REDIS_HOST** / **REDIS_PORT**: Same as chat cache (default 127.0.0.1:6379). Used for model blob storage and **agent state** (session/task state so memory survives restarts and multi-day tasks).
- **HF_TOKEN** / **HUGGING_FACE_HUB_TOKEN**: Optional; higher rate limits for Hugging Face.
- **MULTI_MODEL_SEMANTIC_MODELS**: Comma-separated model IDs (default: mukaj/fin-mpnet-base, ProsusAI/finbert, sentence-transformers/all-mpnet-base-v2).

## Interaction with sg-agent

- **sg-agent**: Generates example sentences using DataHub schema and LLM (e.g. "I'm feeling lucky" in the Compare UI).
- **modelmgmt-agent**: Annotates any sentence with semantic tags. Backend can call sg-agent for a sentence, then modelmgmt-agent to annotate it (e.g. lucky + `?annotate=true` returns sentence and annotation).

## Backend

The OpenInt backend imports semantic logic from modelmgmt-agent when `openint-agents` is on `PYTHONPATH`. Semantic API endpoints (`/api/semantic/interpret`, `/api/semantic/preview-multi`, etc.) use modelmgmt-agent’s analyzer and model registry. If modelmgmt-agent is unavailable, the backend falls back to local `multi_model_semantic` and `model_registry` (if present).

## See also

- [openint-backend/MULTI_MODEL_SEMANTIC.md](../../openint-backend/MULTI_MODEL_SEMANTIC.md) — API and usage for multi-model semantic analysis
- [openint-backend/README.md](../../openint-backend/README.md) — Backend API and Redis model registry
- [openint-agents/README.md](../README.md) — Agent system and orchestrator
