# Model Recommendations for OpenInt

This guide explains why graph results are often more accurate than vector results, and recommends open-source LLM and embedding models to improve OpenInt.

## Why Graph Results Are More Accurate Than Vector

| Aspect | Graph (Neo4j) | Vector (Milvus) |
|--------|---------------|-----------------|
| **Lookup type** | Exact ID match (customer_id, transaction_id, dispute_id) | Semantic similarity (embedding cosine distance) |
| **Precision** | High — returns the exact entity when you have the ID | Variable — returns "similar" content, which may not match the query |
| **Use case** | "Show customer 1026847926404610462" → exact customer | "Show me customers with disputed transactions" → semantically similar records |

**Recommendation:** For ID-based queries, graph lookups are inherently more accurate. To improve overall results:

1. **LLM (Ollama):** Use a model with stronger instruction following for query refinement, Cypher generation, and enrichment.
2. **Embedding model:** Use a finance-tuned model for vector search so semantic results align better with banking terminology.

---

## Recommended LLM Models (Ollama)

OpenInt uses Ollama for: sg-agent (sentence fix/generation), graph-agent (Cypher generation), enrich-agent (entity type inference), sentiment-agent, and the aggregator (answer synthesis).

### Primary recommendation: `qwen2.5:7b`

- **Why:** Strong instruction following, good at structured output (JSON, Cypher), handles banking terminology well.
- **Install:** `ollama pull qwen2.5:7b`
- **Config:** `OLLAMA_MODEL=qwen2.5:7b`

### Finance-specific: `martain7r/finance-llama-8b` (Finance-Llama-8B)

- **Why:** Fine-tuned on 500k+ financial examples (QA, reasoning, sentiment, NER). Built for banking assistants.
- **Install:** `ollama pull martain7r/finance-llama-8b`
- **Config:** `OLLAMA_MODEL=martain7r/finance-llama-8b`

### Alternative: `mistral:7b-instruct`

- **Why:** Strong general performance, good instruction following.
- **Install:** `ollama pull mistral:7b-instruct`
- **Config:** `OLLAMA_MODEL=mistral:7b-instruct`

### Previous default: `llama3.2`

- General-purpose; can be used if qwen2.5 is unavailable: `OLLAMA_MODEL=llama3.2`

---

## Recommended Embedding Models (Vector Search)

The embedding model determines how well Milvus vector search matches queries to data. See `docs/EMBEDDING_MODELS.md` for full details.

### Primary recommendation: `mukaj/fin-mpnet-base`

- **Why:** Finance-tuned (79.91 FiQA score vs 49.96 for general models). 768 dimensions.
- **Config:** `EMBEDDING_MODEL=mukaj/fin-mpnet-base`
- **Note:** Requires re-loading Milvus data (different dimensions). Run:
  ```bash
  python openint-testdata/loaders/load_openint_data_to_milvus.py --clean
  ```

### Current default: `all-MiniLM-L6-v2`

- Fast, 384 dimensions, but lower quality for finance-specific queries.

---

## Quick Setup

### 1. Upgrade LLM (enrichment, Cypher, aggregator)

```bash
# Pull the recommended model
ollama pull qwen2.5:7b

# Set in .env or export
export OLLAMA_MODEL=qwen2.5:7b
```

### 2. Upgrade embedding model (vector search)

```bash
# Set in .env
EMBEDDING_MODEL=mukaj/fin-mpnet-base

# Re-load Milvus (creates new collection with 768d)
python openint-testdata/loaders/load_openint_data_to_milvus.py --clean
```

---

## Integration Points

| Component | Uses | Env / config |
|-----------|------|--------------|
| sg-agent | Ollama (sentence fix, generation) | `OLLAMA_MODEL` |
| graph-agent | Ollama (Cypher generation) | `OLLAMA_MODEL` |
| enrich-agent | Ollama (entity type inference) | `OLLAMA_MODEL` |
| sentiment-agent | Ollama | `OLLAMA_MODEL` |
| main.py aggregator | Ollama (answer synthesis, Cypher) | `OLLAMA_MODEL` |
| a2a.py | Ollama (sg-agent fix/generate) | `OLLAMA_MODEL` |
| vectordb-agent | Embedding model | `EMBEDDING_MODEL` |
| Milvus load | Embedding model | `EMBEDDING_MODEL` |
