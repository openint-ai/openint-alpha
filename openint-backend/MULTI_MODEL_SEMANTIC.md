# Multi-Model Semantic Analysis

This feature allows you to process queries through multiple embedding models simultaneously to understand and tag semantics from different perspectives.

## Implementation

The backend uses **modelmgmt-agent** (openint-agents) for multi-model semantic analysis when `openint-agents` is on the path. modelmgmt-agent provides the analyzer and model registry (Hugging Face + Redis). If modelmgmt-agent is unavailable, the backend falls back to local `multi_model_semantic` and `model_registry` in openint-backend. See [openint-agents/modelmgmt_agent/README.md](../openint-agents/modelmgmt_agent/README.md).

## Overview

The multi-model semantic analyzer processes queries through multiple embedding models (e.g., BGE, E5, MPNet) and extracts semantic tags, entities, intents, and other semantic information. This provides:

1. **Consensus Analysis**: Tags detected by multiple models are more reliable
2. **Model Comparison**: See how different models interpret the same query
3. **Rich Semantic Tags**: Extract entities, actions, amounts, locations, transaction types, etc.
4. **Embedding Statistics**: Analyze embedding properties across models

## Usage

### API Endpoint: `/api/semantic/analyze`

**POST** request to analyze a query with multiple models:

```bash
curl -X POST http://localhost:3001/api/semantic/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find customers in California with transactions over $1000",
    "parallel": true
  }'
```

**Response:**
```json
{
  "success": true,
  "query": "Find customers in California with transactions over $1000",
  "models_analyzed": 5,
  "models": {
    "BAAI/bge-base-en-v1.5": {
      "model": "BAAI/bge-base-en-v1.5",
      "tags": [
        {
          "type": "entity",
          "label": "Entity Type",
          "value": "Customer",
          "snippet": "customers",
          "confidence": 0.85
        },
        {
          "type": "state",
          "label": "State",
          "value": "CA",
          "snippet": "California",
          "confidence": 0.95
        },
        {
          "type": "amount_min",
          "label": "Amount over",
          "value": 1000.0,
          "snippet": "over $1000",
          "confidence": 0.9
        }
      ],
      "detected_entities": ["Customer"],
      "detected_actions": ["Search"],
      "embedding_stats": {
        "dimension": 768,
        "norm": 12.34,
        "mean": 0.01,
        "std": 0.05
      }
    },
    ...
  },
  "aggregated": {
    "consensus_tags": [
      {
        "type": "entity",
        "label": "Entity Type",
        "value": "Customer",
        "confidence": 0.87,
        "detected_by_models": 5,
        "models": ["BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5", ...]
      }
    ],
    "tag_counts": {
      "entity": 5,
      "state": 5,
      "amount_min": 5
    },
    "entity_counts": {
      "Customer": 5,
      "Transaction": 3
    }
  },
  "summary": {
    "most_common_entity": "Customer",
    "most_common_action": "Search",
    "most_common_tag_type": "entity"
  }
}
```

### Integration with Chat Endpoint

You can also request multi-model analysis as part of the chat endpoint:

```bash
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me high-value customers in Texas",
    "multi_model_analysis": true
  }'
```

The response will include a `multi_model_analysis` field with the semantic analysis results.

### List Available Models

```bash
curl http://localhost:3001/api/semantic/models
```

## Available Models

By default, the system uses these models:

1. **BAAI/bge-base-en-v1.5** - Best general balance (768 dims)
2. **BAAI/bge-large-en-v1.5** - Highest quality (1024 dims)
3. **intfloat/e5-base-v2** - Excellent semantic understanding (768 dims)
4. **sentence-transformers/all-mpnet-base-v2** - Strong semantic capabilities (768 dims)
5. **mukaj/fin-mpnet-base** - Finance-specific (768 dims, if available)

You can customize models via environment variable:
```bash
export MULTI_MODEL_SEMANTIC_MODELS="BAAI/bge-base-en-v1.5,BAAI/bge-large-en-v1.5"
```

## Semantic Tags Extracted

The system extracts various semantic tags:

### Entity Types
- `customer` - Customer/account references
- `transaction` - Transaction/payment references
- `dispute` - Dispute/chargeback references
- `location` - Location references (state, ZIP, city)

### Transaction Types
- `ach` - ACH transactions
- `wire` - Wire transfers
- `credit` - Credit card transactions
- `debit` - Debit card transactions
- `check` - Check transactions

### Intents/Actions
- `search` - Search/find operations
- `filter` - Filtering operations
- `aggregate` - Aggregation operations (count, sum, etc.)
- `sort` - Sorting operations (top, largest, etc.)

### Amount Patterns
- `amount_min` - Amounts over/above a threshold
- `amount_max` - Amounts under/below a threshold
- `amount_between` - Amount ranges

### Location Patterns
- `state` - US state codes or names
- `zip_code` - ZIP codes

## Python Usage

When the backend runs with openint-agents, the analyzer is provided by **modelmgmt-agent** (`modelmgmt_agent.semantic_analyzer`). When using the backend alone (fallback), you can call the local module:

```python
# With openint-agents on path (modelmgmt-agent):
from modelmgmt_agent.semantic_analyzer import analyze_query_multi_model

# Or backend fallback (openint-backend only):
# from multi_model_semantic import analyze_query_multi_model

# Analyze a query
result = analyze_query_multi_model(
    "Find top 10 customers in California with transactions over $5000",
    parallel=True
)

# Access results
print(f"Models analyzed: {result['models_analyzed']}")
print(f"Consensus tags: {result['aggregated']['consensus_tags']}")
print(f"Summary: {result['summary']}")

# Access individual model results
for model_name, model_result in result['models'].items():
    print(f"\n{model_name}:")
    print(f"  Tags: {model_result['tags']}")
    print(f"  Entities: {model_result['detected_entities']}")
    print(f"  Actions: {model_result['detected_actions']}")
```

## Performance

- **Parallel Processing**: By default, models are processed in parallel for faster analysis
- **Model Loading**: Models are loaded once and reused (lazy loading on first request)
- **Memory**: Each model uses ~400MB-1.5GB of memory depending on size

## Use Cases

1. **Query Understanding**: Understand what users are asking for across multiple semantic perspectives
2. **Query Routing**: Route queries to appropriate agents based on detected entities/actions
3. **Query Enhancement**: Enhance queries with semantic tags for better search results
4. **Analytics**: Analyze query patterns and semantic understanding across models
5. **Debugging**: Compare how different models interpret the same query

## Example Queries

```python
# Financial queries
"Show me customers with transactions over $10,000 in California"
"Find disputes for customer CUST12345"
"List top 5 states by transaction volume"

# Location-based queries
"Customers in Texas with ZIP code 75001"
"Transactions in New York state"

# Transaction type queries
"Show ACH transactions over $5000"
"Wire transfers to California"
```

## Notes

- Models are downloaded automatically on first use (requires internet connection)
- Finance-specific model (`mukaj/fin-mpnet-base`) may not be available in all environments
- Consensus tags (detected by 2+ models) are more reliable than single-model tags
- Embedding statistics can help understand how different models encode queries
