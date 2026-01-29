# Multi-Model Semantic Analysis with Debug Highlights

This document describes the enhanced debug experience that showcases semantic highlights from different models and automatically selects the best model for vector database search.

## Overview

When `debug=true` is set in the chat API request, the system:

1. **Analyzes the query with multiple models** (BGE-base, BGE-large, E5, MPNet, Finance-specific)
2. **Shows semantic highlights** from each model's perspective
3. **Scores each model** based on semantic understanding quality
4. **Selects the best model** for the query
5. **Uses the best model** for vector database search

## API Usage

```bash
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find customers in California with transactions over $1000",
    "debug": true
  }'
```

## Debug Response Format

The debug response includes a `debug` object with multi-model analysis:

```json
{
  "success": true,
  "answer": "...",
  "sources": [...],
  "debug": {
    "query": "Find customers in California with transactions over $1000",
    "semantics": [...],
    "embedding_model": "BAAI/bge-large-en-v1.5",
    "recommended_model": "BAAI/bge-large-en-v1.5",
    "multi_model_analysis": {
      "models_analyzed": 5,
      "best_model": "BAAI/bge-large-en-v1.5",
      "best_model_score": 0.85,
      "model_scores": {
        "BAAI/bge-base-en-v1.5": 0.75,
        "BAAI/bge-large-en-v1.5": 0.85,
        "intfloat/e5-base-v2": 0.70,
        "sentence-transformers/all-mpnet-base-v2": 0.72,
        "mukaj/fin-mpnet-base": 0.80
      },
      "model_highlights": {
        "BAAI/bge-base-en-v1.5": {
          "highlighted_segments": [
            {
              "text": "Find ",
              "type": "text",
              "tag": null
            },
            {
              "text": "customers",
              "type": "highlight",
              "tag": {
                "type": "entity",
                "label": "Entity Type",
                "value": "Customer",
                "confidence": 0.85
              },
              "tag_type": "entity",
              "label": "Entity Type",
              "confidence": 0.85
            },
            {
              "text": " in ",
              "type": "text",
              "tag": null
            },
            {
              "text": "California",
              "type": "highlight",
              "tag": {
                "type": "state",
                "label": "State",
                "value": "CA",
                "confidence": 0.95
              },
              "tag_type": "state",
              "label": "State",
              "confidence": 0.95
            },
            {
              "text": " with ",
              "type": "text",
              "tag": null
            },
            {
              "text": "transactions",
              "type": "highlight",
              "tag": {
                "type": "entity",
                "label": "Entity Type",
                "value": "Transaction",
                "confidence": 0.90
              },
              "tag_type": "entity",
              "label": "Entity Type",
              "confidence": 0.90
            },
            {
              "text": " over ",
              "type": "text",
              "tag": null
            },
            {
              "text": "$1000",
              "type": "highlight",
              "tag": {
                "type": "amount_min",
                "label": "Amount over",
                "value": 1000.0,
                "confidence": 0.90
              },
              "tag_type": "amount_min",
              "label": "Amount over",
              "confidence": 0.90
            }
          ],
          "tag_count": 4,
          "highlighted_count": 4,
          "score": 0.75,
          "is_best": false
        },
        "BAAI/bge-large-en-v1.5": {
          "highlighted_segments": [...],
          "tag_count": 4,
          "highlighted_count": 4,
          "score": 0.85,
          "is_best": true
        },
        ...
      }
    }
  }
}
```

## Visual Representation

The `highlighted_segments` array can be used to render the query with semantic highlights:

```
Find [customers] in [California] with [transactions] over [$1000]
     ^^^^^^^^^^      ^^^^^^^^^^^      ^^^^^^^^^^^^^      ^^^^^^
     Entity         State            Entity            Amount
```

Each model may highlight different parts or have different confidence levels.

## Model Selection Criteria

The system scores models based on:

1. **Number of tags detected** (0-0.4 points)
2. **Average tag confidence** (0-0.3 points)
3. **Detected entities** (0-0.2 points)
4. **Detected actions** (0-0.1 points)
5. **Finance-specific bonus** (+0.1 if finance model)

The model with the highest score is selected for vector search.

## Frontend Integration

To display semantic highlights in the UI:

```javascript
function renderHighlightedQuery(segments) {
  return segments.map(segment => {
    if (segment.type === 'highlight') {
      return (
        <span 
          key={segment.text}
          className={`highlight highlight-${segment.tag_type}`}
          title={`${segment.label}: ${segment.value} (confidence: ${segment.confidence})`}
        >
          {segment.text}
        </span>
      );
    }
    return <span key={segment.text}>{segment.text}</span>;
  });
}

// Usage
const bestModel = debug.multi_model_analysis.best_model;
const highlights = debug.multi_model_analysis.model_highlights[bestModel];
const highlightedQuery = renderHighlightedQuery(highlights.highlighted_segments);
```

## Model Comparison View

You can also show a comparison of all models:

```javascript
function ModelComparison({ modelHighlights, bestModel }) {
  return (
    <div className="model-comparison">
      <h3>Model Comparison</h3>
      {Object.entries(modelHighlights).map(([modelName, data]) => (
        <div key={modelName} className={data.is_best ? 'best-model' : ''}>
          <h4>{modelName} {data.is_best && '⭐ (Best)'}</h4>
          <div>Score: {data.score.toFixed(2)}</div>
          <div>Tags: {data.tag_count}</div>
          <div className="highlighted-query">
            {renderHighlightedQuery(data.highlighted_segments)}
          </div>
        </div>
      ))}
    </div>
  );
}
```

## Automatic Model Selection

The system automatically:

1. Analyzes the query with all available models
2. Scores each model's semantic understanding
3. Selects the best model
4. Uses that model for vector database search
5. Returns results using the optimal model

This ensures the best semantic understanding for each query without manual model selection.

## Benefits

1. **Better Search Results**: Uses the model best suited for each query
2. **Transparency**: Shows how different models interpret the query
3. **Debugging**: Easy to see which semantic tags are detected
4. **Adaptive**: Automatically adapts to query type (finance, general, etc.)

## Example Queries

### Finance Query
```
Query: "Show me high-value customers in Texas"
Best Model: mukaj/fin-mpnet-base (finance-specific)
Highlights: [high-value] [customers] [Texas]
```

### General Query
```
Query: "Find transactions over $5000"
Best Model: BAAI/bge-large-en-v1.5 (highest quality)
Highlights: [transactions] [over $5000]
```

### Location Query
```
Query: "Customers in ZIP code 75001"
Best Model: intfloat/e5-base-v2 (excellent semantic understanding)
Highlights: [Customers] [ZIP code 75001]
```
