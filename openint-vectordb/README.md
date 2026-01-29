# OpenInt Vector Database

Vector database clients and utilities for OpenInt system.

## Structure

```
openint-vectordb/
├── milvus/              # Milvus vector database client
│   ├── milvus_client.py
│   └── __init__.py
└── README.md
```

## Usage

### Milvus Client

```python
from openint_vectordb.milvus import MilvusClient

# Initialize client
client = MilvusClient()

# Use the client
results = client.search("your query", top_k=10)
```

### Configuration

Set environment variables in `.env`:
- `MILVUS_HOST`: Milvus host (default: localhost)
- `MILVUS_PORT`: Milvus port (default: 19530)
- `MILVUS_COLLECTION`: Collection name (default: openint_data)

## Installation

This package is part of the OpenInt project. Install dependencies:

```bash
pip install pymilvus>=2.3.0 sentence-transformers>=2.2.0
```
