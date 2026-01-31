#!/usr/bin/env python3
"""
Milvus connectivity and data diagnostic script.

Run from repo root (or with PYTHONPATH including openint-vectordb):
  python openint-vectordb/check_milvus.py

Checks:
  - Connection to Milvus
  - Collection name and existence
  - Entity count in collection
  - Sample query to verify data is searchable
"""

import os
import sys
from pathlib import Path

# Ensure we can import milvus_client; load .env from repo root
_repo_root = Path(__file__).resolve().parent.parent
_env = _repo_root / ".env"
if _env.exists():
    from dotenv import load_dotenv
    load_dotenv(_env, override=True)

# Add openint-vectordb/milvus to path so milvus_client can be imported
_milvus_dir = Path(__file__).resolve().parent / "milvus"
if str(_milvus_dir) not in sys.path:
    sys.path.insert(0, str(_milvus_dir))


def main():
    print("=" * 60)
    print("Milvus Vector DB Diagnostic")
    print("=" * 60)
    print(f"  .env loaded from: {_env}" if _env.exists() else "  No .env found")
    print()

    try:
        from milvus_client import MilvusClient
    except ImportError as e:
        print(f"  Could not import MilvusClient: {e}")
        print("  Install: pip install pymilvus")
        return 1

    # Connection
    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")
    collection_name = os.getenv("MILVUS_COLLECTION", "default_collection")
    print(f"  MILVUS_HOST: {host}")
    print(f"  MILVUS_PORT: {port}")
    print(f"  MILVUS_COLLECTION: {collection_name}")
    print()

    try:
        client = MilvusClient()
        print("  1. Connection: OK")
        print(f"     Collection name (sanitized): {client.collection_name}")
    except Exception as e:
        print(f"  1. Connection: FAILED - {e}")
        print("\n  Make sure Milvus is running: docker run -d -p 19530:19530 milvusdb/milvus")
        return 1

    # List collections
    try:
        from pymilvus import utility
        collections = utility.list_collections(using=client.alias)
        print(f"\n  2. Collections in Milvus: {collections}")
        if client.collection_name not in collections:
            print(f"     WARNING: '{client.collection_name}' not in list. Loader may have created a different collection.")
    except Exception as e:
        print(f"\n  2. List collections: {e}")

    # Get collection and entity count
    try:
        collection = client.get_or_create_collection()
        # num_entities can be stale; flush first for accuracy
        collection.flush()
        count = collection.num_entities
        print(f"\n  3. Entity count: {count:,}")
        if count == 0:
            print("     Collection is empty. Run the loader:")
            print("       cd openint-testdata && python generators/generate_openint_test_data.py  # writes to openint-testdata/data/")
            print("       python loaders/load_openint_data_to_milvus.py")
    except Exception as e:
        print(f"\n  3. Entity count: FAILED - {e}")

    # Try list_records (requires load)
    try:
        records = client.list_records(limit=3)
        print(f"\n  4. Sample records (up to 3): {len(records)} found")
        for i, r in enumerate(records[:3]):
            print(f"     [{i+1}] id={r.get('id', '')[:50]}... file_type={r.get('metadata', {}).get('file_type', '')}")
        if not records and count and count > 0:
            print("     WARNING: Entity count > 0 but query returned no records. Try collection.load() refresh.")
    except Exception as e:
        print(f"\n  4. List records: {e}")

    # Try search (requires embedding model)
    try:
        results, t_ms, _, _ = client.search("customer transaction", top_k=2)
        print(f"\n  5. Search test: {len(results)} results in {t_ms}ms")
        if results:
            for i, r in enumerate(results[:2]):
                print(f"     [{i+1}] {r.get('id', '')[:40]}... (score={r.get('score', 0):.4f})")
        elif count > 0:
            print("     WARNING: Data exists but search returned nothing. Check embedding model.")
    except Exception as e:
        print(f"\n  5. Search test: {e}")
        print("     Install sentence-transformers: pip install sentence-transformers")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
