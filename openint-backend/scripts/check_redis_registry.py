#!/usr/bin/env python3
"""
Check Redis connectivity for the model registry and optionally list keys or pre-warm a model.

Usage:
  python scripts/check_redis_registry.py                    # Ping Redis, list model keys
  python scripts/check_redis_registry.py --warm mukaj/fin-mpnet-base   # Pre-warm one model
  python scripts/check_redis_registry.py --warm-all        # Pre-warm all dropdown models (slow)

Uses REDIS_HOST and REDIS_PORT (default 127.0.0.1:6379). Start Redis with:
  docker compose -f docker-compose.redis.yml up -d
"""

import argparse
import os
import sys

# Run from openint-backend so model_registry is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

os.chdir(BACKEND_DIR)


def main():
    parser = argparse.ArgumentParser(description="Check Redis model registry and optionally pre-warm models")
    parser.add_argument("--warm", metavar="MODEL", help="Pre-warm a single model (e.g. mukaj/fin-mpnet-base)")
    parser.add_argument("--warm-all", action="store_true", help="Pre-warm all dropdown semantic models")
    args = parser.parse_args()

    from model_registry import (
        check_redis_registry,
        load_model_from_registry,
        MODEL_KEY_PREFIX,
        _get_redis_binary,
    )

    connected, host, port, err = check_redis_registry()
    if not connected:
        print(f"Redis model registry: NOT CONNECTED at {host}:{port}")
        if err:
            print(f"  Error: {err}")
        print("  Start Redis: docker compose -f docker-compose.redis.yml up -d")
        print("  Backend deps: pip install redis (in openint-backend)")
        return 1
    print(f"Redis model registry: connected at {host}:{port}")

    # List model keys
    r = _get_redis_binary()
    if r:
        try:
            keys = []
            for k in r.scan_iter(match=f"{MODEL_KEY_PREFIX}:*:blob", count=100):
                keys.append(k.decode("utf-8") if isinstance(k, bytes) else k)
            if keys:
                print(f"  Cached models ({len(keys)}):")
                for k in sorted(keys):
                    # key format: model_registry:mukaj_fin-mpnet-base:blob
                    name = k.replace(f"{MODEL_KEY_PREFIX}:", "").replace(":blob", "").replace("_", "/", 1)
                    print(f"    - {name}")
            else:
                print("  No models cached yet. Use --warm MODEL or --warm-all to pre-warm.")
        except Exception as e:
            print(f"  Could not list keys: {e}")

    if args.warm:
        print(f"\nPre-warming model: {args.warm}")
        model = load_model_from_registry(args.warm)
        if model is not None:
            print(f"  OK: {args.warm} loaded and cached in Redis")
        else:
            print(f"  Failed to load {args.warm}")
            return 1

    if args.warm_all:
        models = [
            "mukaj/fin-mpnet-base",
            "ProsusAI/finbert",
            "sentence-transformers/all-mpnet-base-v2",
        ]
        print(f"\nPre-warming {len(models)} models (this may take several minutes)...")
        for name in models:
            print(f"  Loading {name}...")
            model = load_model_from_registry(name)
            if model is not None:
                print(f"    OK: cached")
            else:
                print(f"    Failed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
