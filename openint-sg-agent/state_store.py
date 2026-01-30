"""
Optional Redis-backed schema cache for openint-sg-agent.
Uses REDIS_HOST/REDIS_PORT (default 127.0.0.1:6379). Schema cache TTL 24h.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

KEY_PREFIX = "agent_state"
AGENT_NAME = "sg-agent"
REDIS_DEFAULT_HOST = "127.0.0.1"
REDIS_DEFAULT_PORT = 6379
REDIS_CONNECT_TIMEOUT = 5
REDIS_RETRY_AFTER_FAILURE_SECONDS = 30
SCHEMA_CACHE_TTL = 24 * 3600

_redis_client = None
_redis_unavailable_until = 0.0


def _get_redis():
    global _redis_client, _redis_unavailable_until
    if _redis_client is not None:
        return _redis_client
    if time.time() < _redis_unavailable_until:
        return None
    host = (os.getenv("REDIS_HOST") or "").strip() or REDIS_DEFAULT_HOST
    try:
        port = int(os.getenv("REDIS_PORT") or str(REDIS_DEFAULT_PORT))
    except (TypeError, ValueError):
        port = REDIS_DEFAULT_PORT
    try:
        import redis
        r = redis.Redis(
            host=host, port=port, db=0, decode_responses=True,
            socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
        )
        r.ping()
        _redis_client = r
        return r
    except Exception as e:
        _redis_unavailable_until = time.time() + REDIS_RETRY_AFTER_FAILURE_SECONDS
        logger.warning("Redis unavailable at %s:%s: %s", host, port, e)
        return None


def get_schema_cache() -> Optional[Dict[str, Any]]:
    key = f"{KEY_PREFIX}:{AGENT_NAME}:schema"
    r = _get_redis()
    if r is None:
        return None
    try:
        raw = r.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception:
        return None


def set_schema_cache(value: Dict[str, Any]) -> bool:
    key = f"{KEY_PREFIX}:{AGENT_NAME}:schema"
    r = _get_redis()
    if r is None:
        return False
    try:
        r.setex(key, SCHEMA_CACHE_TTL, json.dumps(value))
        return True
    except Exception as e:
        logger.warning("Redis set %s: %s", key, e)
        return False


def is_available() -> bool:
    return _get_redis() is not None
