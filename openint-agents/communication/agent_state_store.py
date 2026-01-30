"""
Redis-backed Agent State Store.

Persistent state for sg-agent and modelmgmt-agent so agent memory survives restarts
and maintains state across multi-day tasks. Uses REDIS_HOST/REDIS_PORT (default
127.0.0.1:6379). Keys are prefixed with agent_state:; values are JSON.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

KEY_PREFIX = "agent_state"
REDIS_DEFAULT_HOST = "127.0.0.1"
REDIS_DEFAULT_PORT = 6379
REDIS_CONNECT_TIMEOUT = 5
REDIS_RETRY_AFTER_FAILURE_SECONDS = 30
# Default TTL for schema cache (24h); None = no expiry for long-lived state
SCHEMA_CACHE_TTL = 24 * 3600
SESSION_STATE_TTL = 7 * 24 * 3600  # 7 days for session/task state

_redis_client = None
_redis_unavailable_until = 0.0


def _get_redis():
    """Get Redis client (decode_responses=True for string/JSON). Returns None if unavailable."""
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
            host=host,
            port=port,
            db=0,
            decode_responses=True,
            socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
        )
        r.ping()
        _redis_client = r
        logger.info("Agent state store connected to Redis at %s:%s", host, port)
        return r
    except Exception as e:
        _redis_unavailable_until = time.time() + REDIS_RETRY_AFTER_FAILURE_SECONDS
        logger.warning(
            "Agent state store Redis unavailable at %s:%s: %s (retry in %ss)",
            host, port, e, REDIS_RETRY_AFTER_FAILURE_SECONDS,
        )
        return None


def _key(agent: str, *parts: str) -> str:
    """Build key: agent_state:{agent}:{part1}:{part2}..."""
    return ":".join([KEY_PREFIX, agent] + [str(p) for p in parts])


def get_state(agent: str, *key_parts: str) -> Optional[Dict[str, Any]]:
    """
    Get JSON state for agent. key_parts form the key suffix, e.g. get("sg-agent", "schema").
    Returns None if key missing, Redis down, or invalid JSON.
    """
    r = _get_redis()
    if r is None:
        return None
    k = _key(agent, *key_parts)
    try:
        raw = r.get(k)
        if raw is None:
            return None
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Agent state store get %s: invalid JSON: %s", k, e)
        return None
    except Exception as e:
        logger.warning("Agent state store get %s: %s", k, e)
        return None


def set_state(
    agent: str,
    *key_parts: str,
    value: Dict[str, Any],
    ttl_seconds: Optional[int] = None,
) -> bool:
    """
    Set JSON state for agent. Returns True on success, False on Redis error.
    ttl_seconds: optional TTL; None = no expiry.
    """
    r = _get_redis()
    if r is None:
        return False
    k = _key(agent, *key_parts)
    try:
        payload = json.dumps(value)
        if ttl_seconds is not None:
            r.setex(k, ttl_seconds, payload)
        else:
            r.set(k, payload)
        return True
    except Exception as e:
        logger.warning("Agent state store set %s: %s", k, e)
        return False


def delete(agent: str, *key_parts: str) -> bool:
    """Delete state key. Returns True on success or key missing."""
    r = _get_redis()
    if r is None:
        return False
    k = _key(agent, *key_parts)
    try:
        r.delete(k)
        return True
    except Exception as e:
        logger.warning("Agent state store delete %s: %s", k, e)
        return False


def is_available() -> bool:
    """Return True if Redis is reachable."""
    return _get_redis() is not None
