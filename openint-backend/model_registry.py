"""
Redis-backed Model Registry for fast scaling.

Treats Redis as the authoritative artifact store for ML models so new replicas
can hydrate from the internal cache in O(1) time instead of waiting on
HuggingFace downloads. Includes thundering-herd protection so only one pod
downloads and writes to Redis; others wait and read.

Uses same REDIS_HOST/REDIS_PORT as chat cache; connect timeout and retry-after
avoid blocking when Redis is down.
"""

import os
import io
import time
import zipfile
import tempfile
import logging
from typing import Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Use HF_TOKEN when set (higher rate limits, faster downloads; suppresses unauthenticated warning)
_hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if _hf_token and _hf_token.strip():
    try:
        import huggingface_hub
        huggingface_hub.login(token=_hf_token.strip())
    except Exception:
        pass

# Redis key prefix for model artifacts
MODEL_KEY_PREFIX = "model_registry"
# Lock TTL (seconds) – writer must populate within this time
MODEL_LOCK_TTL = 600
# How long waiters poll for the model to appear (seconds)
MODEL_WAIT_TIMEOUT = 620
# Poll interval while waiting for another pod to finish writing
MODEL_WAIT_POLL_INTERVAL = 2.0

# Same env as main.py chat cache; fail fast when Redis is down
REDIS_DEFAULT_HOST = "127.0.0.1"
REDIS_DEFAULT_PORT = 6379
REDIS_CONNECT_TIMEOUT = 5  # Seconds to wait for TCP connect (model blobs can be large, so slightly higher than chat)
REDIS_RETRY_AFTER_FAILURE_SECONDS = 30  # Don't retry connect for this long after a failure

_redis_binary_client = None
_redis_unavailable_until = 0.0


def check_redis_registry() -> Tuple[bool, str, int, Optional[str]]:
    """
    Check if Redis is reachable for the model registry (same host/port as chat cache).
    Does not create or cache a client; use for health checks only.

    Returns:
        (connected, host, port, error_message)
    """
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
            decode_responses=False,
            socket_connect_timeout=2,
        )
        r.ping()
        return True, host, port, None
    except Exception as e:
        return False, host, port, str(e)


def _get_redis_binary():
    """Redis client with decode_responses=False for storing binary model blobs.
    Uses REDIS_HOST, REDIS_PORT; same defaults as chat cache (127.0.0.1:6379).
    Fails fast and backs off for REDIS_RETRY_AFTER_FAILURE_SECONDS when Redis is down.
    """
    global _redis_binary_client, _redis_unavailable_until
    if _redis_binary_client is not None:
        return _redis_binary_client
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
            decode_responses=False,
            socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
        )
        r.ping()
        _redis_binary_client = r
        logger.info("Redis model registry connected at %s:%s", host, port)
        return r
    except Exception as e:
        _redis_unavailable_until = time.time() + REDIS_RETRY_AFTER_FAILURE_SECONDS
        logger.warning(
            "Redis model registry unavailable at %s:%s: %s (retry in %ss)",
            host, port, e, REDIS_RETRY_AFTER_FAILURE_SECONDS,
        )
        return None


def _model_key(name: str) -> str:
    """Redis key for model blob (sanitize name for use as key)."""
    safe = name.replace("/", "_").replace("\\", "_").strip()
    return f"{MODEL_KEY_PREFIX}:{safe}:blob"


def _model_lock_key(name: str) -> str:
    return f"{MODEL_KEY_PREFIX}:{name.replace('/', '_').replace('\\\\', '_').strip()}:lock"


def _model_ready_key(name: str) -> str:
    return f"{MODEL_KEY_PREFIX}:{name.replace('/', '_').replace('\\\\', '_').strip()}:ready"


def load_model_from_registry(model_name: str):
    """
    Load a SentenceTransformer model, hydrating from Redis when present.

    - If the model blob exists in Redis: unzip to a temp dir and load from path (O(1) boot).
    - If not: acquire a Redis lock (thundering-herd protection). The holder downloads
      from HuggingFace, saves to a temp dir, zips, stores in Redis, releases lock.
      Other replicas wait for the blob to appear, then load from Redis.

    Returns:
        SentenceTransformer instance, or None if Redis and HuggingFace both fail.
    """
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass
    redis_client = _get_redis_binary()
    if redis_client is None:
        return _load_from_huggingface(model_name)

    blob_key = _model_key(model_name)
    lock_key = _model_lock_key(model_name)
    ready_key = _model_ready_key(model_name)
    instance_id = os.getenv("HOSTNAME", os.getenv("POD_NAME", str(id(redis_client))))

    # 1) Try to load from Redis (fast path)
    try:
        raw = redis_client.get(blob_key)
        if raw is not None and len(raw) > 0:
            logger.info("Model %s found in Redis, hydrating", model_name)
            with tempfile.TemporaryDirectory(prefix="model_registry_") as tmpdir:
                with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
                    zf.extractall(tmpdir)
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(tmpdir)
                logger.info("Model %s loaded from Redis (O(1) boot)", model_name)
                return model
    except Exception as e:
        logger.warning("Failed to load model %s from Redis: %s", model_name, e)

    # 2) Try to become the writer (thundering-herd protection)
    acquired = False
    try:
        acquired = redis_client.set(lock_key, instance_id.encode("utf-8"), nx=True, ex=MODEL_LOCK_TTL)
    except Exception as e:
        logger.warning("Redis lock failed for %s: %s", model_name, e)

    if acquired:
        try:
            logger.info("Acquired model registry lock for %s (this pod will populate Redis)", model_name)
            model = _load_from_huggingface(model_name)
            if model is None:
                return None
            # Save to temp dir, zip, store in Redis
            with tempfile.TemporaryDirectory(prefix="model_registry_save_") as tmpdir:
                model.save(tmpdir)
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for root, _, files in os.walk(tmpdir):
                        for f in files:
                            path = os.path.join(root, f)
                            arcname = os.path.relpath(path, tmpdir)
                            zf.write(path, arcname)
                blob = buf.getvalue()
            redis_client.set(blob_key, blob, ex=7 * 24 * 3600)  # 7 days TTL
            redis_client.set(ready_key, b"1", ex=7 * 24 * 3600)
            logger.info("Model %s stored in Redis (%d bytes)", model_name, len(blob))
            return model
        finally:
            try:
                redis_client.delete(lock_key)
            except Exception:
                pass
    else:
        # 3) Wait for another pod to populate Redis (thundering-herd: waiters)
        logger.info("Waiting for model %s to appear in Redis (another pod is populating)", model_name)
        deadline = time.monotonic() + MODEL_WAIT_TIMEOUT
        while time.monotonic() < deadline:
            try:
                raw = redis_client.get(blob_key)
                if raw is not None and len(raw) > 0:
                    logger.info("Model %s now in Redis, hydrating", model_name)
                    with tempfile.TemporaryDirectory(prefix="model_registry_") as tmpdir:
                        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
                            zf.extractall(tmpdir)
                        from sentence_transformers import SentenceTransformer
                        model = SentenceTransformer(tmpdir)
                        logger.info("Model %s loaded from Redis after wait", model_name)
                        return model
            except Exception as e:
                logger.warning("Failed to load model %s from Redis (retrying): %s", model_name, e)
            time.sleep(MODEL_WAIT_POLL_INTERVAL)
        logger.warning("Timeout waiting for model %s in Redis, falling back to HuggingFace", model_name)
        return _load_from_huggingface(model_name)


def _load_from_huggingface(model_name: str):
    """Load model directly from HuggingFace (no Redis)."""
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading model %s from HuggingFace", model_name)
        model = SentenceTransformer(model_name)
        logger.info("Model %s loaded from HuggingFace", model_name)
        return model
    except Exception as e:
        logger.warning("Failed to load model %s from HuggingFace: %s", model_name, e)
        try:
            from sentence_transformers import SentenceTransformer
            fallback = "all-MiniLM-L6-v2"
            logger.info("Trying fallback model %s", fallback)
            return SentenceTransformer(fallback)
        except Exception as e2:
            logger.error("Fallback model also failed: %s", e2)
            return None
