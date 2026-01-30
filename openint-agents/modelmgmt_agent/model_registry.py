"""
Redis-backed Model Registry for modelmgmt-agent.

Downloads models from Hugging Face, stores in Redis for in-memory lookup so
replicas hydrate in O(1). Thundering-herd protection: one writer, others wait.
Uses REDIS_HOST/REDIS_PORT (default 127.0.0.1:6379).
"""

import os
import io
import time
import zipfile
import tempfile
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if _hf_token and _hf_token.strip():
    try:
        import huggingface_hub
        huggingface_hub.login(token=_hf_token.strip())
    except Exception:
        pass

MODEL_KEY_PREFIX = "model_registry"
MODEL_LOCK_TTL = 600
MODEL_WAIT_TIMEOUT = 620
MODEL_WAIT_POLL_INTERVAL = 2.0
REDIS_DEFAULT_HOST = "127.0.0.1"
REDIS_DEFAULT_PORT = 6379
REDIS_CONNECT_TIMEOUT = 5
REDIS_RETRY_AFTER_FAILURE_SECONDS = 30

_redis_binary_client = None
_redis_unavailable_until = 0.0


def check_redis_registry() -> Tuple[bool, str, int, Optional[str]]:
    """Check if Redis is reachable. Returns (connected, host, port, error_message)."""
    host = (os.getenv("REDIS_HOST") or "").strip() or REDIS_DEFAULT_HOST
    try:
        port = int(os.getenv("REDIS_PORT") or str(REDIS_DEFAULT_PORT))
    except (TypeError, ValueError):
        port = REDIS_DEFAULT_PORT
    try:
        import redis
        r = redis.Redis(host=host, port=port, db=0, decode_responses=False, socket_connect_timeout=2)
        r.ping()
        return True, host, port, None
    except Exception as e:
        return False, host, port, str(e)


def _get_redis_binary():
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
        r = redis.Redis(host=host, port=port, db=0, decode_responses=False, socket_connect_timeout=REDIS_CONNECT_TIMEOUT)
        r.ping()
        _redis_binary_client = r
        logger.info("Redis model registry connected at %s:%s", host, port)
        return r
    except Exception as e:
        _redis_unavailable_until = time.time() + REDIS_RETRY_AFTER_FAILURE_SECONDS
        logger.warning("Redis model registry unavailable at %s:%s: %s (retry in %ss)", host, port, e, REDIS_RETRY_AFTER_FAILURE_SECONDS)
        return None


def _model_key(name: str) -> str:
    safe = name.replace("/", "_").replace("\\", "_").strip()
    return f"{MODEL_KEY_PREFIX}:{safe}:blob"


def _model_lock_key(name: str) -> str:
    return f"{MODEL_KEY_PREFIX}:{name.replace('/', '_').replace('\\\\', '_').strip()}:lock"


def _model_ready_key(name: str) -> str:
    return f"{MODEL_KEY_PREFIX}:{name.replace('/', '_').replace('\\\\', '_').strip()}:ready"


def load_model_from_registry(model_name: str):
    """
    Load a SentenceTransformer model from Redis (or download from HuggingFace and store in Redis).
    Returns the model instance or None.
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

    try:
        raw = redis_client.get(blob_key)
        if raw is not None and len(raw) > 0:
            logger.info("modelmgmt-agent: Model %s found in Redis, hydrating", model_name)
            with tempfile.TemporaryDirectory(prefix="model_registry_") as tmpdir:
                with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
                    zf.extractall(tmpdir)
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(tmpdir)
                logger.info("modelmgmt-agent: Model %s loaded from Redis (O(1) boot)", model_name)
                return model
    except Exception as e:
        logger.warning("Failed to load model %s from Redis: %s", model_name, e)

    acquired = False
    try:
        acquired = redis_client.set(lock_key, instance_id.encode("utf-8"), nx=True, ex=MODEL_LOCK_TTL)
    except Exception as e:
        logger.warning("Redis lock failed for %s: %s", model_name, e)

    if acquired:
        try:
            logger.info("modelmgmt-agent: Acquired model registry lock for %s (this pod will populate Redis)", model_name)
            model = _load_from_huggingface(model_name)
            if model is None:
                return None
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
            redis_client.set(blob_key, blob, ex=7 * 24 * 3600)
            redis_client.set(ready_key, b"1", ex=7 * 24 * 3600)
            logger.info("modelmgmt-agent: Model %s stored in Redis (%d bytes)", model_name, len(blob))
            return model
        finally:
            try:
                redis_client.delete(lock_key)
            except Exception:
                pass
    else:
        deadline = time.monotonic() + MODEL_WAIT_TIMEOUT
        while time.monotonic() < deadline:
            try:
                raw = redis_client.get(blob_key)
                if raw is not None and len(raw) > 0:
                    logger.info("modelmgmt-agent: Model %s now in Redis, hydrating", model_name)
                    with tempfile.TemporaryDirectory(prefix="model_registry_") as tmpdir:
                        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
                            zf.extractall(tmpdir)
                        from sentence_transformers import SentenceTransformer
                        model = SentenceTransformer(tmpdir)
                        logger.info("modelmgmt-agent: Model %s loaded from Redis after wait", model_name)
                        return model
            except Exception as e:
                logger.warning("Failed to load model %s from Redis (retrying): %s", model_name, e)
            time.sleep(MODEL_WAIT_POLL_INTERVAL)
        logger.warning("Timeout waiting for model %s in Redis, falling back to HuggingFace", model_name)
        return _load_from_huggingface(model_name)


def _load_from_huggingface(model_name: str):
    os.environ["TQDM_DISABLE"] = "1"
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("modelmgmt-agent: Loading model %s from HuggingFace", model_name)
        model = SentenceTransformer(model_name)
        logger.info("modelmgmt-agent: Model %s loaded from HuggingFace", model_name)
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
