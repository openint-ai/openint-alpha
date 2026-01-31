"""
DataHub client for sa-agent.
Connects to DataHub running locally and reads dataset schemas.
Falls back to openint-datahub/schemas.py when DataHub is unavailable.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Repo root: openint-agents/sa_agent -> openint-agents -> repo root
_AGENTS_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _AGENTS_DIR.parent
_OPENINT_DATAHUB = _REPO_ROOT / "openint-datahub"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

# Known dataset names (from openint-datahub/schemas.py) for building URNs when reading from DataHub
DEFAULT_DATASET_NAMES = [
    "customers", "ach_transactions", "wire_transactions", "credit_transactions",
    "debit_transactions", "check_transactions", "disputes",
    "country_codes", "state_codes", "zip_codes",
]

DATAHUB_GMS_URL = os.getenv("DATAHUB_GMS_URL", "http://localhost:8080")
DATAHUB_TOKEN = os.getenv("DATAHUB_TOKEN", "")
PLATFORM = "openint"
ENVIRONMENT = "PROD"


def _load_schema_from_module() -> Dict[str, Dict[str, Any]]:
    """Load schema from openint-datahub/schemas.py (no DataHub required)."""
    if _OPENINT_DATAHUB.exists():
        if str(_OPENINT_DATAHUB) not in sys.path:
            sys.path.insert(0, str(_OPENINT_DATAHUB))
    try:
        from schemas import get_dataset_schemas
        return get_dataset_schemas()
    except Exception as e:
        logger.warning("Could not load schema from openint-datahub/schemas.py: %s", e)
    return {}


def _schema_from_datahub_entity(schema_aspect: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Build a schema dict (description, category, fields) from DataHub SchemaMetadata aspect."""
    if schema_aspect is None:
        return None
    fields = []
    for f in getattr(schema_aspect, "fields", []) or []:
        field_path = getattr(f, "fieldPath", None) or getattr(f, "fieldPath", "")
        native_type = getattr(f, "nativeDataType", None) or "STRING"
        desc = getattr(f, "description", None) or ""
        type_str = (native_type if isinstance(native_type, str) else str(native_type)).upper()
        fields.append({
            "name": field_path,
            "type": type_str if type_str else "STRING",
            "description": desc or (str(field_path).replace("_", " ").title() if field_path else ""),
        })
    return {
        "description": getattr(schema_aspect, "schemaName", "") or "Dataset",
        "category": "dataset",
        "fields": fields,
    }


def get_schema_from_datahub() -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Connect to DataHub and read dataset schemas.
    Returns dict mapping dataset name -> { description, category, fields } or None on failure.
    """
    try:
        from datahub.ingestion.graph.client import DataHubGraph
        from datahub.ingestion.graph.config import DatahubClientConfig
    except ImportError:
        logger.debug("DataHub SDK not installed; cannot read from DataHub")
        return None

    config_data: Dict[str, Any] = {"server": DATAHUB_GMS_URL}
    if DATAHUB_TOKEN:
        config_data["token"] = DATAHUB_TOKEN
    token_file = _OPENINT_DATAHUB / ".datahub_token"
    if not DATAHUB_TOKEN and token_file.exists():
        try:
            config_data["token"] = token_file.read_text().strip()
        except Exception:
            pass

    try:
        config = DatahubClientConfig(**config_data)
        graph = DataHubGraph(config)
    except Exception as e:
        logger.warning("DataHub connection failed: %s", e)
        return None

    out: Dict[str, Dict[str, Any]] = {}
    for name in DEFAULT_DATASET_NAMES:
        urn = f"urn:li:dataset:(urn:li:dataPlatform:{PLATFORM},{name},{ENVIRONMENT})"
        try:
            entity = graph.get_entity(entity_urn=urn)
            if not entity:
                continue
            # Entity can be a dict with 'dataset' key or similar; aspect may be under aspects
            aspects = getattr(entity, "get", None)
            if aspects is None:
                aspects = getattr(entity, "aspects", {})
            else:
                aspects = entity.get("aspects", entity) if isinstance(entity, dict) else getattr(entity, "aspects", {})
            if isinstance(aspects, dict):
                schema_aspect = aspects.get("schemaMetadata") or aspects.get("SchemaMetadata")
            else:
                schema_aspect = getattr(aspects, "schemaMetadata", None) or getattr(aspects, "SchemaMetadata", None)
            if schema_aspect is None and isinstance(entity, dict):
                schema_aspect = entity.get("schemaMetadata") or entity.get("SchemaMetadata")
            parsed = _schema_from_datahub_entity(schema_aspect)
            if parsed:
                out[name] = parsed
        except Exception as e:
            logger.debug("Could not get schema for %s from DataHub: %s", name, e)
    return out if out else None


def get_schema() -> Dict[str, Dict[str, Any]]:
    """
    Get dataset schema: from DataHub if available, else from openint-datahub/schemas.py.
    Returns dict mapping dataset name -> { description, category, fields }.
    """
    schema, _ = get_schema_and_source()
    return schema


def get_schema_and_source() -> tuple[Dict[str, Dict[str, Any]], str]:
    """
    Get dataset schema and its source for LLM context.
    Returns (schema, source) where source is "datahub" (DataHub assets and schema)
    or "openint-datahub" (fallback when DataHub unavailable).
    Ensures Ollama/LLM context is always provided via DataHub assets and schema when available.
    """
    schema = get_schema_from_datahub()
    if schema:
        logger.debug("Loaded schema from DataHub (%s datasets)", len(schema))
        return (schema, "datahub")
    schema = _load_schema_from_module()
    logger.debug("Loaded schema from openint-datahub/schemas.py (%s datasets)", len(schema))
    return (schema, "openint-datahub")
