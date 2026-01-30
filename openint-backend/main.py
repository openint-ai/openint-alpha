"""
OpenInt Backend
Flask/FastAPI backend for API gateway to agent system
"""

from flask import Flask, Blueprint, request, jsonify, send_from_directory, g
from flask_cors import CORS
import hashlib
import json
import logging
import os
import re
import sys
import time
from typing import Dict, Any, Optional, List

# Observability: structured logging + OpenTelemetry (must import before other app code)
try:
    from observability import get_logger, setup as setup_observability
except ImportError:
    import logging
    def get_logger(name: str):
        return logging.getLogger(name)
    def setup_observability(app=None):
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

# Hugging Face token (before any HF download): enables higher rate limits and faster downloads; suppresses unauthenticated warning
# Load .env from repo root and from backend dir so OLLAMA_HOST, OLLAMA_MODEL, HF_TOKEN, etc. are available
current_file = os.path.abspath(__file__)
_backend_dir = os.path.dirname(current_file)
_repo_root = os.path.abspath(os.path.join(_backend_dir, ".."))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_repo_root, ".env"))  # repo root (e.g. when using start_backend.sh)
    load_dotenv(os.path.join(_backend_dir, ".env"))  # openint-backend/.env
except ImportError:
    pass
_hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if _hf_token and _hf_token.strip():
    try:
        import huggingface_hub
        huggingface_hub.login(token=_hf_token.strip())
    except Exception:
        pass

# Add openint-agents and openint-vectordb to path so modelmgmt-agent and search_agent (milvus_client) resolve
current_dir = _backend_dir
_repo_root = os.path.abspath(os.path.join(current_dir, '..'))
# Merged UI: serve openint-ui/dist when SERVE_UI=true or FLASK_ENV=production and dist exists
_ui_dist = os.getenv("OPENINT_UI_DIST") or os.path.join(_repo_root, "openint-ui", "dist")
_serve_ui = (os.getenv("SERVE_UI") or "").strip().lower() in ("1", "true", "yes") or (
    os.path.isdir(_ui_dist) and (os.getenv("FLASK_ENV") or "").strip().lower() == "production"
)
agent_system_path = os.path.abspath(os.path.join(_repo_root, 'openint-agents'))
if agent_system_path not in sys.path:
    sys.path.insert(0, agent_system_path)
_vectordb_milvus_path = os.path.abspath(os.path.join(_repo_root, 'openint-vectordb', 'milvus'))
if os.path.isdir(_vectordb_milvus_path) and _vectordb_milvus_path not in sys.path:
    sys.path.insert(0, _vectordb_milvus_path)
_openint_graph_path = os.path.abspath(os.path.join(_repo_root, 'openint-graph'))
if os.path.isdir(_openint_graph_path) and _openint_graph_path not in sys.path:
    sys.path.insert(0, _openint_graph_path)

# Semantic analysis: backend only calls modelmgmt-agent. Model loading, Redis, and annotation are done by the agent.
MULTI_MODEL_AVAILABLE = False
get_analyzer = None
analyze_query_multi_model = None
MODEL_METADATA = []
DROPDOWN_MODEL_IDS = []

try:
    from modelmgmt_agent.semantic_analyzer import (
        get_analyzer as _get_analyzer,
        analyze_query_multi_model as _analyze_multi,
        MODEL_METADATA as _MODEL_METADATA,
        DROPDOWN_MODEL_IDS as _DROPDOWN_MODEL_IDS,
    )
    get_analyzer = _get_analyzer
    analyze_query_multi_model = _analyze_multi
    MODEL_METADATA = _MODEL_METADATA
    DROPDOWN_MODEL_IDS = _DROPDOWN_MODEL_IDS
    MULTI_MODEL_AVAILABLE = True
except ImportError as e:
    get_logger(__name__).warning(
        "modelmgmt-agent not available; semantic endpoints will return 503. Install openint-agents and ensure modelmgmt_agent is on path.",
        extra={"error": str(e)},
    )


def _get_embedding_model_from_modelmgmt(model_name: str):
    """
    Return a SentenceTransformer model only if modelmgmt-agent has loaded it (Redis/HF).
    Backend does not load models; modelmgmt-agent owns all downloads and Redis.
    """
    if not get_analyzer or model_name not in DROPDOWN_MODEL_IDS:
        return None
    logger.info(
        "modelmgmt-agent: loading embedding models (preload=True). models=%s",
        DROPDOWN_MODEL_IDS,
    )
    analyzer = get_analyzer(models=DROPDOWN_MODEL_IDS, preload=True)
    return analyzer.loaded_models.get(model_name) if analyzer else None


# Try to import agent system
AgentOrchestrator = None
AGENT_SYSTEM_AVAILABLE = False

try:
    from communication.orchestrator import AgentOrchestrator
    AGENT_SYSTEM_AVAILABLE = True
except ImportError as e:
    AGENT_SYSTEM_AVAILABLE = False
    _log = get_logger(__name__)
    _log.warning(
        "Agent system not available",
        extra={
            "error": str(e),
            "agent_system_path": agent_system_path,
            "path_exists": os.path.exists(agent_system_path),
            "contents": os.listdir(agent_system_path) if os.path.exists(agent_system_path) else None,
        },
    )
    # Create a dummy class for type hints if import fails
    class AgentOrchestrator:
        def process_query(self, *args, **kwargs):
            return {"error": "Agent system not available"}
        def get_query_result(self, *args, **kwargs):
            return None
        @property
        def registry(self):
            class DummyRegistry:
                def list_agents(self):
                    return []
            return DummyRegistry()

app = Flask(__name__)
# Enable CORS for frontend (running separately)
CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
CORS(app, origins=CORS_ORIGINS)

# MODEL_METADATA and DROPDOWN_MODEL_IDS are imported from modelmgmt_agent when available (single source of truth for the 3 UI dropdown models).

# Observability: JSON logging + OpenTelemetry tracing (idempotent)
setup_observability(app)
logger = get_logger(__name__)

# --- UI interaction logging: log all API requests (method, path, key params, status, duration) ---
def _ui_interaction_summary():
    """Build a short summary of the request for logging. GET query params only (no body read)."""
    path = request.path or ""
    summary = {"method": request.method, "path": path}
    if request.args and path.startswith("/api/"):
        q = dict(request.args)
        if "sentence" in q:
            s = str(q["sentence"])
            summary["sentence_preview"] = (s[:80] + "…") if len(s) > 80 else s
        if "model" in q:
            summary["model"] = q["model"]
        if "query" in q:
            s = str(q["query"])
            summary["query_preview"] = (s[:80] + "…") if len(s) > 80 else s
    return summary


@app.before_request
def _log_ui_request():
    """Log every API request (UI interaction) with method, path, and GET params."""
    if request.path and request.path.startswith("/api/"):
        g._ui_request_start = time.time()
        summary = _ui_interaction_summary()
        logger.info("UI request", extra={"ui_interaction": summary})


@app.after_request
def _log_ui_response(response):
    """Log API response status and duration for UI interactions."""
    if request.path and request.path.startswith("/api/") and getattr(g, "_ui_request_start", None) is not None:
        duration_ms = (time.time() - g._ui_request_start) * 1000
        logger.info(
            "UI response",
            extra={
                "path": request.path,
                "method": request.method,
                "status": response.status_code,
                "duration_ms": round(duration_ms, 1),
            },
        )
    return response


# Redis cache for chat responses: vector DB results stored in Redis, served from Redis if same query within TTL.
# Default: 127.0.0.1:6379 (backend on host → Redis in Docker with -p 6379:6379).
# Backend in Docker: set REDIS_HOST to the Redis service name (e.g. redis) and ensure same Docker network.
REDIS_CACHE_TTL_SECONDS = 60  # Same query served from Redis if repeated within 1 minute
REDIS_DEFAULT_HOST = "127.0.0.1"
REDIS_DEFAULT_PORT = 6379
REDIS_CONNECT_TIMEOUT = 1  # Fail fast when Redis is down so we don't block every request
REDIS_RETRY_AFTER_FAILURE_SECONDS = 30  # Don't retry connection for this long after a failure
_redis_client = None
_redis_unavailable_until = 0.0  # Skip connect attempts until this time (avoid blocking every request)

def get_redis():
    """Return a Redis client for localhost:6379 (or REDIS_HOST/REDIS_PORT), or None if Redis is unavailable."""
    global _redis_client, _redis_unavailable_until
    if _redis_client is not None:
        return _redis_client
    if time.time() < _redis_unavailable_until:
        return None  # Recently failed; don't block this request with another connect attempt
    host = (os.getenv("REDIS_HOST") or "").strip() or REDIS_DEFAULT_HOST
    try:
        port = int(os.getenv("REDIS_PORT") or REDIS_DEFAULT_PORT)
    except (TypeError, ValueError):
        port = REDIS_DEFAULT_PORT
    try:
        import redis
        r = redis.Redis(host=host, port=port, db=0, decode_responses=True, socket_connect_timeout=REDIS_CONNECT_TIMEOUT)
        r.ping()
        _redis_client = r
        logger.info("Redis chat cache connected at %s:%s – responses cached before returning to UI", host, port, extra={"redis_host": host, "redis_port": port})
        return r
    except Exception as e:
        _redis_unavailable_until = time.time() + REDIS_RETRY_AFTER_FAILURE_SECONDS
        logger.info("Redis chat cache disabled – could not connect to %s:%s: %s (retry in %ss)", host, port, e, REDIS_RETRY_AFTER_FAILURE_SECONDS, extra={"redis_host": host, "redis_port": port, "error": str(e)})
        return None


@app.errorhandler(500)
def handle_500(e):
    """Ensure 500 always returns JSON so the UI can show a real error message."""
    import traceback
    error_msg = str(getattr(e, 'description', None) or e)
    logger.error("500 Internal Server Error", extra={"error": error_msg}, exc_info=True)
    return jsonify({
        "success": False,
        "error": error_msg or "Internal server error",
        "answer": f"Error: {error_msg or 'Internal server error'}",
        "sources": [],
        "query_time_ms": 0,
    }), 500


@app.errorhandler(Exception)
def handle_uncaught(e):
    """Catch uncaught exceptions and return JSON (preserve 404/405 etc.)."""
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return jsonify({"success": False, "error": e.description or str(e)}), e.code
    error_msg = str(e)
    logger.error("Uncaught exception", extra={"error": error_msg}, exc_info=True)
    return jsonify({
        "success": False,
        "error": error_msg,
        "answer": f"Error: {error_msg}",
        "sources": [],
        "query_time_ms": 0,
    }), 500

# Model loading and Redis are handled by modelmgmt-agent on first semantic request; backend does not preload.

# Initialize agents first, then orchestrator with agent_instances for LangGraph
orchestrator: Optional[Any] = None
_agent_instances: List[Any] = []
if AGENT_SYSTEM_AVAILABLE:
    try:
        from agents.search_agent import SearchAgent
        from agents.graph_agent import GraphAgent
        from sg_agent.schema_generator_agent import SchemaGeneratorAgent
        from modelmgmt_agent.modelmgmt_agent import ModelMgmtAgent
        search_agent = SearchAgent()
        _agent_instances.append(search_agent)
        graph_agent = GraphAgent()
        _agent_instances.append(graph_agent)
        sg_agent = SchemaGeneratorAgent()
        _agent_instances.append(sg_agent)
        modelmgmt_agent = ModelMgmtAgent()
        _agent_instances.append(modelmgmt_agent)
        agent_instances_map = {a.name: a for a in _agent_instances}
        agent_runner = None
        try:
            import a2a as _a2a
            _a2a.set_agent_instances(agent_instances_map)
            agent_runner = lambda name, query, ctx: _a2a.invoke_agent_via_a2a(name, query, ctx)
        except ImportError:
            pass
        orchestrator = AgentOrchestrator(
            agent_instances=agent_instances_map,
            agent_runner=agent_runner,
        )
        logger.info(
            "Initialized agents and LangGraph orchestrator (A2A for agent communication)",
            extra={
                "agent_count": len(_agent_instances),
                "agent_names": list(agent_instances_map.keys()),
            },
        )
    except Exception as agent_error:
        logger.warning("Could not initialize agents/orchestrator", extra={"error": str(agent_error)}, exc_info=True)


def _build_debug_info(query: str, orchestrator: Any, selected_model: Optional[str] = None) -> Dict[str, Any]:
    """
    Build debug information including semantic analysis with finance model.
    Shows semantic highlights from the finance-specific model.
    """
    # Always use finance model
    finance_model = "mukaj/fin-mpnet-base"
    selected_model = finance_model
    import json
    
    # Use multi-model semantic analysis ONLY if analyzer is already loaded
    # Don't trigger model loading here - it should only happen when explicitly requested in chat endpoint
    multi_model_result = None
    best_model = None
    
    # If user selected a specific model, use it for semantic analysis
    # Otherwise, use multi-model analysis if available
    if selected_model and MULTI_MODEL_AVAILABLE:
        try:
            # Analyze with the selected model only
            analyzer = get_analyzer()
            if analyzer:
                # Create a single-model analysis result
                if selected_model in analyzer.loaded_models:
                    model = analyzer.loaded_models[selected_model]
                    embedding = model.encode(query, convert_to_numpy=True)
                    semantic_analysis = analyzer._extract_semantic_tags(query, selected_model, embedding)
                    highlighted = analyzer._highlight_query_with_tags(query, semantic_analysis.get("tags", []))
                    
                    multi_model_result = {
                        "query": query,
                        "models_analyzed": 1,
                        "models": {
                            selected_model: {
                                **semantic_analysis,
                                "highlighted_query": highlighted,
                                "embedding_preview": embedding[:10].tolist(),
                            }
                        },
                        "best_model": selected_model,
                        "best_model_score": analyzer._score_model_quality({
                            **semantic_analysis,
                            "highlighted_query": highlighted,
                        }),
                        "model_scores": {
                            selected_model: analyzer._score_model_quality({
                                **semantic_analysis,
                                "highlighted_query": highlighted,
                            })
                        },
                        "aggregated": {
                            "all_tags": semantic_analysis.get("tags", []),
                            "consensus_tags": semantic_analysis.get("tags", []),
                            "tag_counts": {},
                            "entity_counts": {},
                            "action_counts": {},
                            "total_tags": len(semantic_analysis.get("tags", [])),
                            "consensus_count": len(semantic_analysis.get("tags", []))
                        },
                        "summary": {}
                    }
                    best_model = selected_model
                else:
                    # Get model from modelmgmt-agent (already loads via Redis/HF)
                    try:
                        loaded = _get_embedding_model_from_modelmgmt(selected_model)
                        if loaded is not None:
                            model = loaded
                            embedding = model.encode(query, convert_to_numpy=True)
                            semantic_analysis = analyzer._extract_semantic_tags(query, selected_model, embedding)
                            highlighted = analyzer._highlight_query_with_tags(query, semantic_analysis.get("tags", []))
                        
                            multi_model_result = {
                                "query": query,
                                "models_analyzed": 1,
                                "models": {
                                    selected_model: {
                                        **semantic_analysis,
                                        "highlighted_query": highlighted,
                                        "embedding_preview": embedding[:10].tolist(),
                                    }
                                },
                                "best_model": selected_model,
                                "best_model_score": analyzer._score_model_quality({
                                    **semantic_analysis,
                                    "highlighted_query": highlighted,
                                }),
                                "model_scores": {
                                    selected_model: analyzer._score_model_quality({
                                        **semantic_analysis,
                                        "highlighted_query": highlighted,
                                    })
                                },
                                "aggregated": {
                                    "all_tags": semantic_analysis.get("tags", []),
                                    "consensus_tags": semantic_analysis.get("tags", []),
                                    "tag_counts": {},
                                    "entity_counts": {},
                                    "action_counts": {},
                                    "total_tags": len(semantic_analysis.get("tags", [])),
                                    "consensus_count": len(semantic_analysis.get("tags", []))
                                },
                                "summary": {}
                            }
                            best_model = selected_model
                    except Exception as load_error:
                        logger.warning("Could not load model for semantic analysis", extra={"model": selected_model, "error": str(load_error)})
        except Exception as e:
            logger.warning("Single-model analysis failed", extra={"error": str(e)}, exc_info=True)
    
    # Fallback to multi-model analysis if no specific model selected
    elif MULTI_MODEL_AVAILABLE:
        try:
            # Check if analyzer is already loaded (don't trigger loading here)
            analyzer = get_analyzer()
            if analyzer and hasattr(analyzer, '_models_loaded') and analyzer._models_loaded:
                logger.info("modelmgmt-agent: running semantic analysis for chat debug (best_model)")
                multi_model_result = analyzer.analyze_query(query, parallel=True)
                if multi_model_result and multi_model_result.get("best_model"):
                    best_model = multi_model_result["best_model"]
                    logger.info("modelmgmt-agent: semantic analysis completed for chat debug", extra={"best_model": best_model})
        except Exception as e:
            # Silently skip if analyzer not ready - don't block debug output
            pass
    
    # Extract semantics from query (simplified version matching web_ui.py patterns)
    semantics = []
    query_lower = query.lower()
    
    # Amount patterns
    amount_over = re.search(r"\b(?:over|above|greater\s+than|more\s+than)\s+\$?([\d,]+(?:\.\d+)?)\b", query, re.I)
    if amount_over:
        try:
            val = float(amount_over.group(1).replace(",", ""))
            semantics.append({
                "type": "amount_min",
                "label": "Amount over",
                "value": val,
                "snippet": amount_over.group(0)
            })
        except:
            pass
    
    amount_under = re.search(r"\b(?:under|below|less\s+than)\s+\$?([\d,]+(?:\.\d+)?)\b", query, re.I)
    if amount_under:
        try:
            val = float(amount_under.group(1).replace(",", ""))
            semantics.append({
                "type": "amount_max",
                "label": "Amount under",
                "value": val,
                "snippet": amount_under.group(0)
            })
        except:
            pass
    
    amount_between = re.search(r"\bbetween\s+\$?([\d,]+(?:\.\d+)?)\s+and\s+\$?([\d,]+(?:\.\d+)?)\b", query, re.I)
    if amount_between:
        try:
            lo, hi = float(amount_between.group(1).replace(",", "")), float(amount_between.group(2).replace(",", ""))
            semantics.append({
                "type": "amount_between",
                "label": "Amount between",
                "value": f"{lo} and {hi}",
                "snippet": amount_between.group(0)
            })
        except:
            pass
    
    # State detection
    state_codes = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"]
    state_match = re.search(r"\b([A-Z]{2})\b", query)
    if state_match and state_match.group(1) in state_codes:
        semantics.append({
            "type": "state",
            "label": "State",
            "value": state_match.group(1),
            "snippet": state_match.group(1)
        })
    
    # State names (California, New York, etc.) - comprehensive list
    state_names = {
        "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
        "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
        "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
        "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
        "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
        "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH",
        "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
        "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
        "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
        "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
        "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
        "district of columbia": "DC", "dc": "DC",
    }
    # Sort by length (longest first) to match "new york" before "new jersey"
    for name, code in sorted(state_names.items(), key=lambda x: -len(x[0])):
        if name in query_lower:
            match = re.search(re.escape(name), query, re.I)
            if match:
                # Check if we already added this state
                if not any(s.get("type") == "state" and s.get("value") == code for s in semantics):
                    semantics.append({
                        "type": "state",
                        "label": "State",
                        "value": code,
                        "snippet": match.group(0)
                    })
                break
    
    # Customer ID patterns
    cust_id_full = re.search(r"\b(CUST\d+)\b", query, re.I)
    if cust_id_full:
        semantics.append({
            "type": "customer_id",
            "label": "Customer",
            "value": cust_id_full.group(1).upper(),
            "snippet": cust_id_full.group(0)
        })
    
    # Top N pattern
    top_n_match = re.search(r"\btop\s+(\d+)\b", query, re.I)
    if top_n_match:
        semantics.append({
            "type": "top_n",
            "label": "Top N",
            "value": int(top_n_match.group(1)),
            "snippet": top_n_match.group(0).strip()
        })
    
    # Analytical patterns
    analytical_match = re.search(r"\b(largest|high-value|high\s*value|biggest|most|highest)\b", query, re.I)
    if analytical_match:
        semantics.append({
            "type": "analytical",
            "label": "Analytical",
            "value": analytical_match.group(1).strip(),
            "snippet": analytical_match.group(0).strip()
        })
    
    # Comprehensive token-based entity detection
    # Map each token to potential semantic meanings
    
    # Entity type keywords (what the user is asking about)
    entity_keywords = {
        "customer": ["customer", "customers", "client", "clients", "account", "accounts", "user", "users"],
        "transaction": ["transaction", "transactions", "payment", "payments", "transfer", "transfers", 
                       "ach", "wire", "credit", "debit", "check", "checks", "deposit", "deposits",
                       "withdrawal", "withdrawals"],
        "dispute": ["dispute", "disputes", "chargeback", "chargebacks", "complaint", "complaints"],
        "location": ["state", "states", "zip", "zipcode", "zip code", "city", "cities", "address", "addresses",
                    "location", "locations", "region", "regions", "area", "areas"],
        "country": ["country", "countries", "nation", "nations", "usa", "us", "united states"],
    }
    
    # Action/verb keywords (what the user wants to do)
    action_keywords = {
        "search": ["search", "find", "look", "show", "list", "get", "fetch", "retrieve", "query"],
        "filter": ["filter", "where", "with", "having", "that", "which"],
        "aggregate": ["count", "total", "sum", "average", "avg", "how many", "number of"],
        "sort": ["top", "largest", "biggest", "highest", "most", "best", "greatest", "maximum", "max"],
        "compare": ["compare", "versus", "vs", "difference", "between"],
    }
    
    # Status/state keywords
    status_keywords = {
        "active": ["active", "open", "enabled", "live", "current"],
        "inactive": ["inactive", "closed", "disabled", "cancelled", "cancelled", "terminated"],
        "pending": ["pending", "waiting", "processing", "in progress"],
        "completed": ["completed", "done", "finished", "settled", "resolved"],
    }
    
    # Transaction type keywords
    transaction_types = {
        "ach": ["ach"],
        "wire": ["wire", "wires", "wire transfer"],
        "credit": ["credit", "credit card", "creditcard"],
        "debit": ["debit", "debit card", "debitcard"],
        "check": ["check", "checks", "cheque", "cheques"],
    }
    
    # Amount-related keywords
    amount_keywords = {
        "over": ["over", "above", "greater than", "more than", "exceeds", "exceeding"],
        "under": ["under", "below", "less than", "fewer than", "under"],
        "between": ["between", "range", "from", "to"],
        "exact": ["exactly", "equal to", "equals", "="],
    }
    
    # Date/time keywords
    date_keywords = {
        "recent": ["recent", "latest", "newest", "last", "latest"],
        "old": ["old", "older", "earliest", "first"],
        "today": ["today"],
        "yesterday": ["yesterday"],
        "this_week": ["this week", "thisweek"],
        "this_month": ["this month", "thismonth"],
        "this_year": ["this year", "thisyear"],
    }
    
    # Detect entities (what user is asking about)
    detected_entities = []
    entity_snippets = {}
    
    for entity_type, keywords in entity_keywords.items():
        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            match = re.search(pattern, query, re.I)
            if match:
                detected_entities.append(entity_type.title())
                if entity_type not in entity_snippets:
                    entity_snippets[entity_type] = match.group(0)
                break
    
    # Detect transaction types
    detected_transaction_types = []
    for trans_type, keywords in transaction_types.items():
        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            match = re.search(pattern, query, re.I)
            if match:
                detected_transaction_types.append(trans_type.title())
                semantics.append({
                    "type": "transaction_type",
                    "label": "Transaction Type",
                    "value": trans_type.title(),
                    "snippet": match.group(0)
                })
                break
    
    # Detect actions
    detected_actions = []
    for action_type, keywords in action_keywords.items():
        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            match = re.search(pattern, query, re.I)
            if match:
                detected_actions.append(action_type.title())
                break
    
    # Detect status
    detected_statuses = []
    for status_type, keywords in status_keywords.items():
        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            match = re.search(pattern, query, re.I)
            if match:
                detected_statuses.append(status_type.title())
                semantics.append({
                    "type": "status",
                    "label": "Status",
                    "value": status_type.title(),
                    "snippet": match.group(0)
                })
                break
    
    # ZIP code detection (5-digit numbers)
    zip_match = re.search(r"\b(\d{5})(?:-\d{4})?\b", query)
    if zip_match:
        semantics.append({
            "type": "zip_code",
            "label": "ZIP Code",
            "value": zip_match.group(1),
            "snippet": zip_match.group(0)
        })
    
    # Customer ID patterns (more comprehensive)
    cust_patterns = [
        (r"\b(CUST\d+)\b", "CUST format"),
        (r"(?:customer|cust)(?:\s*(?:id)?)?\s*(\d{4,8})\b", "Customer ID number"),
        (r"\b(\d{4,8})\s*(?:customer|cust)\b", "ID before customer"),
    ]
    for pattern, label in cust_patterns:
        match = re.search(pattern, query, re.I)
        if match:
            cust_id = match.group(1) if match.lastindex else match.group(0)
            if not any(s.get("type") == "customer_id" for s in semantics):
                semantics.append({
                    "type": "customer_id",
                    "label": "Customer",
                    "value": cust_id.upper() if cust_id.startswith("CUST") else cust_id,
                    "snippet": match.group(0)
                })
            break
    
    # Date range detection
    date_patterns = [
        (r"\blast\s+(\d+)\s+day(s)?\b", "last_n_days"),
        (r"\bthis\s+month\b", "this_month"),
        (r"\bthis\s+year\b", "this_year"),
        (r"\bin\s+(\d{4})\b", "in_year"),
        (r"\bbetween\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})\b", "date_range"),
    ]
    for pattern, date_type in date_patterns:
        match = re.search(pattern, query, re.I)
        if match:
            if date_type == "last_n_days":
                semantics.append({
                    "type": "date_range",
                    "label": "Date Range",
                    "value": f"Last {match.group(1)} days",
                    "snippet": match.group(0)
                })
            elif date_type == "in_year":
                semantics.append({
                    "type": "date_range",
                    "label": "Date Range",
                    "value": f"Year {match.group(1)}",
                    "snippet": match.group(0)
                })
            else:
                semantics.append({
                    "type": "date_range",
                    "label": "Date Range",
                    "value": match.group(0),
                    "snippet": match.group(0)
                })
            break
    
    # Intent detection (search scope) - comprehensive
    intent_keywords = list(set(detected_entities))
    
    # Add search action if present
    if detected_actions:
        if "search" in [a.lower() for a in detected_actions]:
            if "Search" not in intent_keywords:
                intent_keywords.insert(0, "Search")
    
    if intent_keywords:
        # Find first keyword in query for snippet
        snippet = None
        earliest_pos = len(query) + 1
        for entity_type, snippet_text in entity_snippets.items():
            match = re.search(re.escape(snippet_text), query, re.I)
            if match and match.start() < earliest_pos:
                earliest_pos = match.start()
                snippet = snippet_text
        
        if not snippet:
            # Try to find any entity keyword
            for entity_type, keywords in entity_keywords.items():
                for keyword in keywords:
                    match = re.search(r"\b" + re.escape(keyword) + r"\b", query, re.I)
                    if match:
                        snippet = match.group(0)
                        break
                if snippet:
                    break
        
        if not snippet:
            snippet = query.split()[0] if query.split() else "query"
        
        semantics.append({
            "type": "intent",
            "label": "Search scope",
            "value": ", ".join(intent_keywords[:5]),
            "snippet": snippet
        })
    
    # Token-level semantic mapping (for debugging)
    # Extract all words/tokens from query for token-level analysis
    query_words = re.findall(r'\b\w+\b', query_lower)  # Extract all words/tokens
    
    token_semantics = []
    for word in query_words:
        token_meanings = []
        
        # Check entity types
        for entity_type, keywords in entity_keywords.items():
            if word in keywords:
                token_meanings.append(f"Entity:{entity_type}")
        
        # Check transaction types
        for trans_type, keywords in transaction_types.items():
            if word in keywords:
                token_meanings.append(f"Transaction:{trans_type}")
        
        # Check actions
        for action_type, keywords in action_keywords.items():
            if word in keywords:
                token_meanings.append(f"Action:{action_type}")
        
        # Check status
        for status_type, keywords in status_keywords.items():
            if word in keywords:
                token_meanings.append(f"Status:{status_type}")
        
        # Check amount keywords
        for amount_type, keywords in amount_keywords.items():
            if word in keywords:
                token_meanings.append(f"Amount:{amount_type}")
        
        # Check date keywords
        for date_type, keywords in date_keywords.items():
            if word in keywords:
                token_meanings.append(f"Date:{date_type}")
        
        # Check if it's a number (could be amount, ID, ZIP, etc.)
        if word.isdigit():
            if len(word) == 5:
                token_meanings.append("Possible:ZIP")
            elif len(word) >= 4:
                token_meanings.append("Possible:ID")
            else:
                token_meanings.append("Possible:Amount")
        
        if token_meanings:
            token_semantics.append({
                "token": word,
                "meanings": token_meanings
            })
    
    # Generate query vector using finance model
    query_vector = []
    # Always use finance-specific model
    finance_model = "mukaj/fin-mpnet-base"
    embedding_model = finance_model
    embedding_dims = 768  # Finance MPNet model dimension
    
    try:
        # Try to get MilvusClient from search agent
        if orchestrator and hasattr(orchestrator, 'registry'):
            agents = orchestrator.registry.list_agents()
            for agent_info in agents:
                if agent_info.name == "search_agent":
                    # Try to access the agent's milvus_client
                    # For now, create a temporary client
                    break
        
        # Try to import and use MilvusClient directly
        vectordb_path = os.path.abspath(os.path.join(current_dir, '..', 'openint-vectordb', 'milvus'))
        if vectordb_path not in sys.path:
            sys.path.insert(0, vectordb_path)
        
        try:
            # Use finance model via modelmgmt-agent (no direct loading in backend)
            if MULTI_MODEL_AVAILABLE:
                try:
                    model = _get_embedding_model_from_modelmgmt(finance_model)
                    if model is not None:
                        query_vector = model.encode(query, convert_to_numpy=True).tolist()
                        embedding_dims = len(query_vector)
                        embedding_model = finance_model
                except Exception as model_error:
                    logger.warning("Could not use finance model for debug vector", extra={"model": finance_model, "error": str(model_error)})
                    # Fallback to MilvusClient
                    from milvus_client import MilvusClient
                    temp_client = MilvusClient(embedding_model=finance_model)
                    if hasattr(temp_client, '_generate_embeddings'):
                        query_vector = temp_client._generate_embeddings([query])[0]
                        embedding_dims = len(query_vector)
                        embedding_model = getattr(temp_client, 'embedding_model_name', finance_model)
            else:
                # Use MilvusClient with finance model
                from milvus_client import MilvusClient
                temp_client = MilvusClient(embedding_model=finance_model)
                if hasattr(temp_client, '_generate_embeddings'):
                    query_vector = temp_client._generate_embeddings([query])[0]
                    embedding_dims = len(query_vector)
                    embedding_model = getattr(temp_client, 'embedding_model_name', finance_model)
        except Exception as e:
            logger.warning("Could not generate query vector", extra={"error": str(e)})
            query_vector = [0.0] * embedding_dims
    except Exception as e:
        logger.warning("Error generating debug vector", extra={"error": str(e)})
        query_vector = [0.0] * embedding_dims
    
    # Build result with multi-model highlights
    # Always use finance model
    finance_model = "mukaj/fin-mpnet-base"
    final_embedding_model = finance_model
    
    result = {
        "query": query,
        "semantics": semantics,
        "query_vector": query_vector,
        "embedding_dims": embedding_dims,
        "embedding_model": final_embedding_model
    }
    
    # Add multi-model analysis if available
    if multi_model_result:
        result["multi_model_analysis"] = {
            "models_analyzed": multi_model_result.get("models_analyzed", 0),
            "best_model": best_model,
            "best_model_score": multi_model_result.get("best_model_score", 0.0),
            "model_highlights": {}
        }
        
        # Extract highlights from each model
        for model_name, model_result in multi_model_result.get("models", {}).items():
            if "error" not in model_result and "highlighted_query" in model_result:
                highlighted = model_result["highlighted_query"]
                result["multi_model_analysis"]["model_highlights"][model_name] = {
                    "highlighted_segments": highlighted.get("highlighted_segments", []),
                    "tag_count": highlighted.get("tag_count", 0),
                    "highlighted_count": highlighted.get("highlighted_count", 0),
                    "score": multi_model_result.get("model_scores", {}).get(model_name, 0.0),
                    "is_best": model_name == best_model
                }
        
        # Add model scores
        result["multi_model_analysis"]["model_scores"] = multi_model_result.get("model_scores", {})
        
        # Always use finance model
        finance_model = "mukaj/fin-mpnet-base"
        result["embedding_model"] = finance_model
        result["recommended_model"] = finance_model
    
    return result


@app.route('/api/health', methods=['GET'])
def health():
    """Health check. Redis cache is for chat; model loading/Redis is owned by modelmgmt-agent."""
    redis_client = get_redis()
    redis_ok = redis_client is not None
    try:
        redis_client.ping() if redis_client else None
    except Exception:
        redis_ok = False

    return jsonify({
        "status": "healthy",
        "agent_system": AGENT_SYSTEM_AVAILABLE,
        "orchestrator_ready": orchestrator is not None,
        "modelmgmt_agent_available": MULTI_MODEL_AVAILABLE,
        "redis_cache": {
            "connected": redis_ok,
            "host": os.getenv("REDIS_HOST", REDIS_DEFAULT_HOST),
            "port": int(os.getenv("REDIS_PORT") or REDIS_DEFAULT_PORT),
        },
    })


@app.route('/api/ready', methods=['GET'])
def ready():
    """Readiness: 200 only when chat can be served. Typically ready 30–90s after startup."""
    if orchestrator is not None:
        return jsonify({"ready": True, "message": "Backend is ready for chat requests"}), 200
    return jsonify({
        "ready": False,
        "message": "Agent system not ready yet. Backend typically takes 30–90 seconds to initialize. Try again in a minute.",
    }), 503


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat endpoint - sends query to agent system
    
    Request body:
    {
        "message": "user query",
        "session_id": "optional session id",
        "user_id": "optional user id"
    }
    """
    if not orchestrator:
        return jsonify({
            "error": "Agent system not available. The backend may still be starting (typically 30–90 seconds). Check GET /api/ready or try again in a minute.",
        }), 503
    
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' in request"}), 400
    
    user_query = data["message"]
    session_id = data.get("session_id")
    user_id = data.get("user_id")
    metadata = data.get("metadata", {})
    debug_mode = data.get("debug", False)
    multi_model_analysis = data.get("multi_model_analysis", False)  # New flag for multi-model analysis
    # Always use finance model
    selected_embedding_model = "mukaj/fin-mpnet-base"
    
    # Always use finance-specific model
    finance_model = "mukaj/fin-mpnet-base"
    embedding_model = finance_model
    embedding_dims = 768  # Finance MPNet model dimension

    # Redis cache: same query + model + debug => return cached response
    cache_key_raw = f"{user_query}|{embedding_model}|{debug_mode}"
    cache_key = hashlib.sha256(cache_key_raw.encode()).hexdigest()
    redis_client = get_redis()
    if redis_client:
        t0 = time.perf_counter()
        try:
            raw = redis_client.get(f"chat:cache:{cache_key}")
            redis_query_time_ms = (time.perf_counter() - t0) * 1000
            if raw:
                cached = json.loads(raw)
                cached["from_cache"] = True
                cached["redis_query_time_ms"] = round(redis_query_time_ms, 2)
                logger.info("Chat cache hit", extra={"cache_key_preview": cache_key[:16], "redis_query_time_ms": round(redis_query_time_ms, 2)})
                return jsonify(cached)
        except Exception as e:
            logger.warning("Redis cache get failed, continuing without cache", extra={"error": str(e)})
    
    try:
        # Try to get actual model info from MilvusClient
        try:
            vectordb_path = os.path.abspath(os.path.join(current_dir, '..', 'openint-vectordb', 'milvus'))
            if vectordb_path not in sys.path:
                sys.path.insert(0, vectordb_path)
            from milvus_client import MilvusClient
            temp_client = MilvusClient(embedding_model=finance_model)
            embedding_model = getattr(temp_client, 'embedding_model_name', finance_model)
            embedding_dims = getattr(temp_client, 'embedding_dim', 768)
        except:
            pass  # Use defaults if can't get from client
        
        # Perform multi-model analysis ONLY if explicitly requested (debug mode or multi_model_analysis flag)
        # Don't block query processing - do this in parallel or skip if not needed
        best_model = None
        multi_model_result = None
        
        # Always use finance model for search
        finance_model = "mukaj/fin-mpnet-base"
        metadata["embedding_model"] = finance_model
        metadata["best_model"] = finance_model
        
        # Process query through orchestrator (LangGraph: single call returns aggregated; else polling)
        agents_list = [a.name for a in orchestrator.registry.list_agents()]
        logger.info("Processing query", extra={"query_preview": (user_query or "")[:80], "agent_count": len(agents_list)})
        _query_start = time.time()
        result = orchestrator.process_query(
            user_query=user_query,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata
        )

        # LangGraph path: result is already aggregated (status completed/no_responses)
        # Legacy path: result has status "processing", poll get_query_result until done
        aggregated = None
        waited = 0.0
        start_time = time.time()
        if result.get("status") in ("completed", "no_responses"):
            aggregated = result
            waited = time.time() - _query_start
            logger.info("Query completed (LangGraph)", extra={"query_id": result.get("query_id"), "status": result.get("status")})
        else:
            query_id = result.get("query_id")
            max_wait = 10
            wait_interval = 0.05
            logger.debug("Waiting for agent responses", extra={"max_wait": max_wait, "query_id": query_id})
            while waited < max_wait:
                aggregated = orchestrator.get_query_result(query_id)
                if aggregated:
                    status = aggregated.get("status", "unknown")
                    if status != "processing":
                        break
                time.sleep(wait_interval)
                waited = time.time() - start_time
            if not aggregated:
                aggregated = orchestrator.get_query_result(query_id)
            if not aggregated:
                agents_queried = result.get("agents_queried", [])
                all_agents = [a.name for a in orchestrator.registry.list_agents()]
                return jsonify({
                    "success": False,
                    "answer": f"Query timed out after {waited:.1f}s. Agents queried: {', '.join(agents_queried)}. Available agents: {', '.join(all_agents)}",
                    "sources": [],
                    "query_time_ms": int(waited * 1000),
                    "error": "Query timeout - no response from agents",
                    "agents_queried": agents_queried,
                    "available_agents": all_agents
                })

        # Safety: if LangGraph returned completed/no_responses, we must have aggregated
        if aggregated is None and result.get("status") in ("completed", "no_responses"):
            aggregated = result

        # Transform to UI-expected format (same for LangGraph and legacy polling)
        if aggregated:
                status = aggregated.get("status", "unknown")
                if status == "no_responses":
                    # Debug: Check what agents were queried
                    agents_queried = result.get("agents_queried", [])
                    all_agents = [a.name for a in orchestrator.registry.list_agents()]
                    debug_info = f"Agents queried: {agents_queried}, Available agents: {all_agents}"
                    logger.warning("No agent responses", extra={"agents_queried": agents_queried, "available_agents": all_agents})
                    return jsonify({
                        "success": False,
                        "answer": f"No agents were able to process your query. Available agents: {', '.join(all_agents)}. Please try rephrasing your question.",
                        "sources": [],
                        "query_time_ms": int(waited * 1000),
                        "error": aggregated.get("message", "No agents responded"),
                        "debug": debug_info
                    })
                elif status == "completed":
                    # Extract answer from aggregated results - match web_ui.py format exactly
                    # Check both "responses" and "agent_responses" fields
                    responses = aggregated.get("agent_responses", aggregated.get("responses", []))
                    sources = []
                    
                    # Group results by entity type (like web_ui.py does)
                    by_entity = {}
                    entity_labels = {
                        "customers": "Customer",
                        "ach_transactions": "Transaction (ACH)",
                        "wire_transactions": "Transaction (Wire)",
                        "credit_transactions": "Transaction (Credit)",
                        "debit_transactions": "Transaction (Debit)",
                        "check_transactions": "Transaction (Check)",
                        "disputes": "Dispute",
                        "graph_path": "Related (from graph)",
                        "country_codes": "Country",
                        "state_codes": "State",
                        "zip_codes": "Address / ZIP",
                    }
                    
                    for resp in responses:
                        content = resp.get("content", {})
                        results = content.get("results", [])
                        
                        for r in results:
                            result_content = r.get("content", "")
                            result_metadata = r.get("metadata", {})
                            file_type = (result_metadata.get("file_type") or "").strip().lower()
                            entity_label = entity_labels.get(file_type, file_type.replace("_", " ").title() if file_type else "Record")
                            
                            # Extract structured data (matching web_ui.py _extract_structured_from_content)
                            import json
                            import re
                            structured_data = None
                            
                            # Fallback: parse "key: value | key: value" format from Milvus loader so
                            # frontend can show table and one-line summaries work
                            if result_content and "Structured Data:" not in result_content:
                                parsed = {}
                                for part in result_content.split(" | "):
                                    part = part.strip()
                                    if ": " in part:
                                        k, v = part.split(": ", 1)
                                        k, v = k.strip(), v.strip()
                                        if k:
                                            try:
                                                if "." in v:
                                                    parsed[k] = float(v)
                                                else:
                                                    parsed[k] = int(v)
                                            except ValueError:
                                                parsed[k] = v
                                if parsed:
                                    result_content = result_content.rstrip() + "\nStructured Data: " + json.dumps(parsed)
                            
                            # Find "Structured Data:" marker
                            if "Structured Data:" in result_content:
                                json_start = result_content.find("Structured Data:") + len("Structured Data:")
                                json_str = result_content[json_start:].strip()
                                
                                # Remove trailing ... if present
                                if json_str.endswith("..."):
                                    json_str = json_str[:-3].strip()
                                
                                # Try to find complete JSON object by counting braces
                                brace_count = 0
                                end_pos = len(json_str)
                                found_start = False
                                
                                for i, char in enumerate(json_str):
                                    if char == '{':
                                        brace_count += 1
                                        found_start = True
                                    elif char == '}':
                                        brace_count -= 1
                                        if brace_count == 0 and found_start:
                                            end_pos = i + 1
                                            break
                                
                                json_str = json_str[:end_pos]
                                
                                # Try to parse JSON
                                if json_str.startswith('{'):
                                    try:
                                        structured_data = json.loads(json_str)
                                    except json.JSONDecodeError:
                                        # If JSON is truncated, try to extract what we can
                                        # Look for key-value pairs we can extract
                                        structured_data = {}
                                        # Extract known patterns
                                        patterns = {
                                            "customer_id": r'"customer_id"\s*:\s*"([^"]+)"',
                                            "first_name": r'"first_name"\s*:\s*"([^"]+)"',
                                            "last_name": r'"last_name"\s*:\s*"([^"]+)"',
                                            "email": r'"email"\s*:\s*"([^"]+)"',
                                            "phone": r'"phone"\s*:\s*"([^"]+)"',
                                            "city": r'"city"\s*:\s*"([^"]+)"',
                                            "state_code": r'"state_code"\s*:\s*"([^"]+)"',
                                            "account_status": r'"account_status"\s*:\s*"([^"]+)"',
                                            "zip_code": r'"zip_code"\s*:\s*"([^"]+)"',
                                        }
                                        for key, pattern in patterns.items():
                                            match = re.search(pattern, json_str)
                                            if match:
                                                structured_data[key] = match.group(1)
                            
                            # Build one-line summary (matching web_ui.py _one_line_summary)
                            one_line = None
                            if structured_data:
                                pairs = []
                                if file_type == "customers":
                                    first = " ".join(filter(None, [str(structured_data.get("first_name") or ""), str(structured_data.get("last_name") or "")])).strip()
                                    if first:
                                        pairs.append(("Name", first))
                                    if structured_data.get("customer_id"):
                                        pairs.append(("Customer ID", str(structured_data["customer_id"])))
                                    loc = None
                                    if structured_data.get("city") and structured_data.get("state_code"):
                                        loc = f"{structured_data['city']}, {structured_data['state_code']}"
                                    elif structured_data.get("state_code"):
                                        loc = str(structured_data["state_code"])
                                    if loc:
                                        pairs.append(("Location", loc))
                                    if structured_data.get("zip_code"):
                                        pairs.append(("ZIP", str(structured_data["zip_code"])))
                                    if structured_data.get("account_status"):
                                        pairs.append(("Status", str(structured_data["account_status"])))
                                    if structured_data.get("email"):
                                        pairs.append(("Email", str(structured_data["email"])))
                                elif file_type in ("ach_transactions", "wire_transactions", "credit_transactions", "debit_transactions", "check_transactions"):
                                    if structured_data.get("transaction_id"):
                                        pairs.append(("Transaction ID", str(structured_data["transaction_id"])))
                                    if structured_data.get("customer_id"):
                                        pairs.append(("Customer", str(structured_data["customer_id"])))
                                    if structured_data.get("amount") is not None:
                                        currency = structured_data.get("currency") or "USD"
                                        pairs.append(("Currency", currency))
                                        pairs.append(("Amount", str(structured_data["amount"])))
                                    if structured_data.get("transaction_type"):
                                        pairs.append(("Type", str(structured_data["transaction_type"])))
                                    if structured_data.get("status"):
                                        pairs.append(("Status", str(structured_data["status"])))
                                    if structured_data.get("transaction_date"):
                                        pairs.append(("Date", str(structured_data["transaction_date"])[:10]))
                                elif file_type == "disputes":
                                    if structured_data.get("dispute_id"):
                                        pairs.append(("Dispute ID", str(structured_data["dispute_id"])))
                                    if structured_data.get("customer_id"):
                                        pairs.append(("Customer", str(structured_data["customer_id"])))
                                    if structured_data.get("amount_disputed") is not None:
                                        currency = structured_data.get("currency") or "USD"
                                        pairs.append(("Amount", f"{structured_data['amount_disputed']} {currency}"))
                                    if structured_data.get("dispute_status"):
                                        pairs.append(("Status", str(structured_data["dispute_status"])))
                                elif file_type == "state_codes":
                                    if structured_data.get("state_code"):
                                        pairs.append(("Code", str(structured_data["state_code"])))
                                    if structured_data.get("state_name"):
                                        pairs.append(("State", str(structured_data["state_name"])))
                                elif file_type == "zip_codes":
                                    if structured_data.get("zip_code"):
                                        pairs.append(("ZIP", str(structured_data["zip_code"])))
                                    if structured_data.get("city"):
                                        pairs.append(("City", str(structured_data["city"])))
                                    if structured_data.get("state_code"):
                                        pairs.append(("State", str(structured_data["state_code"])))
                                
                                if pairs:
                                    one_line = " · ".join(f"{label} | {value}" for label, value in pairs)
                            
                            # Fallback to content preview if no structured data
                            if not one_line and result_content:
                                text_part = result_content.split("Structured Data:")[0].strip() if "Structured Data:" in result_content else result_content
                                one_line = (text_part[:120] + "…") if len(text_part) > 120 else text_part
                                one_line = one_line.replace("\n", " ").strip()
                            
                            if one_line:
                                if entity_label not in by_entity:
                                    by_entity[entity_label] = []
                                by_entity[entity_label].append(one_line)
                            
                            # Add to sources
                            sources.append({
                                "id": r.get("id", ""),
                                "content": result_content,
                                "score": r.get("score", 0),
                                "metadata": result_metadata
                            })
                    
                    # Build answer in web_ui.py format (matching _synthesize_answer_from_results)
                    if not by_entity:
                        answer = "I couldn't find any relevant data for that question in the database. Try asking about a specific customer, transaction type, location (state or ZIP), or amount."
                    else:
                        lines = []
                        lines.append(f"Based on **{user_query}**, here's what I found:\n")
                        
                        # Order entities as in web_ui.py
                        entity_order = ["Customer", "Transaction (ACH)", "Transaction (Wire)", "Transaction (Credit)", "Transaction (Debit)", "Transaction (Check)", "Dispute", "Related (from graph)", "State", "Address / ZIP", "Country"]
                        for entity_label in entity_order:
                            if entity_label not in by_entity:
                                continue
                            items = by_entity[entity_label]
                            lines.append(f"**{entity_label}** ({len(items)})")
                            for one_line in items:
                                lines.append(f"  • {one_line}")
                            lines.append("")
                        
                        answer = "\n".join(lines).strip()
                    
                    # Build debug information if requested
                    debug_info = None
                    if debug_mode:
                        debug_info = _build_debug_info(user_query, orchestrator, finance_model)
                    
                    # Always use finance model
                    final_embedding_model = finance_model
                    
                    # Elapsed time (never 0 when we have a result)
                    elapsed_ms = max(1, int((time.time() - start_time) * 1000))
                    # Search timing from agent: embedding vs vector DB (so UI can show breakdown)
                    vector_db_query_time_ms = None
                    embedding_time_ms = None
                    for resp in responses:
                        meta = (resp.get("content") or {}).get("metadata") or {}
                        if "vector_db_query_time_ms" in meta:
                            vector_db_query_time_ms = meta["vector_db_query_time_ms"]
                            embedding_time_ms = meta.get("embedding_time_ms")
                            break
                    
                    response_data = {
                        "success": True,
                        "answer": answer,
                        "sources": sources,
                        "query_time_ms": elapsed_ms,
                        "embedding_dims": embedding_dims,
                        "embedding_model": final_embedding_model
                    }
                    if vector_db_query_time_ms is not None:
                        response_data["vector_db_query_time_ms"] = vector_db_query_time_ms
                    if embedding_time_ms is not None:
                        response_data["embedding_time_ms"] = embedding_time_ms
                    
                    if debug_info:
                        response_data["debug"] = debug_info
                    
                    if multi_model_result:
                        response_data["multi_model_analysis"] = multi_model_result
                    
                    # Store vector DB result in Redis with 1-minute TTL; same query within 1 min served from Redis
                    if redis_client:
                        try:
                            redis_client.set(
                                f"chat:cache:{cache_key}",
                                json.dumps(response_data),
                                ex=REDIS_CACHE_TTL_SECONDS,
                            )
                            logger.info(
                                "Chat response cached in Redis (TTL %ss)",
                                REDIS_CACHE_TTL_SECONDS,
                                extra={"cache_key_preview": cache_key[:16], "ttl_seconds": REDIS_CACHE_TTL_SECONDS},
                            )
                        except Exception as e:
                            logger.warning("Redis cache set failed", extra={"error": str(e)})
                    
                    return jsonify(response_data)
                else:
                    # Still processing or error - return what we have
                    agents_queried = result.get("agents_queried", [])
                    all_agents = [a.name for a in orchestrator.registry.list_agents()]
                    
                    return jsonify({
                        "success": False,
                        "answer": f"Query is still being processed (status: {status}). Agents queried: {', '.join(agents_queried)}. Available agents: {', '.join(all_agents)}",
                        "sources": [],
                        "query_time_ms": int(waited * 1000),
                        "error": aggregated.get("message", f"Status: {status}"),
                        "status": status,
                        "agents_queried": agents_queried,
                        "available_agents": all_agents
                    })
        
        # Fallback when aggregated is None (unexpected result shape or no response)
        logger.warning(
            "Chat fallback: no aggregated result",
            extra={
                "result_status": result.get("status"),
                "result_keys": list(result.keys()) if isinstance(result, dict) else None,
                "result_error": result.get("error"),
                "result_message": result.get("message"),
            },
        )
        # Build debug information if requested
        debug_info = None
        finance_model = "mukaj/fin-mpnet-base"
        if debug_mode:
            debug_info = _build_debug_info(user_query, orchestrator, finance_model)
        
        # Always use finance model
        final_embedding_model = finance_model
        
        # Prefer explicit error/message from orchestrator for the user
        fallback_error = result.get("error") or result.get("message") or "Unknown error"
        fallback_answer = f"Failed to process query: {fallback_error}" if fallback_error != "Unknown error" else "Failed to process query"
        
        response_data = {
            "success": False,
            "answer": fallback_answer,
            "sources": [],
            "query_time_ms": 0,
            "error": fallback_error,
            "embedding_model": final_embedding_model,
            "embedding_dims": embedding_dims
        }
        
        if debug_info:
            response_data["debug"] = debug_info
        
        return jsonify(response_data)
    
    except Exception as e:
        error_msg = str(e)
        logger.error("Error processing query", extra={"error": error_msg}, exc_info=True)
        debug_info = None
        if debug_mode:
            try:
                debug_info = _build_debug_info(user_query, orchestrator, finance_model)
            except:
                pass
        
        # Ensure we return valid JSON even on error
        # Always use finance model
        final_embedding_model = finance_model
        
        response_data = {
            "success": False,
            "answer": f"Error: {error_msg}",
            "sources": [],
            "query_time_ms": 0,
            "error": error_msg,
            "embedding_model": final_embedding_model,
            "embedding_dims": embedding_dims
        }
        
        if debug_info:
            response_data["debug"] = debug_info
        
        return jsonify(response_data), 500


@app.route('/api/agents', methods=['GET'])
def list_agents():
    """List available agents"""
    if not orchestrator:
        return jsonify({"agents": []})
    
    agents = orchestrator.registry.list_agents()
    return jsonify({
        "agents": [
            {
                "name": agent.name,
                "description": agent.description,
                "capabilities": [c.name for c in agent.capabilities],
                "status": agent.status.value
            }
            for agent in agents
        ]
    })


@app.route('/api/query/<query_id>', methods=['GET'])
def get_query_result(query_id: str):
    """Get result for a query"""
    if not orchestrator:
        return jsonify({"error": "Agent system not available"}), 503
    
    result = orchestrator.get_query_result(query_id)
    if result:
        return jsonify(result)
    else:
        return jsonify({"error": "Query not found"}), 404


@app.route('/api/semantic/analyze', methods=['POST'])
def analyze_semantics():
    """
    Semantic analysis via modelmgmt-agent (all 3 dropdown models).
    
    Request body:
    {
        "query": "user query text",
        "models": ["model1", "model2"],  # Optional: specific models to use
        "parallel": true  # Optional: process in parallel (default: true)
    }
    
    Returns:
    {
        "query": "original query",
        "models_analyzed": 5,
        "models": {
            "model_name": {
                "tags": [...],
                "detected_entities": [...],
                "detected_actions": [...],
                "embedding_stats": {...}
            }
        },
        "aggregated": {
            "consensus_tags": [...],
            "tag_counts": {...},
            ...
        },
        "summary": {...}
    }
    """
    if not MULTI_MODEL_AVAILABLE:
        return jsonify({
            "error": "modelmgmt-agent not available",
            "message": "Install sentence-transformers: pip install sentence-transformers"
        }), 503
    
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request"}), 400
    
    query = data["query"]
    models = data.get("models")  # Optional: specific models
    parallel = data.get("parallel", True)
    model_list = models if models else DROPDOWN_MODEL_IDS
    logger.info(
        "modelmgmt-agent: running semantic analysis (POST /api/semantic/analyze). models=%s",
        model_list,
    )
    try:
        result = analyze_query_multi_model(query, models=models, parallel=parallel)
        models_analyzed = result.get("models_analyzed", 0) or len(result.get("models") or {})
        logger.info(
            "modelmgmt-agent: semantic analysis completed (POST /api/semantic/analyze)",
            extra={"models_analyzed": models_analyzed},
        )
        return jsonify({
            "success": True,
            **result
        })
    except Exception as e:
        error_msg = str(e)
        logger.error("modelmgmt-agent: semantic analysis failed", extra={"error": error_msg, "query_preview": (query or "")[:100]}, exc_info=True)
        return jsonify({
            "success": False,
            "error": error_msg,
            "query": query
        }), 500


@app.route('/api/semantic/models', methods=['GET'])
def list_semantic_models():
    """List available models (from modelmgmt-agent; the 3 dropdown models)."""
    if not MULTI_MODEL_AVAILABLE:
        return jsonify({
            "error": "modelmgmt-agent not available",
            "models": []
        }), 503

    try:
        logger.info("modelmgmt-agent: listing loaded models (GET /api/semantic/models)")
        analyzer = get_analyzer()
        model_info = analyzer.get_model_info()
        logger.info("modelmgmt-agent: listed %s model(s)", len(model_info))
        return jsonify({
            "success": True,
            "models": model_info,
            "count": len(model_info)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "models": []
        }), 500


@app.route('/api/semantic/models-with-meta', methods=['GET'])
def list_semantic_models_with_meta():
    """List supported models with metadata (from modelmgmt-agent). Use these ids as the 'model' parameter in interpret/preview APIs."""
    if not MULTI_MODEL_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "modelmgmt-agent not available",
            "models": [],
            "count": 0,
        }), 503
    return jsonify({
        "success": True,
        "models": MODEL_METADATA,
        "count": len(MODEL_METADATA),
    })


def _run_semantic_interpret(query: str, model_name: str) -> tuple:
    """
    Run semantic interpretation for a sentence with the given model.
    Returns (response_dict, http_status_code).
    """
    if not MULTI_MODEL_AVAILABLE:
        return {
            "success": False,
            "error": "modelmgmt-agent not available"
        }, 503

    default_model = "mukaj/fin-mpnet-base"
    if not model_name or not model_name.strip():
        model_name = default_model
    if not query or not query.strip():
        return {
            "success": True,
            "query": query or "",
            "model": model_name,
            "tags": [],
            "highlighted_segments": [],
            "token_semantics": [],
            "embedding_stats": {}
        }, 200

    try:
        if model_name not in DROPDOWN_MODEL_IDS:
            return {
                "success": False,
                "error": f"Model {model_name!r} not supported. Use one of: {', '.join(DROPDOWN_MODEL_IDS)}",
            }, 400
        logger.info(
            "modelmgmt-agent: loading embedding models (preload=True) for semantic interpret. models=%s",
            DROPDOWN_MODEL_IDS,
        )
        analyzer = get_analyzer(models=DROPDOWN_MODEL_IDS, preload=True)
        if not analyzer:
            return {"success": False, "error": "modelmgmt-agent not available"}, 503
        if model_name not in analyzer.loaded_models:
            return {
                "success": False,
                "error": f"Model {model_name!r} not loaded by modelmgmt-agent. Check backend logs.",
            }, 503

        model = analyzer.loaded_models[model_name]
        t0 = time.perf_counter()
        try:
            embedding = model.encode(query, convert_to_numpy=True)
        except Exception as e:
            logger.warning("Encode failed in semantic preview", extra={"model": model_name, "error": str(e)})
            return {"success": False, "error": f"Encoding failed: {str(e)}"}, 503
        try:
            semantic_analysis = analyzer._extract_semantic_tags(query, model_name, embedding)
            highlighted = analyzer._highlight_query_with_tags(query, semantic_analysis.get("tags", []))
        except Exception as e:
            logger.warning("Semantic extraction failed", extra={"model": model_name, "error": str(e)})
            return {"success": False, "error": f"Analysis failed: {str(e)}"}, 500

        semantic_annotation_time_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(
            "modelmgmt-agent: semantic interpret completed for model %s",
            model_name,
            extra={"time_ms": semantic_annotation_time_ms},
        )
        token_semantics = []
        query_words = re.findall(r'\b\w+\b', query.lower())
        for word in query_words:
            meanings = []
            for tag in semantic_analysis.get("tags", []):
                snippet_lower = tag.get("snippet", "").lower()
                if word in snippet_lower or snippet_lower in word:
                    meanings.append(f"{tag.get('label')}:{tag.get('type')}")
            if meanings:
                token_semantics.append({"token": word, "meanings": meanings})

        return {
            "success": True,
            "query": query,
            "model": model_name,
            "tags": semantic_analysis.get("tags", []),
            "highlighted_segments": highlighted.get("highlighted_segments", []),
            "token_semantics": token_semantics,
            "embedding_stats": semantic_analysis.get("embedding_stats", {}),
            "semantic_annotation_time_ms": semantic_annotation_time_ms,
        }, 200
    except Exception as e:
        error_msg = str(e)
        logger.error("Error in semantic interpret", extra={"error": error_msg, "query_preview": (query or "")[:100], "model": model_name}, exc_info=True)
        return {"success": False, "error": error_msg, "query": query, "model": model_name}, 500


@app.route('/api/semantic/interpret', methods=['GET'])
def semantic_interpret_get():
    """
    Semantic interpretation of a sentence by the given model (query params).
    GET /api/semantic/interpret?sentence=...&model=...
    Easy to use from browser, curl, and server-to-server.
    """
    sentence = request.args.get("sentence", "").strip()
    model = request.args.get("model", "mukaj/fin-mpnet-base").strip()
    if not sentence:
        return jsonify({"success": False, "error": "Missing query parameter 'sentence'"}), 400
    data, status = _run_semantic_interpret(sentence, model or "mukaj/fin-mpnet-base")
    return jsonify(data), status


def _run_interpret_all(query: str) -> tuple:
    """Run semantic interpretation for a sentence with all supported models (the 3 dropdown models). Returns (response_dict, status_code)."""
    if not MULTI_MODEL_AVAILABLE:
        return {"success": False, "error": "modelmgmt-agent not available", "query": query, "models": {}}, 503
    if not query or not query.strip():
        return {"success": True, "query": query or "", "models": {}}, 200
    model_ids = DROPDOWN_MODEL_IDS
    models_out = {}
    for model_id in model_ids:
        data, _ = _run_semantic_interpret(query, model_id)
        models_out[model_id] = data
    return {"success": True, "query": query.strip(), "models": models_out}, 200


@app.route('/api/semantic/interpret-all', methods=['GET', 'POST'])
def semantic_interpret_all():
    """
    Semantic interpretation of a sentence by all supported models (same as UI dropdown).
    GET: ?sentence=...
    POST: body { "query": "sentence text" }
    Returns { success, query, models: { model_id: { success, query, model, tags, highlighted_segments, token_semantics, ... } } }.
    """
    if request.method == 'GET':
        sentence = request.args.get("sentence", "").strip()
    else:
        data = request.get_json() or {}
        sentence = str(data.get("query", "")).strip()
    if not sentence:
        err = "Missing query parameter 'sentence'" if request.method == 'GET' else "Missing 'query' in request"
        return jsonify({"success": False, "error": err, "models": {}}), 400
    result, status = _run_interpret_all(sentence)
    return jsonify(result), status


@app.route('/api/semantic/preview', methods=['POST'])
def preview_semantic_analysis():
    """
    Real-time semantic analysis preview for input text.
    Returns semantic tags and highlighted segments for the selected model.
    Request body: { "query": "sentence text", "model": "mukaj/fin-mpnet-base" }
    """
    if not MULTI_MODEL_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "modelmgmt-agent not available"
        }), 503

    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request"}), 400

    query = data["query"]
    model_name = data.get("model", "mukaj/fin-mpnet-base")
    result, status = _run_semantic_interpret(query, model_name)
    return jsonify(result), status


@app.route('/api/semantic/preview-multi', methods=['POST'])
def preview_semantic_analysis_multi():
    """
    Run semantic analysis with all configured models and return tags + highlighted
    segments for each. Used when debug is on to show multi-model semantic preview.
    Request body: { "query": "sentence text" }
    """
    if not MULTI_MODEL_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "modelmgmt-agent not available"
        }), 503

    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request"}), 400

    query = data["query"]
    if not query or not query.strip():
        return jsonify({
            "success": True,
            "query": query,
            "models": {}
        })

    try:
        logger.info(
            "modelmgmt-agent: running semantic analysis (POST /api/semantic/preview-multi). models=%s",
            DROPDOWN_MODEL_IDS,
        )
        analyzer = get_analyzer()
        if not analyzer:
            return jsonify({
                "success": False,
                "error": "modelmgmt-agent not available. Install sentence-transformers and ensure at least one embedding model can load (e.g. mukaj/fin-mpnet-base)."
            }), 503

        t0 = time.perf_counter()
        try:
            result = analyzer.analyze_query(query, parallel=True)
        except Exception as e:
            err = str(e)
            logger.warning("modelmgmt-agent: analyze_query failed", extra={"error": err, "query_preview": (query or "")[:100]})
            return jsonify({
                "success": False,
                "error": err or "Semantic analysis failed.",
                "query": query,
                "models": {}
            }), 503
        semantic_annotation_time_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(
            "modelmgmt-agent: semantic analysis completed (preview-multi)",
            extra={"models": len(result.get("models") or {}), "time_ms": semantic_annotation_time_ms},
        )

        if "error" in result:
            return jsonify({
                "success": False,
                "error": result["error"],
                "query": query,
                "models": {}
            }), 400

        result_models = result.get("models") or {}
        if not isinstance(result_models, dict):
            result_models = {}
        models_out = {}
        for model_name, model_result in result_models.items():
            if not isinstance(model_result, dict):
                model_result = {}
            tags = model_result.get("tags") if isinstance(model_result.get("tags"), list) else []
            highlighted = model_result.get("highlighted_query") or {}
            if not isinstance(highlighted, dict):
                highlighted = {}
            segments = highlighted.get("highlighted_segments") if isinstance(highlighted.get("highlighted_segments"), list) else []
            models_out[model_name] = {
                "tags": tags,
                "highlighted_segments": segments,
                "error": model_result.get("error"),
                "semantic_annotation_time_ms": model_result.get("semantic_annotation_time_ms"),
            }
        schema_assets = result.get("schema_assets") or []
        if not isinstance(schema_assets, list):
            schema_assets = []

        return jsonify({
            "success": True,
            "query": query,
            "models": models_out,
            "best_model": result.get("best_model"),
            "semantic_annotation_time_ms": semantic_annotation_time_ms,
            "schema_assets": schema_assets,
        })
    except Exception as e:
        error_msg = str(e)
        logger.error("modelmgmt-agent: multi-model semantic preview failed", extra={"error": error_msg, "query_preview": (query or "")[:100]}, exc_info=True)
        return jsonify({
            "success": False,
            "error": error_msg,
            "query": query,
            "models": {}
        }), 500


# --- A2A (Agent-to-Agent) protocol ---
try:
    from a2a import (
        get_agent_card as a2a_get_agent_card,
        handle_json_rpc as a2a_handle_json_rpc,
    )
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False


@app.route('/api/a2a/agents/<agent_id>/card', methods=['GET'])
def a2a_agent_card(agent_id):
    """Return A2A Agent Card for discovery (Google A2A spec)."""
    if not A2A_AVAILABLE:
        return jsonify({"error": "A2A not available"}), 503
    card = a2a_get_agent_card(agent_id)
    if card is None:
        return jsonify({"error": f"Unknown agent: {agent_id}"}), 404
    return jsonify(card)


@app.route('/api/a2a/agents/<agent_id>', methods=['POST'])
def a2a_agent_json_rpc(agent_id):
    """A2A message/send via JSON-RPC 2.0 (Google A2A spec). All agents use A2A."""
    if not A2A_AVAILABLE:
        return jsonify({"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": "A2A not available"}}), 503
    body = request.get_data()
    resp, status = a2a_handle_json_rpc(body, agent_id)
    return jsonify(resp), status


@app.route('/api/a2a/run', methods=['POST'])
def a2a_run():
    """
    Run A2A flow: sg-agent generates sentences → modelmgmt-agent annotates each.
    Body: { "sentence_count": 3 } (optional, default 3).
    Returns: { "success", "steps", "sentences", "annotations", "error" }.
    """
    if not A2A_AVAILABLE:
        return jsonify({"success": False, "error": "A2A not available"}), 503
    try:
        data = request.get_json() or {}
        sentence_count = min(5, max(1, int(data.get("sentence_count", 3))))
    except (TypeError, ValueError):
        sentence_count = 3
    steps = []
    sentences = []
    annotations = []
    try:
        # Ensure agents path
        agent_system_path = os.path.abspath(os.path.join(current_dir, '..', 'openint-agents'))
        if agent_system_path not in sys.path:
            sys.path.insert(0, agent_system_path)
        datahub_path = os.path.abspath(os.path.join(_repo_root, 'openint-datahub'))
        if os.path.isdir(datahub_path) and datahub_path not in sys.path:
            sys.path.insert(0, datahub_path)
        from a2a import handle_sg_agent_message_send, handle_modelmgmt_agent_message_send
        # Step 1: sg-agent — generate sentences via A2A message/send
        steps.append({"agent": "sg-agent", "action": "generate", "status": "running"})
        t0_sg = time.perf_counter()
        msg = {"message": {"role": "user", "parts": [{"kind": "text", "text": f"Generate {sentence_count} sentences"}], "messageId": "a2a-run-1", "kind": "message"}}
        task_sg = handle_sg_agent_message_send(msg)
        sg_agent_time_ms = round((time.perf_counter() - t0_sg) * 1000)
        if task_sg.get("status", {}).get("state") == "failed":
            steps[-1]["status"] = "failed"
            err_detail = "sg-agent failed to generate sentences"
            msg = task_sg.get("status", {}).get("message") or {}
            for part in msg.get("parts") or []:
                if part.get("kind") == "text" and part.get("text"):
                    err_detail = part["text"].strip()
                    break
            return jsonify({
                "success": False,
                "steps": steps,
                "sentences": [],
                "annotations": [],
                "sg_agent_time_ms": sg_agent_time_ms,
                "error": err_detail,
            }), 503
        steps[-1]["status"] = "completed"
        # Extract sentences from task artifacts
        for art in task_sg.get("artifacts") or []:
            for p in art.get("parts") or []:
                if p.get("kind") == "text" and p.get("text"):
                    sentences.append({"text": p["text"], "category": p.get("metadata", {}).get("category", "Analyst")})
        if not sentences:
            return jsonify({"success": False, "steps": steps, "sentences": [], "annotations": [], "sg_agent_time_ms": sg_agent_time_ms, "error": "No sentences from sg-agent"}), 503
        # Step 2: modelmgmt-agent — annotate each sentence via A2A message/send
        steps.append({"agent": "modelmgmt-agent", "action": "annotate", "status": "running", "count": len(sentences)})
        t0_mm = time.perf_counter()
        for i, sent in enumerate(sentences):
            msg = {"message": {"role": "user", "parts": [{"kind": "text", "text": sent["text"]}], "messageId": f"a2a-annot-{i}", "kind": "message"}}
            task_mm = handle_modelmgmt_agent_message_send(msg)
            ann = None
            if task_mm.get("status", {}).get("state") == "completed":
                for art in task_mm.get("artifacts") or []:
                    for p in art.get("parts") or []:
                        if p.get("kind") == "data":
                            ann = p.get("data")
                            break
            annotations.append({"sentence": sent["text"], "annotation": ann, "success": ann is not None})
        modelmgmt_agent_time_ms = round((time.perf_counter() - t0_mm) * 1000)
        steps[-1]["status"] = "completed"
        return jsonify({
            "success": True,
            "steps": steps,
            "sentences": sentences,
            "annotations": annotations,
            "sg_agent_time_ms": sg_agent_time_ms,
            "modelmgmt_agent_time_ms": modelmgmt_agent_time_ms,
        })
    except Exception as e:
        logger.exception("A2A run failed")
        if steps:
            steps[-1]["status"] = "failed"
        return jsonify({
            "success": False,
            "steps": steps,
            "sentences": sentences,
            "annotations": annotations,
            "sg_agent_time_ms": None,
            "modelmgmt_agent_time_ms": None,
            "error": str(e),
        }), 500


# --- Neo4j Graph Demo (schema from DataHub + openint-testdata loader) ---
NEO4J_CLIENT = None
try:
    from neo4j_client import get_neo4j_client
    NEO4J_CLIENT = get_neo4j_client()
except Exception as e:
    get_logger(__name__).info("Neo4j graph demo: client not available (%s)", e)

# Graph schema: nodes Customer, Transaction, Dispute; relationships HAS_TRANSACTION, OPENED_DISPUTE, REFERENCES (from DataHub + load_openint_data_to_neo4j)
GRAPH_SCHEMA = {
    "nodes": [
        {"label": "Customer", "id_property": "id", "description": "Customer dimension (from DataHub customers schema)"},
        {"label": "Transaction", "id_property": "id", "type_property": "type", "description": "Transaction facts: ach, wire, credit, debit, check (from DataHub fact tables)"},
        {"label": "Dispute", "id_property": "id", "description": "Dispute fact (from DataHub disputes schema)"},
    ],
    "relationships": [
        {"type": "HAS_TRANSACTION", "from": "Customer", "to": "Transaction", "description": "Customer has transactions"},
        {"type": "OPENED_DISPUTE", "from": "Customer", "to": "Dispute", "description": "Customer opened a dispute"},
        {"type": "REFERENCES", "from": "Dispute", "to": "Transaction", "description": "Dispute references a transaction"},
    ],
    "source": "DataHub schema (openint-datahub/schemas.py) + openint-testdata/loaders/load_openint_data_to_neo4j.py",
}


def _build_graph_schema_summary(schema: Dict[str, Any]) -> str:
    """Build a short text summary of the Neo4j schema for the LLM (nodes + relationships)."""
    parts = ["Neo4j schema (node labels and relationship types):"]
    for node in schema.get("nodes", []):
        label = node.get("label", "")
        desc = node.get("description", "")
        id_prop = node.get("id_property", "id")
        type_prop = node.get("type_property", "")
        line = f"- Node label: {label}. id property: {id_prop}."
        if type_prop:
            line += f" type property: {type_prop} (e.g. ach, wire, credit, debit, check for Transaction)."
        line += f" {desc}"
        parts.append(line)
    parts.append("Relationship types (from -> to):")
    for rel in schema.get("relationships", []):
        parts.append(f"- {rel.get('type', '')}: ({rel.get('from', '')})-[:{rel.get('type', '')}]->({rel.get('to', '')}). {rel.get('description', '')}")
    return "\n".join(parts)


def _extract_cypher_from_llm_response(text: str) -> str:
    """Extract Cypher from LLM response; strip markdown code blocks if present."""
    text = (text or "").strip()
    if not text:
        return ""
    # If wrapped in ```cypher ... ``` or ``` ... ```, take content between first and second ```
    if "```" in text:
        parts = text.split("```", 2)
        if len(parts) >= 2:
            block = parts[1].strip()
            # Remove optional language tag on first line (e.g. "cypher" or "neo4j")
            if "\n" in block:
                first, rest = block.split("\n", 1)
                if first.strip().lower() in ("cypher", "neo4j"):
                    block = rest.strip()
            if block and "MATCH" in block.upper():
                return block
    return text.strip()


def _generate_cypher_with_ollama(schema_summary: str, question: str) -> tuple[Optional[str], Optional[str]]:
    """Use Ollama to generate a Cypher query from natural language. Returns (cypher, error)."""
    import urllib.request
    import urllib.error
    import ssl
    host = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL") or "llama3.2"
    prompt = f"""You are a Neo4j Cypher expert. Given the schema below and the user question, return ONLY a valid Cypher query. No explanation, no markdown, no code block wrapper.

Schema:
{schema_summary}

Rules:
- In MATCH, use exactly these variable names: c for Customer, t for Transaction, d for Dispute. Example: MATCH (c:Customer)-[:OPENED_DISPUTE]->(d:Dispute)-[:REFERENCES]->(t:Transaction)
- In RETURN, only use the node variables c, d, t and their properties: c.id, t.id, d.id, t.amount, t.type, d.amount_disputed, d.dispute_status, t.currency, etc. Never use a relationship variable in RETURN (e.g. no r.target; use t.id for transaction_id).
- Relationship types: HAS_TRANSACTION (Customer->Transaction), OPENED_DISPUTE (Customer->Dispute), REFERENCES (Dispute->Transaction).
- Every variable in RETURN must appear in MATCH. Add LIMIT 50 for exploration queries.
- Return ONLY the Cypher statement, nothing else.

User question: {question}"""
    try:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 1024},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        message = data.get("message") or {}
        text = (message.get("content") or "").strip()
        cypher = _extract_cypher_from_llm_response(text)
        if not cypher:
            return (None, "Ollama returned empty or invalid Cypher.")
        return (cypher, None)
    except urllib.error.URLError as e:
        msg = str(e.reason) if getattr(e, "reason", None) else str(e)
        if "Connection refused" in msg or "111" in msg:
            msg = "Ollama is not running. Start Ollama (ollama serve) and pull a model: ollama pull " + model
        return (None, msg)
    except Exception as e:
        return (None, str(e))


def _get_neo4j():
    """Return Neo4j client (may be None)."""
    global NEO4J_CLIENT
    if NEO4J_CLIENT is not None:
        return NEO4J_CLIENT
    try:
        from neo4j_client import get_neo4j_client
        NEO4J_CLIENT = get_neo4j_client()
    except Exception:
        pass
    return NEO4J_CLIENT


# Graph API as Blueprint so routes take precedence over SPA catch-all (avoid 404 for /api/graph/*)
graph_bp = Blueprint("graph_api", __name__, url_prefix="/api")


@graph_bp.route("/graph/stats", methods=["GET"])
def graph_stats():
    """
    Neo4j graph demo: connectivity and node/relationship counts.
    Schema: Customer, Transaction, Dispute; HAS_TRANSACTION, OPENED_DISPUTE, REFERENCES.
    Returns 200 with success: false when Neo4j is unavailable so the UI can show schema + message.
    """
    client = _get_neo4j()
    if not client:
        return jsonify({
            "success": False,
            "connected": False,
            "error": "Neo4j client not available. Install neo4j driver and set NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD.",
            "node_counts": {},
            "relationship_counts": {},
        }), 200
    try:
        if not client.verify_connectivity():
            return jsonify({
                "success": False,
                "connected": False,
                "error": "Cannot connect to Neo4j. Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.",
                "node_counts": {},
                "relationship_counts": {},
            }), 200
        # Node counts
        counts = {}
        for label in ("Customer", "Transaction", "Dispute"):
            rows = client.run(f"MATCH (n:{label}) RETURN count(n) AS c")
            counts[label] = (rows[0].get("c", 0) or 0) if rows else 0
        # Relationship counts
        rel_counts = {}
        for rel in ("HAS_TRANSACTION", "OPENED_DISPUTE", "REFERENCES"):
            rows = client.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c")
            rel_counts[rel] = (rows[0].get("c", 0) or 0) if rows else 0
        return jsonify({
            "success": True,
            "connected": True,
            "node_counts": counts,
            "relationship_counts": rel_counts,
        })
    except Exception as e:
        logger.warning("Neo4j graph stats failed", extra={"error": str(e)})
        return jsonify({
            "success": False,
            "connected": False,
            "error": str(e),
            "node_counts": {},
            "relationship_counts": {},
        }), 200


@graph_bp.route("/graph/schema", methods=["GET"])
def graph_schema():
    """Neo4j graph demo: schema summary from DataHub + loader (nodes and relationship types)."""
    return jsonify({
        "success": True,
        "schema": GRAPH_SCHEMA,
    })


@graph_bp.route("/graph/sample", methods=["GET"])
def graph_sample():
    """
    Neo4j graph demo: sample data — disputes overview and customer–transaction–dispute paths.
    Uses same Cypher patterns as graph_agent and loader schema.
    Returns 200 with success: false and empty arrays when Neo4j is unavailable.
    """
    client = _get_neo4j()
    if not client:
        return jsonify({
            "success": False,
            "error": "Neo4j client not available",
            "disputes_overview": [],
            "paths": [],
        }), 200
    try:
        if not client.verify_connectivity():
            return jsonify({
                "success": False,
                "error": "Cannot connect to Neo4j",
                "disputes_overview": [],
                "paths": [],
            }), 200
        disputes_overview = client.run("""
            MATCH (c:Customer)-[:OPENED_DISPUTE]->(d:Dispute)-[:REFERENCES]->(t:Transaction)
            RETURN c.id AS customer_id, d.id AS dispute_id, t.id AS transaction_id,
                   d.dispute_status AS status, d.amount_disputed AS amount_disputed, d.currency AS currency
            LIMIT 25
        """)
        paths = client.run("""
            MATCH (c:Customer)-[:HAS_TRANSACTION]->(t:Transaction)<-[:REFERENCES]-(d:Dispute)
            RETURN c.id AS customer_id, t.id AS transaction_id, d.id AS dispute_id,
                   t.amount AS tx_amount, d.amount_disputed, d.dispute_status
            LIMIT 20
        """)
        return jsonify({
            "success": True,
            "disputes_overview": disputes_overview or [],
            "paths": paths or [],
        })
    except Exception as e:
        logger.warning("Neo4j graph sample failed", extra={"error": str(e)})
        return jsonify({
            "success": False,
            "error": str(e),
            "disputes_overview": [],
            "paths": [],
        }), 200


# Predefined graph queries for the demo (left-pane suggestions). Each returns list of dicts.
GRAPH_QUERIES = {
    "disputes_overview": {
        "label": "Customer → Dispute → Transaction",
        "cypher": """
            MATCH (c:Customer)-[:OPENED_DISPUTE]->(d:Dispute)-[:REFERENCES]->(t:Transaction)
            RETURN c.id AS customer_id, d.id AS dispute_id, t.id AS transaction_id,
                   d.dispute_status AS status, d.amount_disputed AS amount_disputed, d.currency AS currency
            LIMIT 50
        """,
    },
    "paths": {
        "label": "Customer → Transaction ← Dispute (paths)",
        "cypher": """
            MATCH (c:Customer)-[:HAS_TRANSACTION]->(t:Transaction)<-[:REFERENCES]-(d:Dispute)
            RETURN c.id AS customer_id, t.id AS transaction_id, d.id AS dispute_id,
                   t.amount AS tx_amount, d.amount_disputed, d.dispute_status
            LIMIT 50
        """,
    },
    "wire_over_10k": {
        "label": "Wire transfers over $10,000",
        "cypher": """
            MATCH (c:Customer)-[:HAS_TRANSACTION]->(t:Transaction)
            WHERE t.type = 'wire' AND t.amount > 10000
            RETURN c.id AS customer_id, t.id AS transaction_id, t.amount, t.currency
            ORDER BY t.amount DESC
            LIMIT 50
        """,
    },
    "credit_disputes": {
        "label": "Credit card disputes",
        "cypher": """
            MATCH (c:Customer)-[:OPENED_DISPUTE]->(d:Dispute)-[:REFERENCES]->(t:Transaction)
            WHERE t.type = 'credit'
            RETURN c.id AS customer_id, d.id AS dispute_id, t.id AS transaction_id,
                   d.dispute_status, d.amount_disputed, d.currency
            LIMIT 50
        """,
    },
    "ach_disputes": {
        "label": "ACH disputes",
        "cypher": """
            MATCH (c:Customer)-[:OPENED_DISPUTE]->(d:Dispute)-[:REFERENCES]->(t:Transaction)
            WHERE t.type = 'ach'
            RETURN c.id AS customer_id, d.id AS dispute_id, t.id AS transaction_id,
                   d.dispute_status, d.amount_disputed, d.currency
            LIMIT 50
        """,
    },
    "wire_disputes": {
        "label": "Wire disputes",
        "cypher": """
            MATCH (c:Customer)-[:OPENED_DISPUTE]->(d:Dispute)-[:REFERENCES]->(t:Transaction)
            WHERE t.type = 'wire'
            RETURN c.id AS customer_id, d.id AS dispute_id, t.id AS transaction_id,
                   d.dispute_status, d.amount_disputed, d.currency
            LIMIT 50
        """,
    },
    "open_disputes": {
        "label": "Open disputes",
        "cypher": """
            MATCH (c:Customer)-[:OPENED_DISPUTE]->(d:Dispute)-[:REFERENCES]->(t:Transaction)
            WHERE d.dispute_status = 'Open'
            RETURN c.id AS customer_id, d.id AS dispute_id, t.id AS transaction_id,
                   d.dispute_status, d.amount_disputed, d.currency
            LIMIT 50
        """,
    },
    "top_customers_by_tx": {
        "label": "Top customers by transaction count",
        "cypher": """
            MATCH (c:Customer)-[:HAS_TRANSACTION]->(t:Transaction)
            RETURN c.id AS customer_id, count(t) AS transaction_count
            ORDER BY transaction_count DESC
            LIMIT 25
        """,
    },
    "international_wires": {
        "label": "International wire transfers",
        "cypher": """
            MATCH (c:Customer)-[:HAS_TRANSACTION]->(t:Transaction)
            WHERE t.type = 'wire' AND t.currency <> 'USD'
            RETURN c.id AS customer_id, t.id AS transaction_id, t.amount, t.currency
            LIMIT 50
        """,
    },
}


@graph_bp.route("/graph/query", methods=["GET", "POST"])
def graph_query():
    """
    Run a predefined graph query by query_id.
    GET: ?query_id=disputes_overview
    POST: body { "query_id": "disputes_overview" | ... }
    Returns { success, query_id, label, cypher, columns, rows, error? }.
    """
    client = _get_neo4j()
    if not client:
        return jsonify({
            "success": False,
            "query_id": None,
            "label": None,
            "cypher": None,
            "columns": [],
            "rows": [],
            "error": "Neo4j client not available",
        }), 200
    if request.method == "GET":
        query_id = (request.args.get("query_id") or "").strip()
    else:
        data = request.get_json() or {}
        query_id = (data.get("query_id") or "").strip()
    if not query_id or query_id not in GRAPH_QUERIES:
        return jsonify({
            "success": False,
            "query_id": query_id,
            "label": None,
            "cypher": None,
            "columns": [],
            "rows": [],
            "error": "Unknown query_id. Use one of: " + ", ".join(sorted(GRAPH_QUERIES.keys())),
        }), 200
    spec = GRAPH_QUERIES[query_id]
    cypher = spec["cypher"].strip()
    try:
        if not client.verify_connectivity():
            return jsonify({
                "success": False,
                "query_id": query_id,
                "label": spec["label"],
                "cypher": cypher,
                "columns": [],
                "rows": [],
                "error": "Cannot connect to Neo4j",
            }), 200
        rows = client.run(cypher) or []
        columns = list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
        return jsonify({
            "success": True,
            "query_id": query_id,
            "label": spec["label"],
            "cypher": cypher,
            "columns": columns,
            "rows": rows,
        })
    except Exception as e:
        logger.warning("Neo4j graph query failed", extra={"query_id": query_id, "error": str(e)})
        return jsonify({
            "success": False,
            "query_id": query_id,
            "label": spec["label"],
            "cypher": cypher,
            "columns": [],
            "rows": [],
            "error": str(e),
        }), 200


@graph_bp.route("/graph/query-natural", methods=["GET", "POST"])
def graph_query_natural():
    """
    Run a graph query from natural language using the Neo4j schema and an LLM (Ollama).
    POST: body { "query": "natural language question" }.
    GET: ?query=natural+language+question (for proxies/tools that only allow GET).
    Returns { success, query, cypher, columns, rows, error?, llm_model? }.
    """
    client = _get_neo4j()
    if request.method == "POST":
        data = request.get_json() or {}
        question = (data.get("query") or "").strip()
    else:
        question = (request.args.get("query") or "").strip()
    if not question:
        return jsonify({
            "success": False,
            "query": None,
            "cypher": None,
            "columns": [],
            "rows": [],
            "error": "Missing 'query'. Use POST body {\"query\": \"...\"} or GET ?query=...",
        }), 400
    if not client:
        return jsonify({
            "success": False,
            "query": question,
            "cypher": None,
            "columns": [],
            "rows": [],
            "error": "Neo4j client not available. Install neo4j driver and set NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD.",
        }), 200
    schema_summary = _build_graph_schema_summary(GRAPH_SCHEMA)
    logger.info("graph query-natural: generating Cypher via Ollama for question (length=%d)", len(question))
    t0 = time.perf_counter()
    cypher, llm_error = _generate_cypher_with_ollama(schema_summary, question)
    llm_ms = round((time.perf_counter() - t0) * 1000)
    if llm_error or not cypher:
        logger.warning("graph query-natural: Ollama Cypher generation failed", extra={"error": llm_error})
        return jsonify({
            "success": False,
            "query": question,
            "cypher": cypher or None,
            "columns": [],
            "rows": [],
            "error": llm_error or "Could not generate Cypher from question.",
        }), 200
    try:
        if not client.verify_connectivity():
            return jsonify({
                "success": False,
                "query": question,
                "cypher": cypher,
                "columns": [],
                "rows": [],
                "error": "Cannot connect to Neo4j.",
            }), 200
        rows = client.run(cypher) or []
        columns = list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
        payload = {
            "success": True,
            "query": question,
            "cypher": cypher,
            "columns": columns,
            "rows": rows,
        }
        if llm_ms:
            payload["llm_time_ms"] = llm_ms
        payload["llm_model"] = os.getenv("OLLAMA_MODEL", "llama3.2")
        logger.info("graph query-natural: ran Cypher successfully", extra={"rows": len(rows), "llm_ms": llm_ms})
        return jsonify(payload)
    except Exception as e:
        logger.warning("graph query-natural: Neo4j execution failed", extra={"error": str(e)})
        return jsonify({
            "success": False,
            "query": question,
            "cypher": cypher,
            "columns": [],
            "rows": [],
            "error": f"Neo4j error: {str(e)}",
        }), 200


app.register_blueprint(graph_bp)


@app.route('/api/suggestions/lucky', methods=['GET'])
def suggestions_lucky():
    """
    Return one random suggestion from sg-agent: bank business analytics, customer support,
    or regulatory. Uses DataHub schemas (or openint-datahub/schemas.py) and generates
    a sentence via sg-agent. For Compare "I'm feeling lucky!" button.
    """
    try:
        agent_system_path = os.path.abspath(os.path.join(current_dir, '..', 'openint-agents'))
        if agent_system_path not in sys.path:
            sys.path.insert(0, agent_system_path)
        # Ensure openint-datahub is on path so get_schema() can load schemas.py
        datahub_path = os.path.abspath(os.path.join(_repo_root, 'openint-datahub'))
        if os.path.isdir(datahub_path) and datahub_path not in sys.path:
            sys.path.insert(0, datahub_path)
        from sg_agent.datahub_client import get_schema_and_source
        from sg_agent.sentence_generator import generate_one_lucky
        logger.info("sg-agent: fetching schema for lucky suggestion (DataHub assets and schema)")
        t0_sg = time.perf_counter()
        schema, schema_source = get_schema_and_source()
        if not schema:
            logger.warning("sg-agent: no schema available for lucky suggestion")
            return jsonify({
                "success": False,
                "error": "No schema available. Ensure DataHub is running or openint-datahub/schemas.py exists.",
            }), 503
        if schema_source == "datahub":
            logger.info("sg-agent: using DataHub assets and schema as context for LLM (Ollama)")
        else:
            logger.info("sg-agent: using openint-datahub schema as context for LLM (DataHub unavailable)")
        logger.info("sg-agent: generating sentence (Ollama or template fallback)")
        result = generate_one_lucky(schema, prefer_llm=True, schema_source=schema_source)
        sg_agent_time_ms = round((time.perf_counter() - t0_sg) * 1000)
        sentence = (result.get("sentence") or "").strip()
        err = result.get("error")
        if err or not sentence:
            logger.warning("sg-agent: lucky suggestion failed", extra={"error": err or "no sentence"})
            hint = "Start Ollama (ollama serve) and pull a model, e.g. ollama pull llama3.2. Set OLLAMA_HOST if not localhost:11434."
            return jsonify({
                "success": False,
                "error": err or "Sentence generation failed. " + hint,
            }), 503
        source = result.get("source") or "ollama"
        payload = {
            "success": True,
            "sentence": sentence,
            "category": result.get("category", "Analyst"),
            "source": source,
            "sg_agent_time_ms": sg_agent_time_ms,
        }
        if source == "ollama":
            payload["llm_model"] = os.getenv("OLLAMA_MODEL", "llama3.2")
        logger.info("sg-agent: lucky suggestion ready", extra={"source": source, "category": result.get("category", "Analyst")})
        # sg-agent + modelmgmt-agent: optionally annotate the sentence (for Compare "I'm feeling lucky")
        if sentence and MULTI_MODEL_AVAILABLE and get_analyzer and request.args.get("annotate", "").lower() in ("1", "true", "yes"):
            try:
                logger.info("modelmgmt-agent: annotating lucky sentence (semantic analysis)")
                analysis = analyze_query_multi_model(sentence, parallel=True)
                if "error" not in analysis:
                    models_out = {}
                    for model_name, model_result in analysis.get("models", {}).items():
                        tags = model_result.get("tags", [])
                        highlighted = (model_result.get("highlighted_query") or {}).get("highlighted_segments", [])
                        models_out[model_name] = {"tags": tags, "highlighted_segments": highlighted, "semantic_annotation_time_ms": model_result.get("semantic_annotation_time_ms")}
                    payload["annotation"] = {
                        "success": True,
                        "query": sentence,
                        "models": models_out,
                        "best_model": analysis.get("best_model"),
                        "schema_assets": analysis.get("schema_assets", []),
                    }
                    logger.info("modelmgmt-agent: annotation completed for lucky sentence", extra={"models": len(models_out)})
            except Exception as ann_e:
                logger.warning("modelmgmt-agent: annotation for lucky sentence failed", extra={"error": str(ann_e)})
                payload["annotation"] = {"success": False, "error": str(ann_e)}
        return jsonify(payload)
    except ImportError as e:
        logger.warning("sg-agent: not available for lucky suggestion", extra={"error": str(e)})
        return jsonify({
            "success": False,
            "error": "Suggestions agent not available",
        }), 503
    except Exception as e:
        logger.warning("sg-agent: lucky suggestion failed", extra={"error": str(e)}, exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# --- Merged UI: serve built openint-ui/dist (SPA) when SERVE_UI or FLASK_ENV=production ---
@app.route("/")
def serve_index():
    if not _serve_ui:
        return jsonify({"message": "OpenInt API", "docs": "/api/health"}), 200
    return send_from_directory(_ui_dist, "index.html")


@app.route("/<path:path>")
def serve_spa(path):
    if not _serve_ui:
        return jsonify({"error": "Not found"}), 404
    if path.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    full = os.path.join(_ui_dist, path)
    if os.path.isfile(full):
        return send_from_directory(_ui_dist, path)
    return send_from_directory(_ui_dist, "index.html")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3001))
    logger.info(
        "OpenInt Backend starting",
        extra={"url": f"http://localhost:{port}", "cors_origins": CORS_ORIGINS},
    )
    # Redis for chat cache only; model loading/Redis is done by modelmgmt-agent
    get_redis()
    # Prevent Werkzeug from dumping full request/response JSON to the console
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    if orchestrator is not None:
        logger.info("Backend ready for chat requests (orchestrator initialized)")
    else:
        logger.warning("Backend listening but agent system not initialized – chat will return 503 until ready (typically 30–90s if agents load on first request)")
    flask_debug = (os.getenv("FLASK_DEBUG") or "").strip().lower() in ("1", "true", "yes")
    app.run(host='0.0.0.0', port=port, debug=flask_debug)
