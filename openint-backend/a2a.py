"""
A2A (Agent-to-Agent) protocol support — Google's A2A spec.
Exposes sg-agent and modelmgmt-agent as A2A servers (Agent Card + message/send).
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Base URL for A2A agents (backend origin); override via env if needed
def _base_url() -> str:
    import os
    return (os.environ.get("A2A_BASE_URL") or "http://localhost:3001").rstrip("/")


# --- A2A types (minimal subset of spec) ---

def _agent_card_sg_agent() -> Dict[str, Any]:
    return {
        "name": "Sentence Generation Agent (sg-agent)",
        "description": "Generates natural-language example sentences for banking, analytics, and regulatory use cases. Uses DataHub schema and an LLM (Ollama) to produce questions that real users would ask.",
        "url": f"{_base_url()}/api/a2a/agents/sg-agent",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["application/json", "text/plain"],
        "defaultOutputModes": ["application/json"],
        "skills": [
            {
                "id": "generate-sentences",
                "name": "Generate example sentences",
                "description": "Generate N example sentences (banking/data/analytics/regulatory) from DataHub schema. Send a message with text like 'Generate 3 sentences' or 'Generate 5 example questions'.",
                "tags": ["sentence", "generation", "schema", "banking", "analytics"],
                "examples": ["Generate 3 sentences", "Generate 5 example questions for analysts"],
            }
        ],
    }


def _agent_card_modelmgmt_agent() -> Dict[str, Any]:
    return {
        "name": "Model Management Agent (modelmgmt-agent)",
        "description": "Annotates sentences with semantic tags using multiple embedding models (SLMs). Returns tags, highlighted segments, and schema assets. Used to interpret natural language for the semantic layer.",
        "url": f"{_base_url()}/api/a2a/agents/modelmgmt-agent",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["application/json", "text/plain"],
        "defaultOutputModes": ["application/json"],
        "skills": [
            {
                "id": "semantic-annotate",
                "name": "Semantic annotation",
                "description": "Annotate a sentence with semantic tags and highlighted segments using one or all embedding models. Send the sentence as message text.",
                "tags": ["annotation", "semantic", "embedding", "tags", "highlight"],
                "examples": ["Show me transactions in California over $1000", "List wire transfers for the last quarter"],
            }
        ],
    }


def _text_from_message(message: Dict[str, Any]) -> str:
    """Extract plain text from A2A Message parts (TextPart or first part)."""
    parts = message.get("parts") or []
    for p in parts:
        if p.get("kind") == "text" and "text" in p:
            return (p.get("text") or "").strip()
        if p.get("kind") == "data" and "data" in p:
            d = p["data"]
            if isinstance(d, dict) and "sentence" in d:
                return str(d.get("sentence", "")).strip()
            if isinstance(d, str):
                return d.strip()
    return ""


def _make_task(
    task_id: str,
    context_id: str,
    state: str,
    message_parts: Optional[List[Dict[str, Any]]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    from datetime import datetime, timezone
    status = {
        "state": state,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if message_parts:
        status["message"] = {
            "kind": "message",
            "role": "agent",
            "messageId": str(uuid.uuid4()),
            "taskId": task_id,
            "contextId": context_id,
            "parts": message_parts,
        }
    return {
        "id": task_id,
        "contextId": context_id,
        "status": status,
        "artifacts": artifacts or [],
    }


def handle_sg_agent_message_send(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    A2A message/send for sg-agent. Params: { message: Message }.
    User message text like "Generate 3 sentences" → generate sentences, return Task with artifacts.
    """
    task_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())
    message = params.get("message") or {}
    text = _text_from_message(message)
    if not text:
        return _make_task(
            task_id, context_id, "failed",
            message_parts=[{"kind": "text", "text": "No text in message. Send e.g. 'Generate 3 sentences'."}],
        )
    # Parse "Generate N sentences" or default to 3
    count = 3
    lower = text.lower()
    if "generate" in lower:
        import re
        m = re.search(r"(\d+)\s*(?:sentence|question)s?", lower)
        if m:
            count = min(10, max(1, int(m.group(1))))
    try:
        from sg_agent.datahub_client import get_schema_and_source
        from sg_agent.sentence_generator import generate_sentences
        schema, schema_source = get_schema_and_source()
        if not schema:
            return _make_task(
                task_id, context_id, "failed",
                message_parts=[{"kind": "text", "text": "No schema available. Ensure DataHub or openint-datahub is configured."}],
            )
        sentences_list, error_msg = generate_sentences(schema, count=count, prefer_llm=True, fast_lucky=True, schema_source=schema_source)
        if error_msg and not sentences_list:
            return _make_task(
                task_id, context_id, "failed",
                message_parts=[{"kind": "text", "text": f"Generation failed: {error_msg}"}],
            )
        # Build artifacts: one artifact per sentence (or one artifact with data array)
        parts = []
        for i, item in enumerate(sentences_list[:count]):
            s = (item.get("sentence") or "").strip()
            if s:
                parts.append({"kind": "text", "text": s, "metadata": {"category": item.get("category", "Analyst"), "index": i}})
        if not parts:
            parts = [{"kind": "text", "text": "No sentences generated."}]
        artifacts = [{
            "artifactId": str(uuid.uuid4()),
            "name": "generated_sentences",
            "description": f"{len(parts)} example sentence(s) from sg-agent",
            "parts": parts,
        }]
        return _make_task(task_id, context_id, "completed", artifacts=artifacts)
    except Exception as e:
        logger.exception("sg-agent A2A message/send failed")
        return _make_task(
            task_id, context_id, "failed",
            message_parts=[{"kind": "text", "text": str(e)}],
        )


def handle_modelmgmt_agent_message_send(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    A2A message/send for modelmgmt-agent. Params: { message: Message }.
    User message text = sentence to annotate → return Task with annotation artifact.
    """
    task_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())
    message = params.get("message") or {}
    sentence = _text_from_message(message)
    if not sentence:
        return _make_task(
            task_id, context_id, "failed",
            message_parts=[{"kind": "text", "text": "No sentence in message. Send the sentence to annotate as message text."}],
        )
    try:
        from modelmgmt_agent.semantic_analyzer import analyze_query_multi_model
        analysis = analyze_query_multi_model(sentence, parallel=True)
        if analysis.get("error"):
            return _make_task(
                task_id, context_id, "failed",
                message_parts=[{"kind": "text", "text": analysis["error"]}],
            )
        artifacts = [{
            "artifactId": str(uuid.uuid4()),
            "name": "semantic_annotation",
            "description": "Multi-model semantic annotation",
            "parts": [{"kind": "data", "data": analysis}],
        }]
        return _make_task(task_id, context_id, "completed", artifacts=artifacts)
    except ImportError:
        return _make_task(
            task_id, context_id, "failed",
            message_parts=[{"kind": "text", "text": "modelmgmt-agent not available."}],
        )
    except Exception as e:
        logger.exception("modelmgmt-agent A2A message/send failed")
        return _make_task(
            task_id, context_id, "failed",
            message_parts=[{"kind": "text", "text": str(e)}],
        )


def handle_json_rpc(body: bytes, agent_id: str) -> tuple[Dict[str, Any], int]:
    """
    Dispatch JSON-RPC 2.0 request to the right agent. Returns (response_dict, status_code).
    """
    try:
        data = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as e:
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {e}"}}, 400
    method = data.get("method")
    params = data.get("params") or {}
    rpc_id = data.get("id")
    if method != "message/send":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": -32601, "message": f"Method not supported: {method}"},
        }, 200
    if agent_id == "sg-agent":
        result = handle_sg_agent_message_send(params)
    elif agent_id == "modelmgmt-agent":
        result = handle_modelmgmt_agent_message_send(params)
    else:
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": -32601, "message": f"Unknown agent: {agent_id}"},
        }, 200
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}, 200
