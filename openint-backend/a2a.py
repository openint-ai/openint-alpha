"""
A2A (Agent-to-Agent) protocol support — Google's A2A spec.
All agent communication in this project uses A2A (Agent Card + message/send).
Exposes sg-agent, modelmgmt-agent, search_agent, and graph_agent as A2A servers.
"""

import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Registry of agent instances for A2A handlers that need to call process_query (search_agent, graph_agent)
_agent_instances: Dict[str, Any] = {}


def set_agent_instances(instances: Dict[str, Any]) -> None:
    """Register agent instances so A2A handlers can invoke search_agent and graph_agent."""
    global _agent_instances
    _agent_instances = dict(instances or {})


def get_agent_instance(agent_id: str) -> Optional[Any]:
    """Return the registered agent instance for agent_id, or None."""
    return _agent_instances.get(agent_id)


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


def _agent_card_search_agent() -> Dict[str, Any]:
    return {
        "name": "Search Agent (search_agent)",
        "description": "Semantic search over the vector database (Milvus). Returns relevant documents and sources for natural-language queries.",
        "url": f"{_base_url()}/api/a2a/agents/search_agent",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["application/json", "text/plain"],
        "defaultOutputModes": ["application/json"],
        "skills": [
            {
                "id": "search",
                "name": "Semantic search",
                "description": "Run semantic search for the given query. Send the search query as message text.",
                "tags": ["search", "semantic", "vector", "milvus"],
                "examples": ["Show me transactions in California", "Find customers with high balance"],
            }
        ],
    }


def _agent_card_graph_agent() -> Dict[str, Any]:
    return {
        "name": "Graph Agent (graph_agent)",
        "description": "Graph and relationship queries over Neo4j. Returns connected entities and paths for natural-language queries.",
        "url": f"{_base_url()}/api/a2a/agents/graph_agent",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["application/json", "text/plain"],
        "defaultOutputModes": ["application/json"],
        "skills": [
            {
                "id": "graph",
                "name": "Graph query",
                "description": "Run graph/relationship query for the given question. Send the question as message text.",
                "tags": ["graph", "neo4j", "relationships", "entities"],
                "examples": ["How is customer X related to account Y?", "Find path between two entities"],
            }
        ],
    }


def _agent_card_enrich_agent() -> Dict[str, Any]:
    return {
        "name": "Enrich Agent (enrich_agent)",
        "description": "Extracts customer_id, transaction_id, dispute_id from results and enriches with Neo4j graph lookups. Used by aggregator to add entity details.",
        "url": f"{_base_url()}/api/a2a/agents/enrich_agent",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["application/json", "text/plain"],
        "defaultOutputModes": ["application/json"],
        "skills": [
            {
                "id": "enrich",
                "name": "Enrich IDs",
                "description": "Extract IDs from results and look up full details in Neo4j. Send query with context.results or context.text containing IDs.",
                "tags": ["enrich", "graph", "neo4j", "customer", "transaction", "dispute"],
                "examples": ["Enrich customer 1000000001", "Lookup details for dispute 1000005678"],
            }
        ],
    }


def _agent_card_sentiment_agent() -> Dict[str, Any]:
    return {
        "name": "Sentiment Agent (sentiment-agent)",
        "description": "Analyzes the sentiment/tone of a sentence or question via LLM. Returns sentiment label, confidence, and emoji. Used after sg-agent in the multi-agent demo.",
        "url": f"{_base_url()}/api/a2a/agents/sentiment-agent",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["application/json", "text/plain"],
        "defaultOutputModes": ["application/json"],
        "skills": [
            {
                "id": "analyze-sentiment",
                "name": "Analyze sentiment",
                "description": "Analyze sentiment/tone of a sentence. Send the sentence as message text. Returns sentiment, confidence, and emoji.",
                "tags": ["sentiment", "tone", "llm", "analysis"],
                "examples": ["Show me disputes over $1000", "What transactions are in California?"],
            }
        ],
    }


def get_agent_card(agent_id: str) -> Optional[Dict[str, Any]]:
    """Return A2A Agent Card for agent_id, or None if unknown."""
    cards = {
        "sg-agent": _agent_card_sg_agent,
        "modelmgmt-agent": _agent_card_modelmgmt_agent,
        "search_agent": _agent_card_search_agent,
        "graph_agent": _agent_card_graph_agent,
        "enrich_agent": _agent_card_enrich_agent,
        "sentiment-agent": _agent_card_sentiment_agent,
    }
    fn = cards.get(agent_id)
    return fn() if fn else None


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


def _normalize_ids_to_10_digits(text: str) -> str:
    """Replace CUST/TX/DBT-prefixed IDs with 10-digit numeric form. E.g. CUST00631102 -> 1000631102."""
    import re
    if not text or not isinstance(text, str):
        return text

    def _repl(m):
        digits = "".join(c for c in m.group(0) if c.isdigit())
        if not digits:
            return m.group(0)
        try:
            n = int(digits)
            return str(1000000000 + (n % 1000000000))
        except (TypeError, ValueError):
            return m.group(0)

    # Match CUST, TX, DBT, DSP followed by digits (with optional spaces)
    text = re.sub(r"\b(?:CUST|TX|DBT|DSP)\s*\d+\b", _repl, text, flags=re.IGNORECASE)
    return text


def _extract_original_from_prompt(prompt: str) -> str:
    """Extract original user query from prompt (between --- and ---)."""
    import re
    m = re.search(r"---\s*\n(.*?)\n\s*---", prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def _restore_ssn_from_original(prompt: str, reply: str) -> str:
    """
    Restore SSN (XXX-XX-XXXX) from original user query into the LLM reply.
    The LLM sometimes corrupts SSNs (e.g. 432-10-8327 -> 432-10827).
    """
    import re
    if not reply or not prompt:
        return reply
    original = _extract_original_from_prompt(prompt)
    if not original:
        return reply
    ssn_list = re.findall(r"\d{3}-\d{2}-\d{4}", original)
    if not ssn_list:
        return reply
    ssn = ssn_list[0]
    pat = re.compile(
        r"(\b(?:ssn|social\s*security)\s*[:\.]?\s*)([\d\-\.]{6,15})",
        re.IGNORECASE,
    )
    def repl(match):
        prefix, value = match.group(1), match.group(2)
        if value == ssn:
            return match.group(0)
        return prefix + ssn
    return pat.sub(repl, reply)


def _restore_phone_from_original(prompt: str, reply: str) -> str:
    """
    Restore phone/mobile/cell/telephone numbers from original user query into the LLM reply.
    Matches after mobile, phone, telephone, cell, tel. Phone formats: XXX-XXX-XXXX, (XXX) XXX-XXXX, etc.
    """
    import re
    if not reply or not prompt:
        return reply
    original = _extract_original_from_prompt(prompt)
    if not original:
        return reply
    phone_keywords = re.compile(
        r"\b(?:mobile|phone|telephone|cell|tel|contact)\s*[:\.]?\s*([\d\-\.\(\)\s]{10,20})",
        re.IGNORECASE,
    )
    matches = phone_keywords.findall(original)
    if not matches:
        return reply
    phone = matches[0].strip()
    if not phone or len("".join(c for c in phone if c.isdigit())) < 10:
        return reply
    # Match same keywords in reply followed by a value (possibly corrupted)
    pat = re.compile(
        r"(\b(?:mobile|phone|telephone|cell|tel|contact)\s*[:\.]?\s*)([\d\-\.\(\)\s]{8,25})",
        re.IGNORECASE,
    )
    def repl(match):
        prefix, value = match.group(1), match.group(2)
        if value.strip() == phone:
            return match.group(0)
        return prefix + phone
    return pat.sub(repl, reply)


def _restore_10digit_ids_from_original(prompt: str, reply: str) -> str:
    """
    Restore plain 10-digit numbers (customer IDs, transaction IDs, dispute IDs) from original
    into the LLM reply. The LLM sometimes corrupts them (e.g. 1000003621 -> 100000362 or 1,000,003,621),
    or wrongly replaces them with SSN placeholder (XXX-XX-XXXX). Preserves the exact original form.
    """
    import re
    if not reply or not prompt:
        return reply
    original = _extract_original_from_prompt(prompt)
    if not original:
        return reply
    # Find all 10-digit numbers in original (plain, no SSN/phone - those are handled separately)
    # Exclude SSN format XXX-XX-XXXX and phone-like patterns
    plain_10 = re.findall(r"\b\d{10}\b", original)
    if not plain_10:
        return reply
    # If LLM wrongly turned a customer/transaction/dispute ID into SSN placeholder, restore it
    # Only when original has no real SSN (XXX-XX-XXXX), so we don't overwrite real SSN restoration
    has_ssn_in_original = bool(re.search(r"\d{3}-\d{2}-\d{4}", original))
    ssn_placeholder = re.compile(r"XXX-XX-XXXX", re.IGNORECASE)
    if not has_ssn_in_original and ssn_placeholder.search(reply):
        reply = ssn_placeholder.sub(plain_10[0], reply, count=1)
    for orig in plain_10:
        # Replace corrupted forms in reply with exact original
        # 1. Missing last digit: 1000003621 -> 100000362
        reply = re.sub(r"\b" + re.escape(orig[:-1]) + r"\b", orig, reply)
        # 2. Missing first digit: 1000003621 -> 000003621 (only if orig doesn't start with 0)
        if orig[0] != "0":
            reply = re.sub(r"\b" + re.escape(orig[1:]) + r"\b", orig, reply)
        # 3. Extra trailing digit: 1000003621 -> 10000036210
        reply = re.sub(r"\b" + re.escape(orig) + r"0\b", orig, reply)
        # 4. Extra leading zero: 1000003621 -> 01000003621
        reply = re.sub(r"\b0" + re.escape(orig) + r"\b", orig, reply)
        # 5. Comma-formatted: 1,000,003,621 -> 1000003621
        comma_fmt = re.sub(r"(\d)(?=(\d{3})+(?!\d))", r"\1,", orig)
        reply = reply.replace(comma_fmt, orig)
    return reply


def _sg_agent_fix_or_generate_via_ollama(prompt: str) -> Optional[str]:
    """Call Ollama with the given prompt; return the model reply (single improved/generated sentence) or None."""
    import os
    import ssl
    import urllib.request
    host = (os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL") or "llama3.2"
    try:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 256},
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
        content = (data.get("message") or {}).get("content") or ""
        return content.strip() if content else None
    except Exception as e:
        logger.warning("sg-agent Ollama fix/generate call failed: %s", e)
        return None


def _is_llm_refusal(reply: str) -> bool:
    """True if the LLM returned a refusal/error instead of the requested content."""
    r = (reply or "").lower()
    refusal_phrases = (
        "not provided",
        "please provide",
        "provide the",
        "send me",
        "send the",
        "i'd be happy to assist",
        "i would be happy",
        "i cannot",
        "i'm unable",
        "i am unable",
        "i don't have",
        "i do not have",
        "could you please provide",
        "could you provide",
        "would you please",
    )
    return any(p in r for p in refusal_phrases)


def handle_sg_agent_message_send(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    A2A message/send for sg-agent. Params: { message: Message }.
    - If message is a "fix/improve user sentence" request (multi-agent-demo): call Ollama with that prompt and return the improved sentence.
    - Else (e.g. "Generate 3 sentences"): generate sentences from schema, return Task with artifacts.
    """
    task_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())
    message = params.get("message") or {}
    text = _text_from_message(message)
    if not text:
        return _make_task(
            task_id, context_id, "failed",
            message_parts=[{"kind": "text", "text": "No text in message. Send e.g. 'Generate 3 sentences' or a fix/improve prompt."}],
        )
    lower = text.lower()
    # Multi-agent-demo: "fix user sentence" or "generate one question" — pass full prompt to Ollama and return single reply
    if (
        "the user typed the following" in lower
        or "output only the improved sentence" in lower
        or "fix any spelling" in lower
        or "edit this user query" in lower
        or "fix spelling and grammar" in lower
        or ("generate a single natural-language question" in lower and "output only the question" in lower)
    ):
        reply = _sg_agent_fix_or_generate_via_ollama(text)
        # Reject LLM refusals/confusion (e.g. "not provided", "please provide") — treat as failure
        if reply and _is_llm_refusal(reply):
            reply = None
        if reply:
            reply = _restore_ssn_from_original(text, reply)
            reply = _restore_phone_from_original(text, reply)
            reply = _restore_10digit_ids_from_original(text, reply)
            reply = _normalize_ids_to_10_digits(reply)
            artifacts = [{
                "artifactId": str(uuid.uuid4()),
                "name": "improved_or_generated_sentence",
                "description": "Single sentence from sg-agent (fix/context or generated)",
                "parts": [{"kind": "text", "text": reply}],
            }]
            return _make_task(task_id, context_id, "completed", artifacts=artifacts)
        return _make_task(
            task_id, context_id, "failed",
            message_parts=[{"kind": "text", "text": "Ollama did not return an improved or generated sentence."}],
        )
    # Parse "Generate N sentences" or default to 3
    count = 3
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
        # Fallback to template sentences when Ollama fails (same as Compare "I'm feeling lucky")
        if not sentences_list and schema:
            try:
                from sg_agent.sentence_generator import _generate_templates
                templates = _generate_templates(schema)
                if templates:
                    sentences_list = templates[:count]
                    error_msg = None
                    logger.info("sg-agent A2A: using template fallback (%s sentences)", len(sentences_list))
            except Exception as tb:
                logger.debug("sg-agent A2A template fallback failed: %s", tb)
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
                s = _normalize_ids_to_10_digits(s)
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


def handle_search_agent_message_send(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    A2A message/send for search_agent. Params: { message: Message, context?: Dict }.
    Runs semantic search via registered agent instance; returns Task with results artifact.
    """
    task_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())
    message = params.get("message") or {}
    text = _text_from_message(message)
    context = params.get("context") or {}
    if not text:
        return _make_task(
            task_id, context_id, "failed",
            message_parts=[{"kind": "text", "text": "No search query in message."}],
        )
    agent = get_agent_instance("search_agent")
    if not agent:
        return _make_task(
            task_id, context_id, "failed",
            message_parts=[{"kind": "text", "text": "search_agent not registered."}],
        )
    try:
        resp = agent.process_query(text, context)
        artifacts = [{
            "artifactId": str(uuid.uuid4()),
            "name": "search_results",
            "description": "Semantic search results",
            "parts": [{"kind": "data", "data": {"results": resp.results, "message": resp.message, "metadata": resp.metadata or {}}}],
        }]
        return _make_task(task_id, context_id, "completed", artifacts=artifacts)
    except Exception as e:
        logger.exception("search_agent A2A message/send failed")
        return _make_task(
            task_id, context_id, "failed",
            message_parts=[{"kind": "text", "text": str(e)}],
        )


def handle_enrich_agent_message_send(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    A2A message/send for enrich_agent. Params: { message: Message, context?: Dict }.
    Extracts IDs from context.results/context.text and enriches via Neo4j.
    """
    task_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())
    message = params.get("message") or {}
    text = _text_from_message(message)
    context = params.get("context") or {}
    if text:
        context = dict(context)
        context["text"] = text
    agent = get_agent_instance("enrich_agent")
    if not agent:
        return _make_task(
            task_id, context_id, "failed",
            message_parts=[{"kind": "text", "text": "enrich_agent not registered."}],
        )
    try:
        resp = agent.process_query(text or "enrich", context)
        artifacts = [{
            "artifactId": str(uuid.uuid4()),
            "name": "enrich_results",
            "description": "Enriched entity details from Neo4j",
            "parts": [{"kind": "data", "data": {"results": resp.results, "message": resp.message, "metadata": resp.metadata or {}}}],
        }]
        return _make_task(task_id, context_id, "completed", artifacts=artifacts)
    except Exception as e:
        logger.exception("enrich_agent A2A message/send failed")
        return _make_task(
            task_id, context_id, "failed",
            message_parts=[{"kind": "text", "text": str(e)}],
        )


def handle_graph_agent_message_send(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    A2A message/send for graph_agent. Params: { message: Message, context?: Dict }.
    Runs graph query via registered agent instance; returns Task with results artifact.
    """
    task_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())
    message = params.get("message") or {}
    text = _text_from_message(message)
    context = params.get("context") or {}
    if not text:
        return _make_task(
            task_id, context_id, "failed",
            message_parts=[{"kind": "text", "text": "No query in message."}],
        )
    agent = get_agent_instance("graph_agent")
    if not agent:
        return _make_task(
            task_id, context_id, "failed",
            message_parts=[{"kind": "text", "text": "graph_agent not registered."}],
        )
    try:
        resp = agent.process_query(text, context)
        artifacts = [{
            "artifactId": str(uuid.uuid4()),
            "name": "graph_results",
            "description": "Graph query results",
            "parts": [{"kind": "data", "data": {"results": resp.results, "message": resp.message, "metadata": resp.metadata or {}}}],
        }]
        return _make_task(task_id, context_id, "completed", artifacts=artifacts)
    except Exception as e:
        logger.exception("graph_agent A2A message/send failed")
        return _make_task(
            task_id, context_id, "failed",
            message_parts=[{"kind": "text", "text": str(e)}],
        )


def handle_sentiment_agent_message_send(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    A2A message/send for sentiment-agent. Params: { message: Message }.
    Analyzes sentiment of the sentence and returns Task with sentiment artifact.
    """
    task_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())
    message = params.get("message") or {}
    sentence = _text_from_message(message)
    if not sentence:
        return _make_task(
            task_id, context_id, "failed",
            message_parts=[{"kind": "text", "text": "No sentence in message. Send the sentence to analyze as message text."}],
        )
    try:
        from sentiment_agent.sentiment_analyzer import analyze_sentence_sentiment
        sentiment, confidence, emoji, reasoning, err = analyze_sentence_sentiment(sentence)
        if err:
            return _make_task(
                task_id, context_id, "failed",
                message_parts=[{"kind": "text", "text": err}],
            )
        data = {"sentiment": sentiment or "", "confidence": confidence, "emoji": emoji}
        if reasoning:
            data["reasoning"] = reasoning
        artifacts = [{
            "artifactId": str(uuid.uuid4()),
            "name": "sentiment_analysis",
            "description": "Sentiment analysis of the sentence",
            "parts": [{"kind": "data", "data": data}],
        }]
        return _make_task(task_id, context_id, "completed", artifacts=artifacts)
    except ImportError:
        return _make_task(
            task_id, context_id, "failed",
            message_parts=[{"kind": "text", "text": "sentiment-agent not available."}],
        )
    except Exception as e:
        logger.exception("sentiment-agent A2A message/send failed")
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
    elif agent_id == "sentiment-agent":
        result = handle_sentiment_agent_message_send(params)
    elif agent_id == "search_agent":
        result = handle_search_agent_message_send(params)
    elif agent_id == "graph_agent":
        result = handle_graph_agent_message_send(params)
    elif agent_id == "enrich_agent":
        result = handle_enrich_agent_message_send(params)
    else:
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": -32601, "message": f"Unknown agent: {agent_id}"},
        }, 200
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}, 200


def invoke_agent_via_a2a(
    agent_id: str,
    user_query: str,
    context: Dict[str, Any],
) -> Tuple[List[Any], str, Dict[str, Any]]:
    """
    Invoke an agent via the A2A protocol (message/send) and return (results, message, metadata).
    Used by the LangGraph orchestrator so all agent communication goes over A2A.
    Supports search_agent and graph_agent only (orchestrator uses these).
    """
    if agent_id not in ("search_agent", "graph_agent", "enrich_agent"):
        return [], "Unsupported agent for A2A invoke", {}
    params = {
        "message": {"parts": [{"kind": "text", "text": user_query}]},
        "context": context,
    }
    if agent_id == "search_agent":
        task = handle_search_agent_message_send(params)
    elif agent_id == "graph_agent":
        task = handle_graph_agent_message_send(params)
    else:
        task = handle_enrich_agent_message_send(params)
    status = task.get("status") or {}
    if status.get("state") == "failed":
        msg = ""
        if status.get("message") and isinstance(status["message"], dict):
            parts = status["message"].get("parts") or []
            for p in parts:
                if p.get("kind") == "text" and "text" in p:
                    msg = p["text"]
                    break
        return [], msg or "A2A task failed", {}
    artifacts = task.get("artifacts") or []
    for art in artifacts:
        for part in art.get("parts") or []:
            if part.get("kind") == "data" and "data" in part:
                d = part["data"]
                if isinstance(d, dict):
                    return (
                        d.get("results", []),
                        d.get("message", ""),
                        d.get("metadata", {}),
                    )
    return [], "", {}
