"""
Sentence generator for openint-sg-agent.
Uses Ollama (open-source LLM) to generate banking, data, analytics, and regulatory
example questions. Context comes from the DataHub catalog schema (datasets and fields).
No template fallback.
"""

import os
import json
import logging
import urllib.request
import urllib.error
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def _get_ollama_config() -> tuple[str, str]:
    """OLLAMA_HOST and OLLAMA_MODEL at call time."""
    host = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL") or "llama3.2"
    return (host, model)


def _build_schema_summary(schema: Dict[str, Dict[str, Any]], max_datasets: Optional[int] = None, max_fields: int = 15) -> str:
    """Build a short text summary of datasets and fields from the DataHub catalog for the LLM.
    Use max_datasets to limit size for faster Ollama response (e.g. 6 for lucky)."""
    parts = []
    items = list(schema.items())
    if max_datasets is not None and max_datasets > 0:
        items = items[:max_datasets]
    for name, meta in items:
        desc = (meta.get("description", "") or "")[:80]
        category = meta.get("category", "dataset")
        fields = meta.get("fields", [])[:max_fields]
        field_names = [f.get("name", "") for f in fields]
        parts.append(f"- {name} ({category}): {desc}. Fields: {', '.join(field_names)}")
    return "\n".join(parts)


def _build_prompt(schema_summary: str, count: int, short_for_lucky: bool = False, schema_source: str = "datahub") -> str:
    """Prompt for sentence generation. Context to the LLM is always from DataHub assets and schema
    (or openint-datahub fallback). schema_source is 'datahub' or 'openint-datahub'."""
    context_label = "DataHub assets and schema (from DataHub catalog)" if schema_source == "datahub" else "schema (from openint-datahub; DataHub unavailable)"
    if short_for_lucky:
        return f"""You are helping a banking data platform. Context to the LLM: {context_label}. Given the schema below, generate exactly {count} natural-language questions (1-3 sentences each). Banking/data/analytics/regulatory only. Return a JSON array of objects with "sentence" and "category" (one of: Analyst, Customer Care, Business Analyst, Regulatory). Use only table/field names from the schema. Return ONLY the JSON array.

Schema:
{schema_summary}"""
    return f"""You are helping a banking data platform. Context to the LLM: {context_label}. Given the following dataset schema (tables and fields), generate exactly {count} natural-language questions that real users would ask. Focus strictly on banking, data, analytics, and regulatory use cases. Return a JSON array of objects, each with "sentence" (the question) and "category" (one of: "Analyst", "Customer Care", "Business Analyst", "Regulatory").

Schema:
{schema_summary}

Rules:
- Analyst: data exploration, counts, top N, filters, multi-dimensional breakdowns, dashboards.
- Customer Care: lookup by customer, account, transaction, with full context and related records.
- Business Analyst: KPIs, trends, segments, comparisons, year-over-year, regional breakdowns.
- Regulatory: compliance, reporting, audit, SAR, CTR, multi-condition filters, audit trails.
- Use only table/field names from the schema. All questions must be banking/data/analytics/regulatory in nature.
- CRITICAL — Length and complexity: Each question MUST be at least 5x longer than a one-line question (aim for 50–120+ words). Make every query complex: use multiple clauses, conditions, and groupings. Include at least 3–5 of: time ranges, breakdowns (by region/type/state), comparisons (vs last period), explicit fields to return, filters (amounts, statuses), and purpose (e.g. "for executive summary", "for compliance review"). No short one-liners.
- Return ONLY the JSON array, no other text."""


def _parse_llm_json_response(text: str) -> Optional[List[Dict[str, Any]]]:
    """Extract JSON array from LLM response; strip markdown code blocks if present."""
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        return None


def _generate_with_ollama(schema_summary: str, count: int = 20, short_for_lucky: bool = False, schema_source: str = "datahub") -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Generate example sentences using Ollama (open-source LLM). Context to the LLM is provided via DataHub assets and schema (or openint-datahub). Returns (sentences, None) or (None, error_message).
    When short_for_lucky=True, use a shorter prompt and lower num_predict for faster response."""
    host, model = _get_ollama_config()
    num_predict = min(8000, max(512, 200 * count)) if short_for_lucky else min(8000, max(512, 350 * count))
    try:
        import ssl
        prompt = _build_prompt(schema_summary, count, short_for_lucky=short_for_lucky, schema_source=schema_source)
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": num_predict},
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
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        message = data.get("message") or {}
        text = (message.get("content") or "").strip()
        parsed = _parse_llm_json_response(text)
        if isinstance(parsed, list):
            out = []
            for item in parsed:
                if isinstance(item, dict) and item.get("sentence"):
                    out.append({
                        "sentence": item["sentence"],
                        "category": item.get("category", "Analyst"),
                        "source": "ollama",
                    })
            return (out[:count] if out else [], None)
        return (None, "Ollama returned invalid JSON. Try a different model or check the response.")
    except urllib.error.URLError as e:
        logger.warning("Ollama sentence generation failed: %s", e)
        msg = str(e.reason) if getattr(e, "reason", None) else str(e)
        if "Connection refused" in msg or "111" in msg:
            msg = "Ollama is not running. Start Ollama (e.g. ollama serve) and pull a model: ollama pull " + model
        return (None, msg)
    except Exception as e:
        logger.warning("Ollama sentence generation failed: %s", e)
        return (None, str(e))


def _generate_templates(schema: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Template-based example sentences from schema (no LLM). Fallback when Ollama unavailable."""
    sentences: List[Dict[str, Any]] = []
    ds = "disputes" if "disputes" in schema else "transactions"
    templates_analyst = [
        "I need a report showing the top {n} customers by total number of transactions for the last quarter, broken down by transaction type (ACH, wire, check, and card), including their region and account status, and compared to the same period last year so we can see growth trends.",
        "How many {dataset} did we have in the last month, broken down by status (pending, completed, failed, reversed), by state, and by transaction type, with a comparison to the prior month for trend analysis?",
        "Give me a full summary of transaction mix by state for the last quarter: ACH, wire, check, and card volume and count per state, with year-over-year change and top 10 states by total volume for executive review.",
    ]
    for t in templates_analyst:
        sentences.append({"sentence": t.format(n=10, dataset=ds), "category": "Analyst", "source": "template"})
    templates_care = [
        "Show me all transactions for customer {cid} in the last 30 days including amount, status, transaction type (ACH, wire, card), date, and counterparty, and list any related disputes or reversals so I can explain the activity to the customer.",
    ]
    for t in templates_care:
        sentences.append({"sentence": t.format(cid="CUST00000001"), "category": "Customer Care", "source": "template"})
    templates_ba = [
        "I need the top 10 largest wire transfers for the last quarter with originator, beneficiary, amount, date, and country, plus a breakdown of domestic vs international and how that compares to the prior quarter for executive summary.",
    ]
    for s in templates_ba:
        sentences.append({"sentence": s, "category": "Business Analyst", "source": "template"})
    templates_regulatory = [
        "For SAR reporting, list all wire transfers over $10,000 in the last 30 days with originator name and ID, beneficiary name and country, amount, date, and transaction ID, grouped by originator.",
    ]
    for s in templates_regulatory:
        sentences.append({"sentence": s, "category": "Regulatory", "source": "template"})
    return sentences


def generate_one_lucky(
    schema: Dict[str, Dict[str, Any]],
    prefer_llm: bool = True,
    schema_source: str = "datahub",
) -> Dict[str, Any]:
    """Generate one random sentence for 'I'm feeling lucky!'. Prefers Ollama; falls back to templates."""
    import random
    sentences, error_msg = generate_sentences(schema, count=5, prefer_llm=prefer_llm, fast_lucky=True, schema_source=schema_source)
    if sentences:
        return random.choice(sentences)
    templates = _generate_templates(schema)
    if templates:
        one = random.choice(templates)
        return {"sentence": one["sentence"], "category": one.get("category", "Analyst"), "source": "template", "fallback": True, "ollama_error": error_msg}
    return {"sentence": "", "category": "", "source": "ollama", "error": error_msg or "Ollama is not available. Start Ollama and pull a model (e.g. ollama pull llama3.2)."}


def generate_sentences(
    schema: Dict[str, Dict[str, Any]],
    count: int = 25,
    prefer_llm: bool = True,
    fast_lucky: bool = False,
    schema_source: str = "datahub",
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Generate example sentences via Ollama. Returns (sentences, error). When fast_lucky=True, use shorter schema summary."""
    if fast_lucky:
        summary = _build_schema_summary(schema, max_datasets=6, max_fields=8)
        sentences_list, error_msg = _generate_with_ollama(summary, count=count, short_for_lucky=True, schema_source=schema_source)
    else:
        summary = _build_schema_summary(schema)
        sentences_list, error_msg = _generate_with_ollama(summary, count=count, schema_source=schema_source)
    return (sentences_list or [], error_msg)
