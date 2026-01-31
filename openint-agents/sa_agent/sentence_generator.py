"""
Sentence generator for sa-agent.
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
        return f"""You are helping a banking data platform. Context to the LLM: {context_label}. Given the schema below, generate exactly {count} natural-language questions (1-3 sentences each). Banking/data/analytics/regulatory only. Return a JSON array of objects with "sentence" and "category" (one of: Analyst, Customer Care, Business Analyst, Regulatory). Use only table/field names from the schema. If you include example IDs, use exactly 10-digit numeric IDs: customer_id (e.g. 1000000001), transaction_id (e.g. 1000001234), dispute_id (e.g. 1000005678). Never use prefixes like CUST or TX. Return ONLY the JSON array.

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
- If you include example IDs (customer, transaction, dispute), use exactly 10-digit numeric IDs only: customer_id (e.g. 1000000001), transaction_id (e.g. 1000001234), dispute_id (e.g. 1000005678). Never use prefixes like CUST or TX.
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
    # Lower num_predict for fewer sentences to speed up response (≈200–400 tokens per sentence; less for short)
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
        # Allow self-signed / localhost
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

    # Analyst-style
    templates_analyst = [
        "I need a report showing the top {n} customers by total number of transactions for the last quarter, broken down by transaction type (ACH, wire, check, and card), including their region and account status, and compared to the same period last year so we can see growth trends.",
        "Give me the top {n} cities by customer count for the past 6 months, with a breakdown of active vs closed accounts per city, and show month-over-month change in customer count so we can identify which markets are growing or shrinking.",
        "How many {dataset} did we have in the last month, broken down by status (pending, completed, failed, reversed), by state, and by transaction type, with a comparison to the prior month for trend analysis?",
        "I need to compare ACH vs wire transaction volumes by state for the last quarter, with total count and total amount per state, and show which states have the highest share of international wires versus domestic so we can assess regional mix.",
        "Which ZIP codes have the most failed or reversed transactions in the last 90 days, with count and total amount per ZIP, broken down by transaction type and by month, so we can prioritize fraud and operational review?",
        "Give me a full summary of transaction mix by state for the last quarter: ACH, wire, check, and card volume and count per state, with year-over-year change and top 10 states by total volume for executive review.",
        "Show me the top {n} customers by total transaction amount for the last quarter with their region, account status, and a breakdown by transaction type (ACH, wire, card), plus how that compares to the same quarter last year.",
        "I need a breakdown of active vs closed accounts by region for the last quarter, with customer count and total transaction volume per region, and flag any regions where closed-account rate increased by more than 10% vs prior quarter.",
        "Which transaction types have the most pending status right now, with count and total amount per type, broken down by state and by age (e.g. 0–7 days, 8–30 days, 30+ days), so we can prioritize back-office clearing?",
        "List the top {n} states by transaction volume for the last 6 months including ACH and wire separately, with month-over-month change and year-over-year change, and identify the top 5 states with the largest YoY growth for planning.",
    ]
    for t in templates_analyst:
        s = t.format(n=10, dataset=ds)
        sentences.append({"sentence": s, "category": "Analyst", "source": "template"})

    # Customer care-style
    templates_care = [
        "Show me all transactions for customer {cid} in the last 30 days including amount, status, transaction type (ACH, wire, card), date, and counterparty, and list any related disputes or reversals so I can explain the activity to the customer.",
        "List all disputes for customer {cid} that are still open or were closed in the last 90 days, with filing date, amount, status, resolution date if closed, and the associated transaction IDs so I can give a full dispute history.",
        "What is the current account status for customer {cid}, including active vs closed, total transaction count and volume for the last 12 months, and any pending ACH or wire transactions that haven't settled yet?",
        "Show me all pending ACH payments for the account associated with customer {cid}, with amount, expected settlement date, originator or beneficiary details, and how long each has been pending so we can follow up if needed.",
        "I need to see all wire transfers over $10,000 for customer {cid} in the last 90 days with beneficiary name and country, amount, date, and status, and flag any that are still pending or failed for compliance or customer inquiry.",
        "List all credit card disputes for customer {cid} that are still open, with filing date, amount, merchant, dispute reason, and days since filing, plus the status of each so I can prioritize and update the customer.",
        "Show me the full transaction history for the account associated with customer {cid} for the past 90 days: all ACH, wire, and card transactions with amount, date, status, and counterparty, grouped by month, so I can reconcile and answer the customer.",
        "For customer {cid}, give me a summary of failed or reversed transactions in the last 60 days with transaction type, amount, date, and reason if available, and list any related disputes so we can resolve in one call.",
    ]
    for t in templates_care:
        s = t.format(cid="1000000001")
        sentences.append({"sentence": s, "category": "Customer Care", "source": "template"})

    # Business analyst style
    templates_ba = [
        "I need the top 10 largest wire transfers for the last quarter with originator, beneficiary, amount, date, and country, plus a breakdown of domestic vs international and how that compares to the prior quarter for executive summary.",
        "Which states have the highest share of international wires versus domestic for the last 6 months, with total volume and count per state, and show the trend month over month so we can see where international is growing.",
        "Give me credit card transactions over $1,000 by merchant category for the last quarter, with count and total amount per category, and compare to the same period last year to identify which categories are growing or shrinking.",
        "Where do we have the most disputed credit card transactions: by state, by merchant category, and by month for the last 12 months, with count and total disputed amount, so we can target fraud and dispute resolution efforts?",
        "Show me all ACH transactions over $5,000 for the last month with originator, beneficiary, date, and status, broken down by state and by transaction type (credit vs debit), and flag any that failed or were reversed for review.",
        "I need failed or reversed ACH transactions by type and state for the last quarter, with count and total amount per combination, and a comparison to the prior quarter to see if failure rates are improving or worsening.",
        "Give me a comparison of pending wire transactions vs the same period last year by region, with count and total amount per region, and show the percentage change so we can explain backlog or improvement to leadership.",
        "Which regions have the most check usage and how has that changed over the last 6 months: volume and count by region by month, with year-over-year comparison so we can plan for check decline and digital migration.",
        "I need disputed credit card transactions by state with count and total amount for the last quarter, plus top 5 states by dispute rate and by total disputed amount, for executive summary and board reporting.",
        "List all customers in California with active accounts and their total transaction volume (ACH, wire, card) for this quarter, with a breakdown by transaction type and comparison to last quarter for growth analysis.",
    ]
    for s in templates_ba:
        sentences.append({"sentence": s, "category": "Business Analyst", "source": "template"})

    # Regulatory style
    templates_regulatory = [
        "For SAR reporting, list all wire transfers over $10,000 in the last 30 days with originator name and ID, beneficiary name and country, amount, date, and transaction ID, grouped by originator, and flag any originators with multiple such transfers in the period.",
        "I need disputes filed in Q1 by state with count and total amount per state, plus the breakdown by dispute type and by resolution status (open, closed, won, lost), for regulatory and board reporting.",
        "Give me ACH transaction volume by originator for the last 90 days for SAR reporting: total count and total amount per originator, with originator ID and name, and list any originators with volume over $100,000 in the period for further review.",
        "Credit card chargebacks by merchant category for this month: count and total amount per category, with merchant IDs for the top 10 categories by chargeback volume, for compliance and merchant risk review.",
        "List all customers with multiple disputes (2 or more) in the last 12 months, with customer ID, dispute count, total disputed amount, and dates of each dispute, grouped by customer for fraud and compliance review.",
        "International wire transfers by beneficiary country for the last quarter: count and total amount per country, with a breakdown by amount band (e.g. under $10k, $10k–$50k, over $50k) for regulatory filing and sanctions screening.",
        "I need a full audit trail of failed or reversed transactions by type and date for the last quarter, with transaction ID, amount, date, customer or originator, and reason code if available, for internal audit and regulatory exam.",
        "List all pending disputes older than 45 days with customer ID, amount, filing date, dispute type, and days pending, grouped by state, for compliance review and escalation to meet SLA.",
        "For regulatory filing, list international wire transfers by beneficiary country and amount band (e.g. $10k–$50k, over $50k) for the last 30 days, with count and total amount per country per band, and include originator count per country for risk assessment.",
        "Give me all wire transfers over $10,000 in the last 30 days with full detail: originator, beneficiary, amount, date, country, and transaction ID, plus a summary by beneficiary country and by day for CTR and SAR support.",
    ]
    for s in templates_regulatory:
        sentences.append({"sentence": s, "category": "Regulatory", "source": "template"})

    return sentences


def generate_one_lucky(
    schema: Dict[str, Dict[str, Any]],
    prefer_llm: bool = True,
    schema_source: str = "datahub",
) -> Dict[str, Any]:
    """
    Generate one random sentence for "I'm feeling lucky!" — banking, data,
    analytics, or regulatory. Context to the LLM is provided via DataHub assets and schema (schema_source).
    Prefers Ollama; falls back to template sentences when Ollama is unavailable or returns invalid JSON.
    Returns {"sentence", "category", "source": "ollama"|"template"}.
    """
    import random
    # Request only 5 sentences for "I'm feeling lucky" to keep Ollama response fast; we pick one.
    sentences, error_msg = generate_sentences(schema, count=5, prefer_llm=prefer_llm, fast_lucky=True, schema_source=schema_source)
    if sentences:
        return random.choice(sentences)
    # Fallback: use template sentences so the button always works when schema is available
    templates = _generate_templates(schema)
    if templates:
        one = random.choice(templates)
        return {
            "sentence": one["sentence"],
            "category": one.get("category", "Analyst"),
            "source": "template",
            "fallback": True,
            "ollama_error": error_msg,
        }
    return {
        "sentence": "",
        "category": "",
        "source": "ollama",
        "error": error_msg or "Ollama is not available. Start Ollama and pull a model (e.g. ollama pull llama3.2).",
    }


def generate_sentences(
    schema: Dict[str, Dict[str, Any]],
    count: int = 25,
    prefer_llm: bool = True,
    fast_lucky: bool = False,
    schema_source: str = "datahub",
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Generate example sentences via Ollama (open-source LLM). Context to the LLM is provided via
    DataHub assets and schema (schema_source='datahub') or openint-datahub (schema_source='openint-datahub').
    Returns (sentences, error). If error is set, sentences will be empty.
    When fast_lucky=True, use a shorter schema summary for faster Ollama response (e.g. for "I'm feeling lucky").
    """
    if not prefer_llm:
        logger.warning("Sentence generation is LLM-only; prefer_llm=False ignored.")
    if fast_lucky:
        summary = _build_schema_summary(schema, max_datasets=6, max_fields=8)
        sentences_list, error_msg = _generate_with_ollama(summary, count=count, short_for_lucky=True, schema_source=schema_source)
    else:
        summary = _build_schema_summary(schema)
        sentences_list, error_msg = _generate_with_ollama(summary, count=count, schema_source=schema_source)
    return (sentences_list or [], error_msg)
