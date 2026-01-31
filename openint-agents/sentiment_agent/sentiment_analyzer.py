"""
Sentiment analysis for sentences/questions via Ollama LLM.
Identifies tone/sentiment (e.g. neutral, curious, analytical, urgent) and returns emoji.
When Ollama fails or returns invalid JSON, uses rule-based fallback so sentiment always succeeds.
"""

from __future__ import annotations

import json
import logging
import os
import re
import ssl
import urllib.error
import urllib.request

from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _try_extract_from_raw(content: str) -> Optional[Tuple[str, float, str, str]]:
    """
    Try to extract sentiment fields from malformed LLM output using regex.
    Returns (sentiment, confidence, emoji, reasoning) or None if extraction fails.
    """
    if not content or not isinstance(content, str):
        return None
    content = content.strip()
    sentiment = None
    confidence = 0.6
    emoji = None
    reasoning = None
    # "sentiment": "value" or "sentiment":"value"
    m = re.search(r'"sentiment"\s*:\s*"([^"]*)"', content, re.IGNORECASE | re.DOTALL)
    if m:
        sentiment = (m.group(1) or "").strip()[:200]
    # confidence: 0.8 or "confidence": 0.8
    m = re.search(r'"confidence"\s*:\s*([\d.]+)', content, re.IGNORECASE)
    if m:
        try:
            confidence = max(0.0, min(1.0, float(m.group(1))))
        except (TypeError, ValueError):
            pass
    # "emoji": "😊" or emoji in text
    m = re.search(r'"emoji"\s*:\s*"([^"]*)"', content, re.IGNORECASE)
    if m:
        emoji = (m.group(1) or "").strip()[:20] or None
    # "reasoning": "value"
    m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', content, re.IGNORECASE | re.DOTALL)
    if m:
        reasoning = (m.group(1) or "").strip()[:500] or None
    if sentiment or emoji:
        return (
            sentiment or "neutral",
            confidence,
            emoji or "💭",
            reasoning or "Extracted from LLM response.",
        )
    return None


def _fallback_sentiment(text: str) -> Tuple[str, float, str, str]:
    """
    Rule-based fallback when Ollama is unavailable.
    Returns (sentiment, confidence, emoji, reasoning).
    """
    t = (text or "").lower()
    # Urgent / critical
    if any(w in t for w in ("urgent", "asap", "immediately", "critical", "emergency", "fix now", "help!", "need now")):
        return ("urgent or concerned", 0.6, "⚡", "Keyword match: question suggests urgency or time-sensitivity.")
    # Frustrated / negative
    if any(w in t for w in ("frustrated", "angry", "unhappy", "disappointed", "complaint", "wrong", "error", "broken")):
        return ("frustrated or negative", 0.6, "😟", "Keyword match: question suggests frustration or concern.")
    # Curious / exploratory
    if any(w in t for w in ("show", "list", "find", "explore", "tell me about", "what", "how many", "which")):
        return ("curious and exploratory", 0.65, "🔍", "Keyword match: question appears to be exploratory or data-seeking.")
    # Default: neutral analytical
    return ("neutral and analytical", 0.55, "💭", "Fallback: neutral analytical (Ollama unavailable).")


def analyze_sentence_sentiment(text: str) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[str], Optional[str]]:
    """
    Analyze sentiment/tone of a sentence or question via Ollama.
    Returns (sentiment, confidence, emoji, reasoning, error).
    sentiment: short descriptive phrase (e.g. "neutral and analytical", "curious and exploratory")
    confidence: 0.0–1.0
    emoji: single emoji fitting the sentiment
    reasoning: brief explanation of why this sentiment was detected
    """
    text = (text or "").strip()
    if not text:
        return (None, None, None, None, "No text provided")

    host = (os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL") or "qwen2.5:7b"

    prompt = f"""Classify the sentiment/tone of this sentence. Reply with ONLY valid JSON, no other text:
{{"sentiment":"short phrase","confidence":0.8,"emoji":"🔍","reasoning":"brief why"}}

Options: neutral/analytical, curious/exploratory, urgent, frustrated/negative, positive/hopeful.
Sentence: {text[:1500]}"""

    content = ""
    try:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 128},
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
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        message = data.get("message") or {}
        content = (message.get("content") or "").strip()

        if "```" in content:
            parts = content.split("```", 2)
            if len(parts) >= 2:
                content = parts[1].strip()
                if content.lower().startswith("json"):
                    content = content[4:].strip()

        start = content.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(content)):
                if content[i] == "{":
                    depth += 1
                elif content[i] == "}":
                    depth -= 1
                    if depth == 0:
                        content = content[start : i + 1]
                        break

        obj = json.loads(content)
        sentiment = (obj.get("sentiment") or "").strip()
        if sentiment:
            sentiment = sentiment[:200]

        try:
            confidence = float(obj.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        emoji = (obj.get("emoji") or "").strip()[:20] or None
        reasoning = (obj.get("reasoning") or "").strip()
        if reasoning:
            reasoning = reasoning[:500]
        reasoning = reasoning or None
        return (sentiment or None, confidence, emoji, reasoning, None)

    except urllib.error.URLError as e:
        msg = str(getattr(e, "reason", e) or e)
        if "Connection refused" in msg or "111" in msg:
            msg = "Ollama not available"
        logger.warning("sentiment-agent Ollama call failed, using fallback: %s", msg)
        sent, conf, em, reason = _fallback_sentiment(text)
        return (sent, conf, em, reason, None)
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("sentiment-agent invalid LLM response, using fallback: %s", e)
        # Try regex extraction from raw content before rule-based fallback
        extracted = _try_extract_from_raw(content) if content else None
        if extracted:
            return (extracted[0], extracted[1], extracted[2], extracted[3], None)
        sent, conf, em, reason = _fallback_sentiment(text)
        return (sent, conf, em, reason, None)
    except Exception as e:
        logger.warning("sentiment-agent error, using fallback: %s", e, exc_info=True)
        sent, conf, em, reason = _fallback_sentiment(text)
        return (sent, conf, em, reason, None)
