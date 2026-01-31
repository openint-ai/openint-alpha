"""
Semantic analysis for modelmgmt-agent.
Processes queries through multiple embedding models; uses modelmgmt_agent.model_registry
(Hugging Face + Redis) for model loading.
"""

import os
import re
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    SentenceTransformer = None
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")


def _get_schema_for_semantics() -> Optional[Dict[str, Dict[str, Any]]]:
    """Get dataset schema from DataHub (sa-agent) or openint-datahub/schemas.py. Returns None if unavailable."""
    try:
        modelmgmt_agent_dir = Path(__file__).resolve().parent
        agents_root = modelmgmt_agent_dir.parent
        repo_root = agents_root.parent
        openint_datahub = repo_root / "openint-datahub"
        if str(agents_root) not in sys.path:
            sys.path.insert(0, str(agents_root))
        try:
            from sa_agent.datahub_client import get_schema
            schema = get_schema()
            if schema:
                return schema
        except Exception:
            pass
        if openint_datahub.exists() and str(openint_datahub) not in sys.path:
            sys.path.insert(0, str(openint_datahub))
        try:
            from schemas import get_dataset_schemas
            return get_dataset_schemas()
        except Exception:
            pass
    except Exception as e:
        logger.debug("Schema not available for semantic tagging: %s", e)
    return None


# Single source of truth for the 3 UI dropdown models: IDs + metadata for /api/semantic/models-with-meta
MODEL_METADATA = [
    {
        "id": "mukaj/fin-mpnet-base",
        "display_name": "Finance MPNet",
        "author": "mukaj",
        "description": "Fin-MPNET-Base: state-of-the-art for financial documents (79.91 FiQA). Fine-tuned from all-mpnet-base-v2 on 150k+ financial QA examples. Downloaded from Hugging Face; stored in Redis when the model registry is enabled.",
        "details": "768 dimensions · Fast · Use for banking/finance semantic search.",
        "url": "https://huggingface.co/mukaj/fin-mpnet-base",
    },
    {
        "id": "ProsusAI/finbert",
        "display_name": "FinBERT",
        "author": "Prosus AI",
        "description": "FinBERT: financial sentiment analysis. Pre-trained BERT fine-tuned on Financial PhraseBank. Downloaded from Hugging Face when this option is selected.",
        "details": "Finance domain · Sentiment (positive/negative/neutral) · Use for financial text understanding.",
        "url": "https://huggingface.co/ProsusAI/finbert",
    },
    {
        "id": "sentence-transformers/all-mpnet-base-v2",
        "display_name": "General MPNet",
        "author": "sentence-transformers",
        "description": "Popular, powerful open source model. Strong general-purpose embeddings (768d). Good balance of quality and speed.",
        "details": "768 dimensions · Popular · Use for general semantic tasks.",
        "url": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
    },
]
DROPDOWN_MODEL_IDS = [m["id"] for m in MODEL_METADATA]


class MultiModelSemanticAnalyzer:
    """
    Analyzes queries using multiple embedding models to extract semantic tags.
    Loads models via modelmgmt_agent.model_registry (Hugging Face + Redis).
    """

    DEFAULT_MODELS = DROPDOWN_MODEL_IDS

    def __init__(self, models: Optional[List[str]] = None, lazy_load: bool = False):
        if not EMBEDDING_AVAILABLE:
            raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
        self.models_to_use = models or self.DEFAULT_MODELS
        self.loaded_models: Dict[str, Any] = {}
        self._models_loaded = False
        if not lazy_load:
            logger.info("modelmgmt-agent: Loading embedding models for multi-model analysis (count=%s)", len(self.models_to_use))
            self._load_models()
        else:
            logger.info("modelmgmt-agent: Multi-model analyzer initialized (models will load on first use, count=%s)", len(self.models_to_use))

    def preload_all_models(self):
        if not self._models_loaded:
            logger.info("modelmgmt-agent: Preloading embedding models (count=%s)", len(self.models_to_use))
            self._load_models()

    def _load_models(self):
        if self._models_loaded:
            return
        # Suppress sentence-transformers/transformers progress bar (Loading weights: 100%)
        os.environ["TQDM_DISABLE"] = "1"
        try:
            import transformers
            transformers.logging.set_verbosity_error()
        except Exception:
            pass
        try:
            from modelmgmt_agent.model_registry import load_model_from_registry
            use_registry = True
        except ImportError:
            use_registry = False
        if use_registry:
            logger.info("modelmgmt-agent: Using Redis model registry (O(1) boot when scaling)")
        for model_name in self.models_to_use:
            try:
                logger.info("modelmgmt-agent: Loading embedding model: %s", model_name)
                if use_registry:
                    model = load_model_from_registry(model_name)
                    if model is not None:
                        self.loaded_models[model_name] = model
                if model_name not in self.loaded_models and SentenceTransformer:
                    self.loaded_models[model_name] = SentenceTransformer(model_name)
                if model_name in self.loaded_models:
                    dim = self.loaded_models[model_name].get_sentence_embedding_dimension()
                    logger.info("modelmgmt-agent: Embedding model loaded (dimension: %s). Model: %s", dim, model_name)
            except Exception as e:
                logger.warning("Failed to load embedding model, skipping", extra={"model": model_name, "error": str(e)})
        if not self.loaded_models:
            raise RuntimeError("No models could be loaded!")
        self._models_loaded = True
        logger.info("modelmgmt-agent: Loaded %s embedding model(s): %s", len(self.loaded_models), list(self.loaded_models.keys()))

    _NON_QUERY_TAG_TYPES = frozenset({"model", "embedding_norm", "embedding_dim", "embedding_peak"})

    def _highlight_query_with_tags(self, query: str, tags: List[Dict[str, Any]]) -> Dict[str, Any]:
        tag_positions = []
        for tag in tags:
            if tag.get("type") in self._NON_QUERY_TAG_TYPES:
                continue
            snippet = tag.get("snippet", "")
            if snippet:
                pos = query.lower().find(snippet.lower())
                if pos >= 0:
                    tag_positions.append({"tag": tag, "start": pos, "end": pos + len(snippet), "snippet": snippet})
        tag_positions.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
        seen_spans = set()
        unique_positions = []
        for tp in tag_positions:
            key = (tp["start"], tp["end"])
            if key in seen_spans:
                continue
            seen_spans.add(key)
            unique_positions.append(tp)
        unique_positions.sort(key=lambda x: x["start"])
        highlighted_segments = []
        last_end = 0
        for tp in unique_positions:
            start, end = tp["start"], tp["end"]
            if end <= last_end:
                continue
            clip_start = max(start, last_end)
            if clip_start >= end:
                continue
            if clip_start > last_end:
                highlighted_segments.append({"text": query[last_end:clip_start], "type": "text", "tag": None})
            highlighted_segments.append({
                "text": query[clip_start:end],
                "type": "highlight",
                "tag": tp["tag"],
                "tag_type": tp["tag"].get("type"),
                "label": tp["tag"].get("label"),
                "confidence": tp["tag"].get("confidence", 0.0),
            })
            last_end = end
        if last_end < len(query):
            highlighted_segments.append({"text": query[last_end:], "type": "text", "tag": None})
        return {
            "original_query": query,
            "highlighted_segments": highlighted_segments,
            "tag_count": len(tags),
            "highlighted_count": len([s for s in highlighted_segments if s["type"] == "highlight"]),
        }

    def _extract_schema_tags(
        self, query: str, model: Any, model_name: str, schema: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not schema or not hasattr(model, "encode"):
            return []
        phrases = []
        phrase_labels = {}
        for ds_name, meta in schema.items():
            ds_label = ds_name.replace("_", " ").title()
            phrases.append(ds_name.replace("_", " "))
            phrase_labels[phrases[-1]] = f"Dataset: {ds_label}"
            for f in meta.get("fields", [])[:20]:
                fname = f.get("name", "")
                if not fname:
                    continue
                readable = fname.replace("_", " ")
                if readable not in phrase_labels:
                    phrases.append(readable)
                    phrase_labels[readable] = f"Field: {readable.title()}"
        if not phrases:
            return []
        words = query.split()
        ngrams = []
        seen = set()
        for n in range(1, min(4, len(words) + 1)):
            for i in range(len(words) - n + 1):
                ng = " ".join(words[i : i + n])
                if ng.lower() not in seen and len(ng) >= 2:
                    seen.add(ng.lower())
                    ngrams.append(ng)
                if len(ngrams) >= 45:
                    break
            if len(ngrams) >= 45:
                break
        if not ngrams:
            return []
        try:
            emb_ng = model.encode(ngrams, convert_to_numpy=True)
            emb_ph = model.encode(phrases[:50], convert_to_numpy=True)
        except Exception as e:
            logger.debug("Schema-tag encode failed: %s", e)
            return []
        base_threshold = 0.38
        model_offset = (hash(model_name) % 80) / 100.0 * 0.12
        threshold = base_threshold + model_offset
        tags_out = []
        for idx, ng in enumerate(ngrams):
            a = emb_ng[idx]
            norms_ph = np.linalg.norm(emb_ph, axis=1)
            norms_ph = np.where(norms_ph == 0, 1e-9, norms_ph)
            sims = np.dot(emb_ph, a) / (np.linalg.norm(a) + 1e-9) / norms_ph
            best = int(np.argmax(sims))
            sim = float(sims[best])
            if sim >= threshold and query.lower().find(ng.lower()) >= 0:
                label = phrase_labels.get(phrases[best], phrases[best])
                tags_out.append({"type": "schema_field", "label": label, "value": phrases[best], "snippet": ng, "confidence": round(min(1.0, sim), 3)})
        return tags_out

    def _score_model_quality(self, model_result: Dict[str, Any]) -> float:
        if "error" in model_result:
            return 0.0
        score = 0.0
        score += min(len(model_result.get("tags", [])) * 0.1, 0.4)
        tags = model_result.get("tags", [])
        if tags:
            score += sum(t.get("confidence", 0.0) for t in tags) / len(tags) * 0.3
        score += min(len(model_result.get("detected_entities", [])) * 0.1, 0.2)
        score += min(len(model_result.get("detected_actions", [])) * 0.05, 0.1)
        if "fin" in (model_result.get("model") or "").lower():
            score += 0.1
        return min(score, 1.0)

    def _extract_semantic_tags(self, query: str, model_name: str, embedding: np.ndarray) -> Dict[str, Any]:
        query_lower = query.lower()
        tags = []
        amount_over = re.search(r"\b(?:over|above|greater\s+than|more\s+than)\s+\$?([\d,]+(?:\.\d+)?)\b", query, re.I)
        if amount_over:
            try:
                val = float(amount_over.group(1).replace(",", ""))
                tags.append({"type": "amount_min", "label": "Amount over", "value": val, "snippet": amount_over.group(0), "confidence": 0.9})
            except Exception:
                pass
        amount_under = re.search(r"\b(?:under|below|less\s+than)\s+\$?([\d,]+(?:\.\d+)?)\b", query, re.I)
        if amount_under:
            try:
                val = float(amount_under.group(1).replace(",", ""))
                tags.append({"type": "amount_max", "label": "Amount under", "value": val, "snippet": amount_under.group(0), "confidence": 0.9})
            except Exception:
                pass
        state_codes = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"]
        state_match = re.search(r"\b([A-Z]{2})\b", query)
        if state_match and state_match.group(1) in state_codes:
            tags.append({"type": "state", "label": "State", "value": state_match.group(1), "snippet": state_match.group(1), "confidence": 0.95})
        entity_keywords = {"customer": ["customer", "customers", "client", "clients", "account", "accounts"], "transaction": ["transaction", "transactions", "payment", "payments", "transfer", "transfers"], "dispute": ["dispute", "disputes", "chargeback", "chargebacks"], "location": ["state", "states", "zip", "zipcode", "city", "cities", "address"]}
        detected_entities = []
        for entity_type, keywords in entity_keywords.items():
            for keyword in keywords:
                if re.search(r"\b" + re.escape(keyword) + r"\b", query, re.I):
                    detected_entities.append(entity_type.title())
                    tags.append({"type": "entity", "label": "Entity Type", "value": entity_type.title(), "snippet": keyword, "confidence": 0.85})
                    break
        transaction_types = {"ach": ["ach"], "wire": ["wire", "wires"], "credit": ["credit", "credit card"], "debit": ["debit", "debit card"], "check": ["check", "checks"]}
        for trans_type, keywords in transaction_types.items():
            for keyword in keywords:
                if re.search(r"\b" + re.escape(keyword) + r"\b", query, re.I):
                    tags.append({"type": "transaction_type", "label": "Transaction Type", "value": trans_type.title(), "snippet": keyword, "confidence": 0.9})
                    break
        action_keywords = {"search": ["search", "find", "look", "show", "list"], "filter": ["filter", "where", "with"], "aggregate": ["count", "total", "sum", "average", "how many"], "sort": ["top", "largest", "biggest", "highest", "most"]}
        detected_actions = []
        for action_type, keywords in action_keywords.items():
            for keyword in keywords:
                if re.search(r"\b" + re.escape(keyword) + r"\b", query, re.I):
                    detected_actions.append(action_type.title())
                    tags.append({"type": "intent", "label": "Intent", "value": action_type.title(), "snippet": keyword, "confidence": 0.8})
                    break
        norm = float(np.linalg.norm(embedding))
        dim = len(embedding)
        peak_dim = int(np.argmax(np.abs(embedding)))
        embedding_stats = {"dimension": dim, "norm": norm, "mean": float(np.mean(embedding)), "std": float(np.std(embedding)), "min": float(np.min(embedding)), "max": float(np.max(embedding))}
        tags.append({"type": "embedding_norm", "label": "Embedding norm", "value": round(norm, 2), "snippet": str(round(norm, 2)), "confidence": 1.0})
        tags.append({"type": "embedding_dim", "label": "Embedding dim", "value": dim, "snippet": str(dim), "confidence": 1.0})
        tags.append({"type": "embedding_peak", "label": "Peak dim", "value": peak_dim, "snippet": str(peak_dim), "confidence": 1.0})
        for t in tags:
            base = t.get("confidence", 0.9)
            t["confidence"] = round(base * (0.92 + 0.08 * (norm % 100) / 100.0), 3)
        tags.append({"type": "model", "label": "Model", "value": model_name, "snippet": model_name, "confidence": 1.0})
        return {"model": model_name, "tags": tags, "detected_entities": detected_entities, "detected_actions": detected_actions, "embedding_stats": embedding_stats, "tag_count": len(tags)}

    def analyze_query(self, query: str, parallel: bool = True) -> Dict[str, Any]:
        if not self._models_loaded:
            logger.info("modelmgmt-agent: Loading embedding models for multi-model analysis (count=%s)", len(self.models_to_use))
            self._load_models()
        if not query or not query.strip():
            return {"query": query, "error": "Empty query", "models_analyzed": 0}
        schema = _get_schema_for_semantics()
        results = {}
        if parallel:
            with ThreadPoolExecutor(max_workers=min(len(self.loaded_models), 5)) as executor:
                future_to_model = {executor.submit(self._analyze_with_model, query, model_name, model, schema): model_name for model_name, model in self.loaded_models.items()}
                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        results[model_name] = future.result()
                    except Exception as e:
                        results[model_name] = {"model": model_name, "error": str(e), "tags": [], "embedding_stats": {}}
        else:
            for model_name, model in self.loaded_models.items():
                try:
                    results[model_name] = self._analyze_with_model(query, model_name, model, schema)
                except Exception as e:
                    results[model_name] = {"model": model_name, "error": str(e), "tags": [], "embedding_stats": {}}
        model_scores = {}
        for model_name, result in results.items():
            if "error" not in result:
                model_scores[model_name] = self._score_model_quality(result)
        best_model = max(model_scores.items(), key=lambda x: x[1])[0] if model_scores else None
        all_tags = []
        tag_counts = {}
        entity_counts = {}
        action_counts = {}
        for model_name, result in results.items():
            if "error" not in result:
                all_tags.extend(result.get("tags", []))
                for tag in result.get("tags", []):
                    tag_type = tag.get("type", "unknown")
                    tag_counts[tag_type] = tag_counts.get(tag_type, 0) + 1
                for entity in result.get("detected_entities", []):
                    entity_counts[entity] = entity_counts.get(entity, 0) + 1
                for action in result.get("detected_actions", []):
                    action_counts[action] = action_counts.get(action, 0) + 1
        tag_groups = {}
        for tag in all_tags:
            tag_key = f"{tag.get('type')}:{tag.get('value')}"
            if tag_key not in tag_groups:
                tag_groups[tag_key] = []
            tag_groups[tag_key].append(tag)
        consensus_tags = []
        for tag_key, tag_list in tag_groups.items():
            if len(tag_list) >= 2:
                avg_confidence = sum(t.get("confidence", 0) for t in tag_list) / len(tag_list)
                consensus_tags.append({**tag_list[0], "confidence": avg_confidence, "detected_by_models": len(tag_list), "models": [r.get("model", "unknown") for r in tag_list]})
        schema_assets = sorted(schema.keys()) if schema else []
        return {
            "query": query,
            "models_analyzed": len(results),
            "models": results,
            "model_scores": model_scores,
            "best_model": best_model,
            "best_model_score": model_scores.get(best_model, 0.0) if best_model else 0.0,
            "schema_assets": schema_assets,
            "aggregated": {"all_tags": all_tags, "consensus_tags": consensus_tags, "tag_counts": tag_counts, "entity_counts": entity_counts, "action_counts": action_counts, "total_tags": len(all_tags), "consensus_count": len(consensus_tags)},
            "summary": {"most_common_entity": max(entity_counts.items(), key=lambda x: x[1])[0] if entity_counts else None, "most_common_action": max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else None, "most_common_tag_type": max(tag_counts.items(), key=lambda x: x[1])[0] if tag_counts else None},
        }

    def _analyze_with_model(self, query: str, model_name: str, model: Any, schema: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        embedding = model.encode(query, convert_to_numpy=True)
        semantic_analysis = self._extract_semantic_tags(query, model_name, embedding)
        tags = list(semantic_analysis.get("tags", []))
        if schema:
            tags.extend(self._extract_schema_tags(query, model, model_name, schema))
        highlighted = self._highlight_query_with_tags(query, tags)
        semantic_annotation_time_ms = round((time.perf_counter() - t0) * 1000)
        return {
            **semantic_analysis,
            "tags": tags,
            "embedding_preview": embedding[:10].tolist(),
            "highlighted_query": highlighted,
            "semantic_annotation_time_ms": semantic_annotation_time_ms,
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {name: {"dimension": m.get_sentence_embedding_dimension(), "max_seq_length": getattr(m, "max_seq_length", None), "loaded": True} for name, m in self.loaded_models.items()}


_analyzer: Optional[MultiModelSemanticAnalyzer] = None
_analyzer_lock = False


def get_analyzer(models: Optional[List[str]] = None, force_reload: bool = False, preload: bool = False) -> Optional[MultiModelSemanticAnalyzer]:
    global _analyzer, _analyzer_lock
    if _analyzer is not None and not force_reload:
        if preload and not _analyzer._models_loaded:
            _analyzer.preload_all_models()
        return _analyzer
    if _analyzer_lock:
        time.sleep(0.1)
        if _analyzer and preload and not _analyzer._models_loaded:
            _analyzer.preload_all_models()
        return _analyzer
    try:
        _analyzer_lock = True
        if models is None:
            env_models = os.getenv("MULTI_MODEL_SEMANTIC_MODELS")
            models = [m.strip() for m in env_models.split(",")] if env_models else DROPDOWN_MODEL_IDS
        _analyzer = MultiModelSemanticAnalyzer(models=models, lazy_load=not preload)
        if preload and not _analyzer._models_loaded:
            _analyzer.preload_all_models()
        return _analyzer
    except Exception as e:
        logger.warning("Failed to initialize multi-model analyzer", extra={"error": str(e)})
        _analyzer = None
        return None
    finally:
        _analyzer_lock = False


def analyze_query_multi_model(query: str, models: Optional[List[str]] = None, parallel: bool = True) -> Dict[str, Any]:
    analyzer = get_analyzer(models=models)
    if analyzer is None:
        return {"query": query, "error": "Multi-model analyzer not available", "models_analyzed": 0}
    return analyzer.analyze_query(query, parallel=parallel)
