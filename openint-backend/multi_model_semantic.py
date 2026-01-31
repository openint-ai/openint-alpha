"""
Multi-Model Semantic Analysis
Processes queries through multiple embedding models to understand and tag semantics
"""

import os
import re
import time
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

try:
    from observability import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")


def _get_schema_for_semantics() -> Optional[Dict[str, Dict[str, Any]]]:
    """Get dataset schema from DataHub (or openint-datahub/schemas.py) for schema-aware tagging. Returns None if unavailable."""
    try:
        import sys
        from pathlib import Path
        # Backend root = parent of openint-backend
        backend_dir = Path(__file__).resolve().parent
        repo_root = backend_dir.parent
        openint_agents = repo_root / "openint-agents"
        openint_datahub = repo_root / "openint-datahub"
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        if openint_agents.exists() and str(openint_agents) not in sys.path:
            sys.path.insert(0, str(openint_agents))
        try:
            from sg_agent.datahub_client import get_schema
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


class MultiModelSemanticAnalyzer:
    """
    Analyzes queries using multiple embedding models to extract semantic tags and understanding.
    """
    
    # Three models for multi-model semantic analysis (debug preview)
    DEFAULT_MODELS = [
        "mukaj/fin-mpnet-base",                      # Finance MPNet
        "ProsusAI/finbert",                          # FinBERT
        "sentence-transformers/all-mpnet-base-v2",   # General MPNet (popular, powerful open source)
    ]
    
    def __init__(self, models: Optional[List[str]] = None, lazy_load: bool = False):
        """
        Initialize multi-model semantic analyzer.
        
        Args:
            models: List of model names to use. If None, uses DEFAULT_MODELS.
            lazy_load: If True, don't load models immediately (load on first use)
        """
        if not EMBEDDING_AVAILABLE:
            raise ImportError(
                "sentence-transformers not available. Install with: pip install sentence-transformers"
            )
        
        self.models_to_use = models or self.DEFAULT_MODELS
        self.loaded_models: Dict[str, SentenceTransformer] = {}
        self._models_loaded = False
        
        if not lazy_load:
            logger.info("Loading embedding models for multi-model analysis", extra={"count": len(self.models_to_use)})
            self._load_models()
        else:
            logger.info("Multi-model analyzer initialized (models will load on first use)", extra={"model_count": len(self.models_to_use)})
    
    def preload_all_models(self):
        """
        Preload all models immediately, even if lazy_load was True.
        Useful for startup initialization.
        """
        if not self._models_loaded:
            logger.info("Preloading embedding models", extra={"count": len(self.models_to_use)})
            self._load_models()
    
    def _load_models(self):
        """Load all specified models (Redis registry first for O(1) boot when scaling)."""
        if self._models_loaded:
            return  # Already loaded
        try:
            import transformers
            transformers.logging.set_verbosity_error()
        except Exception:
            pass
        try:
            from model_registry import load_model_from_registry
            use_registry = True
        except ImportError:
            use_registry = False
        for model_name in self.models_to_use:
            try:
                logger.debug("Loading embedding model", extra={"model": model_name})
                if use_registry:
                    model = load_model_from_registry(model_name)
                    if model is not None:
                        self.loaded_models[model_name] = model
                if model_name not in self.loaded_models:
                    self.loaded_models[model_name] = SentenceTransformer(model_name)
                dim = self.loaded_models[model_name].get_sentence_embedding_dimension()
                logger.info("Loaded embedding model", extra={"model": model_name, "dimension": dim, "from_registry": use_registry})
            except Exception as e:
                logger.warning("Failed to load embedding model, skipping", extra={"model": model_name, "error": str(e)})
        if not self.loaded_models:
            raise RuntimeError("No models could be loaded!")
        self._models_loaded = True
        logger.info("Loaded embedding models", extra={"count": len(self.loaded_models), "models": list(self.loaded_models.keys())})
    
    # Tag types that are model metadata, not actual query spans — do not use for highlighting.
    # (embedding_norm/dim/peak and "model" have snippets like "0.42", "768", "mukaj/fin-mpnet-base"
    # which would wrongly highlight numbers or model names if they appear in the sentence.)
    _NON_QUERY_TAG_TYPES = frozenset({"model", "embedding_norm", "embedding_dim", "embedding_peak"})

    def _highlight_query_with_tags(self, query: str, tags: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create highlighted version of query showing semantic tags.
        Only tags whose snippet is a real substring of the query are used; model/embedding
        metadata tags are excluded so each model can show different highlights from its own tags.
        """
        # Only use tags that represent actual query spans (exclude model/embedding metadata)
        tag_positions = []
        for tag in tags:
            if tag.get("type") in self._NON_QUERY_TAG_TYPES:
                continue
            snippet = tag.get("snippet", "")
            if snippet:
                pos = query.lower().find(snippet.lower())
                if pos >= 0:
                    tag_positions.append({
                        "tag": tag,
                        "start": pos,
                        "end": pos + len(snippet),
                        "snippet": snippet
                    })
        
        # Sort by position (start, then longer spans first so we prefer more specific matches)
        tag_positions.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
        # Dedupe by (start, end) so the same span is not highlighted twice (e.g. "Transactions" from entity + schema)
        seen_spans: set = set()
        unique_positions = []
        for tp in tag_positions:
            key = (tp["start"], tp["end"])
            if key in seen_spans:
                continue
            seen_spans.add(key)
            unique_positions.append(tp)
        # Re-sort by start for segment building
        unique_positions.sort(key=lambda x: x["start"])

        # Build segments with no overlapping output: each character appears in exactly one segment.
        # If two tags overlap (e.g. "Transactions" 0:13 and "Transactions between" 0:24), we only
        # output the non-overlapping part of the later span so we never repeat text.
        highlighted_segments = []
        last_end = 0

        for tp in unique_positions:
            start, end = tp["start"], tp["end"]
            if end <= last_end:
                continue  # fully covered already
            clip_start = max(start, last_end)
            if clip_start >= end:
                continue
            # Text before this (possibly clipped) tag
            if clip_start > last_end:
                highlighted_segments.append({
                    "text": query[last_end:clip_start],
                    "type": "text",
                    "tag": None
                })
            # One highlight segment for the non-overlapping part only
            highlighted_segments.append({
                "text": query[clip_start:end],
                "type": "highlight",
                "tag": tp["tag"],
                "tag_type": tp["tag"].get("type"),
                "label": tp["tag"].get("label"),
                "confidence": tp["tag"].get("confidence", 0.0)
            })
            last_end = end
        
        # Add remaining text
        if last_end < len(query):
            highlighted_segments.append({
                "text": query[last_end:],
                "type": "text",
                "tag": None
            })
        
        return {
            "original_query": query,
            "highlighted_segments": highlighted_segments,
            "tag_count": len(tags),
            "highlighted_count": len([s for s in highlighted_segments if s["type"] == "highlight"])
        }

    def _extract_schema_tags(
        self,
        query: str,
        model: Any,
        model_name: str,
        schema: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Use DataHub schema for sentence annotation: match query n-grams to schema phrases
        (dataset names + field names) in this model's embedding space. The same schema is
        used for all models; only the *matches* (which n-gram → which phrase) differ per
        model. Asset/dataset list is returned separately (schema_assets) so the UI shows
        the same DataHub assets for every model; per-model tags are for matched fields only.
        """
        if not schema or not hasattr(model, "encode"):
            return []
        # Build phrases from schema: dataset names + field names (readable)
        phrases: List[str] = []
        phrase_labels: Dict[str, str] = {}  # phrase -> label for tag
        for ds_name, meta in schema.items():
            ds_label = ds_name.replace("_", " ").title()
            phrases.append(ds_name.replace("_", " "))
            phrase_labels[phrases[-1]] = f"Dataset: {ds_label}"
            for f in meta.get("fields", [])[:20]:  # limit fields per dataset
                fname = f.get("name", "")
                if not fname:
                    continue
                readable = fname.replace("_", " ")
                if readable not in phrase_labels:
                    phrases.append(readable)
                    phrase_labels[readable] = f"Field: {readable.title()}"
        if not phrases:
            return []
        # Overlapping word n-grams from query (1–3 words), max 45 to keep encode small
        words = query.split()
        ngrams: List[str] = []
        seen: set = set()
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
        # Cosine similarity; threshold varies by model so highlights differ per model
        base_threshold = 0.38
        model_offset = (hash(model_name) % 80) / 100.0 * 0.12  # 0.00–0.096
        threshold = base_threshold + model_offset
        tags_out: List[Dict[str, Any]] = []
        for idx, ng in enumerate(ngrams):
            a = emb_ng[idx]
            norms_ph = np.linalg.norm(emb_ph, axis=1)
            norms_ph = np.where(norms_ph == 0, 1e-9, norms_ph)
            sims = np.dot(emb_ph, a) / (np.linalg.norm(a) + 1e-9) / norms_ph
            best = int(np.argmax(sims))
            sim = float(sims[best])
            if sim >= threshold and query.lower().find(ng.lower()) >= 0:
                label = phrase_labels.get(phrases[best], phrases[best])
                tags_out.append({
                    "type": "schema_field",
                    "label": label,
                    "value": phrases[best],
                    "snippet": ng,
                    "confidence": round(min(1.0, sim), 3),
                })
        return tags_out

    def _score_model_quality(self, model_result: Dict[str, Any]) -> float:
        """
        Score a model's semantic analysis quality.
        Higher score = better model for this query.
        
        Args:
            model_result: Result from _analyze_with_model
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        if "error" in model_result:
            return 0.0
        
        score = 0.0
        
        # Base score from number of tags detected
        tag_count = len(model_result.get("tags", []))
        score += min(tag_count * 0.1, 0.4)  # Max 0.4 for tags
        
        # Score from confidence of tags
        tags = model_result.get("tags", [])
        if tags:
            avg_confidence = sum(t.get("confidence", 0.0) for t in tags) / len(tags)
            score += avg_confidence * 0.3  # Max 0.3 for confidence
        
        # Score from detected entities (important for banking queries)
        entities = model_result.get("detected_entities", [])
        score += min(len(entities) * 0.1, 0.2)  # Max 0.2 for entities
        
        # Score from detected actions
        actions = model_result.get("detected_actions", [])
        score += min(len(actions) * 0.05, 0.1)  # Max 0.1 for actions
        
        # Bonus for finance-specific model if it's available
        model_name = model_result.get("model", "")
        if "fin" in model_name.lower():
            score += 0.1
        
        return min(score, 1.0)
    
    def select_best_model(self, analysis_result: Dict[str, Any]) -> Optional[str]:
        """
        Select the best model based on semantic analysis quality.
        
        Args:
            analysis_result: Result from analyze_query
            
        Returns:
            Name of the best model, or None if no models available
        """
        if not analysis_result.get("models"):
            return None
        
        best_model = None
        best_score = -1.0
        
        for model_name, model_result in analysis_result["models"].items():
            score = self._score_model_quality(model_result)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model
    
    def _extract_semantic_tags(self, query: str, model_name: str, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Extract semantic tags from a query using pattern matching and heuristics.
        Tags that drive highlighting (snippet = query substring) are the same for every
        model (regex/heuristics), so without schema-based tags all models would highlight
        the same spans. embedding_norm/dim/peak and "model" differ per model but are
        excluded from highlighting (see _NON_QUERY_TAG_TYPES) since their snippet is
        metadata, not a query span. Per-model highlighting is achieved via _extract_schema_tags.
        """
        query_lower = query.lower()
        tags = []
        
        # Amount patterns
        amount_over = re.search(r"\b(?:over|above|greater\s+than|more\s+than)\s+\$?([\d,]+(?:\.\d+)?)\b", query, re.I)
        if amount_over:
            try:
                val = float(amount_over.group(1).replace(",", ""))
                tags.append({
                    "type": "amount_min",
                    "label": "Amount over",
                    "value": val,
                    "snippet": amount_over.group(0),
                    "confidence": 0.9
                })
            except:
                pass
        
        amount_under = re.search(r"\b(?:under|below|less\s+than)\s+\$?([\d,]+(?:\.\d+)?)\b", query, re.I)
        if amount_under:
            try:
                val = float(amount_under.group(1).replace(",", ""))
                tags.append({
                    "type": "amount_max",
                    "label": "Amount under",
                    "value": val,
                    "snippet": amount_under.group(0),
                    "confidence": 0.9
                })
            except:
                pass
        
        # State detection
        state_codes = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"]
        state_match = re.search(r"\b([A-Z]{2})\b", query)
        if state_match and state_match.group(1) in state_codes:
            tags.append({
                "type": "state",
                "label": "State",
                "value": state_match.group(1),
                "snippet": state_match.group(1),
                "confidence": 0.95
            })
        
        # Entity type keywords
        entity_keywords = {
            "customer": ["customer", "customers", "client", "clients", "account", "accounts"],
            "transaction": ["transaction", "transactions", "payment", "payments", "transfer", "transfers"],
            "dispute": ["dispute", "disputes", "chargeback", "chargebacks"],
            "location": ["state", "states", "zip", "zipcode", "city", "cities", "address"],
        }
        
        detected_entities = []
        for entity_type, keywords in entity_keywords.items():
            for keyword in keywords:
                if re.search(r"\b" + re.escape(keyword) + r"\b", query, re.I):
                    detected_entities.append(entity_type.title())
                    tags.append({
                        "type": "entity",
                        "label": "Entity Type",
                        "value": entity_type.title(),
                        "snippet": keyword,
                        "confidence": 0.85
                    })
                    break
        
        # Transaction types
        transaction_types = {
            "ach": ["ach"],
            "wire": ["wire", "wires"],
            "credit": ["credit", "credit card"],
            "debit": ["debit", "debit card"],
            "check": ["check", "checks"],
        }
        
        for trans_type, keywords in transaction_types.items():
            for keyword in keywords:
                if re.search(r"\b" + re.escape(keyword) + r"\b", query, re.I):
                    tags.append({
                        "type": "transaction_type",
                        "label": "Transaction Type",
                        "value": trans_type.title(),
                        "snippet": keyword,
                        "confidence": 0.9
                    })
                    break
        
        # Intent detection
        action_keywords = {
            "search": ["search", "find", "look", "show", "list"],
            "filter": ["filter", "where", "with"],
            "aggregate": ["count", "total", "sum", "average", "how many"],
            "sort": ["top", "largest", "biggest", "highest", "most"],
        }
        
        detected_actions = []
        for action_type, keywords in action_keywords.items():
            for keyword in keywords:
                if re.search(r"\b" + re.escape(keyword) + r"\b", query, re.I):
                    detected_actions.append(action_type.title())
                    tags.append({
                        "type": "intent",
                        "label": "Intent",
                        "value": action_type.title(),
                        "snippet": keyword,
                        "confidence": 0.8
                    })
                    break
        
        # Calculate embedding statistics (model-specific, so results differ per model)
        norm = float(np.linalg.norm(embedding))
        dim = len(embedding)
        peak_dim = int(np.argmax(np.abs(embedding)))
        embedding_stats = {
            "dimension": dim,
            "norm": norm,
            "mean": float(np.mean(embedding)),
            "std": float(np.std(embedding)),
            "min": float(np.min(embedding)),
            "max": float(np.max(embedding))
        }
        # Embedding-derived tags: different per model so the Compare UI shows distinct values
        tags.append({
            "type": "embedding_norm",
            "label": "Embedding norm",
            "value": round(norm, 2),
            "snippet": str(round(norm, 2)),
            "confidence": 1.0
        })
        tags.append({
            "type": "embedding_dim",
            "label": "Embedding dim",
            "value": dim,
            "snippet": str(dim),
            "confidence": 1.0
        })
        tags.append({
            "type": "embedding_peak",
            "label": "Peak dim",
            "value": peak_dim,
            "snippet": str(peak_dim),
            "confidence": 1.0
        })
        # Vary tag confidence slightly by model (using embedding norm) so same rule-based
        # tags have different confidence per model
        for t in tags:
            base = t.get("confidence", 0.9)
            t["confidence"] = round(base * (0.92 + 0.08 * (norm % 100) / 100.0), 3)
        # Model-identifying tag so each model's output is clearly distinct
        tags.append({
            "type": "model",
            "label": "Model",
            "value": model_name,
            "snippet": model_name,
            "confidence": 1.0
        })
        return {
            "model": model_name,
            "tags": tags,
            "detected_entities": detected_entities,
            "detected_actions": detected_actions,
            "embedding_stats": embedding_stats,
            "tag_count": len(tags)
        }
    
    def analyze_query(self, query: str, parallel: bool = True) -> Dict[str, Any]:
        """
        Analyze a query using all loaded models.
        
        Args:
            query: The query text to analyze
            parallel: If True, process models in parallel (faster)
            
        Returns:
            Dictionary with analysis results from all models
        """
        # Lazy load models if not already loaded
        if not self._models_loaded:
            logger.info("Loading embedding models for multi-model analysis", extra={"count": len(self.models_to_use)})
            self._load_models()
        
        if not query or not query.strip():
            return {
                "query": query,
                "error": "Empty query",
                "models_analyzed": 0
            }

        schema = _get_schema_for_semantics()
        results = {}

        if parallel:
            # Process models in parallel (each gets same schema; per-model embedding produces different tags)
            with ThreadPoolExecutor(max_workers=min(len(self.loaded_models), 5)) as executor:
                future_to_model = {
                    executor.submit(self._analyze_with_model, query, model_name, model, schema): model_name
                    for model_name, model in self.loaded_models.items()
                }

                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result = future.result()
                        results[model_name] = result
                    except Exception as e:
                        results[model_name] = {
                            "model": model_name,
                            "error": str(e),
                            "tags": [],
                            "embedding_stats": {}
                        }
        else:
            # Process models sequentially
            for model_name, model in self.loaded_models.items():
                try:
                    result = self._analyze_with_model(query, model_name, model, schema)
                    results[model_name] = result
                except Exception as e:
                    results[model_name] = {
                        "model": model_name,
                        "error": str(e),
                        "tags": [],
                        "embedding_stats": {}
                    }
        
        # Score each model
        model_scores = {}
        for model_name, result in results.items():
            if "error" not in result:
                model_scores[model_name] = self._score_model_quality(result)
        
        # Select best model
        best_model = None
        if model_scores:
            best_model = max(model_scores.items(), key=lambda x: x[1])[0]
        
        # Aggregate results across models
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
        
        # Find consensus tags (detected by multiple models)
        consensus_tags = []
        tag_groups = {}
        for tag in all_tags:
            tag_key = f"{tag.get('type')}:{tag.get('value')}"
            if tag_key not in tag_groups:
                tag_groups[tag_key] = []
            tag_groups[tag_key].append(tag)
        
        for tag_key, tag_list in tag_groups.items():
            if len(tag_list) >= 2:  # Detected by at least 2 models
                # Average confidence
                avg_confidence = sum(t.get("confidence", 0) for t in tag_list) / len(tag_list)
                consensus_tags.append({
                    **tag_list[0],
                    "confidence": avg_confidence,
                    "detected_by_models": len(tag_list),
                    "models": [r.get("model", "unknown") for r in tag_list]
                })
        
        # Same schema (and thus same asset list) is used for all models; expose for UI so "Asset" is consistent
        schema_assets = sorted(schema.keys()) if schema else []

        return {
            "query": query,
            "models_analyzed": len(results),
            "models": results,
            "model_scores": model_scores,
            "best_model": best_model,
            "best_model_score": model_scores.get(best_model, 0.0) if best_model else 0.0,
            "schema_assets": schema_assets,
            "aggregated": {
                "all_tags": all_tags,
                "consensus_tags": consensus_tags,
                "tag_counts": tag_counts,
                "entity_counts": entity_counts,
                "action_counts": action_counts,
                "total_tags": len(all_tags),
                "consensus_count": len(consensus_tags)
            },
            "summary": {
                "most_common_entity": max(entity_counts.items(), key=lambda x: x[1])[0] if entity_counts else None,
                "most_common_action": max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else None,
                "most_common_tag_type": max(tag_counts.items(), key=lambda x: x[1])[0] if tag_counts else None,
            }
        }
    
    def _analyze_with_model(
        self,
        query: str,
        model_name: str,
        model: SentenceTransformer,
        schema: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Analyze query with a specific model. Returns result including semantic_annotation_time_ms."""
        t0 = time.perf_counter()
        embedding = model.encode(query, convert_to_numpy=True)
        semantic_analysis = self._extract_semantic_tags(query, model_name, embedding)
        tags = list(semantic_analysis.get("tags", []))
        # Per-model schema tags so each model highlights different spans (embedding space differs)
        if schema:
            schema_tags = self._extract_schema_tags(query, model, model_name, schema)
            tags.extend(schema_tags)
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
        """Get information about loaded models."""
        info = {}
        for model_name, model in self.loaded_models.items():
            info[model_name] = {
                "dimension": model.get_sentence_embedding_dimension(),
                "max_seq_length": getattr(model, 'max_seq_length', None),
                "loaded": True
            }
        return info


# Global instance (lazy-loaded)
_analyzer: Optional[MultiModelSemanticAnalyzer] = None
_analyzer_lock = False  # Simple flag to prevent concurrent initialization


def get_analyzer(models: Optional[List[str]] = None, force_reload: bool = False, preload: bool = False) -> Optional[MultiModelSemanticAnalyzer]:
    """
    Get or create the global multi-model semantic analyzer instance.
    Uses lazy loading by default - models are only loaded when first needed.
    Set preload=True to load all models immediately.
    
    Args:
        models: Optional list of models to use (only used on first call)
        force_reload: If True, reload models even if already loaded
        preload: If True, load all models immediately (useful for startup)
        
    Returns:
        MultiModelSemanticAnalyzer instance, or None if models can't be loaded
    """
    global _analyzer, _analyzer_lock
    
    # Return cached analyzer if available and not forcing reload
    if _analyzer is not None and not force_reload:
        # If preload requested and models not loaded, load them now
        if preload and not _analyzer._models_loaded:
            _analyzer.preload_all_models()
        return _analyzer
    
    # Prevent concurrent initialization
    if _analyzer_lock:
        # Wait a bit and return None or existing analyzer
        import time
        time.sleep(0.1)
        if _analyzer and preload and not _analyzer._models_loaded:
            _analyzer.preload_all_models()
        return _analyzer
    
    try:
        _analyzer_lock = True
        
        # Allow override via environment variable
        env_models = os.getenv("MULTI_MODEL_SEMANTIC_MODELS")
        if env_models:
            models = env_models.split(",")
        
        # Use lazy_load=False if preload is True
        _analyzer = MultiModelSemanticAnalyzer(models=models, lazy_load=not preload)
        
        # If lazy_load was True but preload requested, load now
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
    """
    Convenience function to analyze a query with multiple models.
    
    Args:
        query: The query text to analyze
        models: Optional list of models to use
        parallel: If True, process models in parallel
        
    Returns:
        Dictionary with analysis results, or error dict if analyzer unavailable
    """
    analyzer = get_analyzer(models=models)
    if analyzer is None:
        return {
            "query": query,
            "error": "Multi-model analyzer not available",
            "models_analyzed": 0
        }
    return analyzer.analyze_query(query, parallel=parallel)
