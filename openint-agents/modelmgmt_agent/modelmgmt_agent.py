"""
modelmgmt-agent: Model Management Agent.

Downloads embedding models from Hugging Face, stores them in Redis for in-memory lookup,
and annotates sentences with semantic tags. Works with sg-agent: sg-agent generates
sentences from DataHub schema + LLM; modelmgmt-agent annotates those (and any) sentences.
"""

import sys
import os
import logging
from typing import Dict, List, Any, Optional

_AGENTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENTS_ROOT not in sys.path:
    sys.path.insert(0, _AGENTS_ROOT)

from agents.base_agent import BaseAgent, AgentResponse
from communication.agent_registry import AgentCapability

from modelmgmt_agent.semantic_analyzer import get_analyzer, analyze_query_multi_model

logger = logging.getLogger(__name__)


class ModelMgmtAgent(BaseAgent):
    """
    Model management agent: loads embedding models from Hugging Face (via Redis when available),
    stores them in Redis for fast lookup, and annotates sentences with semantic tags.
    Capability: semantic_annotate (single or multi-model).
    """

    def __init__(self):
        capabilities = [
            AgentCapability(
                name="semantic_annotate",
                description="Annotate a sentence with semantic tags using one or all embedding models (Hugging Face + Redis)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "sentence": {"type": "string", "description": "Sentence to annotate"},
                        "model": {"type": "string", "description": "Optional model ID; if omitted, use all models"},
                        "parallel": {"type": "boolean", "default": True},
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "query": {"type": "string"},
                        "models": {"type": "object"},
                        "best_model": {"type": "string"},
                        "schema_assets": {"type": "array"},
                    },
                },
            ),
        ]
        super().__init__(
            name="modelmgmt-agent",
            description="Model management agent: downloads models from Hugging Face, stores in Redis, annotates sentences with tags and highlighted segments",
            capabilities=capabilities,
        )

    def process_query(self, query: str, context: Dict[str, Any] = None) -> AgentResponse:
        """
        Process a sentence-annotation request.
        Query is the sentence to annotate; context can have "model" (single model) or use all models.
        """
        context = context or {}
        sentence = (query or "").strip()
        if not sentence:
            return AgentResponse(
                success=False,
                results=[],
                message="Empty sentence",
                metadata={"file_type": "semantic_annotation"},
            )
        single_model = context.get("model", "").strip()
        parallel = context.get("parallel", True)
        try:
            self.update_status("BUSY")
            if single_model:
                analyzer = get_analyzer()
                if not analyzer:
                    return AgentResponse(
                        success=False,
                        results=[],
                        message="Analyzer not available",
                        metadata={"file_type": "semantic_annotation"},
                    )
                if not analyzer._models_loaded:
                    analyzer.preload_all_models()
                if single_model not in analyzer.loaded_models:
                    from modelmgmt_agent.model_registry import load_model_from_registry
                    model = load_model_from_registry(single_model)
                    if model is not None:
                        analyzer.loaded_models[single_model] = model
                if single_model not in analyzer.loaded_models:
                    return AgentResponse(
                        success=False,
                        results=[],
                        message=f"Could not load model {single_model}",
                        metadata={"file_type": "semantic_annotation"},
                    )
                from modelmgmt_agent.semantic_analyzer import _get_schema_for_semantics
                schema = _get_schema_for_semantics()
                result = analyzer._analyze_with_model(sentence, single_model, analyzer.loaded_models[single_model], schema)
                analysis = {
                    "success": True,
                    "query": sentence,
                    "models": {single_model: result},
                    "best_model": single_model,
                    "schema_assets": sorted(schema.keys()) if schema else [],
                }
            else:
                analysis = analyze_query_multi_model(sentence, parallel=parallel)
                if "error" in analysis:
                    return AgentResponse(
                        success=False,
                        results=[],
                        message=analysis.get("error", "Analysis failed"),
                        metadata={"file_type": "semantic_annotation"},
                    )
                analysis["success"] = True
            return AgentResponse(
                success=True,
                results=[{"content": "", "metadata": {"file_type": "semantic_annotation", "analysis": analysis}}],
                message="OK",
                metadata={"file_type": "semantic_annotation", "analysis": analysis},
            )
        except Exception as e:
            logger.exception("modelmgmt-agent annotation failed")
            return AgentResponse(
                success=False,
                results=[],
                message=str(e),
                metadata={"file_type": "semantic_annotation"},
            )
        finally:
            self.update_status("IDLE")
