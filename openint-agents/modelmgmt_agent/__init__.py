"""
modelmgmt-agent: Model Management Agent.

Downloads embedding models from Hugging Face, stores them in Redis for fast in-memory
lookup, and annotates sentences with semantic tags (entities, intents, schema fields).
Works with sg-agent: sg-agent generates sentences from DataHub schema + LLM;
modelmgmt-agent annotates those (and any) sentences.
"""

from modelmgmt_agent.modelmgmt_agent import ModelMgmtAgent  # noqa: F401

__all__ = ["ModelMgmtAgent"]
