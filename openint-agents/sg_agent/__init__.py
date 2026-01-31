"""
Sentence Generation Agent (sg-agent).
Connects to DataHub (or uses local schema), reads dataset schemas, and uses the best
available generative model to produce example sentences that analysts, customer care,
and business users would ask in a banking context.
"""

from sg_agent.schema_generator_agent import SchemaGeneratorAgent

__all__ = ["SchemaGeneratorAgent"]
