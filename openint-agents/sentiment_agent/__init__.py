"""
Sentiment Agent: analyzes sentiment/tone of a sentence or question via LLM.
Used in the multi-agent demo to display sentiment with icon after sa-agent produces the sentence.
"""

from .sentiment_analyzer import analyze_sentence_sentiment

__all__ = ["analyze_sentence_sentiment"]
