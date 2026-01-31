# Sentiment Agent

Analyzes the sentiment/tone of a sentence or question via LLM (Ollama).

## Flow

Runs after **sg-agent** in the multi-agent demo:
1. User types a sentence or sg-agent generates one
2. sg-agent may fix/improve the sentence
3. **sentiment-agent** analyzes the final sentence and returns sentiment, confidence, and emoji

## Usage

```python
from sentiment_agent.sentiment_analyzer import analyze_sentence_sentiment

sentiment, confidence, emoji, reasoning, error = analyze_sentence_sentiment("Show me disputes over $1000")
# e.g. ("neutral and analytical", 0.85, "🤔", "The question uses neutral language and asks for data...", None)
```

## A2A

Exposed as an A2A agent at `/api/a2a/agents/sentiment-agent`. Send the sentence as message text; returns Task with `sentiment`, `confidence`, `emoji`, and optional `reasoning` (why this sentiment was detected) in the artifact.

## Configuration

Uses `OLLAMA_HOST` and `OLLAMA_MODEL` (default: qwen2.5:7b).
