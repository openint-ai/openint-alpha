"""
MCP server for the OpenInt semantic layer.

Exposes tools so any MCP-capable agent (Cursor, Claude Desktop, etc.) can:
- Interpret a sentence with a chosen embedding model (tags, highlighted segments, token semantics).
- List available models.

Uses the existing OpenInt backend HTTP API. Set OPENINT_BACKEND_URL (default http://localhost:3001).
Run with stdio for Cursor/Claude: python server.py
Or: uv run server.py
"""

import os
import urllib.parse
from contextlib import asynccontextmanager

# Apply before importing FastMCP so it uses the patched stdio_server.
# Avoid "Invalid JSON: EOF while parsing" when client sends bare newlines on stdio.
def _patch_mcp_stdio_skip_empty_lines():
    import sys
    from io import TextIOWrapper
    import anyio
    import anyio.lowlevel
    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
    import mcp.types as types
    from mcp.shared.message import SessionMessage
    import mcp.server.stdio as _stdio

    async def _patched_stdio_server(stdin=None, stdout=None):
        if not stdin:
            stdin = anyio.wrap_file(TextIOWrapper(sys.stdin.buffer, encoding="utf-8"))
        if not stdout:
            stdout = anyio.wrap_file(TextIOWrapper(sys.stdout.buffer, encoding="utf-8"))

        read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
        write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

        async def stdin_reader():
            try:
                async with read_stream_writer:
                    async for line in stdin:
                        if not line or not line.strip():
                            continue
                        try:
                            message = types.JSONRPCMessage.model_validate_json(line)
                        except Exception as exc:
                            await read_stream_writer.send(exc)
                            continue
                        session_message = SessionMessage(message)
                        await read_stream_writer.send(session_message)
            except anyio.ClosedResourceError:
                await anyio.lowlevel.checkpoint()

        async def stdout_writer():
            try:
                async with write_stream_reader:
                    async for session_message in write_stream_reader:
                        json_str = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
                        await stdout.write(json_str + "\n")
                        await stdout.flush()
            except anyio.ClosedResourceError:
                await anyio.lowlevel.checkpoint()

        async with anyio.create_task_group() as tg:
            tg.start_soon(stdin_reader)
            tg.start_soon(stdout_writer)
            yield read_stream, write_stream

    _stdio.stdio_server = asynccontextmanager(_patched_stdio_server)


_patch_mcp_stdio_skip_empty_lines()

import requests
from mcp.server.fastmcp import FastMCP

BACKEND_URL = os.environ.get("OPENINT_BACKEND_URL", "http://localhost:3001").rstrip("/")

mcp = FastMCP(
    "OpenInt Semantic Layer",
    json_response=True,
    instructions="Tools to interpret natural language sentences using the OpenInt semantic layer (embedding models). Use semantic_interpret for one model, semantic_interpret_all for all supported models at once, or semantic_list_models to see available model IDs.",
)


def _get(path: str, params: dict | None = None) -> dict:
    url = f"{BACKEND_URL}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def _post(path: str, json: dict) -> dict:
    r = requests.post(f"{BACKEND_URL}{path}", json=json, timeout=60)
    r.raise_for_status()
    return r.json()


@mcp.tool()
def semantic_interpret(
    sentence: str,
    model: str = "mukaj/fin-mpnet-base",
) -> dict:
    """
    Interpret a natural language sentence with the semantic layer.

    Returns tags (e.g. customer_id, state, intent), highlighted segments, token semantics,
    and optional embedding stats. Use this when you need to understand what entities and
    intents a sentence expresses for search, analytics, or downstream services.

    Args:
        sentence: The natural language sentence to interpret (e.g. "Show me transactions for customer 1001 in California").
        model: Embedding model ID. Default mukaj/fin-mpnet-base. Use semantic_list_models to see options.

    Returns:
        JSON with success, query, model, tags, highlighted_segments, token_semantics, embedding_stats.
    """
    if not sentence or not sentence.strip():
        return {
            "success": True,
            "query": sentence or "",
            "model": model,
            "tags": [],
            "highlighted_segments": [],
            "token_semantics": [],
            "embedding_stats": {},
        }
    try:
        return _get("/api/semantic/interpret", params={"sentence": sentence.strip(), "model": model or "mukaj/fin-mpnet-base"})
    except requests.RequestException as e:
        err = e.response.json() if e.response is not None and e.response.text else {}
        return {
            "success": False,
            "error": err.get("error", str(e)),
            "query": sentence,
            "model": model,
        }


@mcp.tool()
def semantic_interpret_all(sentence: str) -> dict:
    """
    Interpret a natural language sentence with all supported models (same as the UI dropdown).

    Returns one result per model: tags, highlighted_segments, token_semantics, embedding_stats for each.
    Use this when you need interpretation from every supported model in a single call.

    Args:
        sentence: The natural language sentence to interpret.

    Returns:
        JSON with success, query, models: { model_id: { success, query, model, tags, highlighted_segments, token_semantics, ... } }.
    """
    if not sentence or not sentence.strip():
        return {"success": True, "query": sentence or "", "models": {}}
    try:
        return _get("/api/semantic/interpret-all", params={"sentence": sentence.strip()})
    except requests.RequestException as e:
        err = e.response.json() if e.response is not None and e.response.text else {}
        return {
            "success": False,
            "error": err.get("error", str(e)),
            "query": sentence,
            "models": {},
        }


@mcp.tool()
def semantic_list_models() -> dict:
    """
    List available embedding model IDs for semantic interpretation.

    Returns the list of model IDs you can pass as the 'model' argument to semantic_interpret.
    """
    try:
        return _get("/api/semantic/models")
    except requests.RequestException as e:
        err = e.response.json() if e.response is not None and e.response.text else {}
        return {"success": False, "error": err.get("error", str(e)), "models": [], "count": 0}


@mcp.tool()
def semantic_list_models_with_meta() -> dict:
    """
    List supported models with metadata: id, display_name, author, description, details, url (Hugging Face).
    Use these ids as the 'model' parameter in semantic_interpret. Author and details are for documentation and display.
    """
    try:
        return _get("/api/semantic/models-with-meta")
    except requests.RequestException as e:
        err = e.response.json() if e.response is not None and e.response.text else {}
        return {"success": False, "error": err.get("error", str(e)), "models": [], "count": 0}


if __name__ == "__main__":
    mcp.run()
