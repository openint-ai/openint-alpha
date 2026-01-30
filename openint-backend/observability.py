"""
Observability: OpenTelemetry tracing + structured JSON logging.

- JSON log lines (one object per line) for log aggregators.
- Trace correlation: trace_id and span_id on each log record when in a span.
- Flask auto-instrumentation for HTTP spans.

Environment:
- OTEL_SERVICE_NAME: service name (default: openint-backend)
- OTEL_EXPORTER_OTLP_ENDPOINT: OTLP HTTP endpoint for traces (e.g. http://localhost:4318/v1/traces). If unset, traces go to console.
- LOG_JSON: set to 1 for JSON logs (default: 1 when OTEL_EXPORTER_OTLP_ENDPOINT is set, else 0 for dev-friendly)
- LOG_LEVEL: WARNING (default, log only issues) or INFO. DEBUG is disabled project-wide.
- OTEL_PYTHON_LOG_CORRELATION: set to true to add trace_id/span_id to log records (default: true)
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Optional OpenTelemetry - avoid hard fail if not installed
_otel_available = False
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.flask import FlaskInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    _otel_available = True
except ImportError:
    pass


LOG_JSON = os.environ.get("LOG_JSON", "").strip() in ("1", "true", "yes")
OTEL_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
# Default to JSON when exporting to OTLP (production-style)
if OTEL_ENDPOINT and not os.environ.get("LOG_JSON"):
    LOG_JSON = True


class JsonLogFormatter(logging.Formatter):
    """
    Emit one JSON object per log line. Observability-friendly: timestamp, level, message,
    logger name, and optional trace_id/span_id (from OpenTelemetry log instrumentation).
    """

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        # UTC ISO8601 for log aggregators
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        log_dict: Dict[str, Any] = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # OpenTelemetry LoggingInstrumentor adds these when OTEL_PYTHON_LOG_CORRELATION=true
        trace_id = getattr(record, "otelTraceID", None)
        span_id = getattr(record, "otelSpanID", None)
        if trace_id is not None:
            log_dict["trace_id"] = trace_id
        if span_id is not None:
            log_dict["span_id"] = span_id
        # Exception info
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)
        # Structured extra (e.g. logger.info("msg", extra={"query_id": "..."}))
        # Sanitize: never log full request/response JSON; skip or summarize only
        if self.include_extra:
            skip = {"name", "msg", "args", "created", "filename", "funcName", "levelname", "levelno", "lineno", "module", "msecs", "pathname", "process", "processName", "relativeCreated", "stack_info", "exc_info", "exc_text", "thread", "threadName", "message", "taskName", "otelTraceID", "otelSpanID", "otelServiceName", "otelTraceSampled"}
            # Never include keys that might hold full request/response body
            body_like_keys = frozenset({"request", "response", "body", "data", "json", "request_body", "response_body", "request_data", "response_data", "payload", "request_json", "response_json"})
            for k, v in record.__dict__.items():
                if k not in skip and v is not None and k.lower() not in body_like_keys:
                    log_dict[k] = _sanitize_log_value(v, max_str_len=100)
        line = json.dumps(log_dict, default=str)
        # Prevent dumping huge JSON: cap line length so logs stay readable
        max_log_line = 2000
        if len(line) > max_log_line:
            line = line[: max_log_line - 20] + " ... (truncated)"
        return line


# Keys that indicate API request/response payloads – never log full content
_RESPONSE_LIKE_KEYS = frozenset({"answer", "sources", "results", "debug", "multi_model_analysis", "annotation", "models", "query", "message", "error", "success", "artifacts", "status"})


def _sanitize_log_value(v: Any, max_str_len: int = 100) -> Any:
    """Avoid logging full JSON blobs or long strings; never log request/response bodies."""
    if v is None:
        return None
    if isinstance(v, bool) or isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        return v[:max_str_len] + ("..." if len(v) > max_str_len else "")
    if isinstance(v, (list, tuple)):
        n = len(v)
        if n <= 4 and all(isinstance(x, (str, int, float, bool)) or x is None for x in v):
            return [x if not isinstance(x, str) or len(x) <= max_str_len else x[:max_str_len] + "..." for x in v]
        return {"_type": "list", "len": n}
    if isinstance(v, dict):
        keys_set = set(v.keys())
        if keys_set & _RESPONSE_LIKE_KEYS and len(v) > 2:
            return {"_type": "api_payload", "keys": list(v.keys())[:12], "len": len(v)}
        n = len(v)
        if n > 8:
            return {"_type": "dict", "keys": list(v.keys())[:10], "len": n}
        out = {}
        for k, val in v.items():
            out[k] = _sanitize_log_value(val, max_str_len)
        return out
    try:
        s = str(v)
        return s[:max_str_len] + ("..." if len(s) > max_str_len else "")
    except Exception:
        return "<unrepresentable>"


def _install_logging_instrumentation() -> None:
    """Add trace_id/span_id to log records when inside a span (OTEL_PYTHON_LOG_CORRELATION)."""
    if not _otel_available:
        return
    try:
        # Ensures otelTraceID, otelSpanID are added to LogRecord when in a span
        os.environ.setdefault("OTEL_PYTHON_LOG_CORRELATION", "true")
        LoggingInstrumentor().instrument(set_logging_format=True)
    except Exception:
        pass


def _configure_logging() -> None:
    """Configure root logger. Default: WARNING. Debug logging is disabled project-wide (DEBUG is capped to INFO)."""
    root = logging.getLogger()
    level_name = (os.environ.get("LOG_LEVEL") or "WARNING").strip().upper()
    if level_name == "DEBUG":
        level_name = "INFO"  # Debug logging turned off project-wide
    level = getattr(logging, level_name, logging.WARNING)
    root.setLevel(level)
    # Remove existing handlers to avoid duplicate lines
    for h in list(root.handlers):
        root.removeHandler(h)

    if LOG_JSON:
        formatter = JsonLogFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Log file in backend dir (backend.log) – do not commit
    try:
        log_file = os.path.join(os.path.dirname(__file__), "backend.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except Exception:
        pass


class _NoopSpanExporter:
    """No-op span exporter: do not print spans to console (avoids any request/response in span attributes)."""
    def export(self, spans):
        pass
    def shutdown(self):
        pass


def _configure_tracing() -> None:
    """Configure OpenTelemetry TracerProvider."""
    if not _otel_available:
        return
    try:
        resource = Resource.create({
            SERVICE_NAME: os.environ.get("OTEL_SERVICE_NAME", "openint-backend"),
        })
        provider = TracerProvider(resource=resource)

        if OTEL_ENDPOINT:
            exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT)
        else:
            exporter = _NoopSpanExporter()

        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
    except Exception:
        pass


def setup(app=None) -> None:
    """
    Initialize observability: JSON logging, trace correlation, and Flask instrumentation.
    Call once at startup, before app.run().
    """
    _configure_logging()
    _install_logging_instrumentation()
    if _otel_available:
        _configure_tracing()
    if _otel_available and app is not None:
        try:
            FlaskInstrumentor().instrument_app(app)
        except Exception:
            pass


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module/component. Use for structured, observability-friendly logs."""
    return logging.getLogger(name)
