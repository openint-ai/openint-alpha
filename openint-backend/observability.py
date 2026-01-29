"""
Observability: OpenTelemetry tracing + structured JSON logging.

- JSON log lines (one object per line) for log aggregators.
- Trace correlation: trace_id and span_id on each log record when in a span.
- Flask auto-instrumentation for HTTP spans.

Environment:
- OTEL_SERVICE_NAME: service name (default: openint-backend)
- OTEL_EXPORTER_OTLP_ENDPOINT: OTLP HTTP endpoint for traces (e.g. http://localhost:4318/v1/traces). If unset, traces go to console.
- LOG_JSON: set to 1 for JSON logs (default: 1 when OTEL_EXPORTER_OTLP_ENDPOINT is set, else 0 for dev-friendly)
- LOG_LEVEL: WARNING (default, log only issues), INFO, or DEBUG for verbose.
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
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
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
        if self.include_extra:
            skip = {"name", "msg", "args", "created", "filename", "funcName", "levelname", "levelno", "lineno", "module", "msecs", "pathname", "process", "processName", "relativeCreated", "stack_info", "exc_info", "exc_text", "thread", "threadName", "message", "taskName", "otelTraceID", "otelSpanID", "otelServiceName", "otelTraceSampled"}
            for k, v in record.__dict__.items():
                if k not in skip and v is not None:
                    try:
                        json.dumps(v)
                    except (TypeError, ValueError):
                        v = str(v)
                    log_dict[k] = v
        return json.dumps(log_dict, default=str)


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
    """Configure root logger with JSON or human-friendly format. Default: WARNING (log only issues)."""
    root = logging.getLogger()
    level_name = (os.environ.get("LOG_LEVEL") or "WARNING").strip().upper()
    level = getattr(logging, level_name, logging.WARNING)
    root.setLevel(level)
    # Remove existing handlers to avoid duplicate lines
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if LOG_JSON:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s"
            )
        )
    root.addHandler(handler)


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
            exporter = ConsoleSpanExporter()

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
