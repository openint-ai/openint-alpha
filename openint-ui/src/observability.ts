/**
 * Frontend observability: OpenTelemetry tracing + structured logging.
 *
 * - Tracing: WebTracerProvider, fetch auto-instrumentation, OTLP or console export.
 * - Logging: Structured log records (JSON) with timestamp, level, message, logger,
 *   optional trace_id/span_id for correlation with backend traces.
 *
 * Environment (Vite: import.meta.env):
 * - VITE_OTEL_SERVICE_NAME: service name (default: openint-ui)
 * - VITE_OTEL_EXPORTER_OTLP_ENDPOINT: OTLP HTTP endpoint (e.g. http://localhost:4318/v1/traces). If unset, traces log to console.
 * - VITE_LOG_JSON: set to "1" for JSON log lines (default: 1 in production build)
 */

import { trace } from '@opentelemetry/api';
import {
  WebTracerProvider,
  SimpleSpanProcessor,
  ConsoleSpanExporter,
} from '@opentelemetry/sdk-trace-web';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { ZoneContextManager } from '@opentelemetry/context-zone';
import { FetchInstrumentation } from '@opentelemetry/instrumentation-fetch';
import { registerInstrumentations } from '@opentelemetry/instrumentation';

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogRecord {
  timestamp: string;
  level: LogLevel;
  logger: string;
  message: string;
  trace_id?: string;
  span_id?: string;
  [key: string]: unknown;
}

const LOG_JSON =
  typeof import.meta !== 'undefined' &&
  (import.meta.env?.VITE_LOG_JSON === '1' || import.meta.env?.MODE === 'production');
const OTEL_ENDPOINT =
  typeof import.meta !== 'undefined' ? (import.meta.env?.VITE_OTEL_EXPORTER_OTLP_ENDPOINT as string) || '' : '';
const SERVICE_NAME =
  typeof import.meta !== 'undefined' ? (import.meta.env?.VITE_OTEL_SERVICE_NAME as string) || 'openint-ui' : 'openint-ui';

function getTraceContext(): { trace_id?: string; span_id?: string } {
  try {
    const span = trace.getActiveSpan();
    if (!span) return {};
    const ctx = span.spanContext();
    return { trace_id: ctx.traceId, span_id: ctx.spanId };
  } catch {
    return {};
  }
}

function emitLog(level: LogLevel, logger: string, message: string, extra: Record<string, unknown> = {}): void {
  const record: LogRecord = {
    timestamp: new Date().toISOString(),
    level,
    logger,
    message,
    ...getTraceContext(),
    ...extra,
  };
  const line = LOG_JSON ? JSON.stringify(record) : `[${record.timestamp}] ${record.level.toUpperCase()} [${logger}] ${message}`;
  switch (level) {
    case 'debug':
      console.debug(line);
      break;
    case 'info':
      console.info(line);
      break;
    case 'warn':
      console.warn(line);
      break;
    case 'error':
      console.error(line);
      break;
    default:
      console.log(line);
  }
}

export interface Logger {
  debug(message: string, extra?: Record<string, unknown>): void;
  info(message: string, extra?: Record<string, unknown>): void;
  warn(message: string, extra?: Record<string, unknown>): void;
  error(message: string, extra?: Record<string, unknown>): void;
}

export function getLogger(name: string): Logger {
  return {
    debug: (message: string, extra?: Record<string, unknown>) => emitLog('debug', name, message, extra),
    info: (message: string, extra?: Record<string, unknown>) => emitLog('info', name, message, extra),
    warn: (message: string, extra?: Record<string, unknown>) => emitLog('warn', name, message, extra),
    error: (message: string, extra?: Record<string, unknown>) => emitLog('error', name, message, extra),
  };
}

let initialized = false;

/** No-op span exporter so we don't log "Object" to console when OTLP is not configured. */
class NoopSpanExporter {
  export(_spans: unknown, resultCallback: (result: { code: number }) => void) {
    resultCallback({ code: 0 });
  }
  shutdown() {
    return Promise.resolve();
  }
}

export function initObservability(): void {
  if (initialized) return;
  try {
    const exporter = OTEL_ENDPOINT
      ? new OTLPTraceExporter({
          url: OTEL_ENDPOINT.endsWith('/v1/traces') ? OTEL_ENDPOINT : `${OTEL_ENDPOINT}/v1/traces`,
        })
      : new NoopSpanExporter();

    const provider = new WebTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    provider.register({ contextManager: new ZoneContextManager() });

    registerInstrumentations({
      instrumentations: [new FetchInstrumentation({})],
    });

    initialized = true;
    getLogger('observability').info('OpenTelemetry tracing initialized', {
      service: SERVICE_NAME,
      otlp: !!OTEL_ENDPOINT,
    });
  } catch (e) {
    getLogger('observability').warn('OpenTelemetry init skipped', { error: String(e) });
  }
}

export function getTracer() {
  return trace.getTracer(SERVICE_NAME, '1.0.0');
}
