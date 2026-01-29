/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_OTEL_SERVICE_NAME?: string;
  readonly VITE_OTEL_EXPORTER_OTLP_ENDPOINT?: string;
  readonly VITE_LOG_JSON?: string;
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
