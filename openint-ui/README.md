# OpenInt UI

React frontend for OpenInt system - user interface for interacting with AI agents.

## Structure

```
openint-ui/
├── src/
│   ├── components/ # React components
│   ├── api/        # API client
│   └── App.tsx     # Main app
├── public/         # Static assets
└── package.json
```

## Setup

```bash
cd openint-ui
npm install
npm run dev
```

## Configuration

The frontend connects to the backend API. Set the API URL in your environment:

- `VITE_API_URL`: Backend API URL (default: http://localhost:3001)

Or update `src/api/` files to point to your backend.

## Observability (OpenTelemetry + structured logging)

The UI uses **structured JSON logging** and **OpenTelemetry tracing** for observability.

- **Logging**: One JSON object per log line (`timestamp`, `level`, `logger`, `message`, optional `trace_id`/`span_id`, and `extra`). Use `VITE_LOG_JSON=1` for JSON; in production build JSON is default.
- **Tracing**: Fetch requests are auto-instrumented. Traces are sent to an OTLP endpoint or printed to console.
- **Correlation**: Logs include `trace_id` and `span_id` when inside a span so they can be correlated with backend traces.

Environment variables (Vite prefix `VITE_`):

- `VITE_OTEL_SERVICE_NAME`: Service name (default: `openint-ui`)
- `VITE_OTEL_EXPORTER_OTLP_ENDPOINT`: OTLP HTTP endpoint (e.g. `http://localhost:4318/v1/traces`). If unset, traces log to console.
- `VITE_LOG_JSON`: Set to `1` for JSON log lines

## Development

```bash
# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Backend Integration

The UI communicates with `openint-backend` via REST API. Ensure the backend is running before starting the frontend.
