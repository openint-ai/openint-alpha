import { Component, type ErrorInfo, type ReactNode } from 'react';
import { Routes, Route } from 'react-router-dom';
import { getLogger } from './observability';
import { Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Marketing from './components/Marketing';
import SemanticCompare from './components/SemanticCompare';
import A2A from './components/A2A';
import Chat from './components/Chat';
import GraphDemo from './components/GraphDemo';
import Help from './components/Help';
import ApiDocs from './components/ApiDocs';
import McpDocs from './components/McpDocs';

const log = getLogger('App');

/** Catches render errors so the app shows a fallback instead of a blank page. */
class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    log.error('ErrorBoundary caught', {
      error: error.message,
      stack: error.stack,
      componentStack: info.componentStack,
    });
  }

  render() {
    if (this.state.error) {
      return (
        <div className="max-w-2xl mx-auto p-6 rounded-xl bg-white border border-red-200 shadow-lg">
          <h2 className="text-lg font-semibold text-red-800 mb-2">Something went wrong</h2>
          <p className="text-sm text-gray-700 mb-4">{this.state.error.message}</p>
          <button
            type="button"
            onClick={() => this.setState({ error: null })}
            className="px-4 py-2 rounded-lg bg-brand-500 text-white text-sm font-medium hover:bg-brand-600"
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Marketing />} />
        <Route path="/chat" element={<ErrorBoundary><Chat /></ErrorBoundary>} />
        <Route path="/compare" element={<SemanticCompare />} />
        <Route path="/a2a" element={<A2A />} />
        <Route path="/graph" element={<GraphDemo />} />
        <Route path="/search" element={<Navigate to="/compare" replace />} />
        <Route path="/help" element={<Help />} />
        <Route path="/docs/api" element={<ApiDocs />} />
        <Route path="/docs/mcp" element={<McpDocs />} />
        <Route path="/api-docs" element={<Navigate to="/docs/api" replace />} />
      </Route>
    </Routes>
  );
}
