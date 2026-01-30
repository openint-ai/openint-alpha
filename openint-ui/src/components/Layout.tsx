import { useState, useRef, useEffect } from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import Logo from './Logo';

const PRODUCTS = [
  { path: '/chat', label: 'Vector DB Chat Demo' },
  { path: '/compare', label: 'Sentence Annotation Compare' },
  { path: '/a2a', label: 'A2A Demo' },
  { path: '/graph', label: 'Neo4J Graph Demo' },
] as const;

export default function Layout() {
  const loc = useLocation();
  const [productsOpen, setProductsOpen] = useState(false);
  const [docsOpen, setDocsOpen] = useState(false);
  const productsRef = useRef<HTMLDivElement>(null);
  const docsRef = useRef<HTMLDivElement>(null);

  const currentProduct = PRODUCTS.find((f) => f.path === loc.pathname);
  const isHelp = loc.pathname === '/help';
  const isDocs = loc.pathname.startsWith('/docs') || loc.pathname === '/help';
  const isDocsApi = loc.pathname === '/docs/api';
  const isDocsMcp = loc.pathname === '/docs/mcp';

  useEffect(() => {
    setProductsOpen(false);
    setDocsOpen(false);
  }, [loc.pathname]);

  useEffect(() => {
    if (!productsOpen) return;
    const onOutside = (e: MouseEvent) => {
      if (productsRef.current && !productsRef.current.contains(e.target as Node)) setProductsOpen(false);
    };
    document.addEventListener('mousedown', onOutside);
    return () => document.removeEventListener('mousedown', onOutside);
  }, [productsOpen]);

  useEffect(() => {
    if (!docsOpen) return;
    const onOutside = (e: MouseEvent) => {
      if (docsRef.current && !docsRef.current.contains(e.target as Node)) setDocsOpen(false);
    };
    document.addEventListener('mousedown', onOutside);
    return () => document.removeEventListener('mousedown', onOutside);
  }, [docsOpen]);

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <header className="border-b border-surface-200 bg-white/95 backdrop-blur-sm sticky top-0 z-30 shadow-soft">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 py-5 sm:py-6 flex items-center justify-between gap-6">
          <Logo />
          <nav className="flex items-center gap-1">
            <div className="relative" ref={productsRef}>
              <button
                type="button"
                onClick={() => setProductsOpen((o) => !o)}
                className={`inline-flex items-center gap-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  currentProduct ? 'bg-brand-100 text-brand-700' : 'text-gray-600 hover:text-gray-900 hover:bg-surface-100'
                }`}
                aria-expanded={productsOpen}
                aria-haspopup="true"
                aria-label="Products menu"
              >
                Products
                <svg className={`w-4 h-4 transition-transform ${productsOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {productsOpen && (
                <div className="absolute left-0 top-full mt-1 min-w-[220px] rounded-xl border border-surface-200 bg-white py-1 shadow-lg z-40">
                  {PRODUCTS.map(({ path, label }) => (
                    <Link
                      key={path}
                      to={path}
                      className={`block px-4 py-2.5 text-sm font-medium transition-colors ${
                        currentProduct?.path === path ? 'bg-brand-50 text-brand-700' : 'text-gray-700 hover:bg-surface-50'
                      }`}
                    >
                      {label}
                    </Link>
                  ))}
                </div>
              )}
            </div>
            <div className="relative" ref={docsRef}>
              <button
                type="button"
                onClick={() => setDocsOpen((o) => !o)}
                className={`inline-flex items-center gap-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isDocs ? 'bg-brand-100 text-brand-700' : 'text-gray-600 hover:text-gray-900 hover:bg-surface-100'
                }`}
                aria-expanded={docsOpen}
                aria-haspopup="true"
                aria-label="Documentation menu"
              >
                Documentation
                <svg className={`w-4 h-4 transition-transform ${docsOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {docsOpen && (
                <div className="absolute right-0 top-full mt-1 min-w-[180px] rounded-xl border border-surface-200 bg-white py-1 shadow-lg z-40">
                  <Link
                    to="/help"
                    className={`block px-4 py-2.5 text-sm font-medium transition-colors ${
                      isHelp ? 'bg-brand-50 text-brand-700' : 'text-gray-700 hover:bg-surface-50'
                    }`}
                  >
                    Help
                  </Link>
                  <Link
                    to="/docs/api"
                    className={`block px-4 py-2.5 text-sm font-medium transition-colors ${
                      isDocsApi ? 'bg-brand-50 text-brand-700' : 'text-gray-700 hover:bg-surface-50'
                    }`}
                  >
                    API
                  </Link>
                  <Link
                    to="/docs/mcp"
                    className={`block px-4 py-2.5 text-sm font-medium transition-colors ${
                      isDocsMcp ? 'bg-brand-50 text-brand-700' : 'text-gray-700 hover:bg-surface-50'
                    }`}
                  >
                    MCP Server
                  </Link>
                </div>
              )}
            </div>
          </nav>
        </div>
      </header>

      <main className={`flex-1 w-full mx-auto py-4 sm:py-6 md:py-8 ${loc.pathname === '/' ? 'max-w-full px-0' : loc.pathname === '/compare' || loc.pathname === '/a2a' || loc.pathname === '/graph' ? 'max-w-full px-2 sm:px-4' : loc.pathname === '/chat' ? 'max-w-full px-2 sm:px-4 md:px-6' : 'max-w-4xl px-2 sm:px-4 md:px-6'}`}>
        <Outlet />
      </main>
    </div>
  );
}
