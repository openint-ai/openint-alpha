import { useState, useRef, useEffect } from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import Logo from './Logo';

/** Icons for nav dropdowns (24x24, stroke 2). */
const IconChat = () => (
  <svg className="w-5 h-5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
  </svg>
);
const IconCompare = () => (
  <svg className="w-5 h-5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
  </svg>
);
const IconA2A = () => (
  <svg className="w-5 h-5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M17 1l4 4-4 4" /><path d="M3 11V9a4 4 0 0 1 4-4h14" /><path d="M7 23l-4-4 4-4" /><path d="M21 13v2a4 4 0 0 1-4 4H3" />
  </svg>
);
const IconGraph = () => (
  <svg className="w-5 h-5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <circle cx="12" cy="12" r="3" /><path d="M12 2v4M12 18v4M2 12h4M18 12h4" /><path d="m4.93 4.93 2.83 2.83m8.48 8.48 2.83 2.83M4.93 19.07l2.83-2.83m8.48-8.48 2.83-2.83" />
  </svg>
);
const IconMultiAgent = () => (
  <svg className="w-5 h-5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <rect x="3" y="11" width="18" height="10" rx="2" /><circle cx="12" cy="5" r="3" />
  </svg>
);
const IconHelp = () => (
  <svg className="w-5 h-5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <circle cx="12" cy="12" r="10" /><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" /><path d="M12 17h.01" />
  </svg>
);
const IconApi = () => (
  <svg className="w-5 h-5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><path d="M14 2v6h6" /><path d="M16 13H8" /><path d="M16 17H8" /><path d="M10 9H8" />
  </svg>
);
const IconMcp = () => (
  <svg className="w-5 h-5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <rect x="2" y="2" width="20" height="8" rx="2" /><rect x="2" y="14" width="20" height="8" rx="2" /><path d="M6 6v.01M6 18v.01M18 6v.01M18 18v.01" />
  </svg>
);

const PRODUCTS = [
  { path: '/chat', label: 'Vector DB Chat Demo', Icon: IconChat },
  { path: '/compare', label: 'Sentence Annotation Compare', Icon: IconCompare },
  { path: '/a2a', label: 'A2A Demo', Icon: IconA2A },
  { path: '/graph', label: 'Neo4J Graph Demo', Icon: IconGraph },
  { path: '/multi-agent', label: 'Multi-Agent Demo', Icon: IconMultiAgent },
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
                  {PRODUCTS.map(({ path, label, Icon }) => (
                    <Link
                      key={path}
                      to={path}
                      className={`flex items-center gap-3 px-4 py-2.5 text-sm font-medium transition-colors ${
                        currentProduct?.path === path ? 'bg-brand-50 text-brand-700' : 'text-gray-700 hover:bg-surface-50'
                      }`}
                    >
                      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-gray-100 text-gray-600" aria-hidden>
                        <Icon />
                      </span>
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
                    className={`flex items-center gap-3 px-4 py-2.5 text-sm font-medium transition-colors ${
                      isHelp ? 'bg-brand-50 text-brand-700' : 'text-gray-700 hover:bg-surface-50'
                    }`}
                  >
                    <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-gray-100 text-gray-600" aria-hidden>
                      <IconHelp />
                    </span>
                    Help
                  </Link>
                  <Link
                    to="/docs/api"
                    className={`flex items-center gap-3 px-4 py-2.5 text-sm font-medium transition-colors ${
                      isDocsApi ? 'bg-brand-50 text-brand-700' : 'text-gray-700 hover:bg-surface-50'
                    }`}
                  >
                    <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-gray-100 text-gray-600" aria-hidden>
                      <IconApi />
                    </span>
                    API
                  </Link>
                  <Link
                    to="/docs/mcp"
                    className={`flex items-center gap-3 px-4 py-2.5 text-sm font-medium transition-colors ${
                      isDocsMcp ? 'bg-brand-50 text-brand-700' : 'text-gray-700 hover:bg-surface-50'
                    }`}
                  >
                    <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-gray-100 text-gray-600" aria-hidden>
                      <IconMcp />
                    </span>
                    MCP Server
                  </Link>
                </div>
              )}
            </div>
          </nav>
        </div>
      </header>

      <main className={`flex-1 w-full mx-auto py-4 sm:py-6 md:py-8 ${loc.pathname === '/' ? 'max-w-full px-0' : loc.pathname === '/compare' || loc.pathname === '/a2a' || loc.pathname === '/graph' || loc.pathname === '/multi-agent' ? 'max-w-full px-2 sm:px-4' : loc.pathname === '/chat' ? 'max-w-full px-2 sm:px-4 md:px-6' : 'max-w-4xl px-2 sm:px-4 md:px-6'}`}>
        <Outlet />
      </main>
    </div>
  );
}
