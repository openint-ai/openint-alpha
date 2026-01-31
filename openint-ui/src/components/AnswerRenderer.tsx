/**
 * AnswerRenderer: Renders LLM answers with semantic highlighting.
 * - Highlights amounts, IDs, dates, locations using clean CSS colors
 * - Adds info icon (i) next to IDs; click opens popup with graph DB details
 * - Uses tables only for clear markdown pipe-style tables (| a | b | c |)
 * - Renders everything else as prose with **bold** support
 */

import { useState, useCallback } from 'react';
import { parseForHighlight, getHighlightClass, type HighlightPart } from '../utils/semanticHighlight';
import { fetchGraphEnrich, type GraphEnrichResponse } from '../api';

type Block = { type: 'paragraph'; text: string } | { type: 'table'; rows: string[][] };

/** Parse answer into ordered blocks: paragraphs and pipe-separated tables. */
function parseBlocks(text: string): Block[] {
  const lines = text.split(/\n/);
  const blocks: Block[] = [];
  let i = 0;
  let paraBuffer: string[] = [];

  const flushPara = () => {
    const s = paraBuffer.join('\n').trim();
    if (s) blocks.push({ type: 'paragraph', text: s });
    paraBuffer = [];
  };

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (trimmed.includes('|')) {
      const cells = trimmed.split(/\|/).map((c) => c.trim()).filter((c) => c !== '');
      const isSeparator = cells.every((c) => /^[\s\-:]+$/.test(c));

      if (cells.length >= 2 && !isSeparator) {
        flushPara();
        const tableRows: string[][] = [cells];
        i++;
        while (i < lines.length) {
          const nextLine = lines[i];
          const nextTrimmed = nextLine.trim();
          if (!nextTrimmed) break;
          const nextCells = nextTrimmed.split(/\|/).map((c) => c.trim()).filter((c) => c !== '');
          const nextIsSep = nextCells.every((c) => /^[\s\-:]+$/.test(c));
          if (nextIsSep) {
            i++;
            continue;
          }
          if (nextCells.length >= 2 && nextCells.length === cells.length) {
            tableRows.push(nextCells);
            i++;
          } else break;
        }
        if (tableRows.length >= 2) blocks.push({ type: 'table', rows: tableRows });
        continue;
      }
    }

    paraBuffer.push(line);
    i++;
  }
  flushPara();
  return blocks;
}

/** Modern Customer icon - user silhouette */
function CustomerIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="8" r="3.5" />
      <path d="M5 20c0-3.5 3.1-6 7-6s7 2.5 7 6" />
    </svg>
  );
}

/** Modern Transaction icon - banknote / money transfer */
function TransactionIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="2" y="6" width="20" height="12" rx="2" />
      <path d="M12 12h.01" strokeWidth="2" />
      <path d="M7 12h.01" strokeWidth="2" />
      <path d="M17 12h.01" strokeWidth="2" />
    </svg>
  );
}

/** Modern Dispute icon - flag / alert */
function DisputeIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" />
      <path d="M4 22v-7" />
    </svg>
  );
}

/** Generic info icon - when entity type is unknown */
function GenericEntityIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5L12 3z" />
    </svg>
  );
}

/** Sparkle icon for enriched entities */
function SparkleIcon({ className, title }: { className?: string; title?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      {title && <title>{title}</title>}
      <path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5L12 3z" />
    </svg>
  );
}

/** Entity-type icon button - Customer, Transaction, or Dispute based on label. Sparkle when enriched. */
function EntityIconButton({
  id,
  label,
  onClick,
  enrichedDetails,
  isEnriched,
}: {
  id: string;
  label?: string | null;
  onClick: () => void;
  enrichedDetails?: Record<string, { label?: string; display_name?: string }>;
  isEnriched?: boolean;
}) {
  const resolvedLabel = label ?? (enrichedDetails && getLabelForId(id, enrichedDetails));
  const l = (resolvedLabel ?? '').toLowerCase();
  const isCustomer = l === 'customer';
  const isTransaction = l === 'transaction';
  const isDispute = l === 'dispute';
  const bgClass = isCustomer ? 'bg-emerald-100 text-emerald-600 hover:bg-emerald-200' : isTransaction ? 'bg-amber-100 text-amber-600 hover:bg-amber-200' : isDispute ? 'bg-rose-100 text-rose-600 hover:bg-rose-200' : 'bg-cyan-100 text-cyan-600 hover:bg-cyan-200';
  const ringClass = isCustomer ? 'focus:ring-emerald-400' : isTransaction ? 'focus:ring-amber-400' : isDispute ? 'focus:ring-rose-400' : 'focus:ring-cyan-400';
  const Icon = isCustomer ? CustomerIcon : isTransaction ? TransactionIcon : isDispute ? DisputeIcon : GenericEntityIcon;
  const displayName = enrichedDetails && getDisplayNameForId(id, enrichedDetails);
  const title = displayName || (isCustomer ? 'Customer' : isTransaction ? 'Transaction' : isDispute ? 'Dispute' : 'View entity details');
  return (
    <span className="inline-flex items-center gap-0.5">
      <button
        type="button"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          onClick();
        }}
        className={`inline-flex align-middle items-center justify-center w-5 h-5 rounded-full ${bgClass} focus:outline-none focus:ring-2 ${ringClass} focus:ring-offset-1 transition-all duration-200 cursor-pointer hover:scale-110 shadow-sm ${isEnriched ? 'ring-2 ring-cyan-300 ring-offset-0.5' : ''}`}
        aria-label={title}
        title={isEnriched ? `✨ ${title} — enriched` : title}
      >
        <Icon className="w-3 h-3" />
      </button>
      {isEnriched && (
        <SparkleIcon className="w-3 h-3 text-cyan-500 sparkle-twinkle" title="Enriched by graph lookup" />
      )}
    </span>
  );
}

function getLabelForId(id: string, enrichedDetails: Record<string, { label?: string }>): string | undefined {
  if (enrichedDetails[id]?.label) return enrichedDetails[id].label;
  const digits = id.replace(/\D/g, '');
  if (digits) {
    const n = parseInt(digits, 10);
    if (!Number.isNaN(n)) {
      const normalized = String(1000000000 + (n % 1000000000));
      return enrichedDetails[normalized]?.label;
    }
  }
  return undefined;
}

function getDisplayNameForId(id: string, enrichedDetails: Record<string, { display_name?: string }>): string | undefined {
  if (enrichedDetails[id]?.display_name) return enrichedDetails[id].display_name;
  const digits = id.replace(/\D/g, '');
  if (digits) {
    const n = parseInt(digits, 10);
    if (!Number.isNaN(n)) {
      const normalized = String(1000000000 + (n % 1000000000));
      return enrichedDetails[normalized]?.display_name;
    }
  }
  return undefined;
}

/** Popup modal for ID details from graph DB */
function IdDetailsPopup({
  open,
  onClose,
  id,
  data,
  loading,
  error,
}: {
  open: boolean;
  onClose: () => void;
  id: string;
  data: GraphEnrichResponse | null;
  loading: boolean;
  error: string | null;
}) {
  if (!open) return null;
  const props = data?.properties ?? {};
  const entries = Object.entries(props).filter(([, v]) => v != null && v !== '');
  return (
    <div
      className="fixed inset-0 z-[200] flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm answer-reveal"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="id-details-title"
    >
      <div
        className="bg-white rounded-xl shadow-2xl max-w-md w-full max-h-[85vh] overflow-hidden flex flex-col border border-slate-200 answer-reveal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-200 bg-gradient-to-r from-slate-50 to-slate-100">
          <h2 id="id-details-title" className="flex items-center gap-2 text-sm font-semibold text-slate-800 min-w-0">
            {data?.label && (() => {
              const l = data.label.toLowerCase();
              const Icon = l === 'customer' ? CustomerIcon : l === 'transaction' ? TransactionIcon : l === 'dispute' ? DisputeIcon : null;
              const bg = l === 'customer' ? 'bg-emerald-500' : l === 'transaction' ? 'bg-amber-500' : l === 'dispute' ? 'bg-rose-500' : 'bg-slate-500';
              return Icon ? (
                <span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-lg ${bg} text-white`}>
                  <Icon className="w-3.5 h-3.5" />
                </span>
              ) : null;
            })()}
            <span className="truncate">{data?.label ? `${data.label} — ${id}` : `Entity details — ${id}`}</span>
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-500 hover:text-slate-700 p-1 rounded focus:outline-none focus:ring-2 focus:ring-indigo-400"
            aria-label="Close"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="p-5 overflow-y-auto flex-1">
          {loading && (
            <p className="text-sm text-slate-500">Loading details for {id}…</p>
          )}
          {error && !loading && (
            <p className="text-sm text-rose-600">{error}</p>
          )}
          {data?.success && entries.length > 0 && !loading && (
            <dl className="space-y-3">
              {entries.map(([key, value]) => (
                <div key={key} className="flex flex-col gap-0.5">
                  <dt className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">{key}</dt>
                  <dd className="text-sm text-slate-800 font-mono break-all">
                    {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                  </dd>
                </div>
              ))}
            </dl>
          )}
          {data?.success && entries.length === 0 && !loading && (
            <p className="text-sm text-slate-500">No additional properties found.</p>
          )}
        </div>
      </div>
    </div>
  );
}

/** Normalize ID to 10-digit form (matches backend: 10^9 + n % 10^9). */
function normalizeIdForCompare(id: string): string {
  const digits = id.replace(/\D/g, '');
  if (!digits) return id;
  const n = parseInt(digits, 10);
  if (Number.isNaN(n)) return id;
  return String(1000000000 + (n % 1000000000));
}

/** Check if id matches any in enriched set (normalize for comparison). */
function isEnriched(id: string, enrichedIds?: Set<string>): boolean {
  if (!enrichedIds || enrichedIds.size === 0) return false;
  if (enrichedIds.has(id)) return true;
  const normalized = normalizeIdForCompare(id);
  return enrichedIds.has(normalized);
}

/** Renders a single line with semantic highlights, **bold**, and entity-type icons. */
function LineWithHighlights({
  line,
  className = '',
  onIdClick,
  enrichedIds,
  enrichedDetails,
}: {
  line: string;
  className?: string;
  onIdClick: (id: string, label?: string) => void;
  enrichedIds?: Set<string>;
  enrichedDetails?: Record<string, { label?: string; display_name?: string }>;
}) {
  const safeLine = line != null && typeof line === 'string' ? line : '';
  const segments: { text: string; bold: boolean }[] = [];
  let remaining = safeLine;
  let idx: number;
  while ((idx = remaining.indexOf('**')) >= 0) {
    if (idx > 0) segments.push({ text: remaining.slice(0, idx), bold: false });
    const end = remaining.indexOf('**', idx + 2);
    if (end < 0) {
      segments.push({ text: remaining.slice(idx), bold: false });
      break;
    }
    segments.push({ text: remaining.slice(idx + 2, end), bold: true });
    remaining = remaining.slice(end + 2);
  }
  if (remaining.length > 0) segments.push({ text: remaining, bold: false });

  return (
    <span className={className}>
      {segments.map((seg, si) => {
        const parts = parseForHighlight(seg.text);
        const content = parts.map((p: HighlightPart, i: number) => {
          // Avoid duplication: when ID appears inside enriched display format, do NOT expand again
          // Customer: "Name (ID: X, mobile: ...)" | Transaction/Dispute: "prefix (X)"
          const textBefore = parts.slice(0, i).map((x) => x.text).join('');
          const isInsideDisplayName =
            textBefore.endsWith(' (ID: ') ||
            textBefore.endsWith(' (');
          const shouldExpand =
            !isInsideDisplayName &&
            isEnriched(p.text, enrichedIds) &&
            enrichedDetails &&
            getDisplayNameForId(p.text, enrichedDetails);
          return p.type === 'text' ? (
            <span key={i}>{p.text}</span>
          ) : p.type === 'id' ? (
            <span key={i} className="inline-flex items-baseline gap-0.5">
              <span
                className={`inline-flex items-center gap-1 ${getHighlightClass(p.type)} ${isEnriched(p.text, enrichedIds) ? 'enriched-id-highlight border-cyan-300 bg-gradient-to-r from-cyan-50 to-sky-50 shadow-sm' : ''} transition-all duration-200 hover:shadow-md`}
                title={isEnriched(p.text, enrichedIds) ? '✨ Enriched by enrich-agent — click to view details' : 'Click to view entity details'}
              >
                {shouldExpand ? getDisplayNameForId(p.text, enrichedDetails!) : p.text}
              </span>
              <EntityIconButton
                id={p.text}
                onClick={() => onIdClick(p.text, enrichedDetails ? getLabelForId(p.text, enrichedDetails) : undefined)}
                enrichedDetails={enrichedDetails}
                isEnriched={isEnriched(p.text, enrichedIds)}
              />
            </span>
          ) : (
            <span key={i} className={getHighlightClass(p.type)}>
              {p.text}
            </span>
          );
        });
        return seg.bold ? (
          <strong key={si} className="font-semibold text-slate-900">
            {content}
          </strong>
        ) : (
          <span key={si}>{content}</span>
        );
      })}
    </span>
  );
}

/** Semantic styling for table cells (light background for important values). */
function getCellClass(value: string): string {
  const isAmount = /^-?\$?[\d,]+(\.\d{2})?/.test(value) || /^-?[\d,]+\.\d{2}$/.test(value);
  const isUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value);
  const isId = isUuid || /^\d{9,19}$/.test(value) || /^CUST|^ACH|^WIRE|^CHK|^CRD|^DBT|^DSP/i.test(value);
  const isDate = /^\d{4}-\d{2}-\d{2}/.test(value);
  if (isAmount) return 'text-amber-800 font-semibold';
  if (isId) return 'text-indigo-800 font-mono text-xs';
  if (isDate) return 'text-emerald-800';
  return '';
}

/** Table cell with entity icons - for table cells that contain IDs */
function TableCellWithIds({
  cell,
  onIdClick,
  enrichedIds,
  enrichedDetails,
}: {
  cell: string;
  onIdClick: (id: string, label?: string) => void;
  enrichedIds?: Set<string>;
  enrichedDetails?: Record<string, { label?: string; display_name?: string }>;
}) {
  return (
    <td className={`px-3 py-2 text-slate-800 break-words ${getCellClass(cell)}`}>
      <LineWithHighlights line={cell} onIdClick={onIdClick} enrichedIds={enrichedIds} enrichedDetails={enrichedDetails} />
    </td>
  );
}

export interface AnswerRendererProps {
  text: string;
  className?: string;
  /** IDs that enrich-agent looked up (for highlight/badge) */
  enrichedIds?: Set<string> | string[];
  /** Per-ID enrich details: { label } for entity-type icons */
  enrichedDetails?: Record<string, { label?: string; display_name?: string }>;
}

/** Renders answer as prose with highlights; IDs get entity-type icon; enriched IDs get highlight; tables supported. */
export function AnswerRenderer({ text, className = '', enrichedIds, enrichedDetails }: AnswerRendererProps) {
  const safeText = text != null && typeof text === 'string' ? text : '';
  const blocks = parseBlocks(safeText);
  const [popup, setPopup] = useState<{
    open: boolean;
    id: string;
    data: GraphEnrichResponse | null;
    loading: boolean;
    error: string | null;
  }>({ open: false, id: '', data: null, loading: false, error: null });

  const handleIdClick = useCallback((id: string, label?: string) => {
    setPopup({ open: true, id, data: null, loading: true, error: null });
    fetchGraphEnrich(id, label)
      .then((data) => {
        setPopup((p) => ({
          ...p,
          data,
          loading: false,
          error: data.success ? null : (data.error ?? 'Not found'),
        }));
      })
      .catch((err) => {
        setPopup((p) => ({
          ...p,
          loading: false,
          error: err instanceof Error ? err.message : 'Failed to fetch details',
        }));
      });
  }, []);

  const closePopup = useCallback(() => {
    setPopup((p) => ({ ...p, open: false }));
  }, []);

  const enrichedSet = enrichedIds instanceof Set
    ? enrichedIds
    : Array.isArray(enrichedIds)
      ? new Set(enrichedIds)
      : undefined;

  return (
    <>
      <div className={`space-y-4 text-slate-800 leading-relaxed answer-reveal ${className}`}>
        {blocks.map((block, bi) =>
          block.type === 'paragraph' ? (
            <div key={bi} className="whitespace-pre-wrap">
              {block.text.split(/\n/).map((line, j) => (
                <div key={j} className={j > 0 ? 'mt-1' : ''}>
                  <LineWithHighlights line={line} onIdClick={handleIdClick} enrichedIds={enrichedSet} enrichedDetails={enrichedDetails} />
                </div>
              ))}
            </div>
          ) : (
            <div
              key={bi}
              className="w-full overflow-x-auto rounded-xl border border-slate-200 bg-gradient-to-br from-slate-50 to-white shadow-sm my-4 answer-reveal overflow-hidden"
            >
              <table className="w-full border-collapse text-sm min-w-0">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-100/80">
                    {block.rows[0].map((cell, ci) => (
                      <th
                        key={ci}
                        className="px-4 py-3 text-left text-xs font-semibold text-slate-700 uppercase tracking-wider whitespace-nowrap"
                      >
                        {cell}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {block.rows.slice(1).map((row, ri) => (
                    <tr key={ri} className={`${ri % 2 === 0 ? 'bg-white' : 'bg-slate-50/60'} hover:bg-slate-50/80 transition-colors`}>
                      {row.map((cell, ci) => (
                        <TableCellWithIds key={ci} cell={cell} onIdClick={handleIdClick} enrichedIds={enrichedSet} enrichedDetails={enrichedDetails} />
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        )}
      </div>
      <IdDetailsPopup
        open={popup.open}
        onClose={closePopup}
        id={popup.id}
        data={popup.data}
        loading={popup.loading}
        error={popup.error}
      />
    </>
  );
}
