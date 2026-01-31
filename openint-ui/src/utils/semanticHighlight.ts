/**
 * Parse answer text into segments for semantic highlighting (entity, amount, id, date, location).
 */

export type HighlightPart = { type: 'text' | 'entity' | 'amount' | 'id' | 'date' | 'location'; text: string };

const PATTERNS: { re: RegExp; type: HighlightPart['type'] }[] = [
  { re: /^-?\$?[\d,]+(?:\.\d{2})?\s*(?:USD|EUR|GBP)?/i, type: 'amount' },
  { re: /\b\d{10,}\b/, type: 'id' }, // 10+ digit IDs (customer_id, transaction_id, dispute_id)
  { re: /\bCUST\d+\b/i, type: 'id' },
  { re: /\b(?:ACH|WIRE|CREDIT|DEBIT|CHECK|DBT|DSP|DISPUTE)\d+\b/i, type: 'id' },
  { re: /\b\d{4}-\d{2}-\d{2}\b/, type: 'date' },
  { re: /\b(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC)\b/, type: 'location' },
  { re: /\b\d{5}(?:-\d{4})?\b/, type: 'location' },
];

export function parseForHighlight(text: string): HighlightPart[] {
  const parts: HighlightPart[] = [];
  const safeText = text != null && typeof text === 'string' ? text : '';
  let remaining = safeText;
  while (remaining.length > 0) {
    let best: { index: number; length: number; type: HighlightPart['type'] } | null = null;
    for (const { re, type } of PATTERNS) {
      const m = remaining.match(re);
      if (m && m.index !== undefined && (best === null || m.index < best.index)) {
        best = { index: m.index, length: m[0].length, type };
      }
    }
    if (best === null) {
      parts.push({ type: 'text', text: remaining });
      break;
    }
    if (best.index > 0) {
      parts.push({ type: 'text', text: remaining.slice(0, best.index) });
    }
    parts.push({ type: best.type, text: remaining.slice(best.index, best.index + best.length) });
    remaining = remaining.slice(best.index + best.length);
  }
  return parts;
}

/** Clean CSS highlight colors for answer text (amounts, IDs, dates, locations). */
export function getHighlightClass(type: string): string {
  switch (type) {
    case 'amount':
      return 'text-amber-700 font-semibold bg-amber-50/90 px-1 py-0.5 rounded';
    case 'id':
      return 'text-indigo-700 font-mono text-[13px] bg-indigo-50/80 px-1 py-0.5 rounded';
    case 'date':
      return 'text-emerald-700 bg-emerald-50/80 px-1 py-0.5 rounded';
    case 'location':
      return 'text-blue-700 bg-blue-50/80 px-1 py-0.5 rounded';
    case 'entity':
      return 'text-slate-800 font-medium';
    default:
      return '';
  }
}
