/**
 * Parse answer text into segments for semantic highlighting (entity, amount, id, date, location).
 */

export type HighlightPart = { type: 'text' | 'entity' | 'amount' | 'id' | 'date' | 'location'; text: string };

const PATTERNS: { re: RegExp; type: HighlightPart['type'] }[] = [
  { re: /^-?\$?[\d,]+(?:\.\d{2})?\s*(?:USD|EUR)?/i, type: 'amount' },
  { re: /\bCUST\d+\b/i, type: 'id' },
  { re: /\b(?:ACH|WIRE|CREDIT|DEBIT|CHECK)\d+\b/i, type: 'id' },
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

export function getHighlightClass(type: string): string {
  switch (type) {
    case 'amount':
      return 'text-amber-800 font-medium';
    case 'id':
      return 'text-indigo-800 font-mono text-xs';
    case 'date':
      return 'text-emerald-800';
    case 'location':
      return 'text-blue-800';
    case 'entity':
      return 'text-brand-700 font-medium';
    default:
      return '';
  }
}
