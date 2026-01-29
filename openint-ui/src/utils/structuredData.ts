/**
 * Parse and format Milvus result content for human-readable display.
 * Handles "Structured Data: {...}" JSON (including truncated payloads).
 */

/** Human-readable labels for known field names (snake_case → Title Case) */
export const FIELD_LABELS: Record<string, string> = {
  customer_id: 'Customer ID',
  transaction_id: 'Transaction ID',
  first_name: 'First name',
  last_name: 'Last name',
  state_code: 'State',
  state_name: 'State name',
  zip_code: 'ZIP code',
  country_code: 'Country code',
  country_name: 'Country',
  account_number: 'Account number',
  routing_number: 'Routing number',
  transaction_date: 'Date',
  transaction_datetime: 'Date & time',
  transaction_type: 'Type',
  amount: 'Amount',
  currency: 'Currency',
  status: 'Status',
  description: 'Description',
  ach_code: 'ACH code',
  wire_type: 'Wire type',
  beneficiary_name: 'Beneficiary',
  card_type: 'Card',
  merchant_name: 'Merchant',
  merchant_category: 'Category',
  check_number: 'Check number',
  payee_name: 'Payee',
  email: 'Email',
  phone: 'Phone',
  street_address: 'Address',
  city: 'City',
  region: 'Region',
  timezone: 'Timezone',
  customer_type: 'Customer type',
  account_status: 'Account status',
  account_opened_date: 'Account opened',
  credit_score: 'Credit score',
  date_of_birth: 'Date of birth',
  created_at: 'Created',
  updated_at: 'Updated',
};

/** Section titles and field order by entity type for readable grouping */
const SECTION_ORDER: Record<string, { section?: string; order: string[] }> = {
  customers: {
    section: 'Customer',
    order: [
      'customer_id', 'first_name', 'last_name', 'email', 'phone',
      'customer_type', 'account_status', 'credit_score', 'account_opened_date',
      'street_address', 'city', 'state_code', 'zip_code', 'country_code',
      'date_of_birth', 'created_at', 'updated_at',
    ],
  },
  ach_transactions: {
    section: 'Transaction',
    order: [
      'transaction_id', 'customer_id', 'transaction_type', 'amount', 'currency',
      'transaction_date', 'transaction_datetime', 'status', 'description',
      'ach_code', 'routing_number', 'account_number', 'created_at',
    ],
  },
  wire_transactions: {
    section: 'Transaction',
    order: [
      'transaction_id', 'customer_id', 'amount', 'currency', 'transaction_date',
      'status', 'description', 'wire_type', 'beneficiary_name',
      'routing_number', 'account_number', 'created_at',
    ],
  },
  credit_transactions: {
    section: 'Transaction',
    order: [
      'transaction_id', 'customer_id', 'amount', 'currency', 'transaction_date',
      'status', 'description', 'card_type', 'merchant_name', 'merchant_category',
      'created_at',
    ],
  },
  debit_transactions: {
    section: 'Transaction',
    order: [
      'transaction_id', 'customer_id', 'transaction_type', 'amount', 'currency',
      'transaction_date', 'status', 'description', 'merchant_name', 'created_at',
    ],
  },
  check_transactions: {
    section: 'Transaction',
    order: [
      'transaction_id', 'customer_id', 'amount', 'currency', 'transaction_date',
      'status', 'description', 'check_number', 'payee_name', 'created_at',
    ],
  },
  disputes: {
    section: 'Dispute',
    order: [
      'dispute_id', 'transaction_id', 'customer_id', 'dispute_date', 'dispute_reason',
      'dispute_status', 'amount_disputed', 'currency', 'description', 'created_at',
    ],
  },
  state_codes: {
    section: 'State',
    order: ['state_code', 'state_name', 'region'],
  },
  zip_codes: {
    section: 'Address',
    order: ['zip_code', 'city', 'state_code', 'timezone'],
  },
  country_codes: {
    section: 'Country',
    order: ['country_code', 'country_name', 'region'],
  },
  graph_path: {
    section: 'Related (from graph)',
    order: [
      'customer_id', 'dispute_id', 'transaction_id', 'status', 'amount_disputed',
      'currency', 'transaction_type', 'transaction_count', 'tx_amount', 'dispute_status',
    ],
  },
};

/** Entity type display order for tables (Customer first, then transactions, then reference). */
export const ENTITY_TABLE_ORDER = [
  'customers', 'ach_transactions', 'wire_transactions', 'credit_transactions',
  'debit_transactions', 'check_transactions', 'disputes', 'graph_path',
  'state_codes', 'zip_codes', 'country_codes',
] as const;

export function getColumnOrder(fileType: string): string[] {
  const config = SECTION_ORDER[fileType as keyof typeof SECTION_ORDER];
  return config?.order ?? [];
}

export function getSectionTitle(fileType: string): string {
  const config = SECTION_ORDER[fileType as keyof typeof SECTION_ORDER];
  return config?.section ?? fileType.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Parse "Structured Data: {...}" from content. Handles truncated JSON (trailing ...).
 */
export function parseStructuredContent(content: string): Record<string, unknown> | null {
  if (!content?.includes('Structured Data:')) return null;
  const start = content.indexOf('Structured Data:') + 'Structured Data:'.length;
  let raw = content.slice(start).trim();
  // Remove trailing ellipsis from truncated payloads
  raw = raw.replace(/\s*\.\.\.\s*$/, '').trim();
  if (!raw.startsWith('{')) return null;
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    return typeof parsed === 'object' && parsed !== null ? parsed : null;
  } catch {
    // Try to fix truncated JSON: close unclosed string/key and add }
    const lastBrace = raw.lastIndexOf('}');
    if (lastBrace > 0) {
      try {
        return JSON.parse(raw.slice(0, lastBrace + 1)) as Record<string, unknown>;
      } catch {
        // ignore
      }
    }
    return null;
  }
}

/**
 * Human-readable label for a field key.
 */
export function formatFieldLabel(key: string): string {
  return FIELD_LABELS[key] ?? key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export interface FormattedEntry {
  label: string;
  key: string;
  value: unknown;
}

export interface FormattedSection {
  title?: string;
  entries: FormattedEntry[];
}

/**
 * Turn raw structured data into ordered sections with human labels.
 * Puts known fields first in a logical order; appends any extra keys at the end.
 */
export function formatStructuredForDisplay(
  data: Record<string, unknown>,
  fileType: string
): FormattedSection[] {
  const config = SECTION_ORDER[fileType];
  const keys = Object.keys(data);
  const ordered: string[] = config
    ? [...config.order.filter((k) => keys.includes(k)), ...keys.filter((k) => !config.order.includes(k))]
    : keys;
  const entries: FormattedEntry[] = ordered
    .filter((k) => data[k] !== undefined && data[k] !== null && data[k] !== '')
    .map((key) => ({ label: formatFieldLabel(key), key, value: data[key] }));
  if (entries.length === 0) return [];
  return [{ title: config?.section, entries }];
}

/**
 * Build a short one-line summary for cards (e.g. "John Doe · CA 90210 · Active").
 */
export function formatOneLineSummary(data: Record<string, unknown>, fileType: string): string {
  const parts: string[] = [];
  switch (fileType) {
    case 'customers': {
      const first = [data.first_name, data.last_name].filter(Boolean).join(' ');
      if (first) parts.push(first);
      if (data.city && data.state_code) parts.push(`${data.city}, ${data.state_code}`);
      else if (data.state_code) parts.push(String(data.state_code));
      if (data.account_status) parts.push(String(data.account_status));
      break;
    }
    case 'ach_transactions':
    case 'wire_transactions':
    case 'credit_transactions':
    case 'debit_transactions':
    case 'check_transactions': {
      if (data.transaction_id) parts.push(String(data.transaction_id));
      if (data.amount != null) parts.push(`${data.currency ?? 'USD'} ${data.amount}`);
      if (data.status) parts.push(String(data.status));
      if (data.transaction_date) parts.push(String(data.transaction_date).slice(0, 10));
      break;
    }
    case 'state_codes':
      if (data.state_code) parts.push(String(data.state_code));
      if (data.state_name) parts.push(String(data.state_name));
      break;
    case 'zip_codes':
      if (data.zip_code) parts.push(String(data.zip_code));
      if (data.city) parts.push(String(data.city));
      if (data.state_code) parts.push(String(data.state_code));
      break;
    case 'country_codes':
      if (data.country_name) parts.push(String(data.country_name));
      if (data.country_code) parts.push(String(data.country_code));
      break;
    default:
      if (data.id) parts.push(String(data.id));
      else parts.push(...Object.values(data).slice(0, 3).map(String));
  }
  return parts.filter(Boolean).join(' · ');
}
