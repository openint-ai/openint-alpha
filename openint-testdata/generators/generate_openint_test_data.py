"""
openInt Test Data Generator
Generates realistic openInt test data including:
- Customer dimension table (default 10K records; override with --num-customers)
- Transaction fact tables (ACH, Wire, Credit, Debit, Check) for those customers only
- Disputes by type: ach_disputes.csv, credit_disputes.csv, debit_disputes.csv,
  wire_disputes.csv, check_disputes.csv, atm_disputes.csv

ID strategy (banking-critical, for agentic analytics):
- customer_id: BIGINT (64-bit), random unique per run; no append, always overwrite.
- transaction_id: UUID (string) per transaction.
- dispute_id: INT (32-bit), random unique per run.

Always overwrites output files; never appends. Ensures no duplicate IDs and no join conflicts.

Schema: Connects to DataHub API to fetch table schema when available; otherwise uses
openint-datahub/schemas.py. Generated CSVs follow schema field order and types.
"""

import json
import os
import sys
import ssl
import argparse
import urllib.request
import urllib.error
import uuid
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional, List, Dict, Any

# Timeout for DataHub schema fetch so generator does not hang when DataHub is down
_SCHEMA_FETCH_TIMEOUT_SEC = 5

# Initialize Faker
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# ID strategy: customer_id = BIGINT (random); transaction_id = UUID; dispute_id = INT (random)
# BIGINT: positive 64-bit (1 to 2^63-1); INT: positive 32-bit (1 to 2^31-1)
CUSTOMER_ID_MIN = 1
CUSTOMER_ID_MAX = 2**63 - 1  # BIGINT max
DISPUTE_ID_MIN = 1
DISPUTE_ID_MAX = 2**31 - 1   # INT max

# Global schema (DataHub or openint-datahub/schemas.py); set in main()
SCHEMA: Dict[str, Dict[str, Any]] = {}

# Data directories: output to project root data/ (repo_root/data)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent  # openint-testdata
_REPO_ROOT = _PROJECT_ROOT.parent   # repo root
BASE_DIR = _REPO_ROOT / "data"
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Subdirectories
DIMENSIONS_DIR = BASE_DIR / "dimensions"
FACTS_DIR = BASE_DIR / "facts"

for dir_path in [DIMENSIONS_DIR, FACTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# DataHub/schema resolution uses repo root
_OPENINT_DATAHUB = _REPO_ROOT / "openint-datahub"


def get_schema() -> Dict[str, Dict[str, Any]]:
    """
    Fetch table schema from DataHub API when available; otherwise use openint-datahub/schemas.py.
    Uses a short timeout so the generator does not hang when DataHub is down.
    Returns dict mapping dataset name -> { description, category, fields }.
    """
    def _fetch_datahub_schema() -> Optional[Dict[str, Dict[str, Any]]]:
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        openint_agents = _REPO_ROOT / "openint-agents"
        if openint_agents.exists() and str(openint_agents) not in sys.path:
            sys.path.insert(0, str(openint_agents))
        from sg_agent.datahub_client import get_schema_from_datahub  # type: ignore[import-not-found]
        return get_schema_from_datahub()

    # Try DataHub API with timeout so we don't hang when DataHub is not running
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            fut = executor.submit(_fetch_datahub_schema)
            schema = fut.result(timeout=_SCHEMA_FETCH_TIMEOUT_SEC)
        if schema:
            return schema
    except (FuturesTimeoutError, Exception):
        pass
    # Fallback: openint-datahub/schemas.py
    if _OPENINT_DATAHUB.exists() and str(_OPENINT_DATAHUB) not in sys.path:
        sys.path.insert(0, str(_OPENINT_DATAHUB))
    try:
        from schemas import get_dataset_schemas
        return get_dataset_schemas()
    except Exception:
        pass
    return {}


def _reorder_df_by_schema(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Reorder DataFrame columns to match schema field order when schema is available."""
    if not SCHEMA or dataset_name not in SCHEMA:
        return df
    fields = SCHEMA[dataset_name].get("fields") or []
    schema_cols = [f["name"] for f in fields if f["name"] in df.columns]
    extra = [c for c in df.columns if c not in schema_cols]
    return df[schema_cols + extra] if schema_cols else df


TX_FACT_FILES = [
    "credit_transactions.csv", "debit_transactions.csv", "ach_transactions.csv",
    "wire_transactions.csv", "check_transactions.csv",
]

# Dispute fact files: one per transaction type (ach, credit, debit, wire, check, atm)
DISPUTE_FACT_FILES = [
    "ach_disputes.csv", "credit_disputes.csv", "debit_disputes.csv",
    "wire_disputes.csv", "check_disputes.csv", "atm_disputes.csv",
]


def _ensure_bigint_id(val: Any) -> Optional[int]:
    """Ensure ID is a valid BIGINT (positive 64-bit) for customer_id. Handles float/NaN from CSV. Returns int or None."""
    if pd.isna(val):
        return None
    try:
        v = int(float(val))
        return v if CUSTOMER_ID_MIN <= v <= CUSTOMER_ID_MAX else None
    except (ValueError, TypeError):
        return None


def _load_valid_customer_ids() -> Optional[set]:
    """Load customer_ids from customers.csv. Returns None if file doesn't exist."""
    customers_path = DIMENSIONS_DIR / "customers.csv"
    if not customers_path.exists():
        return None
    try:
        df = pd.read_csv(customers_path, usecols=["customer_id"])
        return set(int(x) for x in df["customer_id"].dropna().unique() if _ensure_bigint_id(x) is not None)
    except Exception:
        return None


def _get_max_dispute_id() -> int:
    """Return max dispute_id across all dispute fact CSVs, or 0 if none. (Legacy; dispute_id now uses random INT pool.)"""
    mx = 0
    for fname in DISPUTE_FACT_FILES:
        p = FACTS_DIR / fname
        if p.exists():
            try:
                df = pd.read_csv(p, usecols=["dispute_id"], nrows=None)
                if not df.empty and "dispute_id" in df.columns:
                    mx = max(mx, int(df["dispute_id"].max()))
            except Exception:
                pass
    return mx


# Batch size for generation loops: 5K–25K default to avoid OOM; override with GENERATE_BATCH_SIZE or --batch-size
def _default_generate_batch_size() -> int:
    try:
        env_val = os.environ.get("GENERATE_BATCH_SIZE")
        if env_val is not None:
            return max(5_000, min(50_000, int(env_val)))
    except (TypeError, ValueError):
        pass
    cpu = (os.cpu_count() or 4)
    if cpu >= 8:
        return 25_000
    if cpu >= 4:
        return 15_000
    return 5_000


def generate_customers(num_records=1000000, batch_size=None):
    """
    Generate customer dimension table. customer_id: BIGINT (random unique), no duplication.
    Always overwrites; never appends. Ensures no duplicate IDs for agentic analytics.
    """
    if batch_size is None:
        batch_size = _default_generate_batch_size()
    customers_path = DIMENSIONS_DIR / "customers.csv"
    # Generate num_records unique random BIGINTs
    taken: set = set()
    new_ids: List[int] = []
    while len(new_ids) < num_records:
        cid = random.randint(CUSTOMER_ID_MIN, CUSTOMER_ID_MAX)
        if cid not in taken:
            taken.add(cid)
            new_ids.append(cid)

    print(f"\n📊 Generating {num_records:,} customer records (batch size: {batch_size:,})...")
    print(f"   📁 Output: {customers_path.resolve()} (overwrite)")
    print(f"   📋 customer_id: BIGINT (random unique)")

    total_written = 0
    write_header = True
    for i in range(0, num_records, batch_size):
        batch_end = min(i + batch_size, num_records)
        batch = []
        for k in range(i, batch_end):
            customer_id = new_ids[k]
            ssn = fake.ssn()
            first_name = fake.first_name()
            last_name = fake.last_name()
            email = f"{first_name.lower()}.{last_name.lower()}@{fake.domain_name()}"
            phone = fake.phone_number()
            date_of_birth = fake.date_of_birth(minimum_age=18, maximum_age=90)
            account_opened_date = fake.date_between(start_date='-10y', end_date='today')
            street_address = fake.street_address()
            city = fake.city()
            state_code = fake.state_abbr()
            zip_code = fake.zipcode()
            country_code = "US"
            customer_type = np.random.choice(
                ["Individual", "Business", "Premium", "VIP"],
                p=[0.6, 0.25, 0.1, 0.05]
            )
            account_status = np.random.choice(
                ["Active", "Inactive", "Closed", "Suspended"],
                p=[0.85, 0.05, 0.08, 0.02]
            )
            credit_score = np.random.randint(300, 850) if customer_type in ["Individual", "Premium", "VIP"] else None
            batch.append({
                "customer_id": customer_id,
                "ssn": ssn,
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "phone": phone,
                "date_of_birth": date_of_birth,
                "account_opened_date": account_opened_date,
                "street_address": street_address,
                "city": city,
                "state_code": state_code,
                "zip_code": zip_code,
                "country_code": country_code,
                "customer_type": customer_type,
                "account_status": account_status,
                "credit_score": credit_score,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            })
        df_batch = pd.DataFrame(batch)
        df_batch = _reorder_df_by_schema("customers", df_batch)
        df_batch.to_csv(customers_path, mode=("w" if write_header else "a"), index=False, header=write_header)
        write_header = False
        total_written += len(df_batch)
        if (i + batch_size) % 100000 == 0 or batch_end == num_records:
            print(f"   Progress: {batch_end:,}/{num_records:,} customers generated")
    print(f"✅ Generated customers.csv with {total_written:,} records")
    return pd.read_csv(customers_path)


def generate_ach_transactions(num_records, customer_ids, batch_size=None, start_id=None):
    """Generate ACH (Automated Clearing House) transactions. transaction_id = UUID. Always overwrites."""
    if batch_size is None:
        batch_size = _default_generate_batch_size()
    print(f"\n📊 Generating {num_records:,} ACH transaction records (batch size: {batch_size:,})...")
    print(f"   📋 transaction_id: UUID")

    path = FACTS_DIR / "ach_transactions.csv"
    transaction_types = ["Debit", "Credit"]
    ach_codes = ["PPD", "WEB", "TEL", "CCD", "ARC", "BOC", "POP"]
    write_header = True
    total_written = 0
    for i in range(0, num_records, batch_size):
        batch_end = min(i + batch_size, num_records)
        batch = []
        for j in range(i, batch_end):
            transaction_id = str(uuid.uuid4())
            customer_id = np.random.choice(customer_ids)
            transaction_datetime = fake.date_time_between(start_date='-2y', end_date='now')
            transaction_date = transaction_datetime.date()
            transaction_type = np.random.choice(transaction_types)
            amount = round(np.random.lognormal(mean=5, sigma=1.5), 2)
            if transaction_type == "Debit":
                amount = -abs(amount)
            else:
                amount = abs(amount)
            ach_code = np.random.choice(ach_codes)
            routing_number = f"{np.random.randint(100000000, 999999999)}"
            account_number = fake.bban()
            status = np.random.choice(
                ["Completed", "Pending", "Failed", "Reversed"],
                p=[0.92, 0.05, 0.02, 0.01]
            )
            batch.append({
                "transaction_id": transaction_id,
                "customer_id": customer_id,
                "transaction_date": transaction_date,
                "transaction_datetime": transaction_datetime,
                "transaction_type": transaction_type,
                "amount": amount,
                "currency": "USD",
                "ach_code": ach_code,
                "routing_number": routing_number,
                "account_number": account_number,
                "description": f"ACH {transaction_type} - {ach_code}",
                "status": status,
                "created_at": datetime.now(),
            })
        df_batch = pd.DataFrame(batch)
        df_batch = _reorder_df_by_schema("ach_transactions", df_batch)
        df_batch.to_csv(path, mode=("w" if write_header else "a"), index=False, header=write_header)
        write_header = False
        total_written += len(df_batch)
        if (i + batch_size) % 100000 == 0 or batch_end == num_records:
            print(f"   Progress: {batch_end:,}/{num_records:,} ACH transactions generated")
    print(f"✅ Generated ach_transactions.csv with {total_written:,} records")
    return pd.read_csv(path), None


def generate_wire_transactions(num_records, customer_ids, batch_size=None, start_id=None):
    """Generate Wire transfer transactions. transaction_id = UUID. Always overwrites."""
    if batch_size is None:
        batch_size = _default_generate_batch_size()
    print(f"\n📊 Generating {num_records:,} Wire transaction records (batch size: {batch_size:,})...")
    print(f"   📋 transaction_id: UUID")
    path = FACTS_DIR / "wire_transactions.csv"
    wire_types = ["Domestic", "International"]
    countries = ["US", "CA", "GB", "DE", "FR", "JP", "AU", "CH"]
    write_header = True
    total_written = 0
    for i in range(0, num_records, batch_size):
        batch_end = min(i + batch_size, num_records)
        batch = []
        for j in range(i, batch_end):
            transaction_id = str(uuid.uuid4())
            customer_id = np.random.choice(customer_ids)
            transaction_datetime = fake.date_time_between(start_date='-2y', end_date='now')
            transaction_date = transaction_datetime.date()
            wire_type = np.random.choice(wire_types, p=[0.7, 0.3])
            amount = round(np.random.lognormal(mean=7, sigma=1.8), 2)
            if wire_type == "International":
                currency = np.random.choice(["USD", "EUR", "GBP", "JPY", "CAD", "AUD"])
                beneficiary_country = np.random.choice(countries)
                beneficiary_openInt_swift = fake.swift()
            else:
                currency = "USD"
                beneficiary_country = "US"
                beneficiary_openInt_swift = None
            sender_routing = f"{np.random.randint(100000000, 999999999)}"
            beneficiary_routing = f"{np.random.randint(100000000, 999999999)}" if wire_type == "Domestic" else None
            beneficiary_account = fake.bban()
            beneficiary_name = fake.name()
            status = np.random.choice(
                ["Completed", "Pending", "Failed", "Cancelled"],
                p=[0.88, 0.08, 0.03, 0.01]
            )
            fee = round(amount * 0.001, 2) if wire_type == "Domestic" else round(amount * 0.002, 2)
            batch.append({
                "transaction_id": transaction_id,
                "customer_id": customer_id,
                "transaction_date": transaction_date,
                "transaction_datetime": transaction_datetime,
                "wire_type": wire_type,
                "amount": amount,
                "currency": currency,
                "sender_routing": sender_routing,
                "beneficiary_routing": beneficiary_routing,
                "beneficiary_account": beneficiary_account,
                "beneficiary_name": beneficiary_name,
                "beneficiary_country": beneficiary_country,
                "beneficiary_bank_swift": beneficiary_openInt_swift,
                "fee": fee,
                "description": f"Wire Transfer - {wire_type}",
                "status": status,
                "created_at": datetime.now(),
            })
        df_batch = pd.DataFrame(batch)
        df_batch = _reorder_df_by_schema("wire_transactions", df_batch)
        df_batch.to_csv(path, mode=("w" if write_header else "a"), index=False, header=write_header)
        write_header = False
        total_written += len(df_batch)
        if (i + batch_size) % 100000 == 0 or batch_end == num_records:
            print(f"   Progress: {batch_end:,}/{num_records:,} Wire transactions generated")
    print(f"✅ Generated wire_transactions.csv with {total_written:,} records")
    return pd.read_csv(path), None


def generate_credit_transactions(num_records, customer_ids, batch_size=None, start_id=None):
    """Generate Credit card transactions. transaction_id = UUID. Always overwrites."""
    if batch_size is None:
        batch_size = _default_generate_batch_size()
    print(f"\n📊 Generating {num_records:,} Credit card transaction records (batch size: {batch_size:,})...")
    print(f"   📋 transaction_id: UUID")
    path = FACTS_DIR / "credit_transactions.csv"
    card_types = ["Visa", "Mastercard", "American Express", "Discover"]
    merchant_categories = [
        "Retail", "Restaurant", "Gas Station", "Grocery", "Online Shopping",
        "Travel", "Entertainment", "Healthcare", "Utilities", "Education"
    ]
    write_header = True
    total_written = 0
    for i in range(0, num_records, batch_size):
        batch_end = min(i + batch_size, num_records)
        batch = []
        for j in range(i, batch_end):
            transaction_id = str(uuid.uuid4())
            customer_id = np.random.choice(customer_ids)
            transaction_datetime = fake.date_time_between(start_date='-1y', end_date='now')
            transaction_date = transaction_datetime.date()
            card_type = np.random.choice(card_types)
            card_number_last4 = str(np.random.randint(1000, 9999))
            amount = round(np.random.lognormal(mean=4, sigma=1.2), 2)
            merchant_name = fake.company()
            merchant_category = np.random.choice(merchant_categories)
            merchant_city = fake.city()
            merchant_state = fake.state_abbr()
            merchant_country = "US"
            authorization_code = fake.bothify(text='??????', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            status = np.random.choice(
                ["Approved", "Declined", "Pending", "Refunded"],
                p=[0.94, 0.04, 0.01, 0.01]
            )
            batch.append({
                "transaction_id": transaction_id,
                "customer_id": customer_id,
                "transaction_date": transaction_date,
                "transaction_datetime": transaction_datetime,
                "card_type": card_type,
                "card_number_last4": card_number_last4,
                "amount": amount,
                "currency": "USD",
                "merchant_name": merchant_name,
                "merchant_category": merchant_category,
                "merchant_city": merchant_city,
                "merchant_state": merchant_state,
                "merchant_country": merchant_country,
                "authorization_code": authorization_code,
                "description": f"{merchant_category} - {merchant_name}",
                "status": status,
                "created_at": datetime.now(),
            })
        df_batch = pd.DataFrame(batch)
        df_batch = _reorder_df_by_schema("credit_transactions", df_batch)
        df_batch.to_csv(path, mode=("w" if write_header else "a"), index=False, header=write_header)
        write_header = False
        total_written += len(df_batch)
        if (i + batch_size) % 100000 == 0 or batch_end == num_records:
            print(f"   Progress: {batch_end:,}/{num_records:,} Credit transactions generated")
    print(f"✅ Generated credit_transactions.csv with {total_written:,} records")
    return pd.read_csv(path), None


def generate_check_transactions(num_records, customer_ids, batch_size=None, start_id=None):
    """Generate Check transactions. transaction_id = UUID. Always overwrites."""
    if batch_size is None:
        batch_size = _default_generate_batch_size()
    print(f"\n📊 Generating {num_records:,} Check transaction records (batch size: {batch_size:,})...")
    print(f"   📋 transaction_id: UUID")
    path = FACTS_DIR / "check_transactions.csv"
    write_header = True
    total_written = 0
    for i in range(0, num_records, batch_size):
        batch_end = min(i + batch_size, num_records)
        batch = []
        for j in range(i, batch_end):
            transaction_id = str(uuid.uuid4())
            customer_id = np.random.choice(customer_ids)
            transaction_datetime = fake.date_time_between(start_date='-2y', end_date='now')
            transaction_date = transaction_datetime.date()
            check_number = np.random.randint(100, 999999)
            amount = round(np.random.lognormal(mean=5.5, sigma=1.4), 2)
            payee_name = fake.name()
            memo = np.random.choice([
                "Payment", "Rent", "Utilities", "Services", "Invoice Payment",
                "Loan Payment", "Insurance", "Tax Payment"
            ], p=[0.2, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05])
            status = np.random.choice(
                ["Cleared", "Pending", "Bounced", "Cancelled"],
                p=[0.85, 0.1, 0.04, 0.01]
            )
            batch.append({
                "transaction_id": transaction_id,
                "customer_id": customer_id,
                "transaction_date": transaction_date,
                "transaction_datetime": transaction_datetime,
                "check_number": check_number,
                "amount": amount,
                "currency": "USD",
                "payee_name": payee_name,
                "memo": memo,
                "description": f"Check #{check_number} - {payee_name}",
                "status": status,
                "created_at": datetime.now(),
            })
        df_batch = pd.DataFrame(batch)
        df_batch = _reorder_df_by_schema("check_transactions", df_batch)
        df_batch.to_csv(path, mode=("w" if write_header else "a"), index=False, header=write_header)
        write_header = False
        total_written += len(df_batch)
        if (i + batch_size) % 100000 == 0 or batch_end == num_records:
            print(f"   Progress: {batch_end:,}/{num_records:,} Check transactions generated")
    print(f"✅ Generated check_transactions.csv with {total_written:,} records")
    return pd.read_csv(path), None


def generate_debit_transactions(num_records, customer_ids, batch_size=None, start_id=None):
    """Generate Debit card transactions. transaction_id = UUID. Always overwrites."""
    if batch_size is None:
        batch_size = _default_generate_batch_size()
    print(f"\n📊 Generating {num_records:,} Debit card transaction records (batch size: {batch_size:,})...")
    print(f"   📋 transaction_id: UUID")
    path = FACTS_DIR / "debit_transactions.csv"
    transaction_types = ["Purchase", "ATM Withdrawal", "Online Payment", "Recurring Payment"]
    merchant_categories = [
        "Retail", "Restaurant", "Gas Station", "Grocery", "ATM",
        "Online Shopping", "Utilities", "Subscription"
    ]
    write_header = True
    total_written = 0
    for i in range(0, num_records, batch_size):
        batch_end = min(i + batch_size, num_records)
        batch = []
        for j in range(i, batch_end):
            transaction_id = str(uuid.uuid4())
            customer_id = np.random.choice(customer_ids)
            transaction_datetime = fake.date_time_between(start_date='-1y', end_date='now')
            transaction_date = transaction_datetime.date()
            transaction_type = np.random.choice(transaction_types)
            amount = round(np.random.lognormal(mean=3.8, sigma=1.1), 2)
            amount = -abs(amount)
            if transaction_type == "ATM Withdrawal":
                merchant_name = "ATM"
                merchant_category = "ATM"
                merchant_location = fake.address()
            else:
                merchant_name = fake.company()
                merchant_category = np.random.choice(merchant_categories)
                merchant_location = None
            status = np.random.choice(
                ["Completed", "Pending", "Declined", "Reversed"],
                p=[0.93, 0.04, 0.02, 0.01]
            )
            batch.append({
                "transaction_id": transaction_id,
                "customer_id": customer_id,
                "transaction_date": transaction_date,
                "transaction_datetime": transaction_datetime,
                "transaction_type": transaction_type,
                "amount": amount,
                "currency": "USD",
                "merchant_name": merchant_name,
                "merchant_category": merchant_category,
                "merchant_location": merchant_location,
                "description": f"{transaction_type} - {merchant_name}",
                "status": status,
                "created_at": datetime.now(),
            })
        df_batch = pd.DataFrame(batch)
        df_batch = _reorder_df_by_schema("debit_transactions", df_batch)
        df_batch.to_csv(path, mode=("w" if write_header else "a"), index=False, header=write_header)
        write_header = False
        total_written += len(df_batch)
        if (i + batch_size) % 100000 == 0 or batch_end == num_records:
            print(f"   Progress: {batch_end:,}/{num_records:,} Debit transactions generated")
    print(f"✅ Generated debit_transactions.csv with {total_written:,} records")
    return pd.read_csv(path), None


def _generate_dispute_description_via_ollama(
    tx_type: str,
    amount: float,
    dispute_reason: str,
    tx_context: Dict[str, Any],
) -> Optional[str]:
    """
    Call Ollama to generate a creative, contextual dispute description for bank/transaction data.
    Returns None on failure (caller should use template fallback).
    """
    host = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL") or "llama3.2"
    ctx_str = ", ".join(f"{k}={v}" for k, v in tx_context.items() if v is not None and str(v).strip())
    prompt = f"""You are helping a banking platform. Generate ONE short, realistic dispute description (1-2 sentences) for a customer disputing a {tx_type} transaction.

Context: Transaction type={tx_type}, amount=${amount:.2f}, dispute reason="{dispute_reason}". Extra details: {ctx_str or 'N/A'}

Be creative and varied. Examples of good descriptions:
- "Unauthorized credit card charge of ${amount:.2f} at [merchant] — card was in my possession; possible skimming."
- "Duplicate ACH debit — same payroll was deducted twice on the same day."
- "Wire transfer sent to wrong beneficiary account; requested recall immediately."
- "Subscription charge after cancellation; merchant continued billing."
- "Check never received by payee; payment cleared but payee never got funds."

Return ONLY the dispute description text, nothing else. No quotes, no prefix. One or two sentences max."""
    try:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.8, "num_predict": 150},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        ctx_ssl = ssl.create_default_context()
        ctx_ssl.check_hostname = False
        ctx_ssl.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=15, context=ctx_ssl) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = (data.get("message") or {}).get("content") or ""
        desc = content.strip()
        # Strip surrounding quotes if LLM wrapped the response
        while len(desc) >= 2 and desc[0] == '"' and desc[-1] == '"':
            desc = desc[1:-1].strip()
        desc = desc[:500]  # cap length
        return desc if desc else None
    except Exception:
        return None


def _build_tx_context(tx_type: str, tx: pd.Series) -> Dict[str, Any]:
    """Build a small context dict from transaction row for LLM prompt."""
    ctx = {}
    if tx_type == "credit":
        for k in ("merchant_name", "merchant_category", "card_type"):
            if k in tx and pd.notna(tx.get(k)):
                ctx[k] = str(tx[k])[:80]
    elif tx_type == "debit":
        for k in ("merchant_name", "merchant_category", "transaction_type"):
            if k in tx and pd.notna(tx.get(k)):
                ctx[k] = str(tx[k])[:80]
    elif tx_type == "ach":
        for k in ("ach_code", "transaction_type", "description"):
            if k in tx and pd.notna(tx.get(k)):
                ctx[k] = str(tx[k])[:80]
    elif tx_type == "wire":
        for k in ("wire_type", "beneficiary_name", "beneficiary_country"):
            if k in tx and pd.notna(tx.get(k)):
                ctx[k] = str(tx[k])[:80]
    elif tx_type == "check":
        for k in ("payee_name", "memo", "check_number"):
            if k in tx and pd.notna(tx.get(k)):
                ctx[k] = str(tx[k])[:80]
    return ctx


def _generate_disputes_for_type(
    tx_path: Path,
    tx_type: str,
    cols: List[str],
    num_disputes: int,
    dispute_id_pool: List[int],
    use_llm: bool,
    dispute_reasons: List[str],
    dispute_statuses: List[str],
    valid_customer_ids: Optional[set] = None,
) -> tuple:
    """
    Generate disputes for a single transaction type. Returns (df, remaining_dispute_id_pool).
    customer_id: BIGINT from customers.csv. transaction_id: UUID string from transaction CSV.
    dispute_id: random unique INT from pool.
    """
    if not tx_path.exists():
        return None, dispute_id_pool
    header = pd.read_csv(tx_path, nrows=0).columns.tolist()
    usecols = [c for c in cols if c in header]
    if not all(r in usecols for r in ["transaction_id", "customer_id", "amount"]):
        return None, dispute_id_pool
    df = pd.read_csv(tx_path, usecols=usecols)
    if "currency" not in df.columns:
        df["currency"] = "USD"
    df = df.dropna(subset=["transaction_id", "customer_id"])
    if df.empty:
        return None, dispute_id_pool
    # Filter to rows where customer_id exists in customers.csv (if provided)
    if valid_customer_ids is not None:
        df = df[df["customer_id"].apply(lambda x: _ensure_bigint_id(x) in valid_customer_ids)]
    if df.empty:
        return None, dispute_id_pool
    sample = df.sample(n=min(num_disputes, len(df)), replace=False, random_state=42)
    rows = []
    pool_used = min(len(dispute_id_pool), len(sample))
    pool_remaining = dispute_id_pool[pool_used:]
    pool_for_this = dispute_id_pool[:pool_used]
    llm_ok = use_llm
    for idx, (_, tx) in enumerate(sample.iterrows()):
        if idx >= len(pool_for_this):
            break
        dispute_id = pool_for_this[idx]
        cust_id = _ensure_bigint_id(tx["customer_id"])
        # transaction_id is UUID (string) from transaction CSV; keep as-is
        txn_id = tx["transaction_id"] if pd.notna(tx["transaction_id"]) else None
        if cust_id is None or txn_id is None or str(txn_id).strip() == "":
            continue
        txn_id_str = str(txn_id).strip()
        amount = tx.get("amount", 0)
        if pd.isna(amount) or amount <= 0:
            amount = round(np.random.lognormal(4, 1), 2)
        amount_disputed = abs(round(float(amount), 2))
        dispute_reason = np.random.choice(dispute_reasons)
        if llm_ok:
            ctx = _build_tx_context(tx_type, tx)
            desc = _generate_dispute_description_via_ollama(tx_type, amount_disputed, dispute_reason, ctx)
            if desc is None:
                llm_ok = False
        if not llm_ok:
            desc = f"Dispute for {tx_type} transaction {txn_id_str}: {dispute_reason}"
        rows.append({
            "dispute_id": dispute_id,
            "transaction_type": tx_type,
            "transaction_id": txn_id_str,
            "customer_id": int(cust_id),
            "dispute_date": fake.date_between(start_date="-1y", end_date="today"),
            "dispute_reason": dispute_reason,
            "dispute_status": np.random.choice(dispute_statuses, p=[0.15, 0.25, 0.2, 0.25, 0.15]),
            "amount_disputed": amount_disputed,
            "currency": str(tx.get("currency", "USD")),
            "description": desc[:500] if desc else f"Dispute for {tx_type} transaction {txn_id_str}",
            "created_at": datetime.now(),
        })
    return (pd.DataFrame(rows), pool_remaining) if rows else (None, dispute_id_pool)


def generate_disputes(num_disputes=5000, batch_size=None, use_llm=True):
    """
    Generate disputes by transaction type, writing to type-specific CSVs:
    ach_disputes.csv, credit_disputes.csv, debit_disputes.csv, wire_disputes.csv, check_disputes.csv, atm_disputes.csv.
    Each dispute references customer_id (BIGINT) and transaction_id (UUID) from the respective transaction CSV.
    dispute_id: random unique INT. Always overwrites; never appends.
    """
    if batch_size is None:
        batch_size = _default_generate_batch_size()
    print(f"\n📊 Generating up to {num_disputes:,} dispute records (by transaction type)...")
    print(f"   📁 Output: {FACTS_DIR.resolve()}/*_disputes.csv (overwrite)")
    if use_llm:
        print("   📝 Using Ollama for creative dispute descriptions (set --no-llm for templates)")
    print(f"   📋 dispute_id: INT (random unique)")

    dispute_reasons = [
        "Unauthorized charge", "Duplicate charge", "Product not received", "Incorrect amount",
        "Fraud", "Service not as described", "Subscription not cancelled", "Merchant error",
        "Wire sent to wrong account", "ACH duplicate", "Check not received", "Credit not applied",
    ]
    dispute_statuses = ["Open", "Pending", "Under review", "Resolved", "Closed"]

    # Load valid customer_ids from customers.csv for referential integrity
    valid_customer_ids = _load_valid_customer_ids()
    if valid_customer_ids is not None:
        print(f"   📋 Filtering to {len(valid_customer_ids):,} customer_ids from customers.csv")
    else:
        print("   ⚠️ customers.csv not found; disputes may reference customers not in dimension")

    # Pre-generate random unique INT dispute_ids
    max_disputes_needed = num_disputes * 6  # 6 dispute types
    if max_disputes_needed > DISPUTE_ID_MAX:
        max_disputes_needed = DISPUTE_ID_MAX
    dispute_id_pool = random.sample(range(DISPUTE_ID_MIN, DISPUTE_ID_MAX + 1), min(max_disputes_needed, DISPUTE_ID_MAX))

    # Specs: (tx_file, tx_type, columns). ATM is a subset of debit (transaction_type == "ATM Withdrawal").
    tx_specs: List[tuple] = [
        ("ach_transactions.csv", "ach", ["transaction_id", "customer_id", "amount", "currency", "ach_code", "transaction_type", "description"]),
        ("credit_transactions.csv", "credit", ["transaction_id", "customer_id", "amount", "currency", "merchant_name", "merchant_category", "card_type"]),
        ("debit_transactions.csv", "debit", ["transaction_id", "customer_id", "amount", "currency", "merchant_name", "merchant_category", "transaction_type"]),
        ("wire_transactions.csv", "wire", ["transaction_id", "customer_id", "amount", "currency", "wire_type", "beneficiary_name", "beneficiary_country"]),
        ("check_transactions.csv", "check", ["transaction_id", "customer_id", "amount", "currency", "payee_name", "memo", "check_number"]),
    ]
    per_type = max(100, num_disputes // 6)  # spread across 6 dispute types
    total_generated = 0
    dfs_all: List[pd.DataFrame] = []

    for fname, tx_type, cols in tx_specs:
        path = FACTS_DIR / fname
        df_out, dispute_id_pool = _generate_disputes_for_type(
            path, tx_type, cols, per_type, dispute_id_pool, use_llm, dispute_reasons, dispute_statuses,
            valid_customer_ids=valid_customer_ids,
        )
        if df_out is not None and not df_out.empty:
            out_path = FACTS_DIR / f"{tx_type}_disputes.csv"
            df_out = _reorder_df_by_schema(f"{tx_type}_disputes", df_out)
            df_out.to_csv(out_path, index=False)
            total_generated += len(df_out)
            dfs_all.append(df_out)
            print(f"   ✅ {tx_type}_disputes.csv: {len(df_out):,} records")
        else:
            if path.exists():
                print(f"   ⚠️ No disputes for {tx_type} (no valid rows)")
            else:
                print(f"   ⚠️ Skipping {tx_type}_disputes ({(path.name)} not found)")

    # ATM disputes: from debit_transactions where transaction_type == "ATM Withdrawal"
    debit_path = FACTS_DIR / "debit_transactions.csv"
    if debit_path.exists() and len(dispute_id_pool) > 0:
        cols = ["transaction_id", "customer_id", "amount", "currency", "merchant_name", "merchant_category", "transaction_type"]
        header = pd.read_csv(debit_path, nrows=0).columns.tolist()
        usecols = [c for c in cols if c in header]
        if "transaction_type" in usecols:
            df_debit = pd.read_csv(debit_path, usecols=usecols)
            df_atm = df_debit[df_debit["transaction_type"] == "ATM Withdrawal"]
            if valid_customer_ids is not None:
                df_atm = df_atm[df_atm["customer_id"].apply(lambda x: _ensure_bigint_id(x) in valid_customer_ids)]
            if not df_atm.empty:
                per_atm = min(per_type, len(df_atm), len(dispute_id_pool))
                sample = df_atm.sample(n=per_atm, replace=False, random_state=42)
                atm_pool = dispute_id_pool[:per_atm]
                dispute_id_pool = dispute_id_pool[per_atm:]
                rows = []
                for idx, (_, tx) in enumerate(sample.iterrows()):
                    if idx >= len(atm_pool):
                        break
                    dispute_id = atm_pool[idx]
                    cust_id = _ensure_bigint_id(tx["customer_id"])
                    txn_id = tx["transaction_id"] if pd.notna(tx["transaction_id"]) else None
                    if cust_id is None or txn_id is None or str(txn_id).strip() == "":
                        continue
                    txn_id_str = str(txn_id).strip()
                    amount = tx.get("amount", 0)
                    if pd.isna(amount) or amount <= 0:
                        amount = round(np.random.lognormal(4, 1), 2)
                    amount_disputed = abs(round(float(amount), 2))
                    dispute_reason = np.random.choice(dispute_reasons)
                    desc = f"Dispute for ATM transaction {txn_id_str}: {dispute_reason}"
                    rows.append({
                        "dispute_id": dispute_id,
                        "transaction_type": "debit",  # ATM txns are in debit_transactions; use "debit" for REFERENCES match
                        "transaction_id": txn_id_str,
                        "customer_id": int(cust_id),
                        "dispute_date": fake.date_between(start_date="-1y", end_date="today"),
                        "dispute_reason": dispute_reason,
                        "dispute_status": np.random.choice(dispute_statuses, p=[0.15, 0.25, 0.2, 0.25, 0.15]),
                        "amount_disputed": amount_disputed,
                        "currency": str(tx.get("currency", "USD")),
                        "description": desc[:500],
                        "created_at": datetime.now(),
                    })
                if rows:
                    df_atm_out = pd.DataFrame(rows)
                    out_path = FACTS_DIR / "atm_disputes.csv"
                    df_atm_out = _reorder_df_by_schema("atm_disputes", df_atm_out)
                    df_atm_out.to_csv(out_path, index=False)
                    total_generated += len(df_atm_out)
                    dfs_all.append(df_atm_out)
                    print(f"   ✅ atm_disputes.csv: {len(df_atm_out):,} records")

    if total_generated == 0:
        print("   ❌ No disputes created. Generate transactions first.")
        return None
    llm_note = " (LLM descriptions)" if use_llm else ""
    print(f"✅ Generated {total_generated:,} disputes across type-specific CSVs{llm_note}")
    return pd.concat(dfs_all, ignore_index=True) if dfs_all else None


def main():
    """Main function to generate all test data"""
    parser = argparse.ArgumentParser(
        description="Generate openInt test data (customers, transactions, disputes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all data (default)
  python generate_openInt_test_data.py

  # Generate only one category
  python generate_openint_test_data.py --only-customers
  python generate_openint_test_data.py --only-transactions
  python generate_openint_test_data.py --only-disputes

  # With custom counts (capped by --max-*)
  python generate_openint_test_data.py --only-customers --num-customers 50000
  python generate_openint_test_data.py --only-transactions --num-transactions 100000
  python generate_openint_test_data.py --only-disputes --num-disputes 500 --no-llm

  # Use max limits (always overwrites; no append)
  python generate_openint_test_data.py --max-customers 10000 --max-transactions 50000 --max-disputes 20000
        """
    )
    only_group = parser.add_mutually_exclusive_group()
    only_group.add_argument(
        "--only-customers",
        action="store_true",
        help="Generate only customers (dimensions/customers.csv); skip transactions and static"
    )
    only_group.add_argument(
        "--only-transactions",
        action="store_true",
        help="Generate only transaction fact tables; requires existing customers.csv"
    )
    only_group.add_argument(
        "--only-disputes",
        action="store_true",
        help="Generate only disputes (from existing transaction CSVs); requires transaction fact tables"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip Ollama for dispute descriptions; use template descriptions instead"
    )
    parser.add_argument(
        "--num-customers",
        type=int,
        default=10_000,
        help="Number of customer records to generate (default: 10,000)"
    )
    parser.add_argument(
        "--num-transactions",
        type=int,
        default=100_000,
        help="Total number of transactions to generate across all types, drawn from the customer set (default: 100,000)"
    )
    parser.add_argument(
        "--num-disputes",
        type=int,
        default=5_000,
        help="Number of disputes to generate from existing transactions (default: 5,000); used with --only-disputes or after transactions"
    )
    parser.add_argument(
        "--max-customers",
        type=int,
        default=10_000,
        help="Maximum number of customer records to generate; caps --num-customers (default: 10,000)"
    )
    parser.add_argument(
        "--max-transactions",
        type=int,
        default=50_000,
        help="Maximum total transactions to generate; caps --num-transactions (default: 50,000)"
    )
    parser.add_argument(
        "--max-disputes",
        type=int,
        default=20_000,
        help="Maximum number of disputes to generate; caps --num-disputes (default: 20,000)"
    )
    default_batch = _default_generate_batch_size()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_batch,
        help=f"In-memory batch size for generation loops (default: {default_batch:,}; 5K–50K; set GENERATE_BATCH_SIZE to override)"
    )
    args = parser.parse_args()

    # Apply max caps: never generate more than --max-customers, --max-transactions, --max-disputes
    num_customers_eff = min(args.num_customers, args.max_customers)
    num_transactions_eff = min(args.num_transactions, args.max_transactions)
    num_disputes_eff = min(args.num_disputes, args.max_disputes)

    load_customers = args.only_customers or (not args.only_transactions and not args.only_disputes)
    load_transactions = args.only_transactions or (not args.only_customers and not args.only_disputes)
    load_disputes = args.only_disputes or (not args.only_customers and not args.only_transactions)

    global SCHEMA
    SCHEMA = get_schema()
    if SCHEMA:
        print("   📋 Schema: loaded (DataHub API or openint-datahub/schemas.py)")
    else:
        print("   📋 Schema: none (using built-in column order)")

    print("=" * 80)
    print("🏦 openInt Test Data Generator")
    print("=" * 80)
    print(f"   📁 Output directory: {BASE_DIR.resolve()}")
    print("   🧹 Always overwrite: no append (ensures no duplicate IDs for agentic analytics)")
    print(f"   📋 Limits: max-customers={args.max_customers:,}, max-transactions={args.max_transactions:,}, max-disputes={args.max_disputes:,}")
    if num_customers_eff < args.num_customers or num_transactions_eff < args.num_transactions or num_disputes_eff < args.num_disputes:
        print(f"   📋 Effective (capped): customers={num_customers_eff:,}, transactions={num_transactions_eff:,}, disputes={num_disputes_eff:,}")
    if args.only_customers:
        print("   📌 Generating only: customers")
    elif args.only_transactions:
        print("   📌 Generating only: transactions")
    elif args.only_disputes:
        print("   📌 Generating only: disputes (from existing transactions)")
    print(f"   📦 Batch size: {args.batch_size:,} (streams to CSV per batch to reduce memory)")
    print()

    start_time = datetime.now()

    # Generate customers and/or get customer_ids for transactions
    customer_ids = None
    if load_customers:
        customers_df = generate_customers(num_records=num_customers_eff, batch_size=args.batch_size)
        customer_ids = customers_df['customer_id'].tolist()
    elif load_transactions:
        # Transactions need customer_ids from existing customers.csv
        customers_path = DIMENSIONS_DIR / "customers.csv"
        if not customers_path.exists():
            print(f"\n❌ Cannot generate only transactions: {customers_path} not found.")
            print("   Generate customers first: python generate_openInt_test_data.py --only-customers")
            sys.exit(1)
        customers_df = pd.read_csv(customers_path)
        customer_ids = customers_df['customer_id'].tolist()
        print(f"\n📂 Loaded {len(customer_ids):,} customer IDs from {customers_path}")

    # Generate transaction fact tables (transaction_id = GUID per row; dispute_id = 100K+ monotonic)
    transaction_distribution = None
    if load_transactions and customer_ids is not None:
        print("\n💳 Generating transaction fact tables...")
        total_transactions = num_transactions_eff
        transaction_distribution = {
            "credit": int(total_transactions * 0.35),
            "debit": int(total_transactions * 0.30),
            "ach": int(total_transactions * 0.20),
            "wire": int(total_transactions * 0.10),
            "check": int(total_transactions * 0.05),
        }
        generate_credit_transactions(
            transaction_distribution["credit"], customer_ids, args.batch_size, start_id=None
        )
        generate_debit_transactions(
            transaction_distribution["debit"], customer_ids, args.batch_size, start_id=None
        )
        generate_ach_transactions(
            transaction_distribution["ach"], customer_ids, args.batch_size, start_id=None
        )
        generate_wire_transactions(
            transaction_distribution["wire"], customer_ids, args.batch_size, start_id=None
        )
        generate_check_transactions(
            transaction_distribution["check"], customer_ids, args.batch_size, start_id=None
        )
        # Generate disputes from the transactions we just created
        disputes_df = generate_disputes(
            num_disputes=num_disputes_eff, batch_size=args.batch_size, use_llm=not args.no_llm
        )
    else:
        disputes_df = None

    # Generate only disputes (from existing transaction CSVs) when not already generated above
    if load_disputes and not (load_transactions and customer_ids is not None):
        print("\n📋 Generating disputes (ach, credit, debit, wire, check, atm)...")
        disputes_df = generate_disputes(
            num_disputes=num_disputes_eff, batch_size=args.batch_size, use_llm=not args.no_llm
        )
    elif not load_transactions:
        disputes_df = None

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("✅ Test Data Generation Complete!")
    print("=" * 80)
    print(f"\n📊 Summary:")
    if load_customers:
        print(f"   • Customer records: {num_customers_eff:,}")
    if load_transactions and transaction_distribution:
        print(f"   • Credit transactions: {transaction_distribution['credit']:,}")
        print(f"   • Debit transactions: {transaction_distribution['debit']:,}")
        print(f"   • ACH transactions: {transaction_distribution['ach']:,}")
        print(f"   • Wire transactions: {transaction_distribution['wire']:,}")
        print(f"   • Check transactions: {transaction_distribution['check']:,}")
        print(f"   • Total transactions: {num_transactions_eff:,}")
    if disputes_df is not None:
        print(f"   • Disputes: {len(disputes_df):,}")
    print(f"\n📁 Data saved to:")
    print(f"   • Dimensions: {DIMENSIONS_DIR}")
    print(f"   • Facts: {FACTS_DIR}")
    print(f"\n⏱️  Generation time: {duration}")
    print("=" * 80)


if __name__ == "__main__":
    main()
