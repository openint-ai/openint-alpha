"""
Load openInt test data into Neo4j graph database (schema-driven, reverse lookup).
Uses DataHub schema (openint-datahub/schemas.py) for field definitions.
Order: (1) Load all transactions first, creating minimal Customer nodes from transaction
customer_id and HAS_TRANSACTION; (2) Load disputes (OPENED_DISPUTE, REFERENCES); (3) Enrich
Customer nodes from customers.csv only for customer_ids that appear in the facts.
This avoids loading the full customers dimension; only customers with transactions/disputes get full attributes.

ID handling: _to_id_str() normalizes customer_id, transaction_id, dispute_id for consistency
with Milvus (e.g. "1000003621" not "1000003621.0"). Customer nodes store id as string.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

# Resolve repo root and add openint-graph + openint-datahub
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
openint_graph = _repo_root / "openint-graph"
if openint_graph.exists() and str(openint_graph) not in sys.path:
    sys.path.insert(0, str(openint_graph))
openint_datahub = _repo_root / "openint-datahub"
if openint_datahub.exists() and str(openint_datahub) not in sys.path:
    sys.path.insert(0, str(openint_datahub))

try:
    from neo4j_client import Neo4jClient
except ImportError:
    try:
        from openint_graph.neo4j_client import Neo4jClient
    except ImportError:
        print("Error: Could not import Neo4jClient. Ensure openint-graph is on PYTHONPATH.")
        sys.exit(1)

try:
    from schemas import get_dataset_schemas
except ImportError:
    get_dataset_schemas = None

# Data directories (aligned with Milvus loader: generator writes to openint-testdata/data)
_data_root = Path(__file__).resolve().parent.parent  # openint-testdata
_BASE_CANDIDATES = [
    _repo_root / "data",
    _data_root / "data",  # Generator output: openint-testdata/data
    Path("data"),
]
BASE_DIR = next((p for p in _BASE_CANDIDATES if p.exists()), _data_root / "data")
DIMENSIONS_DIR = BASE_DIR / "dimensions"
FACTS_DIR = BASE_DIR / "facts"

# Schema-driven mapping: dataset_name -> (subdir, filename, node_label, id_property, type_property_value)
# type_property_value only for Transaction nodes (ach, wire, credit, debit, check); None for Customer/Dispute
LOAD_SPEC = {
    "customers": ("dimensions", "customers.csv", "Customer", "customer_id", None),
    "ach_transactions": ("facts", "ach_transactions.csv", "Transaction", "transaction_id", "ach"),
    "wire_transactions": ("facts", "wire_transactions.csv", "Transaction", "transaction_id", "wire"),
    "credit_transactions": ("facts", "credit_transactions.csv", "Transaction", "transaction_id", "credit"),
    "debit_transactions": ("facts", "debit_transactions.csv", "Transaction", "transaction_id", "debit"),
    "check_transactions": ("facts", "check_transactions.csv", "Transaction", "transaction_id", "check"),
    # Disputes: type-specific files (ach_disputes.csv, credit_disputes.csv, etc.)
    "ach_disputes": ("facts", "ach_disputes.csv", "Dispute", "dispute_id", None),
    "credit_disputes": ("facts", "credit_disputes.csv", "Dispute", "dispute_id", None),
    "debit_disputes": ("facts", "debit_disputes.csv", "Dispute", "dispute_id", None),
    "wire_disputes": ("facts", "wire_disputes.csv", "Dispute", "dispute_id", None),
    "check_disputes": ("facts", "check_disputes.csv", "Dispute", "dispute_id", None),
    "atm_disputes": ("facts", "atm_disputes.csv", "Dispute", "dispute_id", None),
}

# Batch size for Cypher UNWIND
DEFAULT_BATCH = 1000


def _schema_field_names(dataset_name: str) -> Optional[List[str]]:
    """Return ordered field names for dataset from DataHub schema, or None if schema unavailable."""
    if get_dataset_schemas is None:
        return None
    schemas = get_dataset_schemas()
    if dataset_name not in schemas:
        return None
    return [f["name"] for f in schemas[dataset_name].get("fields", [])]


def collect_customer_ids_from_facts(
    facts_dir: Path,
    dimensions_dir: Path,
    max_transactions: Optional[int],
    max_disputes: Optional[int],
) -> set:
    """Scan transaction and dispute CSVs and return the set of customer_ids (normalized strings, e.g. 1000003621)."""
    seen: set = set()
    for dataset_name, spec in LOAD_SPEC.items():
        if spec[2] == "Transaction":
            subdir, fname, _, _, _ = spec
            csv_path = (dimensions_dir if subdir == "dimensions" else facts_dir) / fname
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path, usecols=["customer_id"])
            if max_transactions:
                df = df.head(max_transactions)
            for v in df["customer_id"].dropna().unique():
                s = _to_id_str(v)
                if s:
                    seen.add(s)
    # disputes (type-specific: ach_disputes, credit_disputes, etc.)
    for dataset_name, spec in LOAD_SPEC.items():
        if spec[2] != "Dispute":
            continue
        subdir, fname, _, _, _ = spec
        csv_path = (dimensions_dir if subdir == "dimensions" else facts_dir) / fname
        if csv_path.exists():
            df = pd.read_csv(csv_path, usecols=["customer_id"])
            if max_disputes:
                df = df.head(max_disputes)
            for v in df["customer_id"].dropna().unique():
                s = _to_id_str(v)
                if s:
                    seen.add(s)
    return seen


def collect_customer_ids_from_graph(client: Neo4jClient) -> set:
    """Return the set of customer ids (normalized strings, e.g. 1000003621) for all Customer nodes in Neo4j."""
    try:
        rows = client.run("MATCH (c:Customer) RETURN c.id AS id")
        return {_to_id_str(r["id"]) for r in (rows or []) if r.get("id") is not None}
    except Exception:
        return set()


def _to_id_str(val: Any) -> str:
    """Normalize ID for Neo4j: handle int, float (1000005586.0), str. Always returns clean numeric string."""
    if pd.isna(val) or val is None:
        return ""
    try:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return str(int(val))
        s = str(val).strip()
        if "." in s and s.replace(".", "", 1).replace("-", "", 1).isdigit():
            return str(int(float(s)))
        return s
    except (ValueError, TypeError):
        return str(val).strip()


def _row_to_props(row: pd.Series, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convert a DataFrame row to a dict of Neo4j-safe properties (exclude keys in exclude)."""
    exclude = set(exclude or [])
    props = {}
    for col in row.index:
        if col in exclude:
            continue
        v = row[col]
        if pd.isna(v) or (isinstance(v, float) and v != v):  # NaN
            continue
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, (pd.Timestamp,)):
            v = str(v)
        props[col] = v
    return props


def load_customers(
    client: Neo4jClient,
    csv_path: Path,
    batch_size: int = DEFAULT_BATCH,
    max_records: Optional[int] = None,
    schema_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load customers.csv into Customer nodes (schema-driven if schema_fields provided)."""
    if not csv_path.exists():
        return {"success": False, "error": "File not found"}
    print(f"\n📂 Loading Customers from {csv_path.name}...")
    try:
        df = pd.read_csv(csv_path)
        if max_records:
            df = df.head(max_records)
        # Property columns: schema fields present in CSV, excluding id key
        id_key = "customer_id"
        if schema_fields:
            keys = [k for k in schema_fields if k != id_key and k in df.columns]
        else:
            keys = [k for k in df.columns if k != id_key]
        total = 0
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            rows = []
            for _, row in batch.iterrows():
                r = {id_key: _to_id_str(row[id_key])}
                r.update(_row_to_props(row, exclude=[id_key]))
                rows.append(r)
            set_parts = [f"c.{k} = row.{k}" for k in keys]
            cypher = """
            UNWIND $rows AS row
            MERGE (c:Customer {id: row.customer_id})
            SET """ + ", ".join(set_parts)
            client.run(cypher, {"rows": rows})
            total += len(rows)
            print(f"   Progress: {total:,} customers")
        print(f"✅ Loaded {total:,} customers")
        return {"success": True, "total_loaded": total, "table_name": "customers"}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e), "table_name": "customers"}


def enrich_customers(
    client: Neo4jClient,
    csv_path: Path,
    allowed_ids: set,
    batch_size: int = DEFAULT_BATCH,
    max_records: Optional[int] = None,
    schema_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Enrich existing Customer nodes (created from transactions) with attributes from customers.csv.
    Only rows with customer_id in allowed_ids are applied; avoids loading the full dimension."""
    if not csv_path.exists():
        return {"success": False, "error": "File not found"}
    print(f"\n📂 Enriching Customers from {csv_path.name} (only {len(allowed_ids):,} ids from facts)...")
    try:
        df = pd.read_csv(csv_path)
        df = df[df["customer_id"].apply(lambda v: _to_id_str(v) in allowed_ids)]
        if max_records:
            df = df.head(max_records)
        if df.empty:
            print("   No customer rows in CSV matching fact customer_ids")
            return {"success": True, "total_loaded": 0, "table_name": "customers_enrich"}
        id_key = "customer_id"
        # Use all CSV columns (except id) so Customer nodes get complete details (name, email, address, etc.)
        keys = [k for k in df.columns if k != id_key]
        if not keys:
            print("   No extra columns in CSV beyond customer_id")
            return {"success": True, "total_loaded": 0, "table_name": "customers_enrich"}
        total = 0
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            rows = []
            for _, row in batch.iterrows():
                r = {id_key: _to_id_str(row[id_key])}
                r.update(_row_to_props(row, exclude=[id_key]))
                rows.append(r)
            set_parts = [f"c.{k} = row.{k}" for k in keys]
            cypher = """
            UNWIND $rows AS row
            MATCH (c:Customer {id: row.customer_id})
            SET """ + ", ".join(set_parts)
            client.run(cypher, {"rows": rows})
            total += len(rows)
            print(f"   Progress: {total:,} customers enriched")
        print(f"✅ Enriched {total:,} customers")
        return {"success": True, "total_loaded": total, "table_name": "customers_enrich"}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e), "table_name": "customers_enrich"}


def load_transactions(
    client: Neo4jClient,
    csv_path: Path,
    transaction_type: str,
    batch_size: int = DEFAULT_BATCH,
    max_records: Optional[int] = None,
    schema_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load a transaction CSV into Transaction nodes and HAS_TRANSACTION (schema-driven if schema_fields)."""
    if not csv_path.exists():
        return {"success": False, "error": "File not found"}
    print(f"\n📂 Loading {transaction_type} from {csv_path.name}...")
    try:
        df = pd.read_csv(csv_path)
        if max_records:
            df = df.head(max_records)
        exclude_keys = {"transaction_id", "customer_id"}
        if schema_fields:
            other = [k for k in schema_fields if k not in exclude_keys and k in df.columns]
        else:
            other = [c for c in df.columns if c not in exclude_keys]
        total = 0
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            rows = []
            for _, row in batch.iterrows():
                r = {
                    "transaction_id": _to_id_str(row["transaction_id"]),
                    "customer_id": _to_id_str(row["customer_id"]),
                    "transaction_type": transaction_type,
                }
                for k in other:
                    v = row[k]
                    if pd.notna(v) and not (isinstance(v, float) and v != v):
                        if hasattr(v, "item"):
                            v = v.item()
                        if isinstance(v, pd.Timestamp):
                            v = str(v)
                        r[k] = v
                rows.append(r)
            set_parts = [f"t.{k} = row.{k}" for k in other]
            set_clause = ", ".join(set_parts) if set_parts else "t.amount = row.amount"
            # Create minimal Customer from transaction customer_id, then Transaction and HAS_TRANSACTION
            cypher = f"""
            UNWIND $rows AS row
            MERGE (c:Customer {{id: row.customer_id}})
            MERGE (t:Transaction {{id: row.transaction_id, type: row.transaction_type}})
            SET {set_clause}
            WITH c, t
            MERGE (c)-[:HAS_TRANSACTION]->(t)
            """
            client.run(cypher, {"rows": rows})
            total += len(rows)
            if total % 20000 == 0 or total == len(df):
                print(f"   Progress: {total:,} transactions")
        print(f"✅ Loaded {total:,} {transaction_type} transactions")
        return {"success": True, "total_loaded": total, "table_name": f"{transaction_type}_transactions"}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e), "table_name": f"{transaction_type}_transactions"}


def load_disputes(
    client: Neo4jClient,
    csv_path: Path,
    batch_size: int = DEFAULT_BATCH,
    max_records: Optional[int] = None,
    schema_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load disputes.csv into Dispute nodes, OPENED_DISPUTE and REFERENCES (schema-driven if schema_fields)."""
    if not csv_path.exists():
        return {"success": False, "error": "File not found"}
    print(f"\n📂 Loading Disputes from {csv_path.name}...")
    try:
        df = pd.read_csv(csv_path)
        if max_records:
            df = df.head(max_records)
        exclude_keys = {"dispute_id", "customer_id", "transaction_id", "transaction_type"}
        if schema_fields:
            other = [k for k in schema_fields if k not in exclude_keys and k in df.columns]
        else:
            other = [c for c in df.columns if c not in exclude_keys]
        total = 0
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            rows = []
            for _, row in batch.iterrows():
                r = {
                    "dispute_id": _to_id_str(row["dispute_id"]),
                    "customer_id": _to_id_str(row["customer_id"]),
                    "transaction_id": _to_id_str(row["transaction_id"]),
                    "transaction_type": str(row["transaction_type"]).lower(),
                }
                for k in other:
                    v = row[k]
                    if pd.notna(v) and not (isinstance(v, float) and v != v):
                        if hasattr(v, "item"):
                            v = v.item()
                        if isinstance(v, pd.Timestamp):
                            v = str(v)
                        r[k] = v
                rows.append(r)
            set_parts = [f"d.{k} = row.{k}" for k in other]
            set_clause = ", ".join(set_parts) if set_parts else "d.amount_disputed = row.amount_disputed"
            # Customer may already exist from transactions; MERGE so disputes-only customers get a node
            cypher = f"""
            UNWIND $rows AS row
            MERGE (c:Customer {{id: row.customer_id}})
            MERGE (d:Dispute {{id: row.dispute_id}})
            SET {set_clause}
            WITH c, d, row
            MERGE (c)-[:OPENED_DISPUTE]->(d)
            WITH d, row
            MATCH (t:Transaction {{id: row.transaction_id, type: row.transaction_type}})
            MERGE (d)-[:REFERENCES]->(t)
            """
            client.run(cypher, {"rows": rows})
            total += len(rows)
            if total % 10000 == 0 or total == len(df):
                print(f"   Progress: {total:,} disputes")
        print(f"✅ Loaded {total:,} disputes")
        return {"success": True, "total_loaded": total, "table_name": "disputes"}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e), "table_name": "disputes"}


def main():
    parser = argparse.ArgumentParser(
        description="Load openInt test data into Neo4j (schema-driven from DataHub schema)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Data root (default: data or repo data)")
    parser.add_argument("--max-customers", type=int, default=None, help="Max customer records")
    parser.add_argument("--max-transactions", type=int, default=None, help="Max transaction records per table")
    parser.add_argument("--max-disputes", type=int, default=None, help="Max dispute records")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Cypher UNWIND batch size")
    parser.add_argument("--only-customers", action="store_true", help="Load only customers")
    parser.add_argument("--only-transactions", action="store_true", help="Load only transaction tables")
    parser.add_argument("--only-disputes", action="store_true", help="Load only disputes")
    parser.add_argument("--skip-disputes", action="store_true", help="Skip disputes (load customers + transactions only)")
    parser.add_argument(
        "--only-enrich",
        action="store_true",
        help="Only enrich existing Customer nodes from dimensions/customers.csv (read customer_ids from transactions + disputes CSVs, or from Neo4j if none)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete all nodes and relationships in Neo4j before loading",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Delete all nodes and relationships in Neo4j and exit (no load)",
    )
    args = parser.parse_args()

    global BASE_DIR, DIMENSIONS_DIR, FACTS_DIR
    if args.data_dir is not None:
        BASE_DIR = Path(args.data_dir)
        DIMENSIONS_DIR = BASE_DIR / "dimensions"
        FACTS_DIR = BASE_DIR / "facts"

    # Clean-only: connect, delete all, exit
    if args.clean_only:
        try:
            client = Neo4jClient()
            client.connect()
            if not client.verify_connectivity():
                print("❌ Cannot connect to Neo4j")
                return
            print("✅ Connected to Neo4j")
            print("🧹 Deleting all nodes and relationships...")
            client.delete_all()
            print("✅ Done.")
        except Exception as e:
            print(f"❌ {e}")
        return

    if not BASE_DIR.exists():
        print(f"❌ Data directory not found: {BASE_DIR}")
        return

    schema = get_dataset_schemas() if get_dataset_schemas else None
    if schema:
        print("📋 Using DataHub schema (openint-datahub/schemas.py) for field definitions")

    print("=" * 60)
    print("🏦 Loading openInt test data into Neo4j")
    print("=" * 60)

    try:
        client = Neo4jClient()
        client.connect()
        if not client.verify_connectivity():
            print("❌ Cannot connect to Neo4j. Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.")
            return
        print("✅ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return

    if args.clean:
        print("\n🧹 Cleaning Neo4j (deleting all nodes and relationships)...")
        client.delete_all()
        print("   Done.\n")

    results = []
    load_tx = args.only_transactions or (not args.only_customers and not args.only_disputes and not args.only_enrich)
    load_disp = args.only_disputes or (not args.skip_disputes and not args.only_customers and not args.only_transactions and not args.only_enrich)
    # Reverse lookup: enrich customers only for ids that appear in facts (unless --only-customers)
    load_cust_full = args.only_customers
    load_cust_enrich = args.only_enrich or (
        (not args.only_customers and not args.only_transactions and not args.only_disputes) or (load_tx or load_disp)
    )

    # Collect customer_ids: from fact CSVs (transactions + disputes) or from Neo4j when --only-enrich
    allowed_customer_ids: set = set()
    if load_cust_enrich:
        if load_tx or load_disp or args.only_enrich:
            max_tx = None if args.only_enrich else args.max_transactions
            max_disp = None if args.only_enrich else args.max_disputes
            allowed_customer_ids = collect_customer_ids_from_facts(
                FACTS_DIR, DIMENSIONS_DIR, max_tx, max_disp
            )
            if allowed_customer_ids:
                print(f"📋 From transactions + disputes CSVs: {len(allowed_customer_ids):,} distinct customer_ids")
        if args.only_enrich and not allowed_customer_ids:
            allowed_customer_ids = collect_customer_ids_from_graph(client)
            if allowed_customer_ids:
                print(f"📋 From Neo4j Customer nodes: {len(allowed_customer_ids):,} distinct customer ids to enrich")
            else:
                print("⚠️ No customer_ids from CSVs or graph. Ensure data/facts/*.csv exist or load transactions/disputes first.")

    # (1) Transactions first: create minimal Customer + Transaction + HAS_TRANSACTION
    if load_tx:
        for dataset_name, spec in LOAD_SPEC.items():
            if spec[2] != "Transaction":
                continue
            subdir, fname, _, _, ttype = spec
            csv_path = (DIMENSIONS_DIR if subdir == "dimensions" else FACTS_DIR) / fname
            r = load_transactions(
                client,
                csv_path,
                ttype,
                batch_size=args.batch_size,
                max_records=args.max_transactions,
                schema_fields=_schema_field_names(dataset_name),
            )
            results.append(r)

    # (2) Disputes: OPENED_DISPUTE, REFERENCES (from type-specific CSVs: ach_disputes, credit_disputes, etc.)
    if load_disp:
        max_per_file = (args.max_disputes // 6) if args.max_disputes else None
        for dataset_name, spec in LOAD_SPEC.items():
            if spec[2] != "Dispute":
                continue
            subdir, fname, _, _, _ = spec
            csv_path = (DIMENSIONS_DIR if subdir == "dimensions" else FACTS_DIR) / fname
            r = load_disputes(
                client,
                csv_path,
                batch_size=args.batch_size,
                max_records=max_per_file,
                schema_fields=_schema_field_names(dataset_name) or _schema_field_names("disputes"),
            )
            results.append(r)

    # (3) Enrich Customer nodes from customers.csv with full details (all properties from schema/CSV)
    if load_cust_enrich and allowed_customer_ids and "customers" in LOAD_SPEC:
        subdir, fname, _, _, _ = LOAD_SPEC["customers"]
        csv_path = (DIMENSIONS_DIR if subdir == "dimensions" else FACTS_DIR) / fname
        if args.only_enrich:
            print(f"📂 Enrich-only: updating Customer nodes with all properties from {csv_path}")
        r = enrich_customers(
            client,
            csv_path,
            allowed_customer_ids,
            batch_size=args.batch_size,
            max_records=args.max_customers,
            schema_fields=_schema_field_names("customers"),
        )
        results.append(r)
    elif load_cust_full and "customers" in LOAD_SPEC:
        # Legacy: load full customers dimension (--only-customers)
        subdir, fname, _, _, _ = LOAD_SPEC["customers"]
        csv_path = (DIMENSIONS_DIR if subdir == "dimensions" else FACTS_DIR) / fname
        r = load_customers(
            client,
            csv_path,
            batch_size=args.batch_size,
            max_records=args.max_customers,
            schema_fields=_schema_field_names("customers"),
        )
        results.append(r)

    client.close()

    print("\n" + "=" * 60)
    print("✅ Neo4j load complete")
    print("=" * 60)
    total = sum(x.get("total_loaded", 0) for x in results if x.get("success"))
    print(f"   Total records: {total:,}")


if __name__ == "__main__":
    main()
