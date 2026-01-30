"""
Load openInt test data into Neo4j graph database (schema-driven, reverse lookup).
Uses DataHub schema (openint-datahub/schemas.py) for field definitions.
Order: (1) Load all transactions first, creating minimal Customer nodes from transaction
customer_id and HAS_TRANSACTION; (2) Load disputes (OPENED_DISPUTE, REFERENCES); (3) Enrich
Customer nodes from customers.csv only for customer_ids that appear in the facts.
This avoids loading the full customers dimension; only customers with transactions/disputes get full attributes.
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

# Data directories (same layout as Milvus loader)
BASE_DIR = Path("testdata")
if not BASE_DIR.exists():
    BASE_DIR = _repo_root / "testdata"
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
    "disputes": ("facts", "disputes.csv", "Dispute", "dispute_id", None),
}

# Batch size for Cypher UNWIND
DEFAULT_BATCH = 5000


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
    """Scan transaction and dispute CSVs and return the set of customer_ids that appear in facts."""
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
            seen.update(df["customer_id"].astype(str).unique())
    # disputes
    subdir, fname, _, _, _ = LOAD_SPEC["disputes"]
    csv_path = (dimensions_dir if subdir == "dimensions" else facts_dir) / fname
    if csv_path.exists():
        df = pd.read_csv(csv_path, usecols=["customer_id"])
        if max_disputes:
            df = df.head(max_disputes)
        seen.update(df["customer_id"].astype(str).unique())
    return seen


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
                r = {id_key: str(row[id_key])}
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
        df = df[df["customer_id"].astype(str).isin(allowed_ids)]
        if max_records:
            df = df.head(max_records)
        if df.empty:
            print("   No customer rows in CSV matching fact customer_ids")
            return {"success": True, "total_loaded": 0, "table_name": "customers_enrich"}
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
                r = {id_key: str(row[id_key])}
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
                    "transaction_id": str(row["transaction_id"]),
                    "customer_id": str(row["customer_id"]),
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
                    "dispute_id": str(row["dispute_id"]),
                    "customer_id": str(row["customer_id"]),
                    "transaction_id": str(row["transaction_id"]),
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
    parser.add_argument("--data-dir", type=Path, default=None, help="Test data root (default: testdata or repo testdata)")
    parser.add_argument("--max-customers", type=int, default=None, help="Max customer records")
    parser.add_argument("--max-transactions", type=int, default=None, help="Max transaction records per table")
    parser.add_argument("--max-disputes", type=int, default=None, help="Max dispute records")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Cypher UNWIND batch size")
    parser.add_argument("--only-customers", action="store_true", help="Load only customers")
    parser.add_argument("--only-transactions", action="store_true", help="Load only transaction tables")
    parser.add_argument("--only-disputes", action="store_true", help="Load only disputes")
    parser.add_argument("--skip-disputes", action="store_true", help="Skip disputes (load customers + transactions only)")
    args = parser.parse_args()

    global BASE_DIR, DIMENSIONS_DIR, FACTS_DIR
    if args.data_dir is not None:
        BASE_DIR = Path(args.data_dir)
        DIMENSIONS_DIR = BASE_DIR / "dimensions"
        FACTS_DIR = BASE_DIR / "facts"

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

    results = []
    load_tx = args.only_transactions or (not args.only_customers and not args.only_disputes)
    load_disp = args.only_disputes or (not args.skip_disputes and not args.only_customers and not args.only_transactions)
    # Reverse lookup: enrich customers only for ids that appear in facts (unless --only-customers)
    load_cust_full = args.only_customers
    load_cust_enrich = (not args.only_customers and not args.only_transactions and not args.only_disputes) or (load_tx or load_disp)

    # Optional: collect customer_ids from facts so we only enrich those (avoids full customers.csv scan in memory for enrich)
    allowed_customer_ids = set()
    if load_cust_enrich and (load_tx or load_disp):
        allowed_customer_ids = collect_customer_ids_from_facts(
            FACTS_DIR, DIMENSIONS_DIR, args.max_transactions, args.max_disputes
        )
        print(f"📋 Reverse lookup: {len(allowed_customer_ids):,} distinct customer_ids from facts")

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

    # (2) Disputes: OPENED_DISPUTE, REFERENCES (Customer MERGEd if not already present)
    if load_disp and "disputes" in LOAD_SPEC:
        subdir, fname, _, _, _ = LOAD_SPEC["disputes"]
        csv_path = (DIMENSIONS_DIR if subdir == "dimensions" else FACTS_DIR) / fname
        r = load_disputes(
            client,
            csv_path,
            batch_size=args.batch_size,
            max_records=args.max_disputes,
            schema_fields=_schema_field_names("disputes"),
        )
        results.append(r)

    # (3) Enrich Customer nodes from customers.csv only for customer_ids that appear in facts
    if load_cust_enrich and allowed_customer_ids and "customers" in LOAD_SPEC:
        subdir, fname, _, _, _ = LOAD_SPEC["customers"]
        csv_path = (DIMENSIONS_DIR if subdir == "dimensions" else FACTS_DIR) / fname
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
