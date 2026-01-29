"""
Load openInt test data into Neo4j graph database.
Reads fact/dimension CSVs, creates nodes (Customer, Transaction, Dispute) and
relationships (HAS_TRANSACTION, OPENED_DISPUTE, REFERENCES).
Uses the same testdata source as load_openint_data_to_milvus.py.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

# Resolve repo root and add openint-graph
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
openint_graph = _repo_root / "openint-graph"
if openint_graph.exists() and str(openint_graph) not in sys.path:
    sys.path.insert(0, str(openint_graph))

try:
    from neo4j_client import Neo4jClient
except ImportError:
    try:
        from openint_graph.neo4j_client import Neo4jClient
    except ImportError:
        print("Error: Could not import Neo4jClient. Ensure openint-graph is on PYTHONPATH.")
        sys.exit(1)

# Data directories (same layout as Milvus loader)
BASE_DIR = Path("testdata")
if not BASE_DIR.exists():
    BASE_DIR = _repo_root / "testdata"
DIMENSIONS_DIR = BASE_DIR / "dimensions"
FACTS_DIR = BASE_DIR / "facts"

# Batch size for Cypher UNWIND
DEFAULT_BATCH = 5000


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
) -> Dict[str, Any]:
    """Load customers.csv into Customer nodes."""
    if not csv_path.exists():
        return {"success": False, "error": "File not found"}
    print(f"\n📂 Loading Customers from {csv_path.name}...")
    try:
        df = pd.read_csv(csv_path)
        if max_records:
            df = df.head(max_records)
        total = 0
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            rows = []
            for _, row in batch.iterrows():
                r = {"customer_id": str(row["customer_id"])}
                r.update(_row_to_props(row, exclude=["customer_id"]))
                rows.append(r)
            # Build SET from CSV columns (exclude customer_id used in MERGE)
            keys = [k for k in batch.columns if k != "customer_id"]
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


def load_transactions(
    client: Neo4jClient,
    csv_path: Path,
    transaction_type: str,
    batch_size: int = DEFAULT_BATCH,
    max_records: Optional[int] = None,
) -> Dict[str, Any]:
    """Load a transaction CSV into Transaction nodes and HAS_TRANSACTION relationships."""
    if not csv_path.exists():
        return {"success": False, "error": "File not found"}
    print(f"\n📂 Loading {transaction_type} from {csv_path.name}...")
    try:
        df = pd.read_csv(csv_path)
        if max_records:
            df = df.head(max_records)
        cols = [c for c in df.columns if c in ("transaction_id", "customer_id")]
        other = [c for c in df.columns if c not in ("transaction_id", "customer_id")]
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
            cypher = f"""
            UNWIND $rows AS row
            MERGE (t:Transaction {{id: row.transaction_id, type: row.transaction_type}})
            SET {set_clause}
            WITH t, row
            MATCH (c:Customer {{id: row.customer_id}})
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
) -> Dict[str, Any]:
    """Load disputes.csv into Dispute nodes, OPENED_DISPUTE and REFERENCES relationships."""
    if not csv_path.exists():
        return {"success": False, "error": "File not found"}
    print(f"\n📂 Loading Disputes from {csv_path.name}...")
    try:
        df = pd.read_csv(csv_path)
        if max_records:
            df = df.head(max_records)
        other = [c for c in df.columns if c not in ("dispute_id", "customer_id", "transaction_id", "transaction_type")]
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
            cypher = f"""
            UNWIND $rows AS row
            MERGE (d:Dispute {{id: row.dispute_id}})
            SET {set_clause}
            WITH d, row
            MATCH (c:Customer {{id: row.customer_id}})
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
        description="Load openInt test data into Neo4j",
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
    load_cust = args.only_customers or (not args.only_transactions and not args.only_disputes)
    load_tx = args.only_transactions or (not args.only_customers and not args.only_disputes)
    load_disp = args.only_disputes or (not args.skip_disputes and not args.only_customers and not args.only_transactions)

    if load_cust:
        r = load_customers(
            client,
            DIMENSIONS_DIR / "customers.csv",
            batch_size=args.batch_size,
            max_records=args.max_customers,
        )
        results.append(r)

    if load_tx:
        for fname, ttype in [
            ("ach_transactions.csv", "ach"),
            ("wire_transactions.csv", "wire"),
            ("credit_transactions.csv", "credit"),
            ("debit_transactions.csv", "debit"),
            ("check_transactions.csv", "check"),
        ]:
            r = load_transactions(
                client,
                FACTS_DIR / fname,
                ttype,
                batch_size=args.batch_size,
                max_records=args.max_transactions,
            )
            results.append(r)

    if load_disp:
        r = load_disputes(
            client,
            FACTS_DIR / "disputes.csv",
            batch_size=args.batch_size,
            max_records=args.max_disputes,
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
