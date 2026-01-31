#!/usr/bin/env python3
"""
Check Neo4j graph data for a specific customer.
Verifies: Customer node exists, properties (first_name, last_name, email, phone),
and relationships (HAS_TRANSACTION, OPENED_DISPUTE).
"""

import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))
_openint_graph = _repo_root / "openint-graph"
if _openint_graph.exists():
    sys.path.insert(0, str(_openint_graph))

try:
    from neo4j_client import Neo4jClient
except ImportError:
    try:
        from openint_graph.neo4j_client import Neo4jClient
    except ImportError:
        print("Error: Could not import Neo4jClient")
        sys.exit(1)


def main():
    customer_id = os.environ.get("CHECK_CUSTOMER_ID", "1000003621")
    print(f"=" * 60)
    print(f"🔍 Neo4j Check: Customer {customer_id}")
    print(f"=" * 60)

    try:
        client = Neo4jClient()
        client.connect()
        if not client.verify_connectivity():
            print("❌ Cannot connect to Neo4j")
            return 1
        print("✅ Connected to Neo4j\n")
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return 1

    # 1. Direct lookup by id (string match - how enrich-agent does it)
    print("1. Customer node lookup (toString(n.id) = $idStr):")
    rows = client.run(
        "MATCH (n:Customer) WHERE toString(n.id) = $idStr RETURN n",
        {"idStr": customer_id},
    )
    if not rows:
        print("   ❌ No Customer node found")
        # Try alternate: id as stored (maybe int or float)
        rows2 = client.run(
            "MATCH (n:Customer) WHERE n.id = $idStr OR n.id = $idInt RETURN n",
            {"idStr": customer_id, "idInt": int(customer_id) if customer_id.isdigit() else 0},
        )
        if rows2:
            print("   (But found with n.id = $idStr - check id type)")
            rows = rows2
        else:
            print("\n2. Sample Customer nodes (first 3) to see id format:")
            sample = client.run("MATCH (c:Customer) RETURN c.id AS id, keys(c) AS keys LIMIT 3")
            for r in sample or []:
                print(f"   id={r.get('id')!r} (type={type(r.get('id')).__name__}), keys={r.get('keys')}")
            return 1
    else:
        node = rows[0].get("n")
        props = dict(node) if node else {}
        print(f"   ✅ Found. Properties: {list(props.keys())}")
        for k in ("first_name", "last_name", "email", "phone", "id"):
            v = props.get(k)
            print(f"      {k}: {v!r}")
        if not any(props.get(k) for k in ("first_name", "last_name", "email", "phone")):
            print("   ⚠️  Missing first_name, last_name, email, phone — Customer likely not enriched from customers.csv")

    # 2. Relationships
    print("\n2. Relationships:")
    rels = client.run(
        """
        MATCH (c:Customer {id: $idStr})-[:HAS_TRANSACTION]->(t:Transaction)
        RETURN count(t) AS tx_count
        """,
        {"idStr": customer_id},
    )
    tx_count = (rels[0]["tx_count"] if rels else 0) or 0
    print(f"   HAS_TRANSACTION -> Transaction: {tx_count}")

    # Try with toString in case id is stored differently
    if tx_count == 0:
        rels2 = client.run(
            "MATCH (c:Customer)-[:HAS_TRANSACTION]->(t:Transaction) WHERE toString(c.id) = $idStr RETURN count(t) AS tx_count",
            {"idStr": customer_id},
        )
        tx_count = (rels2[0]["tx_count"] if rels2 else 0) or 0
        if tx_count > 0:
            print(f"   (with toString match): HAS_TRANSACTION -> Transaction: {tx_count}")

    disp = client.run(
        """
        MATCH (c:Customer {id: $idStr})-[:OPENED_DISPUTE]->(d:Dispute)
        RETURN count(d) AS disp_count
        """,
        {"idStr": customer_id},
    )
    disp_count = (disp[0]["disp_count"] if disp else 0) or 0
    if disp_count == 0:
        disp2 = client.run(
            "MATCH (c:Customer)-[:OPENED_DISPUTE]->(d:Dispute) WHERE toString(c.id) = $idStr RETURN count(d) AS disp_count",
            {"idStr": customer_id},
        )
        disp_count = (disp2[0]["disp_count"] if disp2 else 0) or 0
    print(f"   OPENED_DISPUTE -> Dispute: {disp_count}")

    # 3. ID type check — Customer.id might be int/float vs string
    print("\n3. ID storage format check:")
    type_check = client.run(
        "MATCH (c:Customer) WHERE toString(c.id) = $idStr RETURN c.id AS raw_id, size(keys(c)) AS prop_count",
        {"idStr": customer_id},
    )
    if type_check:
        r = type_check[0]
        print(f"   raw_id={r.get('raw_id')!r}, type={type(r.get('raw_id')).__name__}, prop_count={r.get('prop_count')}")

    # 4. customers.csv check
    print("\n4. customers.csv check:")
    csv_path = _repo_root / "data" / "dimensions" / "customers.csv"
    if not csv_path.exists():
        csv_path = Path("data/dimensions/customers.csv")
    if csv_path.exists():
        import csv
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("customer_id", "")).strip() == customer_id:
                    print(f"   ✅ Found in CSV. Columns: {list(row.keys())}")
                    for k in ("first_name", "last_name", "email", "phone"):
                        print(f"      {k}: {row.get(k)!r}")
                    break
            else:
                print(f"   ⚠️  customer_id {customer_id} NOT in customers.csv")
    else:
        print(f"   ⚠️  customers.csv not found at {csv_path}")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
