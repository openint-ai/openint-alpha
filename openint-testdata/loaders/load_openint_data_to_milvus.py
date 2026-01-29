"""
Load openInt Test Data into Milvus Vector Database
Converts structured CSV data into searchable vector embeddings
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# Add parent directory to path to import milvus_client
parent_dir = os.path.join(os.path.dirname(__file__), '../../')
sys.path.insert(0, parent_dir)

try:
    # Import from openint-vectordb package (handle hyphenated directory name)
    vectordb_path = os.path.join(parent_dir, 'openint-vectordb', 'milvus')
    if vectordb_path not in sys.path:
        sys.path.insert(0, vectordb_path)
    from milvus_client import MilvusClient
except ImportError:
    # Fallback: try old location
    try:
        root_dir = os.path.join(os.path.dirname(__file__), '../../../')
        vectordb_path = os.path.join(root_dir, 'openint-vectordb', 'milvus')
        sys.path.insert(0, vectordb_path)
        from milvus_client import MilvusClient
    except ImportError:
        print("Error: Could not import MilvusClient. Please ensure openint-vectordb/milvus/milvus_client.py exists.")
        sys.exit(1)

# Batch size: 10K–50K based on system; override with MILVUS_LOAD_BATCH_SIZE
def _default_batch_size() -> int:
    try:
        env_val = os.environ.get("MILVUS_LOAD_BATCH_SIZE")
        if env_val is not None:
            return max(10_000, min(50_000, int(env_val)))
    except (TypeError, ValueError):
        pass
    cpu = (os.cpu_count() or 4)
    if cpu >= 8:
        return 50_000
    if cpu >= 4:
        return 25_000
    return 10_000


# Data directories
BASE_DIR = Path("testdata")
DIMENSIONS_DIR = BASE_DIR / "dimensions"
FACTS_DIR = BASE_DIR / "facts"
STATIC_DIR = BASE_DIR / "static"


def row_to_text(row: pd.Series, table_name: str, csv_columns: List[str]) -> str:
    """
    Convert a DataFrame row to a searchable text representation using CSV schema as-is.
    No appending or transformation - uses exact column names and values from CSV.
    
    Args:
        row: pandas Series representing a row
        table_name: Name of the table for context
        csv_columns: List of column names from CSV header (schema)
        
    Returns:
        Text representation using CSV columns exactly as provided
    """
    # Use CSV columns exactly as provided - no filtering or transformation
    parts = []
    
    # Build text from CSV columns in order, using column names and values as-is
    for col in csv_columns:
        if col in row and pd.notna(row[col]) and row[col] != "":
            # Use column name and value exactly as in CSV
            parts.append(f"{col}: {row[col]}")
    
    # If no values, return minimal representation
    if not parts:
        return f"{table_name} record"
    
    return " | ".join(parts)


def row_to_metadata(row: pd.Series, table_name: str) -> Dict[str, Any]:
    """
    Convert a DataFrame row to metadata dictionary compatible with MilvusClient schema.
    
    Args:
        row: pandas Series representing a row
        table_name: Name of the table
        
    Returns:
        Metadata dictionary compatible with MilvusClient schema
    """
    # MilvusClient expects specific metadata fields:
    # file_type, file_name, file_size, category, chunk_index, total_chunks, original_id
    
    # Store the full row data as JSON in content, use metadata for categorization
    category = table_name.split("_")[0] if "_" in table_name else table_name
    
    # Create a compact file_name from key identifiers
    file_name_parts = []
    if 'customer_id' in row and pd.notna(row['customer_id']):
        file_name_parts.append(str(row['customer_id']))
    elif 'transaction_id' in row and pd.notna(row['transaction_id']):
        file_name_parts.append(str(row['transaction_id']))
    elif 'country_code' in row and pd.notna(row['country_code']):
        file_name_parts.append(str(row['country_code']))
    elif 'state_code' in row and pd.notna(row['state_code']):
        file_name_parts.append(str(row['state_code']))
    elif 'zip_code' in row and pd.notna(row['zip_code']):
        file_name_parts.append(str(row['zip_code']))
    
    file_name = "_".join(file_name_parts) if file_name_parts else f"{table_name}_record"
    
    # Limit file_name to 500 chars (Milvus limit)
    if len(file_name) > 500:
        file_name = file_name[:500]
    
    metadata = {
        "file_type": table_name,  # Store table name as file_type
        "file_name": file_name,
        "file_size": 0,  # Not applicable for structured data
        "category": category,
        "chunk_index": 0,
        "total_chunks": 0,
        "original_id": ""
    }
    
    return metadata


def load_csv_to_milvus(
    csv_path: Path,
    table_name: str,
    client: MilvusClient,
    batch_size: int = 20_000,
    max_records: Optional[int] = None
):
    """
    Load a CSV file into Milvus.
    
    Args:
        csv_path: Path to CSV file
        table_name: Name of the table (used for categorization)
        client: MilvusClient instance
        batch_size: Number of records per batch
        max_records: Maximum number of records to load (None for all)
        
    Returns:
        dict with loading statistics
    """
    if not csv_path.exists():
        print(f"⚠️  File not found: {csv_path}")
        return {"success": False, "error": "File not found"}
    
    print(f"\n📂 Loading {table_name} from {csv_path.name}...")
    
    try:
        # Read CSV in chunks for large files
        chunk_size = 10000
        total_loaded = 0
        total_errors = 0
        
        # Get file size for progress tracking
        file_size = csv_path.stat().st_size / (1024 * 1024)  # MB
        print(f"   File size: {file_size:.2f} MB")
        
        # Read first chunk to get CSV schema (header) and total rows estimate
        first_chunk = pd.read_csv(csv_path, nrows=1)
        csv_columns = list(first_chunk.columns)
        print(f"   CSV Schema (columns): {', '.join(csv_columns)}")
        total_rows = sum(1 for _ in open(csv_path)) - 1  # Subtract header
        print(f"   Total rows: {total_rows:,}")
        
        if max_records:
            total_rows = min(total_rows, max_records)
            print(f"   Loading first {max_records:,} records")
        
        # Process in chunks - use CSV schema from header
        records = []
        chunk_num = 0
        
        for chunk_df in pd.read_csv(csv_path, chunksize=chunk_size):
            # Ensure columns match CSV schema (handle any pandas column reordering)
            chunk_df = chunk_df[csv_columns]
            if max_records and total_loaded >= max_records:
                break
            
            chunk_num += 1
            
            # Limit chunk if we're near max_records
            if max_records:
                remaining = max_records - total_loaded
                if remaining < len(chunk_df):
                    chunk_df = chunk_df.head(remaining)
            
            # Get CSV columns (schema) from first chunk - use as-is
            csv_columns = list(chunk_df.columns)
            
            for idx, row in chunk_df.iterrows():
                try:
                    # Convert row to text using CSV schema exactly as provided
                    text_content = row_to_text(row, table_name, csv_columns)
                    
                    # Create metadata using CSV schema
                    metadata = row_to_metadata(row, table_name)
                    
                    # Create record ID (must be unique) - use first column with ID-like name or first column
                    record_id = None
                    id_columns = ['customer_id', 'transaction_id', 'country_code', 'state_code', 'zip_code', 'dispute_id']
                    for id_col in id_columns:
                        if id_col in row and pd.notna(row[id_col]):
                            record_id = f"{table_name}_{row[id_col]}"
                            break
                    
                    # Fallback: use first column value if no ID column found
                    if not record_id and len(csv_columns) > 0:
                        first_col = csv_columns[0]
                        if first_col in row and pd.notna(row[first_col]):
                            record_id = f"{table_name}_{row[first_col]}"
                    
                    # Final fallback: use index
                    if not record_id:
                        record_id = f"{table_name}_{chunk_num}_{idx}_{len(records)}"
                    
                    # Limit record ID to 500 chars (Milvus limit)
                    if len(record_id) > 500:
                        record_id = record_id[:500]
                    
                    # Store searchable text plus "Structured Data: {...}" so UI can show table
                    # text_content = "column: value | column: value"; append JSON for parseStructuredContent
                    row_dict = {}
                    for col in csv_columns:
                        if col in row.index:
                            v = row[col]
                            if pd.isna(v) or v == "":
                                continue
                            if hasattr(v, "item"):  # numpy scalar
                                row_dict[col] = v.item()
                            elif isinstance(v, (pd.Timestamp, datetime)):
                                row_dict[col] = str(v)
                            else:
                                row_dict[col] = v
                    structured_suffix = "\nStructured Data: " + json.dumps(row_dict) if row_dict else ""
                    content = text_content + structured_suffix
                    
                    # Truncate to Milvus limit (65535 chars) if needed
                    MAX_CONTENT_LENGTH = 65535
                    if len(content) > MAX_CONTENT_LENGTH:
                        content = content[:MAX_CONTENT_LENGTH]
                    
                    record = {
                        "id": record_id,
                        "content": content,  # Only CSV data as-is, no appending
                        "metadata": metadata
                    }
                    
                    records.append(record)
                    
                    # Batch insert when batch_size reached
                    if len(records) >= batch_size:
                        result = client.upsert_records(records, batch_size=batch_size, auto_chunk=False)
                        if result["success"]:
                            total_loaded += result["total_upserted"]
                            print(f"   Progress: {total_loaded:,}/{total_rows:,} records loaded")
                        else:
                            total_errors += len(records)
                            print(f"   ⚠️  Batch failed: {result.get('errors', [])}")
                        records = []
                
                except Exception as e:
                    total_errors += 1
                    if total_errors <= 5:  # Only print first few errors
                        print(f"   ⚠️  Error processing row {idx}: {e}")
            
            # Insert remaining records
            if records:
                result = client.upsert_records(records, batch_size=batch_size, auto_chunk=False)
                if result["success"]:
                    total_loaded += result["total_upserted"]
                else:
                    total_errors += len(records)
                records = []
        
        print(f"✅ Loaded {total_loaded:,} records from {table_name}")
        if total_errors > 0:
            print(f"   ⚠️  {total_errors} errors encountered")
        
        return {
            "success": True,
            "total_loaded": total_loaded,
            "total_errors": total_errors,
            "table_name": table_name
        }
    
    except Exception as e:
        error_msg = f"Error loading {table_name}: {str(e)}"
        print(f"❌ {error_msg}")
        return {"success": False, "error": error_msg}


def main():
    """Main function to load all openInt test data into Milvus"""
    parser = argparse.ArgumentParser(
        description="Load openInt test data into Milvus vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load all data
  python load_openInt_data_to_milvus.py

  # Load only one category
  python load_openInt_data_to_milvus.py --only-customers
  python load_openInt_data_to_milvus.py --only-transactions
  python load_openInt_data_to_milvus.py --only-static

  # Load only first 10,000 customers and 100,000 transactions (for testing)
  python load_openInt_data_to_milvus.py --max-customers 10000 --max-transactions 100000

  # Load only dimension tables (no transactions)
  python load_openInt_data_to_milvus.py --skip-transactions
        """
    )
    parser.add_argument(
        "--max-customers",
        type=int,
        default=None,
        help="Maximum number of customer records to load (default: all)"
    )
    parser.add_argument(
        "--max-transactions",
        type=int,
        default=None,
        help="Maximum number of transaction records per table to load (default: all)"
    )
    parser.add_argument(
        "--skip-transactions",
        action="store_true",
        help="Skip loading transaction fact tables"
    )
    parser.add_argument(
        "--skip-dimensions",
        action="store_true",
        help="Skip loading dimension tables (customers, static tables)"
    )
    only_group = parser.add_mutually_exclusive_group()
    only_group.add_argument(
        "--only-customers",
        action="store_true",
        help="Load only customers (dimensions/customers.csv); skip transactions and static"
    )
    only_group.add_argument(
        "--only-transactions",
        action="store_true",
        help="Load only transaction fact tables (ACH, wire, credit, debit, check); skip customers and static"
    )
    only_group.add_argument(
        "--only-static",
        action="store_true",
        help="Load only static dimension tables (country_codes, state_codes, zip_codes); skip customers and transactions"
    )
    default_batch = _default_batch_size()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_batch,
        help=f"Batch size for loading records (default: {default_batch:,}, from system/config; use MILVUS_LOAD_BATCH_SIZE to override; range 10K–50K)"
    )

    args = parser.parse_args()

    # --only-*: load exactly one category; otherwise use skip flags
    load_customers = args.only_customers or (
        not args.skip_dimensions and not args.only_transactions and not args.only_static
    )
    load_static = args.only_static or (
        not args.skip_dimensions and not args.only_customers and not args.only_transactions
    )
    load_transactions = args.only_transactions or (
        not args.skip_transactions and not args.only_customers and not args.only_static
    )

    print("=" * 80)
    print("🏦 Loading openInt Test Data into Milvus")
    print("=" * 80)

    # Check if testdata directory exists
    if not BASE_DIR.exists():
        print(f"\n❌ Test data directory not found: {BASE_DIR}")
        print(f"\n💡 Please generate test data first by running:")
        print(f"   python generate_openInt_test_data.py")
        return

    # Check only the directories we need
    missing_dirs = []
    if load_customers and not DIMENSIONS_DIR.exists():
        missing_dirs.append("dimensions")
    if load_transactions and not FACTS_DIR.exists():
        missing_dirs.append("facts")
    if load_static and not STATIC_DIR.exists():
        missing_dirs.append("static")

    if missing_dirs:
        print(f"\n❌ Missing test data directories: {', '.join(missing_dirs)}")
        print(f"\n💡 Please generate test data first by running:")
        print(f"   python generate_openInt_test_data.py")
        return

    if args.max_customers:
        print(f"   ⚠️  Limiting customers to {args.max_customers:,} records")
    if args.max_transactions:
        print(f"   ⚠️  Limiting transactions to {args.max_transactions:,} records per table")
    if args.only_customers:
        print(f"   📌 Loading only: customers")
    elif args.only_transactions:
        print(f"   📌 Loading only: transactions")
    elif args.only_static:
        print(f"   📌 Loading only: static (country, state, zip)")
    else:
        if args.skip_transactions:
            print(f"   ⚠️  Skipping transaction tables")
        if args.skip_dimensions:
            print(f"   ⚠️  Skipping dimension tables")
    print(f"   📦 Batch size: {args.batch_size:,} (10K–50K recommended)")

    # Initialize Milvus client
    try:
        client = MilvusClient()
        print(f"\n✅ Connected to Milvus")
        print(f"   Collection: {client.collection_name}")
    except Exception as e:
        print(f"\n❌ Failed to connect to Milvus: {e}")
        print("   Please ensure Milvus is running and accessible")
        return
    
    results = []

    # Load customers only
    if load_customers:
        print("\n" + "=" * 80)
        print("📊 Loading Customers")
        print("=" * 80)
        customers_file = DIMENSIONS_DIR / "customers.csv"
        if customers_file.exists():
            result = load_csv_to_milvus(
                customers_file,
                "customers",
                client,
                batch_size=args.batch_size,
                max_records=args.max_customers
            )
            results.append(result)
        else:
            print(f"⚠️  Customers file not found: {customers_file}")

    # Load static dimension tables only (country, state, zip)
    if load_static:
        print("\n" + "=" * 80)
        print("📋 Loading Static Dimension Tables")
        print("=" * 80)
        static_tables = [
            ("country_codes.csv", "country_codes"),
            ("state_codes.csv", "state_codes"),
            ("zip_codes.csv", "zip_codes"),
        ]
        for filename, table_name in static_tables:
            file_path = STATIC_DIR / filename
            if file_path.exists():
                result = load_csv_to_milvus(
                    file_path,
                    table_name,
                    client,
                    batch_size=args.batch_size
                )
                results.append(result)
            else:
                print(f"⚠️  File not found: {file_path}")
                print(f"   💡 Run 'python generate_openInt_test_data.py' to generate test data")

    # Load fact tables (transaction data)
    if load_transactions:
        print("\n" + "=" * 80)
        print("💳 Loading Transaction Fact Tables")
        print("=" * 80)
        
        fact_tables = [
            ("ach_transactions.csv", "ach_transactions"),
            ("wire_transactions.csv", "wire_transactions"),
            ("credit_transactions.csv", "credit_transactions"),
            ("debit_transactions.csv", "debit_transactions"),
            ("check_transactions.csv", "check_transactions"),
        ]
        
        for filename, table_name in fact_tables:
            file_path = FACTS_DIR / filename
            if file_path.exists():
                result = load_csv_to_milvus(
                    file_path, 
                    table_name, 
                    client, 
                    batch_size=args.batch_size,
                    max_records=args.max_transactions
                )
                results.append(result)
            else:
                print(f"⚠️  File not found: {file_path}")
                print(f"   💡 Run 'python generate_openInt_test_data.py' to generate test data")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ Data Loading Complete!")
    print("=" * 80)
    
    total_loaded = sum(r.get("total_loaded", 0) for r in results if r.get("success"))
    total_errors = sum(r.get("total_errors", 0) for r in results if r.get("success"))
    successful_tables = [r.get("table_name") for r in results if r.get("success")]
    failed_tables = [r.get("table_name") for r in results if not r.get("success")]
    
    print(f"\n📊 Summary:")
    print(f"   • Tables loaded successfully: {len(successful_tables)}")
    print(f"   • Total records loaded: {total_loaded:,}")
    print(f"   • Total errors: {total_errors:,}")
    
    if successful_tables:
        print(f"\n✅ Successfully loaded tables:")
        for table in successful_tables:
            result = next((r for r in results if r.get("table_name") == table), {})
            loaded = result.get("total_loaded", 0)
            print(f"   • {table}: {loaded:,} records")
    
    if failed_tables:
        print(f"\n❌ Failed tables:")
        for table in failed_tables:
            result = next((r for r in results if r.get("table_name") == table), {})
            error = result.get("error", "Unknown error")
            print(f"   • {table}: {error}")
    
    print(f"\n💡 You can now search the data using:")
    print(f"   client = MilvusClient()")
    print(f"   results = client.search('your search query', top_k=10)")
    print("=" * 80)


if __name__ == "__main__":
    main()
