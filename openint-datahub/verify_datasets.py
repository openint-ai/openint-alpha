"""
Verify that datasets were successfully ingested into DataHub.
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from datahub.ingestion.graph.client import DataHubGraph
from datahub.ingestion.graph.config import DatahubClientConfig
import os

# Load token
token_file = Path(__file__).parent / ".datahub_token"
DATAHUB_TOKEN = token_file.read_text().strip() if token_file.exists() else os.getenv("DATAHUB_TOKEN", "")
DATAHUB_GMS_URL = os.getenv("DATAHUB_GMS_URL", "http://localhost:8080")
PLATFORM = "openint"

print("=" * 80)
print("🔍 Verifying DataHub Datasets")
print("=" * 80)
print(f"\n🔗 DataHub GMS URL: {DATAHUB_GMS_URL}")
print(f"📦 Platform: {PLATFORM}")
if DATAHUB_TOKEN:
    print(f"🔑 Token: {'*' * 20}...{DATAHUB_TOKEN[-10:]}")
print()

# Connect to DataHub
try:
    config_data = {"server": DATAHUB_GMS_URL}
    if DATAHUB_TOKEN:
        config_data["token"] = DATAHUB_TOKEN
    
    config = DatahubClientConfig(**config_data)
    graph = DataHubGraph(config=config)
    print("✅ Connected to DataHub\n")
except Exception as e:
    print(f"❌ Failed to connect: {e}")
    sys.exit(1)

# Search for datasets
try:
    platform_query = f'platform:{PLATFORM}'
    # Use regular string formatting to avoid f-string brace conflicts
    query = 'query { search(input: {type: DATASET, query: "' + platform_query + '", start: 0, count: 100}) { searchResults { entity { urn } } } }'
    result = graph.execute_graphql(query=query)
    
    datasets = result.get("search", {}).get("searchResults", [])
    
    if datasets:
        print(f"✅ Found {len(datasets)} datasets:\n")
        for idx, item in enumerate(datasets, 1):
            urn = item.get("entity", {}).get("urn", "unknown")
            # Extract dataset name from URN
            dataset_name = urn.split(",")[-2] if "," in urn else urn
            print(f"   {idx:2d}. {dataset_name}")
        
        print(f"\n💡 View in DataHub UI:")
        print(f"   1. Go to: http://localhost:9002")
        print(f"   2. Search for: platform:{PLATFORM}")
        print(f"   3. Or browse: http://localhost:9002/dataset/{PLATFORM}")
        print(f"   4. Or search for specific dataset name (e.g., 'customers', 'ach_transactions')")
    else:
        print("❌ No datasets found")
        print("\n💡 Try running the ingestion script:")
        print("   python ingest_metadata.py")
        
except Exception as e:
    print(f"❌ Error querying datasets: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
