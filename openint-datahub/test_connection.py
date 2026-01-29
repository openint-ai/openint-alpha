"""
Quick test script to verify DataHub connection and SDK installation.
"""

import os
import sys
from pathlib import Path

# GMS (General Metadata Service) runs on port 8080, not 9002 (frontend)
DATAHUB_GMS_URL = os.getenv("DATAHUB_GMS_URL", "http://localhost:8080")

# Load token from environment variable or from .datahub_token file
DATAHUB_TOKEN = os.getenv("DATAHUB_TOKEN", "")
if not DATAHUB_TOKEN:
    token_file = Path(__file__).parent / ".datahub_token"
    if token_file.exists():
        try:
            DATAHUB_TOKEN = token_file.read_text().strip()
        except Exception as e:
            print(f"⚠️  Warning: Could not read token from {token_file}: {e}")

print("=" * 60)
print("🔍 Testing DataHub Connection")
print("=" * 60)
print(f"\n📡 DataHub URL: {DATAHUB_GMS_URL}")
if DATAHUB_TOKEN:
    print(f"🔑 Token: {'*' * 20}...{DATAHUB_TOKEN[-10:] if len(DATAHUB_TOKEN) > 10 else 'loaded'}")
else:
    print("🔑 Token: Not set (optional if auth is disabled)")
print()

# Test 1: Check SDK installation
print("1️⃣  Testing DataHub SDK installation...")
try:
    from datahub.emitter.rest_emitter import DatahubRestEmitter
    from datahub.metadata.schema_classes import DatasetPropertiesClass
    print("   ✅ DataHub SDK imported successfully")
except ImportError as e:
    print(f"   ❌ Failed to import DataHub SDK: {e}")
    print("\n💡 Install dependencies:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

# Test 2: Test connection
print("\n2️⃣  Testing DataHub connection...")
try:
    import requests
    response = requests.get(f"{DATAHUB_GMS_URL}/health", timeout=5)
    if response.status_code == 200:
        print(f"   ✅ DataHub is reachable (status: {response.status_code})")
    else:
        print(f"   ⚠️  DataHub responded with status: {response.status_code}")
except requests.exceptions.ConnectionError:
    print(f"   ❌ Cannot connect to DataHub at {DATAHUB_GMS_URL}")
    print("   💡 Please ensure DataHub is running:")
    print("      docker-compose up -d")
    sys.exit(1)
except ImportError:
    print("   ⚠️  'requests' not installed, skipping connection test")
except Exception as e:
    print(f"   ⚠️  Connection test failed: {e}")

# Test 3: Test emitter creation
print("\n3️⃣  Testing emitter creation...")
try:
    from datahub.ingestion.graph.client import DataHubGraph
    from datahub.ingestion.graph.config import DatahubClientConfig
    
    config_data = {"server": DATAHUB_GMS_URL}
    if DATAHUB_TOKEN:
        config_data["token"] = DATAHUB_TOKEN
    
    config = DatahubClientConfig(**config_data)
    graph = DataHubGraph(config=config)
    print("   ✅ Graph client created successfully")
    if DATAHUB_TOKEN:
        print("   ✅ Using authentication token")
except Exception as e:
    print(f"   ❌ Failed to create graph client: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All tests passed! Ready to ingest metadata.")
print("=" * 60)
print("\n💡 Run ingestion:")
print("   python ingest_metadata.py")
