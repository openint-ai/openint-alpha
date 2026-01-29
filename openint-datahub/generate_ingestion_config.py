"""
Generate DataHub ingestion YAML config file from CSV files.
This can be used with 'datahub ingest -c ingestion_config.yaml' command.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import yaml

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

TESTDATA_DIR = parent_dir / "testdata"
PLATFORM = "openint"
ENVIRONMENT = "PROD"


def discover_csv_files() -> List[Dict[str, Any]]:
    """Discover all CSV files in testdata directory."""
    csv_files = []
    
    if not TESTDATA_DIR.exists():
        print(f"❌ Testdata directory not found: {TESTDATA_DIR}")
        return csv_files
    
    for csv_path in TESTDATA_DIR.rglob("*.csv"):
        rel_path = csv_path.relative_to(TESTDATA_DIR)
        
        category = "unknown"
        if "dimensions" in str(rel_path):
            category = "dimension"
        elif "facts" in str(rel_path):
            category = "fact"
        elif "static" in str(rel_path):
            category = "static"
        
        dataset_name = csv_path.stem
        
        csv_files.append({
            "name": dataset_name,
            "path": csv_path,
            "category": category,
        })
    
    return sorted(csv_files, key=lambda x: (x["category"], x["name"]))


def read_csv_headers(csv_path: Path) -> List[str]:
    """Read headers from CSV file."""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
            return [col.strip() for col in header_line.split(',')]
    except Exception as e:
        print(f"  ⚠️  Error reading headers from {csv_path}: {e}")
        return []


def generate_ingestion_config():
    """Generate DataHub ingestion YAML config."""
    csv_files = discover_csv_files()
    
    if not csv_files:
        print("❌ No CSV files found")
        return
    
    # Build source config for each dataset
    sources = []
    
    for csv_info in csv_files:
        headers = read_csv_headers(csv_info["path"])
        if not headers:
            continue
        
        dataset_name = csv_info["name"]
        csv_path = csv_info["path"]
        category = csv_info["category"]
        
        # Create dataset URN
        dataset_urn = f"urn:li:dataset:(urn:li:dataPlatform:{PLATFORM},{dataset_name},{ENVIRONMENT})"
        
        # Build schema fields
        fields = []
        for idx, header in enumerate(headers):
            fields.append({
                "fieldPath": header,
                "type": {"type": "STRING"},  # Default to STRING, can be refined
                "description": f"Column {idx + 1}: {header.replace('_', ' ').title()}",
            })
        
        # Create source entry
        source_entry = {
            "type": "file",
            "config": {
                "path": str(csv_path.absolute()),
                "file_type": "csv",
                "platform": PLATFORM,
                "dataset_name": dataset_name,
                "env": ENVIRONMENT,
                "schema": {
                    "fields": fields
                },
                "description": f"{category.title()} table: {dataset_name.replace('_', ' ').title()}",
            }
        }
        
        sources.append(source_entry)
    
    # Build complete ingestion config
    config = {
        "source": {
            "type": "file",
            "config": {
                "path": str(TESTDATA_DIR.absolute()),
                "file_type": "csv",
                "platform": PLATFORM,
                "env": ENVIRONMENT,
            }
        },
        "sink": {
            "type": "datahub-rest",
            "config": {
                "server": os.getenv("DATAHUB_GMS_URL", "http://localhost:9002"),
            }
        },
    }
    
    # Write config file
    config_file = Path(__file__).parent / "ingestion_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Generated ingestion config: {config_file}")
    print(f"\n💡 To ingest metadata, run:")
    print(f"   datahub ingest -c {config_file}")
    print(f"\n   Or use the Python script after configuring authentication.")


if __name__ == "__main__":
    generate_ingestion_config()
