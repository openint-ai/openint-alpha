"""
DataHub metadata ingestion script for openInt testdata.
Automatically discovers CSV files and creates DataHub datasets from their headers.

Note: If you get authentication errors, you can either:
1. Enable token authentication in DataHub and set DATAHUB_TOKEN env var
2. Use the DataHub CLI: 'datahub ingest -c ingestion_config.yaml' (handles auth automatically)
3. Configure DataHub to allow unauthenticated API access for local development
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from schemas import get_dataset_schemas

try:
    from datahub.ingestion.graph.client import DataHubGraph
    from datahub.ingestion.graph.config import DatahubClientConfig
    from datahub.emitter.mcp import MetadataChangeProposalWrapper
    from datahub.metadata.schema_classes import (
        DatasetPropertiesClass,
        SchemaMetadataClass,
        SchemaFieldClass,
        SchemaFieldDataTypeClass,
        SchemalessClass,
        ChangeTypeClass,
        BooleanTypeClass,
        StringTypeClass,
        NumberTypeClass,
        DateTypeClass,
        TimeTypeClass,
        BrowsePathsV2Class,
        BrowsePathEntryClass,
        BrowsePathsClass,
        StatusClass,
        GlobalTagsClass,
        TagAssociationClass,
    )
except ImportError as e:
    print(f"❌ Error importing DataHub SDK: {e}")
    print("💡 Please install dependencies:")
    print("   pip install -r requirements.txt")
    print("")
    print("   Or install directly:")
    print("   pip install 'acryl-datahub>=0.12.0'")
    sys.exit(1)


# DataHub configuration
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

PLATFORM = "openint"
ENVIRONMENT = "PROD"
TESTDATA_DIR = parent_dir / "testdata"

# Global emitter instance (will be created in main)
_rest_emitter = None


def infer_field_type_from_name(field_name: str) -> str:
    """
    Infer DataHub field type from column name only (no data sampling).
    Uses only the header row schema information.
    Returns: STRING, NUMBER, DATE, DATETIME, or BOOLEAN
    """
    field_lower = field_name.lower()
    
    # Infer from column name patterns only
    if 'datetime' in field_lower:
        return "DATETIME"
    elif 'date' in field_lower:
        return "DATE"
    elif 'time' in field_lower and 'datetime' not in field_lower:
        return "DATETIME"
    elif any(keyword in field_lower for keyword in ['amount', 'price', 'cost', 'fee', 'balance', 'score', 'count', 'quantity', 'latitude', 'longitude']):
        return "NUMBER"
    elif any(keyword in field_lower for keyword in ['is_', 'has_', 'active', 'enabled', 'status']):
        return "BOOLEAN"
    
    # Default to STRING
    return "STRING"


def get_data_type_class(field_type: str) -> SchemaFieldDataTypeClass:
    """Convert field type string to DataHub SchemaFieldDataTypeClass.
    Avro expects a union of type records (StringType, NumberType, etc.), not a string.
    """
    type_mapping = {
        "STRING": SchemaFieldDataTypeClass(type=StringTypeClass()),
        "NUMBER": SchemaFieldDataTypeClass(type=NumberTypeClass()),
        "DATE": SchemaFieldDataTypeClass(type=DateTypeClass()),
        "DATETIME": SchemaFieldDataTypeClass(type=TimeTypeClass()),
        "BOOLEAN": SchemaFieldDataTypeClass(type=BooleanTypeClass()),
    }
    return type_mapping.get(field_type.upper(), SchemaFieldDataTypeClass(type=StringTypeClass()))


def create_dataset_urn(dataset_name: str) -> str:
    """Create DataHub dataset URN"""
    return f"urn:li:dataset:(urn:li:dataPlatform:{PLATFORM},{dataset_name},{ENVIRONMENT})"


def get_table_type_tag_urn(category: str) -> str:
    """Return DataHub tag URN for dimension/fact/static table type (best practice for discovery)."""
    # Use category as tag name: dimension, fact, static
    tag_id = category if category in ("dimension", "fact", "static") else "dataset"
    return f"urn:li:tag:{tag_id}"


def _emit_via_graphql(mcp: MetadataChangeProposalWrapper, graph: DataHubGraph) -> None:
    """
    Emit metadata change proposal using DataHubGraph client's built-in emit_mcps method.
    This is the recommended way to emit metadata - it handles authentication and API format correctly.
    
    Args:
        mcp: MetadataChangeProposalWrapper to emit
        graph: DataHubGraph client instance (handles authentication)
    """
    try:
        # Use the SDK's built-in emit_mcps method (handles auth and API format correctly)
        # This is the recommended approach - the SDK handles all the API details
        graph.emit_mcps([mcp])
        return
    except Exception as emit_error:
        # Handle exceptions with tuple args (some DataHub exceptions have tuple args)
        if hasattr(emit_error, 'args') and isinstance(emit_error.args, tuple) and len(emit_error.args) > 0:
            # Extract meaningful error message from exception args
            error_parts = []
            for arg in emit_error.args:
                if isinstance(arg, str):
                    error_parts.append(arg)
                elif not isinstance(arg, (type(None), type)):  # Skip None and type objects
                    error_parts.append(str(arg)[:200])  # Truncate long objects
            error_str = " ".join(error_parts) if error_parts else str(emit_error)
        else:
            error_str = str(emit_error)
        
        # Check for authentication errors
        if "401" in error_str or "Unauthorized" in error_str or "403" in error_str:
            instructions = (
                f"\n{'='*70}\n"
                f"⚠️  Authentication Required\n"
                f"{'='*70}\n"
                f"\nThe DataHub API requires authentication.\n"
                f"\n1. ✅ Verify token is set:\n"
                f"   - Check .datahub_token file exists\n"
                f"   - Or set DATAHUB_TOKEN environment variable\n"
                f"\n2. ✅ Enable token authentication in DataHub UI:\n"
                f"   - Go to: http://localhost:9002/settings/authentication (frontend)\n"
                f"   - Note: API calls go to http://localhost:8080 (GMS backend)\n"
                f"   - Enable 'Token-based Authentication'\n"
                f"   - Generate a personal access token\n"
                f"\n3. ✅ Update token:\n"
                f"   - Save token to: openint-datahub/.datahub_token\n"
                f"   - Or export DATAHUB_TOKEN='your-token-here'\n"
                f"\n4. ✅ Run the script again:\n"
                f"   python ingest_metadata.py\n"
                f"\n{'='*70}\n"
                f"Error: {error_str[:300]}\n"
            )
            raise Exception(instructions)
        else:
            # Re-raise other errors as-is
            raise emit_error


def discover_csv_files() -> List[Dict[str, Any]]:
    """
    Discover all CSV files in testdata directory structure.
    Returns list of dicts with: name, path, category
    """
    csv_files = []
    
    if not TESTDATA_DIR.exists():
        print(f"❌ Testdata directory not found: {TESTDATA_DIR}")
        return csv_files
    
    # Walk through testdata directory
    for csv_path in TESTDATA_DIR.rglob("*.csv"):
        # Get relative path from testdata directory
        rel_path = csv_path.relative_to(TESTDATA_DIR)
        
        # Determine category from directory structure
        category = "unknown"
        if "dimensions" in str(rel_path):
            category = "dimension"
        elif "facts" in str(rel_path):
            category = "fact"
        elif "static" in str(rel_path):
            category = "static"
        
        # Get dataset name from filename (without .csv extension)
        dataset_name = csv_path.stem
        
        csv_files.append({
            "name": dataset_name,
            "path": csv_path,
            "category": category,
        })
    
    return sorted(csv_files, key=lambda x: (x["category"], x["name"]))


def read_csv_schema(csv_path: Path) -> Dict[str, Any]:
    """
    Read CSV file schema from header row only (first line).
    Does not sample any data rows - uses only column names from header.
    Does not count rows - only reads the first line.
    Returns dict with: headers, field_types
    """
    try:
        # Read ONLY the header row (first line)
        with open(csv_path, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
            headers = [col.strip() for col in header_line.split(',')]
        
        # Infer types from column names only (no data sampling)
        field_types = {}
        for col in headers:
            field_types[col] = infer_field_type_from_name(col)
        
        # Don't count rows - we only use the header row for schema
        # Counting rows would require reading the entire file
        
        return {
            "headers": headers,
            "field_types": field_types,
        }
    except Exception as e:
        print(f"    ❌ Error reading CSV header: {e}")
        return {
            "headers": [],
            "field_types": {},
        }


def ingest_dataset_from_csv(
    graph: DataHubGraph,
    dataset_name: str,
    csv_path: Path,
    category: str
) -> bool:
    """
    Ingest metadata for a dataset from CSV file.
    
    Args:
        graph: DataHub Graph client instance
        dataset_name: Name of the dataset
        csv_path: Path to CSV file
        category: Category of the dataset (dimension/fact/static)
        
    Returns:
        True if successful, False otherwise
    """
    global _rest_emitter
    try:
        print(f"\n📊 Processing: {dataset_name}")
        print(f"   📄 CSV: {csv_path}")
        
        # Read schema from CSV header row only (first line) - no data processing
        schema_info = read_csv_schema(csv_path)
        headers = schema_info["headers"]
        field_types = schema_info["field_types"]
        
        if not headers:
            print(f"  ❌ No headers found in CSV")
            return False
        
        dataset_urn = create_dataset_urn(dataset_name)
        
        # Create dataset properties
        description = f"{dataset_name.replace('_', ' ').title()} dataset"
        if category == "dimension":
            description = f"Dimension table: {description}"
        elif category == "fact":
            description = f"Fact table: {description}"
        elif category == "static":
            description = f"Reference table: {description}"
        
        properties = DatasetPropertiesClass(
            name=dataset_name,
            description=description,
            customProperties={
                "category": category,
                "platform": PLATFORM,
                "environment": ENVIRONMENT,
                "source": "csv",
                "file_path": str(csv_path.relative_to(parent_dir)),
                "schema_source": "header_row_only",  # Indicates schema comes from header only
                "field_count": str(len(headers)),  # Number of fields from header
            }
        )
        
        # Emit dataset properties using GraphQL mutation
        properties_mcp = MetadataChangeProposalWrapper(
            entityType="dataset",
            changeType=ChangeTypeClass.UPSERT,
            entityUrn=dataset_urn,
            aspect=properties,
        )
        
        # Use GraphQL mutation to ingest proposal (via authenticated graph client)
        try:
            _emit_via_graphql(properties_mcp, graph)
            print(f"  ✅ Emitted properties for {dataset_name}")
        except Exception as emit_error:
            # Format error message properly (handle exceptions with tuple args)
            if hasattr(emit_error, 'args') and isinstance(emit_error.args, tuple):
                error_parts = [str(arg)[:100] for arg in emit_error.args[:2] if not isinstance(arg, type)]
                error_msg = " | ".join(error_parts) if error_parts else str(emit_error)
            else:
                error_msg = str(emit_error)
            print(f"  ❌ Failed to emit properties: {error_msg[:200]}")  # Truncate long errors
            raise emit_error
        
        # Create schema metadata from CSV headers (required for columns to show in DataHub UI)
        schema_fields = []
        for idx, header in enumerate(headers):
            field_type = field_types.get(header, "STRING")
            field = SchemaFieldClass(
                fieldPath=header,
                type=get_data_type_class(field_type),
                nativeDataType=field_type,
                description=f"Column {idx + 1}: {header.replace('_', ' ').title()}",
            )
            schema_fields.append(field)
        
        # Emit schema metadata so columns appear in DataHub. Use SchemalessClass to avoid
        # Avro serialization issues with rawSchema; the UI uses the fields list for columns.
        try:
            schema_metadata = SchemaMetadataClass(
                schemaName=dataset_name,
                platform=f"urn:li:dataPlatform:{PLATFORM}",
                version=0,
                hash="",
                platformSchema=SchemalessClass(),
                fields=schema_fields,
            )
            
            schema_mcp = MetadataChangeProposalWrapper(
                entityType="dataset",
                changeType=ChangeTypeClass.UPSERT,
                entityUrn=dataset_urn,
                aspect=schema_metadata,
            )
            
            _emit_via_graphql(schema_mcp, graph)
            print(f"  ✅ Emitted schema for {dataset_name} ({len(schema_fields)} columns)")
        except Exception as schema_err:
            # Log so we can fix; do not fail the whole run
            err_msg = str(schema_err)
            if hasattr(schema_err, "args") and isinstance(schema_err.args, tuple):
                err_msg = " | ".join(str(a)[:150] for a in schema_err.args[:3] if not isinstance(a, type))
            print(f"  ⚠️  Schema emission failed (columns may be empty in UI): {err_msg[:300]}")
            import traceback
            traceback.print_exc()
        
        # Emit browse paths so datasets appear in Browse tree and on homepage (default view).
        # Path is platform-specific folders only (Platform/Environment come from URN).
        # Hierarchy: Datasets -> PROD -> openint -> category -> dataset_name
        try:
            browse_path_v2 = BrowsePathsV2Class(
                path=[
                    BrowsePathEntryClass(id=category),
                    BrowsePathEntryClass(id=dataset_name),
                ]
            )
            browse_paths_v2_mcp = MetadataChangeProposalWrapper(
                entityType="dataset",
                changeType=ChangeTypeClass.UPSERT,
                entityUrn=dataset_urn,
                aspect=browse_path_v2,
            )
            _emit_via_graphql(browse_paths_v2_mcp, graph)
            # Legacy browse paths (forward-slash) for search and older UI
            browse_paths = BrowsePathsClass(paths=[f"{PLATFORM}/{category}/{dataset_name}"])
            browse_paths_mcp = MetadataChangeProposalWrapper(
                entityType="dataset",
                changeType=ChangeTypeClass.UPSERT,
                entityUrn=dataset_urn,
                aspect=browse_paths,
            )
            _emit_via_graphql(browse_paths_mcp, graph)
        except Exception as browse_err:
            err_msg = str(browse_err)
            if hasattr(browse_err, "args") and isinstance(browse_err.args, tuple):
                err_msg = " | ".join(str(a)[:100] for a in browse_err.args[:2] if not isinstance(a, type))
            print(f"  ⚠️  Browse paths failed: {err_msg[:200]}")
        
        # Emit Status(removed=False) so asset is visible on homepage and not filtered out
        try:
            status_mcp = MetadataChangeProposalWrapper(
                entityType="dataset",
                changeType=ChangeTypeClass.UPSERT,
                entityUrn=dataset_urn,
                aspect=StatusClass(removed=False),
            )
            _emit_via_graphql(status_mcp, graph)
        except Exception as status_err:
            err_msg = str(status_err)
            if hasattr(status_err, "args") and isinstance(status_err.args, tuple):
                err_msg = " | ".join(str(a)[:100] for a in status_err.args[:2] if not isinstance(a, type))
            print(f"  ⚠️  Status emission failed: {err_msg[:200]}")
        
        # Tag dataset as dimension / fact / static for discovery and filtering (best practice)
        try:
            tag_urn = get_table_type_tag_urn(category)
            global_tags = GlobalTagsClass(
                tags=[TagAssociationClass(tag=tag_urn)],
            )
            tags_mcp = MetadataChangeProposalWrapper(
                entityType="dataset",
                changeType=ChangeTypeClass.UPSERT,
                entityUrn=dataset_urn,
                aspect=global_tags,
            )
            _emit_via_graphql(tags_mcp, graph)
            print(f"  ✅ Tagged as {category}")
        except Exception as tag_err:
            err_msg = str(tag_err)
            if hasattr(tag_err, "args") and isinstance(tag_err.args, tuple):
                err_msg = " | ".join(str(a)[:100] for a in tag_err.args[:2] if not isinstance(a, type))
            print(f"  ⚠️  Tag emission failed: {err_msg[:200]}")
        
        return True
        
    except Exception as e:
        # Format error message properly (handle exceptions with tuple args)
        if hasattr(e, 'args') and isinstance(e.args, tuple):
            error_parts = [str(arg)[:150] for arg in e.args[:2] if not isinstance(arg, type)]
            error_msg = " | ".join(error_parts) if error_parts else str(e)
        else:
            error_msg = str(e)
        print(f"  ❌ Error ingesting {dataset_name}: {error_msg[:300]}")  # Truncate long errors
        import traceback
        traceback.print_exc()
        return False


def ingest_dataset_from_schema(
    graph: DataHubGraph,
    dataset_name: str,
    schema_def: Dict[str, Any],
) -> bool:
    """
    Ingest metadata for a dataset from schema definition (schemas.py).
    Used when testdata/ is not present (e.g. fresh clone); loads only schemas.
    """
    global _rest_emitter
    try:
        print(f"\n📊 Processing: {dataset_name}")
        description = schema_def.get("description", dataset_name.replace("_", " ").title())
        category = schema_def.get("category", "unknown")
        fields = schema_def.get("fields", [])

        if not fields:
            print(f"  ❌ No fields in schema for {dataset_name}")
            return False

        dataset_urn = create_dataset_urn(dataset_name)
        properties = DatasetPropertiesClass(
            name=dataset_name,
            description=description,
            customProperties={
                "category": category,
                "platform": PLATFORM,
                "environment": ENVIRONMENT,
                "source": "schemas.py",
                "schema_source": "definition",
                "field_count": str(len(fields)),
            },
        )
        properties_mcp = MetadataChangeProposalWrapper(
            entityType="dataset",
            changeType=ChangeTypeClass.UPSERT,
            entityUrn=dataset_urn,
            aspect=properties,
        )
        try:
            _emit_via_graphql(properties_mcp, graph)
            print(f"  ✅ Emitted properties for {dataset_name}")
        except Exception as emit_error:
            error_msg = str(emit_error)
            if hasattr(emit_error, "args") and isinstance(emit_error.args, tuple):
                error_msg = " | ".join(str(a)[:100] for a in emit_error.args[:2] if not isinstance(a, type))
            print(f"  ❌ Failed to emit properties: {error_msg[:200]}")
            raise emit_error

        schema_fields = []
        for f in fields:
            name = f.get("name", "")
            ftype = f.get("type", "STRING")
            desc = f.get("description", "")
            schema_fields.append(
                SchemaFieldClass(
                    fieldPath=name,
                    type=get_data_type_class(ftype),
                    nativeDataType=ftype,
                    description=desc or name.replace("_", " ").title(),
                )
            )
        schema_metadata = SchemaMetadataClass(
            schemaName=dataset_name,
            platform=f"urn:li:dataPlatform:{PLATFORM}",
            version=0,
            hash="",
            platformSchema=SchemalessClass(),
            fields=schema_fields,
        )
        schema_mcp = MetadataChangeProposalWrapper(
            entityType="dataset",
            changeType=ChangeTypeClass.UPSERT,
            entityUrn=dataset_urn,
            aspect=schema_metadata,
        )
        _emit_via_graphql(schema_mcp, graph)
        print(f"  ✅ Emitted schema for {dataset_name} ({len(schema_fields)} columns)")

        browse_path_v2 = BrowsePathsV2Class(
            path=[
                BrowsePathEntryClass(id=category),
                BrowsePathEntryClass(id=dataset_name),
            ]
        )
        _emit_via_graphql(
            MetadataChangeProposalWrapper(
                entityType="dataset",
                changeType=ChangeTypeClass.UPSERT,
                entityUrn=dataset_urn,
                aspect=browse_path_v2,
            ),
            graph,
        )
        _emit_via_graphql(
            MetadataChangeProposalWrapper(
                entityType="dataset",
                changeType=ChangeTypeClass.UPSERT,
                entityUrn=dataset_urn,
                aspect=BrowsePathsClass(paths=[f"{PLATFORM}/{category}/{dataset_name}"]),
            ),
            graph,
        )
        _emit_via_graphql(
            MetadataChangeProposalWrapper(
                entityType="dataset",
                changeType=ChangeTypeClass.UPSERT,
                entityUrn=dataset_urn,
                aspect=StatusClass(removed=False),
            ),
            graph,
        )
        tag_urn = get_table_type_tag_urn(category)
        _emit_via_graphql(
            MetadataChangeProposalWrapper(
                entityType="dataset",
                changeType=ChangeTypeClass.UPSERT,
                entityUrn=dataset_urn,
                aspect=GlobalTagsClass(tags=[TagAssociationClass(tag=tag_urn)]),
            ),
            graph,
        )
        print(f"  ✅ Tagged as {category}")
        return True
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, "args") and isinstance(e.args, tuple):
            error_msg = " | ".join(str(a)[:150] for a in e.args[:2] if not isinstance(a, type))
        print(f"  ❌ Error ingesting {dataset_name}: {error_msg[:300]}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to ingest all CSV files or schema definitions as DataHub datasets"""
    print("=" * 80)
    print("📊 DataHub Metadata Ingestion for openInt Testdata")
    print("=" * 80)
    print(f"\n🔗 DataHub GMS URL: {DATAHUB_GMS_URL}")
    print(f"📦 Platform: {PLATFORM}")
    print(f"🌍 Environment: {ENVIRONMENT}")
    print(f"📁 Testdata Directory: {TESTDATA_DIR}\n")

    csv_files = discover_csv_files()

    # Initialize DataHub Graph client (shared for CSV and schema-only ingestion)
    try:
        config_data = {"server": DATAHUB_GMS_URL}
        if DATAHUB_TOKEN:
            config_data["token"] = DATAHUB_TOKEN
            print(f"   Using authentication token")
        config = DatahubClientConfig(**config_data)
        graph = DataHubGraph(config=config)
        print(f"\n✅ Connected to DataHub")
        print(f"   Using GraphQL API for ingestion")
        if not DATAHUB_TOKEN:
            print(f"   ⚠️  No authentication token set - if you get 401 errors, set DATAHUB_TOKEN env var")
        print()
    except Exception as e:
        print(f"\n❌ Failed to connect to DataHub: {e}")
        print(f"   Please ensure DataHub is running at {DATAHUB_GMS_URL}")
        import traceback
        traceback.print_exc()
        return

    total_success = 0
    total_failed = 0
    total_count = 0

    if csv_files:
        print(f"📋 Found {len(csv_files)} CSV files to ingest:\n")
        for csv_info in csv_files:
            print(f"   • {csv_info['category']:10s} - {csv_info['name']}")
        total_count = len(csv_files)
        categories = {}
        for csv_info in csv_files:
            c = csv_info["category"]
            if c not in categories:
                categories[c] = []
            categories[c].append(csv_info)
        for category, datasets in categories.items():
            print(f"\n{'=' * 80}")
            print(f"📁 {category.upper()} Tables")
            print("=" * 80)
            for csv_info in datasets:
                success = ingest_dataset_from_csv(
                    graph,
                    csv_info["name"],
                    csv_info["path"],
                    csv_info["category"],
                )
                if success:
                    total_success += 1
                else:
                    total_failed += 1
    else:
        # No testdata/ or no CSVs: load only schemas from schemas.py (repo-friendly)
        schemas = get_dataset_schemas()
        if not schemas:
            print("❌ No CSV files in testdata/ and no schemas in schemas.py")
            return
        print("📋 No testdata CSVs found; ingesting from schemas.py (schema-only)\n")
        for name, schema_def in sorted(schemas.items()):
            print(f"   • {schema_def.get('category', 'unknown'):10s} - {name}")
        total_count = len(schemas)
        categories = {}
        for name, schema_def in schemas.items():
            c = schema_def.get("category", "unknown")
            if c not in categories:
                categories[c] = []
            categories[c].append((name, schema_def))
        for category, items in categories.items():
            print(f"\n{'=' * 80}")
            print(f"📁 {category.upper()} Tables")
            print("=" * 80)
            for name, schema_def in items:
                success = ingest_dataset_from_schema(graph, name, schema_def)
                if success:
                    total_success += 1
                else:
                    total_failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("✅ Metadata Ingestion Complete!")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   • Successfully ingested: {total_success}")
    print(f"   • Failed: {total_failed}")
    print(f"   • Total datasets: {total_count}")
    print(f"\n💡 View your datasets at: http://localhost:9002/dataset/{PLATFORM}")
    print(f"   (Frontend UI: http://localhost:9002, GMS API: {DATAHUB_GMS_URL})")
    print(f"   Search for platform: {PLATFORM}")
    print("\n📝 Note: If you encountered authentication errors:")
    print(f"   1. Enable token auth in DataHub and set DATAHUB_TOKEN env var")
    print(f"   2. Or use: datahub ingest -c ingestion_config.yaml")
    print(f"   3. Or configure DataHub to allow unauthenticated API access")
    print("=" * 80)


if __name__ == "__main__":
    main()
