"""
Nuke (hard-delete) all openint platform datasets from DataHub.
Use this before re-ingesting fresh data so you start from a clean state.

Usage:
  python nuke_datahub.py [--dry-run] [--force]
  --dry-run: List URNs that would be deleted, do not delete.
  --force:   Skip confirmation prompt.
"""

import argparse
import os
import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from datahub.ingestion.graph.client import DataHubGraph
    from datahub.ingestion.graph.config import DatahubClientConfig
    from datahub.ingestion.graph.filters import RemovedStatusFilter
except ImportError as e:
    print(f"❌ Error importing DataHub SDK: {e}")
    print("💡 Run from openint-datahub with venv: ./venv/bin/python nuke_datahub.py")
    sys.exit(1)

DATAHUB_GMS_URL = os.getenv("DATAHUB_GMS_URL", "http://localhost:8080")
DATAHUB_TOKEN = os.getenv("DATAHUB_TOKEN", "")
if not DATAHUB_TOKEN:
    token_file = Path(__file__).parent / ".datahub_token"
    if token_file.exists():
        try:
            DATAHUB_TOKEN = token_file.read_text().strip()
        except Exception:
            pass

PLATFORM = "openint"


def main():
    parser = argparse.ArgumentParser(description="Hard-delete all openint datasets from DataHub")
    parser.add_argument("--dry-run", action="store_true", help="List URNs only, do not delete")
    parser.add_argument("--force", "-y", action="store_true", help="Skip confirmation")
    args = parser.parse_args()

    print("=" * 60)
    print("🗑️  Nuke openint datasets from DataHub")
    print("=" * 60)
    print(f"\n🔗 GMS URL: {DATAHUB_GMS_URL}")
    print(f"📦 Platform: {PLATFORM}")
    if args.dry_run:
        print("   (dry-run: no changes will be made)\n")
    else:
        print("   ⚠️  This will HARD-DELETE all datasets (irreversible).\n")

    config_data = {"server": DATAHUB_GMS_URL}
    if DATAHUB_TOKEN:
        config_data["token"] = DATAHUB_TOKEN
    config = DatahubClientConfig(**config_data)
    graph = DataHubGraph(config=config)

    # Fetch all dataset URNs for platform=openint (including soft-deleted)
    urns = list(
        graph.get_urns_by_filter(
            entity_types=["dataset"],
            platform=PLATFORM,
            status=RemovedStatusFilter.ALL,
        )
    )

    if not urns:
        print("✅ No openint datasets found. Nothing to delete.")
        return 0

    print(f"📋 Found {len(urns)} dataset(s) to delete:\n")
    for urn in urns:
        # Show short name: urn:li:dataset:(urn:li:dataPlatform:openint,NAME,PROD) -> NAME
        name = urn
        if "," in urn:
            name = urn.split(",")[1].strip(")")
        print(f"   • {name}")
    print()

    if args.dry_run:
        print("Dry-run complete. Re-run without --dry-run to delete.")
        return 0

    if not args.force:
        try:
            reply = input("Type 'yes' to hard-delete all of the above: ").strip().lower()
        except EOFError:
            reply = "no"
        if reply != "yes":
            print("Aborted.")
            return 1

    deleted = 0
    failed = 0
    for urn in urns:
        try:
            graph.hard_delete_entity(urn)
            deleted += 1
            name = urn.split(",")[1].strip(")") if "," in urn else urn
            print(f"   ✅ Deleted {name}")
        except Exception as e:
            failed += 1
            print(f"   ❌ Failed {urn}: {e}")

    print()
    print("=" * 60)
    print(f"✅ Nuke complete. Deleted: {deleted}, Failed: {failed}")
    print("=" * 60)
    print("\n💡 Re-ingest with: ./update_schemas.sh  or  python ingest_metadata.py")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
