# openInt DataHub Integration

This project integrates openInt data models with DataHub, an open-source metadata platform for data discovery, observability, and governance.

**Repo-friendly:** The `data/` directory is gitignored. If `data/` is absent (e.g. fresh clone), `ingest_metadata.py` loads **only schemas** from `schemas.py` and pushes dataset metadata to DataHub. When `data/` exists with CSVs, ingestion uses CSV headers (or schemas) as usual.

## Overview

The `openint-datahub` project pushes metadata about all openInt data datasets to your DataHub instance, including:

- **Dimension Tables**: customers
- **Fact Tables**: ach_transactions, wire_transactions, credit_transactions, debit_transactions, check_transactions, disputes (and type-specific: ach_disputes, credit_disputes, etc.)

## Prerequisites

1. **DataHub Running**: Ensure DataHub is running (UI at `http://localhost:9002/`, GMS API at `http://localhost:8080`)
   - Set `DATAHUB_GMS_URL` if your GMS is on a different URL

2. **DataHub Configuration**: Set `METADATA_SERVICE_AUTH_ENABLED=true` for the `datahub-frontend` container
   - This ensures proper communication between frontend and GMS
   - See [DATAHUB_CONFIG.md](./DATAHUB_CONFIG.md) for detailed setup instructions
   - Example for Docker Compose:
     ```yaml
     services:
       datahub-frontend:
         environment:
           - METADATA_SERVICE_AUTH_ENABLED=true
     ```

3. **Python Dependencies**: Install required packages
   ```bash
   pip install -r requirements.txt
   ```

## Installation

Use the project’s virtual environment so the DataHub SDK is available:

```bash
cd openint-datahub
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Or use the driver script (it creates the venv and installs deps if needed): see **Ingest All Metadata** below.

## Usage

### Ingest and verify (recommended)

From `openint-datahub`:

```bash
./update_schemas.sh
```

This script creates/uses the venv, installs dependencies, and runs ingestion; after success it runs `verify_datasets.py` with the venv’s Python.

### Nuke openint data (before fresh ingest)

To hard-delete all openint datasets from DataHub so you can ingest fresh data:

```bash
cd openint-datahub
./nuke_datahub.sh
```

You will be prompted to type `yes` to confirm. To skip the prompt:

```bash
./nuke_datahub.sh --force
```

To only list what would be deleted (no changes):

```bash
./nuke_datahub.sh --dry-run
```

Then re-ingest with `./update_schemas.sh` or `python ingest_metadata.py`.

### Run verify_datasets.py only

If you already ingested and only want to verify datasets, use the **project venv** (otherwise you may get `ModuleNotFoundError: No module named 'datahub'`):

```bash
cd openint-datahub
./venv/bin/python verify_datasets.py
```

Or activate the venv first:

```bash
cd openint-datahub
source venv/bin/activate
python verify_datasets.py
```

### Check Configuration

Before ingesting, verify your DataHub configuration:

```bash
./check_config.sh
```

This will check:
- DataHub health status
- METADATA_SERVICE_AUTH_ENABLED setting
- Environment variables
- Python dependencies

### Ingest All Metadata (manual)

With the venv activated, run the ingestion script:

```bash
source venv/bin/activate
python ingest_metadata.py
```

### Environment Variables

- `DATAHUB_GMS_URL`: DataHub GMS server URL (default: `http://localhost:8080`; UI is on 9002)
  ```bash
  export DATAHUB_GMS_URL=http://localhost:8080
  ```

- `DATAHUB_TOKEN`: Authentication token (optional, if token auth is enabled)
  ```bash
  export DATAHUB_TOKEN="your-token-here"
  ```

**Note**: If you encounter authentication errors, see [DATAHUB_CONFIG.md](./DATAHUB_CONFIG.md) for configuration options.

## What Gets Ingested

For each dataset, the script pushes:

1. **Dataset Properties**:
   - Name and description
   - Category (dimension/fact/static)
   - Row count (if CSV exists)
   - Platform and environment tags

2. **Schema Metadata**:
   - All field definitions with:
     - Field names
     - Data types (STRING, NUMBER, DATE, DATETIME)
     - Field descriptions

3. **Browse Paths (V2 and legacy)**:
   - So datasets appear in the Browse tree and on the default/homepage view
   - Hierarchy: **Datasets → PROD → openint → category → dataset_name** (e.g. dimension → customers)

4. **Status**:
   - `removed: false` so assets are visible on the homepage and not filtered out

5. **Tags (best practice)**:
   - Each dataset is tagged by table type for discovery and filtering:
     - **dimension** – dimension tables (e.g. customers)
     - **fact** – fact tables (e.g. credit_transactions, ach_transactions)
     - **static** – reference/static tables (e.g. country_codes, state_codes, zip_codes)

## Finding Assets on the Homepage

After ingestion, openint datasets appear:

- **Homepage / default view**: Use **Search** or **Browse** from the top nav. The default Browse tree shows **Datasets → PROD → openint**; expand **openint** then **dimension**, **fact**, or **static** to see each dataset (one per data file).
- **Search**: You can still filter by `platform:openint` or search by dataset name (e.g. `customers`, `credit_transactions`).

## Dataset Schemas

### Dimension Tables

- **customers**: Customer profile information (18 fields)
  - Customer IDs, personal info, addresses, account status, credit scores

### Fact Tables

- **ach_transactions**: ACH transaction records (13 fields)
- **wire_transactions**: Wire transfer records (18 fields)
- **credit_transactions**: Credit card transactions (17 fields)
- **debit_transactions**: Debit card transactions (13 fields)
- **check_transactions**: Check transactions (12 fields)
- **disputes**: Transaction dispute records (11 fields)

### Static Tables

- **country_codes**: Country reference data (4 fields)
- **state_codes**: US state reference data (3 fields)
- **zip_codes**: ZIP code geographic data (6 fields)

## Viewing in DataHub

After ingestion, view your datasets in DataHub:

1. **Browse Datasets**: Navigate to `http://localhost:9002/dataset/openint`
2. **Search**: Use DataHub search to find specific datasets
3. **Schema View**: Click on any dataset to see full schema with field descriptions

## Project Structure

```
openint-datahub/
├── requirements.txt          # Python dependencies
├── schemas.py               # Schema definitions for all datasets
├── ingest_metadata.py       # Main ingestion script (uses GraphQL)
├── generate_ingestion_config.py  # Generate YAML config for CLI
├── ingestion_config.yaml     # DataHub CLI ingestion config
├── test_connection.py        # Connection test script
├── check_config.sh          # Configuration verification script
├── update_schemas.sh        # Driver script (recommended)
├── nuke_datahub.sh         # Nuke all openint datasets (before fresh ingest)
├── nuke_datahub.py         # Python nuke script (hard-delete)
├── DATAHUB_CONFIG.md        # Configuration guide
└── README.md               # This file
```

## Troubleshooting

### Authentication Errors (401 Unauthorized)

If you see `401 Client Error: Unauthorized`:

1. **Set METADATA_SERVICE_AUTH_ENABLED**: Ensure `METADATA_SERVICE_AUTH_ENABLED=true` is set for `datahub-frontend` container
   ```bash
   # Check if set
   docker exec datahub-frontend env | grep METADATA_SERVICE_AUTH_ENABLED
   
   # If not set, add to docker-compose.yml and restart
   docker-compose restart datahub-frontend
   ```

2. **Enable Token Authentication**: In DataHub UI → Settings → Authentication, enable token auth and generate a token

3. **Set Token**: Export the token as environment variable:
   ```bash
   export DATAHUB_TOKEN="your-token-here"
   ```

4. **Alternative**: Use DataHub CLI which handles auth automatically:
   ```bash
   datahub ingest -c ingestion_config.yaml
   ```

### Connection Errors

If you see connection errors:

1. **Check DataHub is running** (GMS health):
   ```bash
   curl http://localhost:8080/health
   ```

2. **Verify URL**: Ensure `DATAHUB_GMS_URL` matches your DataHub instance

3. **Check network**: Ensure firewall/network allows connections

4. **Verify Configuration**: See [DATAHUB_CONFIG.md](./DATAHUB_CONFIG.md) for detailed configuration steps

### Schema Errors

If schema ingestion fails:

1. **Schema-only mode:** If `data/` is absent, ingestion uses `schemas.py` only; ensure `schemas.py` defines all datasets you need.
2. **CSV mode:** If using CSVs, check that files exist in `../data/` and headers match definitions in `schemas.py`.
3. Check DataHub logs for detailed error messages.

## Customization

### Adding New Datasets

1. Add schema definition to `schemas.py` in `get_dataset_schemas()`
2. Add CSV path mapping in `ingest_metadata.py` → `get_csv_path()`
3. Run ingestion script

### Modifying Schemas

Edit `schemas.py` to update field definitions, descriptions, or add new fields.

## Integration with Other Systems

This metadata can be used for:

- **Data Discovery**: Find datasets by searching DataHub
- **Data Lineage**: Track data flow between datasets
- **Data Governance**: Apply tags, ownership, and policies
- **Documentation**: Centralized schema documentation
- **Data Quality**: Track data quality metrics

## Next Steps

1. **Add Tags**: Tag datasets with business terms (e.g., "PII", "Financial")
2. **Set Ownership**: Assign dataset owners in DataHub UI
3. **Add Glossary Terms**: Link fields to business glossary
4. **Set Up Lineage**: Document relationships between datasets
5. **Configure Policies**: Set up data access policies

## Support

For issues or questions:
- Check DataHub documentation: https://datahubproject.io/docs/
- Review DataHub Python SDK: https://datahubproject.io/docs/python-sdk/
