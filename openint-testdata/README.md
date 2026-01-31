# OpenInt Test Data Project

Test data generation and loading for OpenInt system.

## Structure

```
OpenInt-testdata/
├── generators/          # Data generation scripts
│   ├── customers.py    # Customer data generator
│   ├── transactions.py # Transaction data generator
├── loaders/            # Data loading scripts
│   ├── milvus_loader.py # Load data into Milvus
│   └── validator.py    # Data validation
└── README.md
```

## Usage

### Generate Test Data

```bash
cd openint-testdata

# Generate full dataset (customers, transactions, disputes)
python generators/generate_openint_test_data.py

# Generate only disputes (requires existing transaction fact tables)
# Uses Ollama for creative dispute descriptions when available
python generators/generate_openint_test_data.py --only-disputes --num-disputes 500
python generators/generate_openint_test_data.py --only-disputes --no-llm  # template descriptions
python generators/generate_openint_test_data.py --only-disputes --clean-run  # overwrite (default: append)

# Generate specific data types
python generators/generate_openint_test_data.py --only-customers
python generators/generate_openint_test_data.py --only-transactions
```

Disputes are created **only** from existing transaction data: each dispute references a valid `customer_id` and `transaction_id` (both 10-digit) from the respective transaction CSVs. Output files mirror transaction types: `ach_disputes.csv`, `credit_disputes.csv`, `debit_disputes.csv`, `wire_disputes.csv`, `check_disputes.csv`, `atm_disputes.csv`.

**Output directory:** Data is always written to `openint-testdata/data/` (regardless of CWD).

**Append vs overwrite:** By default, new records are appended to existing files. Use `--clean-run` to overwrite instead.

### Load Data into Neo4j

From repo root (with `data/` containing `dimensions/` and `facts/` from the generator):

```bash
# Full load: transactions + disputes + customer enrichment (Customer nodes get full details)
python openint-testdata/loaders/load_openint_data_to_neo4j.py

# Only enrich existing Customer nodes from dimensions/customers.csv (fix "Only ID stored" in UI)
# Reads customer_ids from transactions + disputes CSVs, or from Neo4j if no CSVs
python openint-testdata/loaders/load_openint_data_to_neo4j.py --only-enrich
```

Ensure `data/dimensions/customers.csv` and `data/facts/*.csv` exist (run the generator first).

### Load Data into Milvus

```bash
# Load all data
python -m loaders.milvus_loader

# Load specific tables
python -m loaders.milvus_loader --tables customers,transactions
```

## Configuration

Set environment variables in `.env`:
- `MILVUS_HOST`: Milvus host (default: localhost)
- `MILVUS_PORT`: Milvus port (default: 19530)
- `DATA_DIR`: Directory for generated data (default: ./data)
