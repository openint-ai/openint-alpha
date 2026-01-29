# OpenInt Test Data Project

Test data generation and loading for OpenInt system.

## Structure

```
OpenInt-testdata/
├── generators/          # Data generation scripts
│   ├── customers.py    # Customer data generator
│   ├── transactions.py # Transaction data generator
│   └── static.py       # Static reference data
├── loaders/            # Data loading scripts
│   ├── milvus_loader.py # Load data into Milvus
│   └── validator.py    # Data validation
└── README.md
```

## Usage

### Generate Test Data

```bash
# Generate full dataset
python -m generators.main --full

# Generate quick test dataset
python -m generators.main --quick

# Generate specific data types
python -m generators.main --only customers
python -m generators.main --only transactions
```

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
