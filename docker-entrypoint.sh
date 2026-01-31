#!/bin/bash
set -e

echo "[init] Generating test data..."
python /app/openint-testdata/generators/generate_openint_test_data.py \
  --max-customers 100 \
  --max-transactions 5000 \
  --max-disputes 1000 \
  --no-llm \
  --clean-run || echo "[init] Test data generation failed (non-fatal), continuing..."

echo "[init] Starting backend..."
exec python main.py
