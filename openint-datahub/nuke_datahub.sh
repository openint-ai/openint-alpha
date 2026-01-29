#!/bin/bash
# Driver script to nuke (hard-delete) all openint datasets from DataHub.
# Use before re-ingesting fresh data.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DATAHUB_GMS_URL=${DATAHUB_GMS_URL:-"http://localhost:8080"}
export DATAHUB_GMS_URL

if [ -z "$DATAHUB_TOKEN" ] && [ -f ".datahub_token" ]; then
    DATAHUB_TOKEN=$(cat .datahub_token | tr -d '\n\r ')
    export DATAHUB_TOKEN
fi

VENV_PYTHON="${SCRIPT_DIR}/venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}❌ Virtual environment not found. Run: python3 -m venv venv && ./venv/bin/pip install -r requirements.txt${NC}"
    exit 1
fi

# Pass through args (e.g. --dry-run, --force)
"$VENV_PYTHON" nuke_datahub.py "$@"
