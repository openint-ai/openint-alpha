#!/bin/bash

# Driver script to update all openInt data table schemas in DataHub
# Pushes metadata for all datasets (dimensions, facts, static tables)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}📊 Updating openInt Schemas in DataHub${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi

# Check if DataHub URL is set, otherwise use default
# GMS (General Metadata Service) runs on port 8080, not 9002 (frontend)
DATAHUB_GMS_URL=${DATAHUB_GMS_URL:-"http://localhost:8080"}
echo -e "${BLUE}🔗 DataHub URL: ${DATAHUB_GMS_URL}${NC}"

# Load token from .datahub_token file if DATAHUB_TOKEN env var is not set
if [ -z "$DATAHUB_TOKEN" ] && [ -f ".datahub_token" ]; then
    DATAHUB_TOKEN=$(cat .datahub_token | tr -d '\n\r ')
    export DATAHUB_TOKEN
    echo -e "${GREEN}✅ Loaded DataHub token from .datahub_token${NC}"
elif [ -n "$DATAHUB_TOKEN" ]; then
    echo -e "${GREEN}✅ Using DataHub token from environment${NC}"
else
    echo -e "${YELLOW}⚠️  No DataHub token found (optional if auth is disabled)${NC}"
fi
echo ""

# Check if data directory exists
DATA_DIR="../data"
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}❌ Data directory not found: $DATA_DIR${NC}"
    echo -e "${YELLOW}💡 Please ensure data exists before running schema update${NC}"
    exit 1
fi

# Check if required CSV files exist
echo -e "${BLUE}📋 Checking data files...${NC}"
missing_files=0

# Check dimension tables
if [ ! -f "$DATA_DIR/dimensions/customers.csv" ]; then
    echo -e "${YELLOW}⚠️  Missing: dimensions/customers.csv${NC}"
    missing_files=$((missing_files + 1))
fi

# Check fact tables
fact_tables=("ach_transactions" "wire_transactions" "credit_transactions" "debit_transactions" "check_transactions" "disputes")
for table in "${fact_tables[@]}"; do
    if [ ! -f "$DATA_DIR/facts/${table}.csv" ]; then
        echo -e "${YELLOW}⚠️  Missing: facts/${table}.csv${NC}"
        missing_files=$((missing_files + 1))
    fi
done

# Check static tables
static_tables=("country_codes" "state_codes" "zip_codes")
for table in "${static_tables[@]}"; do
    if [ ! -f "$TESTDATA_DIR/static/${table}.csv" ]; then
        echo -e "${YELLOW}⚠️  Missing: static/${table}.csv${NC}"
        missing_files=$((missing_files + 1))
    fi
done

if [ $missing_files -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Found $missing_files missing file(s)${NC}"
    echo -e "${YELLOW}💡 Some schemas may not include row counts${NC}"
    echo ""
fi

# Check if virtual environment exists, create if needed
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Use venv's python and pip explicitly
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
VENV_PIP="$SCRIPT_DIR/venv/bin/pip"

# Install dependencies if needed
echo -e "${BLUE}📦 Checking dependencies...${NC}"
if ! "$VENV_PYTHON" -c "import datahub" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    if "$VENV_PIP" install -r requirements.txt; then
        echo -e "${GREEN}✅ Dependencies installed${NC}"
    else
        echo -e "${RED}❌ Failed to install dependencies${NC}"
        echo -e "${YELLOW}💡 Trying to install DataHub SDK directly...${NC}"
        "$VENV_PIP" install acryl-datahub pandas pyyaml requests || {
            echo -e "${RED}❌ Failed to install dependencies${NC}"
            echo -e "${YELLOW}💡 Please install manually: cd $SCRIPT_DIR && source venv/bin/activate && pip install -r requirements.txt${NC}"
            exit 1
        }
    fi
    # Verify installation
    if ! "$VENV_PYTHON" -c "import datahub" 2>/dev/null; then
        echo -e "${RED}❌ DataHub SDK still not available after installation${NC}"
        echo -e "${YELLOW}💡 Please check your Python environment and try: pip install acryl-datahub${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Dependencies already installed${NC}"
fi

# Verify DataHub SDK is installed (using venv python)
echo -e "${BLUE}🔍 Verifying DataHub SDK installation...${NC}"
if ! "$VENV_PYTHON" -c "import datahub" 2>/dev/null; then
    echo -e "${RED}❌ DataHub SDK not installed in venv${NC}"
    echo -e "${YELLOW}Installing DataHub SDK...${NC}"
    "$VENV_PIP" install acryl-datahub || {
        echo -e "${RED}❌ Failed to install DataHub SDK${NC}"
        exit 1
    }
fi
echo -e "${GREEN}✅ DataHub SDK verified${NC}"

# Test DataHub connection
echo -e "${BLUE}🔍 Testing DataHub connection...${NC}"
if "$VENV_PYTHON" test_connection.py 2>/dev/null; then
    echo -e "${GREEN}✅ DataHub connection successful${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠️  DataHub connection test failed${NC}"
    echo -e "${YELLOW}💡 Continuing anyway - ingestion will show detailed errors${NC}"
    echo ""
fi

# Run ingestion
echo -e "${BLUE}📤 Ingesting metadata to DataHub...${NC}"
echo ""

# Export DATAHUB_GMS_URL for Python script
export DATAHUB_GMS_URL

if "$VENV_PYTHON" ingest_metadata.py; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ Schema Update Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    # Verify datasets (optional; uses same venv)
    if [ -f "verify_datasets.py" ]; then
        echo -e "${BLUE}🔍 Verifying datasets in DataHub...${NC}"
        "$VENV_PYTHON" verify_datasets.py || true
        echo ""
    fi
    echo -e "${BLUE}💡 View your datasets at:${NC}"
    echo -e "   ${DATAHUB_GMS_URL}/dataset/openint"
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ Schema Update Failed${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo -e "${YELLOW}💡 Troubleshooting:${NC}"
    echo -e "   1. Ensure DataHub is running: curl ${DATAHUB_GMS_URL}/health"
    echo -e "   2. Check DataHub logs for errors"
    echo -e "   3. Verify data files exist"
    echo ""
    exit 1
fi
