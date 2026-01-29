#!/bin/bash
# Quick script to check DataHub configuration

echo "=========================================="
echo "🔍 DataHub Configuration Check"
echo "=========================================="
echo ""

# Check if DataHub is running
echo "1️⃣  Checking DataHub health..."
if curl -s http://localhost:9002/health > /dev/null 2>&1; then
    echo "   ✅ DataHub is running"
else
    echo "   ❌ DataHub is not responding at http://localhost:9002"
    echo "      Please ensure DataHub is running"
    exit 1
fi

# Check METADATA_SERVICE_AUTH_ENABLED if Docker is available
echo ""
echo "2️⃣  Checking METADATA_SERVICE_AUTH_ENABLED..."
if command -v docker > /dev/null 2>&1; then
    if docker ps | grep -q datahub-frontend; then
        AUTH_ENABLED=$(docker exec datahub-frontend env 2>/dev/null | grep METADATA_SERVICE_AUTH_ENABLED | cut -d'=' -f2)
        if [ "$AUTH_ENABLED" = "true" ]; then
            echo "   ✅ METADATA_SERVICE_AUTH_ENABLED=true is set"
        else
            echo "   ⚠️  METADATA_SERVICE_AUTH_ENABLED is not set to 'true'"
            echo "      Current value: ${AUTH_ENABLED:-not set}"
            echo ""
            echo "   💡 To fix, add to docker-compose.yml:"
            echo "      services:"
            echo "        datahub-frontend:"
            echo "          environment:"
            echo "            - METADATA_SERVICE_AUTH_ENABLED=true"
            echo ""
            echo "      Then restart: docker-compose restart datahub-frontend"
        fi
    else
        echo "   ⚠️  datahub-frontend container not found"
        echo "      If using Docker, ensure the container is running"
    fi
else
    echo "   ⚠️  Docker not found - skipping container check"
fi

# Check environment variables
echo ""
echo "3️⃣  Checking environment variables..."
if [ -n "$DATAHUB_GMS_URL" ]; then
    echo "   ✅ DATAHUB_GMS_URL=$DATAHUB_GMS_URL"
else
    echo "   ℹ️  DATAHUB_GMS_URL not set (using default: http://localhost:9002)"
fi

if [ -n "$DATAHUB_TOKEN" ]; then
    echo "   ✅ DATAHUB_TOKEN is set"
else
    echo "   ℹ️  DATAHUB_TOKEN not set (optional if token auth is disabled)"
fi

# Check Python dependencies
echo ""
echo "4️⃣  Checking Python dependencies..."
if python3 -c "import datahub" 2>/dev/null; then
    echo "   ✅ DataHub SDK is installed"
else
    echo "   ❌ DataHub SDK not found"
    echo "      Install with: pip install -r requirements.txt"
fi

echo ""
echo "=========================================="
echo "✅ Configuration check complete!"
echo "=========================================="
echo ""
echo "💡 Next steps:"
echo "   1. Ensure METADATA_SERVICE_AUTH_ENABLED=true is set"
echo "   2. Run: python ingest_metadata.py"
echo "   Or use: datahub ingest -c ingestion_config.yaml"
