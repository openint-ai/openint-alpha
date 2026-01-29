#!/bin/bash

# Fix UI installation issues

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔧 Fixing UI Installation..."
echo ""

cd openint-ui

# Remove corrupted node_modules
if [ -d "node_modules" ]; then
    echo "Removing corrupted node_modules..."
    rm -rf node_modules
fi

# Remove lock file
if [ -f "package-lock.json" ]; then
    echo "Removing package-lock.json..."
    rm -f package-lock.json
fi

# Clean npm cache
echo "Cleaning npm cache..."
npm cache clean --force

# Reinstall dependencies
echo "Installing dependencies..."
npm install

echo ""
echo "✅ UI installation fixed!"
echo ""
echo "Now you can start the UI with:"
echo "  ./start_ui.sh"
