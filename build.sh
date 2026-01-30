#!/usr/bin/env bash
# Build merged backend+UI for deployment.
# Run from repo root. Produces openint-ui/dist and leaves repo ready for deploy_to_ec2.sh.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building OpenInt UI (openint-ui)..."
cd openint-ui
if [ ! -d "node_modules" ]; then
  npm install
fi
npm run build
cd ..

echo "Build complete. openint-ui/dist is ready for merged deploy."
echo "Run ./deploy_to_ec2.sh to deploy to EC2 (set OPENINT_EC2_HOST and OPENINT_EC2_KEY)."
