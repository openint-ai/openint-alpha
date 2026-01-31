#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Prerequisites ──────────────────────────────────────────────
for cmd in minikube helm docker; do
  command -v "$cmd" >/dev/null 2>&1 || error "$cmd is required but not installed."
done

# ── Start minikube if not running ──────────────────────────────
if ! minikube status --format='{{.Host}}' 2>/dev/null | grep -q Running; then
  info "Starting minikube..."
  minikube start --cpus=4 --memory=8192 --driver=docker
else
  info "Minikube already running."
fi

# ── Point docker to minikube's daemon ──────────────────────────
info "Configuring docker to use minikube daemon..."
eval $(minikube docker-env)

# ── Build images inside minikube ───────────────────────────────
info "Building openint-backend image..."
docker build -f Dockerfile.backend -t openint-backend:latest .

info "Building openint-ui image..."
docker build -f Dockerfile.ui -t openint-ui:latest .

# ── Deploy with Helm ───────────────────────────────────────────
info "Deploying with Helm..."
helm upgrade --install openint ./helm/openint --wait --timeout 5m

# ── Wait for pods ──────────────────────────────────────────────
info "Waiting for all pods to be ready..."
kubectl wait --for=condition=ready pod -l app=openint-backend --timeout=120s 2>/dev/null || warn "Backend pod not ready yet"
kubectl wait --for=condition=ready pod -l app=openint-ui --timeout=60s 2>/dev/null || warn "UI pod not ready yet"

# ── Print status ───────────────────────────────────────────────
echo ""
info "Pod status:"
kubectl get pods

echo ""
info "Services:"
kubectl get svc

echo ""
URL=$(minikube service openint-ui --url 2>/dev/null || true)
if [ -n "$URL" ]; then
  info "OpenInt UI available at: ${GREEN}${URL}${NC}"
else
  info "Run 'minikube service openint-ui --url' to get the access URL"
fi
