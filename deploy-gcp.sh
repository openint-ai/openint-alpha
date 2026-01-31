#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configuration ──────────────────────────────────────────────
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
REGION="us-central1"
CLUSTER_NAME="openint-cluster"
REPO_NAME="openint"
DOMAIN="ui.openint.ai"
STATIC_IP_NAME="openint-ip"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

[ -z "$PROJECT_ID" ] && error "No GCP project set. Run: gcloud config set project <PROJECT_ID>"

info "Deploying OpenInt to GCP"
info "  Project:  $PROJECT_ID"
info "  Region:   $REGION"
info "  Cluster:  $CLUSTER_NAME"
info "  Domain:   $DOMAIN"
echo ""

# ── Enable required APIs ───────────────────────────────────────
info "Enabling GCP APIs..."
gcloud services enable \
  container.googleapis.com \
  artifactregistry.googleapis.com \
  compute.googleapis.com \
  cloudbuild.googleapis.com \
  --project="$PROJECT_ID" --quiet

# ── Create Artifact Registry repo ──────────────────────────────
info "Creating Artifact Registry repository..."
gcloud artifacts repositories describe "$REPO_NAME" \
  --location="$REGION" --project="$PROJECT_ID" 2>/dev/null || \
gcloud artifacts repositories create "$REPO_NAME" \
  --repository-format=docker \
  --location="$REGION" \
  --project="$PROJECT_ID" \
  --quiet

# ── Build and push images via Cloud Build ──────────────────────
IMAGE_PREFIX="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"

info "Building openint-backend via Cloud Build..."
cat > /tmp/cloudbuild-backend.yaml <<EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', 'Dockerfile.backend', '-t', '${IMAGE_PREFIX}/openint-backend:latest', '.']
images:
  - '${IMAGE_PREFIX}/openint-backend:latest'
EOF
gcloud builds submit \
  --config=/tmp/cloudbuild-backend.yaml \
  --project="$PROJECT_ID" \
  -q .

info "Building openint-ui via Cloud Build..."
cat > /tmp/cloudbuild-ui.yaml <<EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', 'Dockerfile.ui', '-t', '${IMAGE_PREFIX}/openint-ui:latest', '.']
images:
  - '${IMAGE_PREFIX}/openint-ui:latest'
EOF
gcloud builds submit \
  --config=/tmp/cloudbuild-ui.yaml \
  --project="$PROJECT_ID" \
  -q .

# ── Create GKE Autopilot cluster ──────────────────────────────
info "Creating GKE Autopilot cluster (if not exists)..."
if ! gcloud container clusters describe "$CLUSTER_NAME" --region="$REGION" --project="$PROJECT_ID" 2>/dev/null; then
  gcloud container clusters create-auto "$CLUSTER_NAME" \
    --region="$REGION" \
    --project="$PROJECT_ID" \
    --quiet
fi

# ── Get cluster credentials ───────────────────────────────────
info "Getting cluster credentials..."
gcloud container clusters get-credentials "$CLUSTER_NAME" \
  --region="$REGION" \
  --project="$PROJECT_ID"

# ── Reserve static IP for load balancer ────────────────────────
info "Reserving global static IP..."
if ! gcloud compute addresses describe "$STATIC_IP_NAME" --global --project="$PROJECT_ID" 2>/dev/null; then
  gcloud compute addresses create "$STATIC_IP_NAME" \
    --global \
    --project="$PROJECT_ID"
fi

STATIC_IP=$(gcloud compute addresses describe "$STATIC_IP_NAME" \
  --global --project="$PROJECT_ID" --format="value(address)")
info "Static IP: $STATIC_IP"

# ── Substitute PROJECT_ID in values file ───────────────────────
VALUES_FILE="/tmp/openint-values-gcp.yaml"
sed "s|PROJECT_ID|${PROJECT_ID}|g" helm/openint/values-gcp.yaml > "$VALUES_FILE"

# ── Deploy with Helm ──────────────────────────────────────────
info "Deploying with Helm..."
helm upgrade --install openint ./helm/openint \
  -f "$VALUES_FILE" \
  --wait --timeout 10m

# ── Print status ──────────────────────────────────────────────
echo ""
info "Deployment complete!"
echo ""
kubectl get pods
echo ""
kubectl get svc
echo ""
kubectl get ingress
echo ""

info "Static IP: ${GREEN}${STATIC_IP}${NC}"
info "Domain:    ${GREEN}${DOMAIN}${NC}"
echo ""
info "DNS Configuration:"
info "  Add an A record in your DNS provider:"
info "    ${DOMAIN}  →  ${STATIC_IP}"
echo ""
info "SSL certificate will be auto-provisioned by GCP once DNS propagates."
info "  Check cert status: kubectl describe managedcertificate openint-managed-cert"
info "  Check ingress:     kubectl describe ingress openint-ingress"
echo ""
warn "It may take 10-15 minutes for the load balancer and SSL cert to become fully active."
