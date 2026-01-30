#!/usr/bin/env bash
# Deploy merged backend+UI to EC2. Run ./build.sh first.
# Set OPENINT_EC2_HOST and OPENINT_EC2_KEY (or EC2_HOST, EC2_KEY), e.g.:
#   export OPENINT_EC2_KEY=~/.ssh/openint.pem
#   export OPENINT_EC2_HOST=ec2-user@ec2-3-148-183-18.us-east-2.compute.amazonaws.com

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EC2_HOST="${OPENINT_EC2_HOST:-$EC2_HOST}"
EC2_KEY="${OPENINT_EC2_KEY:-$EC2_KEY}"
REMOTE_DIR="${OPENINT_EC2_REMOTE_DIR:-/home/ec2-user/openint}"

if [ -z "$EC2_HOST" ]; then
  echo "Set OPENINT_EC2_HOST (or EC2_HOST), e.g.:"
  echo "  export OPENINT_EC2_HOST=ec2-user@ec2-3-148-183-18.us-east-2.compute.amazonaws.com"
  exit 1
fi

if [ ! -d "openint-ui/dist" ]; then
  echo "openint-ui/dist not found. Run ./build.sh first."
  exit 1
fi

# Build rsync args: exclude large/unnecessary dirs
RSYNC_EXCLUDE=(
  --exclude '.git'
  --exclude 'node_modules'
  --exclude 'venv'
  --exclude '__pycache__'
  --exclude '*.pyc'
  --exclude '.env'
  --exclude '*.log'
  --exclude 'openint-ui/node_modules'
  --exclude '.backend.pid'
  --exclude '.agent_system.pid'
)

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[ -n "$EC2_KEY" ] && SSH_OPTS+=(-i "$EC2_KEY")

echo "Creating directory structure on EC2..."
ssh "${SSH_OPTS[@]}" "$EC2_HOST" "mkdir -p $REMOTE_DIR/openint-backend $REMOTE_DIR/openint-agents $REMOTE_DIR/openint-vectordb $REMOTE_DIR/openint-graph $REMOTE_DIR/shared $REMOTE_DIR/openint-ui/dist $REMOTE_DIR/scripts"

echo "Syncing to $EC2_HOST:$REMOTE_DIR ..."
rsync -avz "${RSYNC_EXCLUDE[@]}" openint-backend/ "$EC2_HOST:$REMOTE_DIR/openint-backend/"
rsync -avz "${RSYNC_EXCLUDE[@]}" openint-agents/ "$EC2_HOST:$REMOTE_DIR/openint-agents/"
rsync -avz "${RSYNC_EXCLUDE[@]}" openint-vectordb/ "$EC2_HOST:$REMOTE_DIR/openint-vectordb/"
rsync -avz "${RSYNC_EXCLUDE[@]}" openint-graph/ "$EC2_HOST:$REMOTE_DIR/openint-graph/"
rsync -avz "${RSYNC_EXCLUDE[@]}" shared/ "$EC2_HOST:$REMOTE_DIR/shared/"
rsync -avz openint-ui/dist/ "$EC2_HOST:$REMOTE_DIR/openint-ui/dist/"
rsync -avz scripts/ec2_install_and_restart.sh "$EC2_HOST:$REMOTE_DIR/scripts/"

echo "Running install and restart on EC2..."
ssh "${SSH_OPTS[@]}" "$EC2_HOST" "cd $REMOTE_DIR && chmod +x scripts/ec2_install_and_restart.sh && nohup scripts/ec2_install_and_restart.sh >> openint.log 2>&1 &"
echo "Deploy triggered. App should be starting on EC2 port ${PORT:-3001}."
echo "Check: ssh $EC2_HOST tail -f $REMOTE_DIR/openint.log"
echo "Or: curl http://<EC2_PUBLIC_IP>:3001/api/health"
echo "UI: http://<EC2_PUBLIC_IP>:3001/"
exit 0
