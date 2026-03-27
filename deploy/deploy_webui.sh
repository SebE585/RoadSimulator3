#!/usr/bin/env bash
# deploy_webui.sh — Deploy RS3 Web UI to OVH server
# Usage: bash deploy/deploy_webui.sh
# Requires: ssh access to ubuntu@51.91.125.143 (passwordless sudo)
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVER="51.91.125.143"
SSH_USER="ubuntu"
REMOTE_BASE="/opt/daxos/rs3"
SERVICE_NAME="rs3-webui"
STREAMLIT_PORT=8503
PYTHON="python3"

LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ssh_cmd() { ssh "${SSH_USER}@${SERVER}" "$@"; }
scp_cmd() { scp -r "$@"; }

info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
error() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Sync project files (non-destructive: rsync, no --delete)
#    We send only the directories needed by the web UI.
# ---------------------------------------------------------------------------
info "Syncing RS3 project to ${SERVER}:${REMOTE_BASE} ..."
ssh_cmd "sudo mkdir -p ${REMOTE_BASE} && sudo chown -R ${SSH_USER}:${SSH_USER} ${REMOTE_BASE}"

RSYNC_OPTS=(-avz --exclude '__pycache__' --exclude '.git' --exclude '*.pyc' --exclude '.venv' --exclude 'node_modules')

# webui, core2, runner, config, and root-level files (pyproject.toml etc.)
rsync "${RSYNC_OPTS[@]}" "${LOCAL_ROOT}/webui/"        "${SSH_USER}@${SERVER}:${REMOTE_BASE}/webui/"
rsync "${RSYNC_OPTS[@]}" "${LOCAL_ROOT}/core2/"         "${SSH_USER}@${SERVER}:${REMOTE_BASE}/core2/"
rsync "${RSYNC_OPTS[@]}" "${LOCAL_ROOT}/runner/"        "${SSH_USER}@${SERVER}:${REMOTE_BASE}/runner/"
rsync "${RSYNC_OPTS[@]}" "${LOCAL_ROOT}/config/"        "${SSH_USER}@${SERVER}:${REMOTE_BASE}/config/"
rsync "${RSYNC_OPTS[@]}" "${LOCAL_ROOT}/simulator/"     "${SSH_USER}@${SERVER}:${REMOTE_BASE}/simulator/"
rsync "${RSYNC_OPTS[@]}" "${LOCAL_ROOT}/core/"          "${SSH_USER}@${SERVER}:${REMOTE_BASE}/core/"

# Root files needed for editable install
for f in pyproject.toml requirements.txt setup.py setup.cfg; do
    [ -f "${LOCAL_ROOT}/${f}" ] && rsync -avz "${LOCAL_ROOT}/${f}" "${SSH_USER}@${SERVER}:${REMOTE_BASE}/"
done

# ---------------------------------------------------------------------------
# 2. Create venv & install dependencies
# ---------------------------------------------------------------------------
info "Setting up virtualenv and installing dependencies ..."
ssh_cmd <<REMOTE
set -euo pipefail

cd "${REMOTE_BASE}"

# Create venv only if it does not exist
if [ ! -d .venv ]; then
    ${PYTHON} -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# Install the RS3 package (editable) + webui deps
pip install -e . || pip install -r requirements.txt
pip install -r webui/requirements.txt
REMOTE

# ---------------------------------------------------------------------------
# 3. Create systemd service (non-destructive: back up existing)
# ---------------------------------------------------------------------------
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

info "Installing systemd service ${SERVICE_NAME} ..."
ssh_cmd <<REMOTE
set -euo pipefail

if [ -f "${SERVICE_FILE}" ]; then
    echo "  -> Backing up existing service file to ${SERVICE_FILE}.bak"
    sudo cp "${SERVICE_FILE}" "${SERVICE_FILE}.bak"
fi

sudo tee "${SERVICE_FILE}" > /dev/null <<'UNIT'
[Unit]
Description=RoadSimulator3 Web UI (Streamlit)
After=network.target

[Service]
Type=simple
User=daxos
Group=daxos
WorkingDirectory=/opt/daxos/rs3/webui
Environment="PATH=/opt/daxos/rs3/.venv/bin:/usr/local/bin:/usr/bin"
Environment="PYTHONPATH=/opt/daxos/rs3"
ExecStart=/opt/daxos/rs3/.venv/bin/streamlit run app.py \
    --server.port=8503 \
    --server.address=127.0.0.1 \
    --server.headless=true \
    --browser.gatherUsageStats=false
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

# Ensure the daxos user exists
id -u daxos &>/dev/null || sudo useradd -r -m -d /opt/daxos -s /bin/bash daxos
sudo chown -R daxos:daxos "${REMOTE_BASE}"

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"
echo "  -> Service ${SERVICE_NAME} started on port ${STREAMLIT_PORT}"
REMOTE

# ---------------------------------------------------------------------------
# 4. Install nginx vhost (non-destructive: back up existing)
# ---------------------------------------------------------------------------
NGINX_CONF="/etc/nginx/sites-available/rs3"
NGINX_ENABLED="/etc/nginx/sites-enabled/rs3"

info "Installing nginx configuration ..."
scp_cmd "${LOCAL_ROOT}/deploy/nginx-rs3.conf" \
        "${SSH_USER}@${SERVER}:/tmp/nginx-rs3.conf"

ssh_cmd <<REMOTE
set -euo pipefail

if [ -f "${NGINX_CONF}" ]; then
    echo "  -> Backing up existing nginx conf to ${NGINX_CONF}.bak"
    sudo cp "${NGINX_CONF}" "${NGINX_CONF}.bak"
fi

sudo mv /tmp/nginx-rs3.conf "${NGINX_CONF}"
sudo ln -sf "${NGINX_CONF}" "${NGINX_ENABLED}"

sudo nginx -t && sudo systemctl reload nginx
echo "  -> nginx reloaded with RS3 vhost"
REMOTE

info "Deployment complete."
info "RS3 UI should be live at https://simulate.roadsimulator3.fr"
