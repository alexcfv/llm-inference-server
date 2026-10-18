#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="embedding-server"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_DIR/.venv"
SERVICE_USER="${SUDO_USER:-$USER}"

info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m✔\033[0m %s\n' "$*"; }
fail()  { printf '\033[1;31m✘\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] && fail "Run without sudo: ./deploy/setup.sh"

info "Creating venv"
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
ok "venv ready"

info "Installing package"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -e "$REPO_DIR"
ok "dependencies installed"

info "Installing systemd unit"
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null <<EOF
[Unit]
Description=Embedding server (Horizon-Labs/multilingual-embedding-small)
After=network.target

[Service]
User=$SERVICE_USER
WorkingDirectory=$REPO_DIR
ExecStart=$VENV/bin/python $REPO_DIR/app.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"
ok "service enabled and started"

info "Waiting for startup"
PORT="$(grep -E '^\s*port:' "$REPO_DIR/config.yaml" | awk '{print $2}')"
PORT="${PORT:-8080}"
for i in $(seq 1 30); do
    if curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" > /dev/null; then
        ok "server is up on port $PORT"
        curl -s "http://127.0.0.1:$PORT/health"
        echo
        exit 0
    fi
    sleep 1
done
fail "server did not respond within 30s — check: journalctl -u $SERVICE_NAME"
