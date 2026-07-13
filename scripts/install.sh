#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/share/codex-wakatime}"
CONFIG_DIR="${CONFIG_DIR:-$HOME/.config/agent-wakatime}"

install -d -m 0755 "${INSTALL_DIR}"
install -m 0755 "${REPO_DIR}/codex_wakatime_sync.py" "${INSTALL_DIR}/codex_wakatime_sync.py"
install -m 0755 "${REPO_DIR}/scripts/codex-wakatime-sync.sh" "${INSTALL_DIR}/codex-wakatime-sync.sh"

install -d -m 0700 "${CONFIG_DIR}"
if [[ ! -e "${CONFIG_DIR}/sources.conf" ]]; then
  install -m 0600 "${REPO_DIR}/sources.example.conf" "${CONFIG_DIR}/sources.conf"
fi

printf 'Installed agent-wakatime files and source configuration.\n'
