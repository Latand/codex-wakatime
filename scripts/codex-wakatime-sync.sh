#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DEFAULT_SYNC_SCRIPT="${SCRIPT_DIR}/codex_wakatime_sync.py"
if [[ ! -f "${DEFAULT_SYNC_SCRIPT}" ]]; then
  DEFAULT_SYNC_SCRIPT="${SCRIPT_DIR}/../codex_wakatime_sync.py"
fi
SYNC_SCRIPT="${SCRIPT_PATH:-${DEFAULT_SYNC_SCRIPT}}"
SESSIONS_DIR="${SESSIONS_DIR:-$HOME/.codex/sessions}"
STATE_DB="${STATE_DB:-$HOME/.codex-wakatime/state.db}"
WAKATIME_BIN="${WAKATIME_BIN:-wakatime-cli}"

SINCE_ARG="${SINCE:-}"
EXTRA_ARGS=("$@")
if [[ $# -gt 0 && ${1} != --* ]]; then
  SINCE_ARG="${1}"
  EXTRA_ARGS=("${@:2}")
fi
SINCE_ARG="${SINCE_ARG:-45m}"

${PYTHON_BIN} "${SYNC_SCRIPT}" sync \
  --since "${SINCE_ARG}" \
  --sessions-dir "${SESSIONS_DIR}" \
  --state-db "${STATE_DB}" \
  --wakatime-bin "${WAKATIME_BIN}" \
  "${EXTRA_ARGS[@]}"

"${WAKATIME_BIN}" --sync-offline-activity 0 >/dev/null 2>&1 || true
