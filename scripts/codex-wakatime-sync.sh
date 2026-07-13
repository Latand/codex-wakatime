#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DEFAULT_SYNC_SCRIPT="${SCRIPT_DIR}/codex_wakatime_sync.py"
if [[ ! -f "${DEFAULT_SYNC_SCRIPT}" ]]; then
  DEFAULT_SYNC_SCRIPT="${SCRIPT_DIR}/../codex_wakatime_sync.py"
fi
SYNC_SCRIPT="${SCRIPT_PATH:-${DEFAULT_SYNC_SCRIPT}}"
STATE_DB="${STATE_DB:-$HOME/.codex-wakatime/state.db}"
PRIVACY_KEY_FILE="${PRIVACY_KEY_FILE:-$HOME/.codex-wakatime/privacy.key}"
WAKATIME_CONFIG="${WAKATIME_CONFIG:-$HOME/.wakatime.cfg}"
SOURCE_FILE="${SOURCE_FILE-$HOME/.config/agent-wakatime/sources.conf}"

SINCE_ARG="${SINCE:-}"
EXTRA_ARGS=("$@")
if [[ $# -gt 0 && ${1} != --* ]]; then
  SINCE_ARG="${1}"
  EXTRA_ARGS=("${@:2}")
fi
SINCE_ARG="${SINCE_ARG:-45m}"

SOURCE_ARGS=()
if [[ -n "${SOURCE_FILE}" && -f "${SOURCE_FILE}" ]]; then
  SOURCE_ARGS=(--source-file "${SOURCE_FILE}")
fi

exec "${PYTHON_BIN}" "${SYNC_SCRIPT}" sync \
  --since "${SINCE_ARG}" \
  --state-db "${STATE_DB}" \
  --privacy-key-file "${PRIVACY_KEY_FILE}" \
  --wakatime-config "${WAKATIME_CONFIG}" \
  "${SOURCE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
