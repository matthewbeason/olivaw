#!/bin/zsh
set -euo pipefail

LABEL="com.beason.olivaw"
ROOT_DIR="/Users/mbeason/olivaw"
SOURCE_PLIST="${ROOT_DIR}/deploy/${LABEL}.plist"
TARGET_DIR="${HOME}/Library/LaunchAgents"
TARGET_PLIST="${TARGET_DIR}/${LABEL}.plist"
LOG_DIR="${HOME}/Library/Logs"
DOMAIN="gui/$(id -u)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This helper only supports macOS launchd." >&2
  exit 1
fi

if [[ ! -f "${SOURCE_PLIST}" ]]; then
  echo "Missing plist template: ${SOURCE_PLIST}" >&2
  exit 1
fi

if [[ ! -x "${ROOT_DIR}/.venv/bin/olivaw" ]]; then
  echo "Missing executable: ${ROOT_DIR}/.venv/bin/olivaw" >&2
  echo "Create the virtualenv and install Olivaw first: python -m pip install -e '.[dev]'" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}" "${LOG_DIR}"
cp "${SOURCE_PLIST}" "${TARGET_PLIST}"

launchctl bootout "${DOMAIN}" "${TARGET_PLIST}" >/dev/null 2>&1 || true
launchctl bootstrap "${DOMAIN}" "${TARGET_PLIST}"
launchctl enable "${DOMAIN}/${LABEL}"
launchctl kickstart -k "${DOMAIN}/${LABEL}"

echo "Installed and started ${LABEL}."
echo "URL: http://127.0.0.1:8765"
echo "Status: scripts/status_launch_agent.sh"
echo "Logs: ${LOG_DIR}/olivaw.log and ${LOG_DIR}/olivaw-error.log"
