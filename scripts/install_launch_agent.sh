#!/bin/zsh
set -euo pipefail

LABEL="com.beason.olivaw"
ROOT_DIR="/Users/mbeason/olivaw"
SOURCE_PLIST="${ROOT_DIR}/deploy/${LABEL}.plist"
TARGET_DIR="${HOME}/Library/LaunchAgents"
TARGET_PLIST="${TARGET_DIR}/${LABEL}.plist"
LOG_DIR="${HOME}/Library/Logs"
DOMAIN="gui/$(id -u)"
HOST="${OLIVAW_WEB_HOST:-127.0.0.1}"
PORT="${OLIVAW_WEB_PORT:-8765}"

usage() {
  cat <<EOF
Usage: scripts/install_launch_agent.sh [--lan] [--host HOST] [--port PORT]

Options:
  --lan         Bind Olivaw to 0.0.0.0 for LAN access.
  --host HOST  Bind host. Default: 127.0.0.1
  --port PORT  Bind port. Default: 8765

LAN mode exposes Olivaw to devices on your local network.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lan)
      HOST="0.0.0.0"
      ;;
    --host)
      shift
      if [[ $# -eq 0 || -z "$1" ]]; then
        echo "Missing value for --host." >&2
        exit 1
      fi
      HOST="$1"
      ;;
    --port)
      shift
      if [[ $# -eq 0 || -z "$1" ]]; then
        echo "Missing value for --port." >&2
        exit 1
      fi
      PORT="$1"
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This helper only supports macOS launchd." >&2
  exit 1
fi

if [[ -z "${HOST}" ]]; then
  echo "Host cannot be empty." >&2
  exit 1
fi

if [[ ! "${PORT}" =~ ^[0-9]+$ ]]; then
  echo "Port must be numeric: ${PORT}" >&2
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
/usr/libexec/PlistBuddy -c "Set :ProgramArguments:3 ${HOST}" "${TARGET_PLIST}"
/usr/libexec/PlistBuddy -c "Set :ProgramArguments:5 ${PORT}" "${TARGET_PLIST}"

launchctl bootout "${DOMAIN}" "${TARGET_PLIST}" >/dev/null 2>&1 || true
launchctl bootstrap "${DOMAIN}" "${TARGET_PLIST}"
launchctl enable "${DOMAIN}/${LABEL}"
launchctl kickstart -k "${DOMAIN}/${LABEL}"

echo "Installed and started ${LABEL}."
echo "Bind: ${HOST}:${PORT}"
echo "Local URL: http://127.0.0.1:${PORT}"
if [[ "${HOST}" == "0.0.0.0" ]]; then
  echo "LAN URL example: http://home:${PORT}"
  echo "Caution: LAN mode exposes Olivaw to devices on your local network."
fi
echo "Status: scripts/status_launch_agent.sh"
echo "Logs: ${LOG_DIR}/olivaw.log and ${LOG_DIR}/olivaw-error.log"
