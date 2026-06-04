#!/bin/zsh
set -euo pipefail

LABEL="com.beason.olivaw"
DOMAIN="gui/$(id -u)"
PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
HOST="127.0.0.1"
PORT="8765"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This helper only supports macOS launchd." >&2
  exit 1
fi

echo "LaunchAgent: ${LABEL}"
echo "Plist: ${PLIST}"
if [[ -f "${PLIST}" ]]; then
  HOST="$(/usr/libexec/PlistBuddy -c "Print :ProgramArguments:3" "${PLIST}" 2>/dev/null || echo "127.0.0.1")"
  PORT="$(/usr/libexec/PlistBuddy -c "Print :ProgramArguments:5" "${PLIST}" 2>/dev/null || echo "8765")"
fi
echo "Bind: ${HOST}:${PORT}"
echo "Local URL: http://127.0.0.1:${PORT}"
if [[ "${HOST}" == "0.0.0.0" ]]; then
  echo "LAN URL example: http://home:${PORT}"
  echo "Caution: LAN mode exposes Olivaw to devices on your local network."
fi
echo

if launchctl print "${DOMAIN}/${LABEL}"; then
  echo
  echo "Logs:"
  echo "  ${HOME}/Library/Logs/olivaw.log"
  echo "  ${HOME}/Library/Logs/olivaw-error.log"
else
  echo "${LABEL} is not currently loaded." >&2
  echo "Install it with: scripts/install_launch_agent.sh" >&2
  exit 1
fi
