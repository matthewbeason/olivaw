#!/bin/zsh
set -euo pipefail

LABEL="com.beason.olivaw"
DOMAIN="gui/$(id -u)"
PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This helper only supports macOS launchd." >&2
  exit 1
fi

echo "LaunchAgent: ${LABEL}"
echo "Plist: ${PLIST}"
echo "URL: http://127.0.0.1:8765"
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
