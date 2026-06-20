#!/bin/zsh
set -euo pipefail

LABEL="com.beason.olivaw"
TARGET_PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
DOMAIN="gui/$(id -u)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This helper only supports macOS launchd." >&2
  exit 1
fi

if [[ ! -f "${TARGET_PLIST}" ]]; then
  echo "${LABEL} is not installed in ~/Library/LaunchAgents." >&2
  echo "Install it with: scripts/install_launch_agent.sh" >&2
  exit 1
fi

launchctl kickstart -k "${DOMAIN}/${LABEL}"
echo "Restarted ${LABEL}."
