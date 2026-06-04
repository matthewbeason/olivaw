#!/bin/zsh
set -euo pipefail

LABEL="com.beason.olivaw"
TARGET_PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
DOMAIN="gui/$(id -u)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This helper only supports macOS launchd." >&2
  exit 1
fi

if [[ -f "${TARGET_PLIST}" ]]; then
  launchctl bootout "${DOMAIN}" "${TARGET_PLIST}" >/dev/null 2>&1 || true
  rm -f "${TARGET_PLIST}"
  echo "Uninstalled ${LABEL}."
else
  launchctl bootout "${DOMAIN}/${LABEL}" >/dev/null 2>&1 || true
  echo "${LABEL} was not installed in ~/Library/LaunchAgents."
fi
