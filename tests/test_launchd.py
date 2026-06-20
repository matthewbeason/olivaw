from __future__ import annotations

import plistlib
from pathlib import Path


PLIST_PATH = Path("deploy/com.beason.olivaw.plist")


def load_plist() -> dict[str, object]:
    with PLIST_PATH.open("rb") as handle:
        return plistlib.load(handle)


def test_launchd_plist_template_exists():
    assert PLIST_PATH.exists()


def test_launchd_plist_contains_expected_service_configuration():
    plist = load_plist()

    assert plist["Label"] == "com.beason.olivaw"
    assert plist["ProgramArguments"] == [
        "/Users/mbeason/olivaw/.venv/bin/olivaw",
        "web",
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
    ]
    assert plist["WorkingDirectory"] == "/Users/mbeason/olivaw"
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] is True
    assert plist["StandardOutPath"] == "/Users/mbeason/Library/Logs/olivaw.log"
    assert (
        plist["StandardErrorPath"]
        == "/Users/mbeason/Library/Logs/olivaw-error.log"
    )


def test_launchd_helper_scripts_exist():
    assert Path("scripts/install_launch_agent.sh").exists()
    assert Path("scripts/restart_launch_agent.sh").exists()
    assert Path("scripts/uninstall_launch_agent.sh").exists()
    assert Path("scripts/status_launch_agent.sh").exists()


def test_install_launch_agent_supports_lan_binding_options():
    script = Path("scripts/install_launch_agent.sh").read_text(encoding="utf-8")

    assert 'HOST="${OLIVAW_WEB_HOST:-127.0.0.1}"' in script
    assert 'PORT="${OLIVAW_WEB_PORT:-8765}"' in script
    assert "--lan" in script
    assert 'HOST="0.0.0.0"' in script
    assert "--host HOST" in script
    assert "--port PORT" in script
    assert "Set :ProgramArguments:3 ${HOST}" in script
    assert "Set :ProgramArguments:5 ${PORT}" in script
    assert "LAN URL example: http://home:${PORT}" in script
    assert "LAN mode exposes Olivaw" in script


def test_status_launch_agent_reports_configured_host_and_lan_url():
    script = Path("scripts/status_launch_agent.sh").read_text(encoding="utf-8")

    assert 'Print :ProgramArguments:3' in script
    assert 'Print :ProgramArguments:5' in script
    assert "Local URL: http://127.0.0.1:${PORT}" in script
    assert "LAN URL example: http://home:${PORT}" in script
    assert "LAN mode exposes Olivaw" in script


def test_restart_launch_agent_kickstarts_installed_service():
    script = Path("scripts/restart_launch_agent.sh").read_text(encoding="utf-8")

    assert 'TARGET_PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"' in script
    assert 'launchctl kickstart -k "${DOMAIN}/${LABEL}"' in script
    assert "Install it with: scripts/install_launch_agent.sh" in script
