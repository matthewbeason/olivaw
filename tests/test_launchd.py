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
    assert Path("scripts/uninstall_launch_agent.sh").exists()
    assert Path("scripts/status_launch_agent.sh").exists()
