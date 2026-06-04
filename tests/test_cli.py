from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from olivaw.cli import main


def clear_config_env(monkeypatch):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_LOCAL_BASE_URL",
        "OLIVAW_LOCAL_MODEL",
        "OLIVAW_CLOUD_ENABLED",
        "OLIVAW_CLOUD_MODEL",
        "OLIVAW_CLOUD_FALLBACK",
        "OPENAI_API_KEY",
        "OLIVAW_OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_cli_reports_missing_config_without_traceback(monkeypatch, tmp_path, capsys):
    missing = tmp_path / "missing.toml"
    monkeypatch.setenv("OLIVAW_CONFIG", str(missing))

    exit_code = main(["health"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Configuration error:" in captured.err
    assert str(missing) in captured.err
    assert "Traceback" not in captured.err


def test_cli_web_defaults_to_localhost_port_8765(monkeypatch):
    captured = {}

    def fake_run(app, host, port, reload):
        captured.update({"app": app, "host": host, "port": port, "reload": reload})

    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=fake_run),
    )

    assert main(["web"]) == 0
    assert captured == {
        "app": "olivaw.web:app",
        "host": "127.0.0.1",
        "port": 8765,
        "reload": False,
    }


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["web", "--host", "0.0.0.0"], {"host": "0.0.0.0", "port": 8765}),
        (["web", "--port", "9876"], {"host": "127.0.0.1", "port": 9876}),
        (
            ["web", "--host", "127.0.0.1", "--port", "8765"],
            {"host": "127.0.0.1", "port": 8765},
        ),
    ],
)
def test_cli_web_accepts_host_and_port(monkeypatch, args, expected):
    captured = {}

    def fake_run(app, host, port, reload):
        captured.update({"app": app, "host": host, "port": port, "reload": reload})

    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=fake_run),
    )

    assert main(args) == 0
    assert captured["host"] == expected["host"]
    assert captured["port"] == expected["port"]


def test_cli_sources_outputs_registered_sources(capsys):
    exit_code = main(["sources"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Olivaw Sources" in captured.out
    assert "Manual example source (manual): ok" in captured.out
    assert "Example item: Demonstrates source plumbing." in captured.out


def test_cli_config_outputs_redacted_user_config(monkeypatch, tmp_path, capsys):
    clear_config_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = (
        tmp_path / "Library" / "Application Support" / "Olivaw" / "config.toml"
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
[providers.cloud]
enabled = true
model = "gpt-4.1"

[secrets]
openai_api_key = "config-secret"
""",
        encoding="utf-8",
    )

    exit_code = main(["config"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Olivaw Configuration" in captured.out
    assert f"Config file path: {config_path}" in captured.out
    assert "Config file exists: yes" in captured.out
    assert "- Enabled: yes" in captured.out
    assert "- API key configured: yes" in captured.out
    assert "config-secret" not in captured.out
