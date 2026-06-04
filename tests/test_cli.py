from __future__ import annotations

from olivaw.cli import main


def test_cli_reports_missing_config_without_traceback(monkeypatch, tmp_path, capsys):
    missing = tmp_path / "missing.toml"
    monkeypatch.setenv("OLIVAW_CONFIG", str(missing))

    exit_code = main(["health"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Configuration error:" in captured.err
    assert str(missing) in captured.err
    assert "Traceback" not in captured.err
