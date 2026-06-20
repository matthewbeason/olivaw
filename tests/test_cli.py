from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from olivaw.cli import main

WEATHER_PROMPT = "Hi could you tell me what the weather is in Phoenix az"


def clear_config_env(monkeypatch):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_LOCAL_BASE_URL",
        "OLIVAW_LOCAL_MODEL",
        "OLIVAW_CLOUD_ENABLED",
        "OLIVAW_CLOUD_MODEL",
        "OLIVAW_CLOUD_FALLBACK",
        "OLIVAW_FILES_DIR",
        "OLIVAW_FILES_MAX_BYTES",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_PRIME_OBSERVER_ENABLED",
        "OLIVAW_CORE_SIGNAL_DIR",
        "OLIVAW_CORE_SIGNAL_ENABLED",
        "OLIVAW_HEALTH_REVIEW_ENABLED",
        "OLIVAW_TEMPLATE_AUTO_RELOAD",
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


def test_cli_restart_web_kickstarts_launch_agent(monkeypatch, tmp_path, capsys):
    target = tmp_path / "Library/LaunchAgents/com.beason.olivaw.plist"
    target.parent.mkdir(parents=True)
    target.write_text("", encoding="utf-8")

    captured = {}

    def fake_run(cmd, check):
        captured["cmd"] = cmd
        captured["check"] = check

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(os, "getuid", lambda: 501)
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(sys, "platform", "darwin")

    assert main(["restart-web"]) == 0
    stdout = capsys.readouterr().out

    assert captured == {
        "cmd": ["launchctl", "kickstart", "-k", "gui/501/com.beason.olivaw"],
        "check": True,
    }
    assert "Restarted gui/501/com.beason.olivaw." in stdout


def test_cli_restart_web_requires_installed_launch_agent(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(sys, "platform", "darwin")

    assert main(["restart-web"]) == 2
    captured = capsys.readouterr()

    assert "LaunchAgent not installed." in captured.err


def test_cli_sources_outputs_registered_sources(capsys):
    # Uses repository/user defaults; only asserts source registration shape.
    exit_code = main(["sources"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Olivaw Sources" in captured.out
    assert "Manual example source (manual): ok" in captured.out
    assert "Local files (files):" in captured.out
    assert "Prime Observer (prime_observer):" in captured.out
    assert "Core Signal (core_signal):" in captured.out


def test_cli_health_review_outputs_diagnostic(monkeypatch, capsys):
    from olivaw.assistant.attribution import SOURCE_BACKED, AttributedResponse
    from olivaw.briefing.health_review import HealthReviewResult

    def fake_briefing(config=None):
        return AttributedResponse(
            text="# Source Briefing\n\n## Core Signal\n- Core Signal test: Stable.\n",
            attribution=SOURCE_BACKED,
            sources=("core_signal",),
            capability="source-backed briefing",
        )

    def fake_generate(dashboard, *, config):
        return HealthReviewResult(
            text="Generated review.",
            status="available",
            provider="fake-local",
            model="fake-model",
            latency_ms=12,
        )

    monkeypatch.setattr("olivaw.cli.compose_source_briefing", fake_briefing)
    monkeypatch.setattr("olivaw.cli.generate_health_review", fake_generate)

    exit_code = main(["health-review"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Olivaw Health Review" in captured.out
    assert "Status: available" in captured.out
    assert "Provider: fake-local" in captured.out
    assert "Model: fake-model" in captured.out
    assert "Accepted: yes" in captured.out
    assert "Rejected: no" in captured.out
    assert "Generated review." in captured.out


def test_cli_health_review_accepts_transient_model_override(monkeypatch, capsys):
    from olivaw.assistant.attribution import SOURCE_BACKED, AttributedResponse
    from olivaw.briefing.health_review import HealthReviewResult

    seen_models: list[str] = []

    def fake_briefing(config=None):
        return AttributedResponse(
            text="# Source Briefing\n\n## Core Signal\n- Core Signal test: Stable.\n",
            attribution=SOURCE_BACKED,
            sources=("core_signal",),
            capability="source-backed briefing",
        )

    def fake_generate(dashboard, *, config):
        seen_models.append(config.local.model)
        return HealthReviewResult(
            text="Generated review.",
            status="available",
            provider="fake-local",
            model=config.local.model,
            latency_ms=12,
        )

    monkeypatch.setattr("olivaw.cli.compose_source_briefing", fake_briefing)
    monkeypatch.setattr("olivaw.cli.generate_health_review", fake_generate)

    exit_code = main(["health-review", "--model", "llama3.1:8b"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert seen_models == ["llama3.1:8b"]
    assert "Model: llama3.1:8b" in captured.out


def test_cli_health_review_model_override_does_not_persist_config(
    monkeypatch,
    tmp_path,
):
    from olivaw.assistant.attribution import SOURCE_BACKED, AttributedResponse
    from olivaw.briefing.health_review import HealthReviewResult

    config_path = tmp_path / "olivaw.toml"
    config_text = "\n".join(
        [
            "[providers.local]",
            'type = "ollama"',
            'base_url = "http://localhost:11434"',
            'model = "llama3.2:3b"',
            "",
        ]
    )
    config_path.write_text(config_text, encoding="utf-8")
    monkeypatch.setenv("OLIVAW_CONFIG", str(config_path))

    def fake_briefing(config=None):
        return AttributedResponse(
            text="# Source Briefing\n\n## Core Signal\n- Core Signal test: Stable.\n",
            attribution=SOURCE_BACKED,
            sources=("core_signal",),
            capability="source-backed briefing",
        )

    def fake_generate(dashboard, *, config):
        return HealthReviewResult(
            text="Generated review.",
            status="available",
            provider="fake-local",
            model=config.local.model,
            latency_ms=12,
        )

    monkeypatch.setattr("olivaw.cli.compose_source_briefing", fake_briefing)
    monkeypatch.setattr("olivaw.cli.generate_health_review", fake_generate)

    assert main(["health-review", "--model", "llama3.1:8b"]) == 0

    assert config_path.read_text(encoding="utf-8") == config_text


def test_cli_health_review_attempts_repeat_diagnostic(monkeypatch, capsys):
    from olivaw.assistant.attribution import SOURCE_BACKED, AttributedResponse
    from olivaw.briefing.health_review import HealthReviewResult

    calls = 0

    def fake_briefing(config=None):
        return AttributedResponse(
            text="# Source Briefing\n\n## Core Signal\n- Core Signal test: Stable.\n",
            attribution=SOURCE_BACKED,
            sources=("core_signal",),
            capability="source-backed briefing",
        )

    def fake_generate(dashboard, *, config):
        nonlocal calls
        calls += 1
        return HealthReviewResult(
            text=f"Generated review {calls}.",
            status="available",
            provider="fake-local",
            model=config.local.model,
            latency_ms=12,
        )

    monkeypatch.setattr("olivaw.cli.compose_source_briefing", fake_briefing)
    monkeypatch.setattr("olivaw.cli.generate_health_review", fake_generate)

    exit_code = main(["health-review", "--attempts", "2"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == 2
    assert "Attempt: 1/2" in captured.out
    assert "Attempt: 2/2" in captured.out


def test_cli_health_review_diagnostic_reports_guardrail_rejection(monkeypatch, capsys):
    from olivaw.assistant.attribution import SOURCE_BACKED, AttributedResponse
    from olivaw.briefing.health_review import HealthReviewResult

    def fake_briefing(config=None):
        return AttributedResponse(
            text="# Source Briefing\n\n## Core Signal\n- Core Signal test: Stable.\n",
            attribution=SOURCE_BACKED,
            sources=("core_signal",),
            capability="source-backed briefing",
        )

    def fake_generate(dashboard, *, config):
        return HealthReviewResult(
            text="Health review unavailable: generated response was rejected by guardrails.",
            status="guardrail_rejected",
            reason="generated response was rejected by guardrails.",
            provider="fake-local",
            model=config.local.model,
            latency_ms=12,
            guardrail_rejected=True,
        )

    monkeypatch.setattr("olivaw.cli.compose_source_briefing", fake_briefing)
    monkeypatch.setattr("olivaw.cli.generate_health_review", fake_generate)

    exit_code = main(["health-review", "--model", "llama3.1:8b"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Status: guardrail_rejected" in captured.out
    assert "Accepted: no" in captured.out
    assert "Rejected: yes" in captured.out
    assert "Reason: generated response was rejected by guardrails." in captured.out


def test_cli_brief_sources_outputs_source_backed_briefing(monkeypatch, tmp_path, capsys):
    clear_config_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    data_path = tmp_path / "Library" / "Application Support" / "Olivaw" / "data"
    (data_path / "notes").mkdir(parents=True)
    (data_path / "notes" / "welcome.md").write_text(
        "# Welcome\nSource note.\n",
        encoding="utf-8",
    )

    exit_code = main(["brief-sources"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "# Source Briefing" in captured.out
    assert "- manual: ok" in captured.out
    assert "- files: ok" in captured.out
    assert "Example item from manual source" in captured.out
    assert "File found: notes/welcome.md" in captured.out
    assert "This briefing is source-backed using: manual, files." in captured.out


def test_cli_brief_sources_accepts_markdown_format(monkeypatch, tmp_path, capsys):
    clear_config_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    exit_code = main(["brief-sources", "--format", "markdown"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "# Source Briefing" in captured.out


def test_cli_brief_sources_outputs_core_signal_event_metadata(
    monkeypatch,
    tmp_path,
    capsys,
):
    clear_config_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    core_dir = tmp_path / "core"
    prime_dir = tmp_path / "prime"
    core_dir.mkdir()
    prime_dir.mkdir()
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    (core_dir / "latest.md").write_text(
        """# Core Signal Morning Brief - 2026-06-08

Status: Attention

The network had 1 sustained slowdown period(s).

Why This Status:
Sustained slowdown was detected.

Issue Location: Likely upstream/ISP issue

Recommended Action: Check provider status if symptoms matched.

Technical Evidence:
- Window: 2026-06-08T11:11:30+00:00 to 2026-06-08T11:12:09+00:00
- Prime Observer investigation: viz/investigate.html?start=1&end=2
- Attribution source: Prime Observer incident attribution
""",
        encoding="utf-8",
    )

    exit_code = main(["brief-sources"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Event: The network had 1 sustained slowdown period(s)." in captured.out
    assert "Severity/status: Attention / attention" in captured.out
    assert (
        "Affected window: 2026-06-08T11:11:30+00:00 "
        "to 2026-06-08T11:12:09+00:00"
    ) in captured.out
    assert (
        "Recommended action: Check provider status if symptoms matched."
        in captured.out
    )
    assert "Issue location: Likely upstream/ISP issue" in captured.out
    assert "View investigation: viz/investigate.html?start=1&end=2" in captured.out


def test_cli_brief_input_still_outputs_fixture_briefing(capsys):
    exit_code = main(["brief", "--input", "examples/daily_context.json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "# Daily Briefing" in captured.out
    assert "Stabilize Olivaw v0 as a local-first assistant foundation." in captured.out


def test_cli_chat_weather_request_uses_weather_source_without_provider(
    monkeypatch, capsys
):
    class FailingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request):
            raise AssertionError("weather request should not call provider")

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", FailingRouter)

    exit_code = main(["chat", WEATHER_PROMPT])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Weather:" in captured.out
    assert "Rain chance" in captured.out
    assert "enable_openai_weather" not in captured.out
    assert "provide weather via cloud OpenAI provider support" not in captured.out
    assert "OpenAI can retrieve live weather" not in captured.out


def test_cli_init_config_creates_config_once(monkeypatch, tmp_path, capsys):
    clear_config_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = (
        tmp_path / "Library" / "Application Support" / "Olivaw" / "config.toml"
    )

    assert main(["init-config"]) == 0
    first = capsys.readouterr()
    assert f"Created: {config_path}" in first.out
    assert config_path.exists()
    assert "openai_api_key = \"\"" in config_path.read_text(encoding="utf-8")

    config_path.write_text("sentinel", encoding="utf-8")
    assert main(["init-config"]) == 0
    second = capsys.readouterr()
    assert f"Configuration already exists: {config_path}" in second.out
    assert config_path.read_text(encoding="utf-8") == "sentinel"


def test_cli_init_data_creates_data_tree_once(monkeypatch, tmp_path, capsys):
    clear_config_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    data_path = tmp_path / "Library" / "Application Support" / "Olivaw" / "data"

    assert main(["init-data"]) == 0
    first = capsys.readouterr()
    assert f"Created: {data_path}" in first.out
    assert (data_path / "notes" / "welcome.md").exists()
    assert (data_path / "reports" / "example.json").exists()
    assert (data_path / "status" / "system.txt").exists()

    sentinel = data_path / "status" / "system.txt"
    sentinel.write_text("sentinel", encoding="utf-8")
    assert main(["init-data"]) == 0
    second = capsys.readouterr()
    assert f"Data directory already exists: {data_path}" in second.out
    assert sentinel.read_text(encoding="utf-8") == "sentinel"


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
