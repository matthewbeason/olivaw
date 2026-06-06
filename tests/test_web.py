from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from olivaw.models import HealthReport, ProviderStatus
from olivaw.web import (
    _briefing_dashboard,
    _dashboard_status,
    _human_generated_time,
    app,
)


client = TestClient(app)
WEATHER_PROMPT = "Hi could you tell me what the weather is in Phoenix az"


@pytest.fixture(autouse=True)
def mock_health_checks(monkeypatch):
    def fake_health(config=None):
        return HealthReport(
            local=ProviderStatus(
                name="ollama",
                kind="local",
                state="unavailable",
                message="Mocked local provider status.",
                detail="Mocked test health check; no local network probe.",
                model="llama3.1:8b",
            ),
            cloud=ProviderStatus(
                name="openai",
                kind="cloud",
                state="disabled",
                message="Mocked cloud provider status.",
                model="gpt-4.1-mini",
            ),
            selected_provider=None,
            cloud_fallback="disabled",
            notes=["Mocked web health check."],
        )

    monkeypatch.setattr("olivaw.web.run_health_checks", fake_health)


def test_home_route_renders():
    response = client.get("/")

    assert response.status_code == 200
    assert "Assistant Home" in response.text
    assert "Example Briefing" in response.text
    assert "Briefing renders without repo fixtures" in response.text


def test_home_route_renders_from_non_repo_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert "Assistant Home" in response.text
    assert "Briefing renders without repo fixtures" in response.text


def test_health_route_renders():
    response = client.get("/health")

    assert response.status_code == 200
    assert "Local Provider" in response.text
    assert "Cloud Provider" in response.text
    assert "Mocked local provider status." in response.text


def test_capabilities_route_renders_identity_grounding():
    response = client.get("/capabilities")

    assert response.status_code == 200
    assert "Implemented" in response.text
    assert "Not Implemented Yet" in response.text
    assert "deterministic briefing generation from structured input" in response.text
    assert "calendar integration" in response.text
    assert "weather lookup" in response.text


def test_sources_route_renders_registered_sources():
    # Route should render both default sources without requiring a real data dir.
    response = client.get("/sources")

    assert response.status_code == 200
    assert "Sources" in response.text
    assert "Manual example source" in response.text
    assert "Local files" in response.text
    assert "Prime Observer" in response.text
    assert "Core Signal" in response.text
    assert "manual" in response.text
    assert "files" in response.text
    assert "prime_observer" in response.text
    assert "core_signal" in response.text
    assert "Example item" in response.text
    assert "Demonstrates source plumbing." in response.text


def test_briefing_route_renders_source_backed_briefing(monkeypatch, tmp_path):
    for name in ("OLIVAW_CONFIG", "OLIVAW_FILES_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    data_path = tmp_path / "Library" / "Application Support" / "Olivaw" / "data"
    (data_path / "status").mkdir(parents=True)
    (data_path / "status" / "system.txt").write_text(
        "System status\nAll local.\n",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert "Today at a glance" in response.text
    assert "source-backed" in response.text
    assert "Generated" in response.text
    assert re.search(r"Generated (just now|\d+ minutes? ago|today at)", response.text)
    assert "Refresh briefing" in response.text
    assert "Overall Status" in response.text
    assert "Healthy" in response.text
    assert "Sources do not report a condition needing attention." in response.text
    assert "Worth Knowing" in response.text
    assert "Recommended Action" in response.text
    assert "Network Status" in response.text
    assert "DNS Activity" in response.text
    assert "Core Signal Findings" in response.text
    assert "Category: context" in response.text
    assert "Category: action" in response.text
    assert "Category: network" in response.text
    assert "Category: DNS" in response.text
    assert "Category: interpretation" in response.text
    assert "Show raw briefing" in response.text
    assert "manual, files" in response.text
    assert "Example item from manual source" in response.text
    assert "File found: status/system.txt" in response.text


def test_briefing_route_reflects_changed_source_data_between_requests(
    monkeypatch,
    tmp_path,
):
    for name in ("OLIVAW_CONFIG", "OLIVAW_FILES_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    data_path = tmp_path / "Library" / "Application Support" / "Olivaw" / "data"
    data_path.mkdir(parents=True)
    source_file = data_path / "status.txt"
    source_file.write_text("Version one\n", encoding="utf-8")

    first = client.get("/briefing")

    source_file.write_text("Version two\n", encoding="utf-8")
    second = client.get("/briefing")

    assert first.status_code == 200
    assert second.status_code == 200
    assert "Version one" in first.text
    assert "Version two" in second.text
    assert "Version one" not in second.text
    assert "This briefing is source-backed using: manual, files." in second.text


def test_briefing_dashboard_promotes_dns_domains_and_collapses_metrics():
    dashboard = _briefing_dashboard(
        """# Source Briefing

## Prime Observer
- DNS summary: available from Prime Observer.
  - Total queries: 1000
  - Blocked queries: 20
  - Encrypted queries: 800
  - Block rate: 2.0%
  - Raw block rate: 0.02
  - Top queried domain: www.example.test (count 300, share 0.3)
  - Top blocked domain: ads.example.test (count 10, share 0.5)
  - Top resolved domain: api.example.test (count 250, share 0.25)

## Core Signal
- Core Signal Morning Brief - 2026-06-05 (2026-06-05) [Healthy]: Stable.
  - Recommended action: No action.

## Attribution
This briefing is source-backed using: prime_observer, core_signal.
""",
        "2026-06-06T00:14:59+00:00",
        ("prime_observer", "core_signal"),
    )

    assert dashboard["dns_activity"] == [
        "Top blocked domain: ads.example.test",
        "Top resolved domain: api.example.test",
        "Top queried domain: www.example.test",
    ]
    assert "Total queries: 1000" in dashboard["dns_details"]
    assert "Block rate: 2.0%" in dashboard["dns_details"]
    assert "Top queried domain: www.example.test (count 300, share 0.3)" in (
        dashboard["dns_details"]
    )
    assert dashboard["sources"] == ("prime_observer", "core_signal")


def test_human_generated_time_formats_relative_and_today():
    generated = datetime(2026, 6, 6, 0, 14, 59, tzinfo=timezone.utc)

    assert _human_generated_time(generated, now=generated + timedelta(minutes=2)) == (
        "Generated 2 minutes ago"
    )

    later = generated + timedelta(hours=2)
    assert "Generated today at" in _human_generated_time(generated, now=later)


def test_dashboard_status_maps_no_action_watch_to_healthy():
    core_lines = [
        "- Core Signal Morning Brief - 2026-06-05 (2026-06-05) [Watch]: "
        "Performance was unusually slow compared with the normal pattern.",
        "- Why/status reasoning: Performance was noticeably different from "
        "historical norms, but it was not actionable because no sustained "
        "instability or user-impacting issue was detected.",
        "- Recommended action: No action unless people noticed slow calls.",
        "- DNS filtering looked normal: 2.4% of queries were blocked.",
    ]
    prime_lines = [
        "- Network attribution generated at 2026-06-05T23:18:43+00:00.",
        "- Current LAN/WAN state: No network issue detected",
        "- Current status: no_issue_detected",
    ]

    status = _dashboard_status(core_lines, prime_lines)

    assert status["label"] == "Healthy"
    assert status["tone"] == "healthy"
    assert "No action is recommended" in status["explanation"]


def test_dashboard_status_keeps_watch_when_monitoring_is_warranted():
    core_lines = [
        "- Core Signal Morning Brief - 2026-06-05 (2026-06-05) [Watch]: "
        "Recurring latency should be watched.",
        "- Why/status reasoning: Confidence: medium; worth monitoring.",
        "- Recommended action: Monitor the next weekly report.",
    ]

    status = _dashboard_status(core_lines, [])

    assert status["label"] == "Watch"
    assert status["tone"] == "watch"
    assert "worth monitoring" in status["explanation"]


def test_dashboard_status_marks_action_needed_for_actionable_recommendation():
    core_lines = [
        "- Core Signal Morning Brief - 2026-06-05 (2026-06-05) [Attention]: "
        "Sustained slowdown detected.",
        "- Why/status reasoning: A sustained slowdown was detected.",
        "- Recommended action: Restart the router and recheck the connection.",
    ]

    status = _dashboard_status(core_lines, [])

    assert status["label"] == "Action Needed"
    assert status["tone"] == "action"
    assert "needs attention" in status["explanation"]


def test_chat_post_renders_chat_response(monkeypatch):
    class FakeResponse:
        text = "mocked OpenAI-capable chat response"

    def fake_run_with_attribution(self, prompt, config=None):
        assert prompt == "hello"
        return FakeResponse()

    monkeypatch.setattr(
        "olivaw.web.ChatCapability.run_with_attribution",
        fake_run_with_attribution,
    )

    response = client.post("/chat", data={"prompt": "hello"})

    assert response.status_code == 200
    assert "mocked OpenAI-capable chat response" in response.text


def test_chat_post_handles_unavailable_capability_without_provider(monkeypatch):
    class FailingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request):
            raise AssertionError("weather request should not call provider")

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", FailingRouter)

    response = client.post("/chat", data={"prompt": "What's the weather in Phoenix?"})

    assert response.status_code == 200
    assert "do not currently have a weather source configured" in response.text
    assert "WeatherSource" in response.text


def test_chat_post_exact_weather_request_matches_cli_guardrails(monkeypatch):
    class FailingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request):
            raise AssertionError("weather request should not call provider")

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", FailingRouter)

    response = client.post("/chat", data={"prompt": WEATHER_PROMPT})

    assert response.status_code == 200
    assert "do not currently have a weather source configured" in response.text
    assert "WeatherSource" in response.text
    assert "enable_openai_weather" not in response.text
    assert "provide weather via cloud OpenAI provider support" not in response.text
    assert "OpenAI can retrieve live weather" not in response.text


def test_settings_does_not_expose_secret(monkeypatch):
    monkeypatch.setenv("OLIVAW_OPENAI_API_KEY", "very-secret")

    response = client.get("/settings")

    assert response.status_code == 200
    assert "API key present" in response.text
    assert "very-secret" not in response.text


def test_config_route_renders_redacted_user_config(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_CLOUD_ENABLED",
        "OPENAI_API_KEY",
        "OLIVAW_OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
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

    response = client.get("/config")

    assert response.status_code == 200
    assert "Configuration" in response.text
    assert str(config_path) in response.text
    assert "API key present" in response.text
    assert "yes" in response.text
    assert "config-secret" not in response.text
