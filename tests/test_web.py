from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from olivaw.models import HealthReport, ProviderStatus
from olivaw.web import app


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
    assert "Source Briefing" in response.text
    assert "source-backed" in response.text
    assert "manual, files" in response.text
    assert "Example item from manual source" in response.text
    assert "File found: status/system.txt" in response.text


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
