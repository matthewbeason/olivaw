from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from olivaw.models import HealthReport, ProviderStatus
from olivaw.web import app


client = TestClient(app)


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


def test_settings_does_not_expose_secret(monkeypatch):
    monkeypatch.setenv("OLIVAW_OPENAI_API_KEY", "very-secret")

    response = client.get("/settings")

    assert response.status_code == 200
    assert "API key present" in response.text
    assert "very-secret" not in response.text
