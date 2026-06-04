from __future__ import annotations

from olivaw.config import OlivawConfig
from olivaw.health import format_health_report
from olivaw.models import HealthReport, ProviderStatus


def test_health_report_mentions_ollama_guidance():
    report = HealthReport(
        local=ProviderStatus(
            name="ollama",
            kind="local",
            state="unavailable",
            message="Unable to connect to Ollama.",
            detail="Expected endpoint: http://localhost:11434. Install Ollama and run: ollama serve.",
            model="llama3.1:8b",
        ),
        cloud=ProviderStatus(
            name="openai",
            kind="cloud",
            state="disabled",
            message="OpenAI cloud provider is disabled.",
            model="gpt-4.1-mini",
        ),
        selected_provider=None,
        cloud_fallback="disabled",
        notes=["Cloud fallback is disabled by policy."],
    )

    text = format_health_report(report)

    assert "Local Provider: Unavailable" in text
    assert "http://localhost:11434" in text
    assert "ollama serve" in text
    assert "Cloud Fallback: disabled" in text


def test_config_constructs_without_external_services():
    config = OlivawConfig()

    assert config.local.base_url == "http://localhost:11434"
    assert config.cloud.enabled is False

