from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from olivaw.briefing.health_review import HealthReviewResult
from olivaw.models import HealthReport, ProviderStatus
from olivaw.web import (
    _briefing_dashboard,
    _dashboard_status,
    _human_generated_time,
    _normalize_briefing_dashboard,
    app,
)


client = TestClient(app)
WEATHER_PROMPT = "Hi could you tell me what the weather is in Phoenix az"


@pytest.fixture(autouse=True)
def mock_health_checks(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("olivaw.web._HEALTH_REVIEW_CACHE", None)

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
    monkeypatch.setattr(
        "olivaw.web.generate_health_review",
        lambda dashboard, *, config: HealthReviewResult(
            text="Health review unavailable: mocked web test.",
            status="generation_failed",
            reason="mocked web test",
            model="llama3.2:3b",
            provider="ollama",
        ),
    )


def test_home_route_renders():
    response = client.get("/")

    assert response.status_code == 200
    assert "Operations Center" in response.text
    assert "Overview" in response.text
    assert "Ask Olivaw" in response.text
    assert "Source Freshness" in response.text
    assert "What Changed Recently" in response.text
    assert "Network Signal" in response.text
    assert "Open Evidence Package" in response.text
    assert 'href="/chat?prompt=How%20was%20the%20network%20overnight%3F"' in (
        response.text
    )
    assert "Health review not generated yet." in response.text
    assert "Event ID" not in response.text
    assert "Raw briefing" not in response.text


def test_home_route_does_not_generate_health_review_synchronously(monkeypatch):
    def fail_generate(dashboard, *, config):
        raise AssertionError("GET / must not synchronously generate Health Review")

    monkeypatch.setattr("olivaw.web.generate_health_review", fail_generate)

    response = client.get("/")

    assert response.status_code == 200
    assert "Health review not generated yet." in response.text


def test_briefing_route_does_not_generate_health_review_synchronously(monkeypatch):
    def fail_generate(dashboard, *, config):
        raise AssertionError("GET /briefing must not synchronously generate Health Review")

    monkeypatch.setattr("olivaw.web.generate_health_review", fail_generate)

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Health review not generated yet." in response.text


def test_home_network_signal_renders_human_readable_fields(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (prime_dir / "network_attribution.json").write_text(
        """
{
  "generated_at": "2026-06-17T14:23:00+00:00",
  "current_attribution": {
    "label": "Likely upstream (ISP / path)",
    "status": "likely_upstream",
    "confidence": "medium",
    "evidence": ["WAN degraded while LAN remained healthy."]
  }
}
""",
        encoding="utf-8",
    )
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Morning Brief",
  "date": "2026-06-17",
  "status": "Watch",
  "summary": "One interpreted slowdown event is present.",
  "recommended_action": "No action unless people noticed issues.",
  "events": [
    {
      "summary": "A sustained slowdown was observed.",
      "status": "attention",
      "severity": "attention",
      "window_start": "2026-06-17 14:20",
      "window_end": "2026-06-17 14:23",
      "confidence": "medium",
      "issue_location": "Likely upstream (ISP / path)",
      "prime_observer_investigation": "viz/investigate.html?start=1&end=2"
    }
  ]
}
""",
        encoding="utf-8",
    )

    response = client.get("/")

    assert response.status_code == 200
    assert "Network Signal" in response.text
    assert "Status" in response.text
    assert "Attribution" in response.text
    assert "Likely upstream (ISP / path)" in response.text
    assert "Confidence" in response.text
    assert "medium" in response.text
    assert "Last incident" in response.text
    assert "2026-06-17 14:20 to 2026-06-17 14:23" in response.text
    assert "Open Evidence Package" in response.text
    assert "spark-strip" not in response.text
    assert "spark-segment" not in response.text


def test_home_navigation_simplifies_primary_routes():
    response = client.get("/")

    assert response.status_code == 200
    assert '<div class="nav-primary">' in response.text
    assert '<a href="/">Overview</a>' in response.text
    assert '<a href="/chat">Ask Olivaw</a>' in response.text
    assert '<a href="/sources">Sources</a>' in response.text
    assert '<a href="/settings">Settings</a>' in response.text
    assert '<div class="nav-secondary" aria-label="Compatibility links">' in (
        response.text
    )
    assert '<a href="/briefing">Briefing</a>' in response.text
    assert '<a href="/health">Health</a>' in response.text
    assert '<a href="/capabilities">Capabilities</a>' in response.text
    assert '<a href="/config">Config</a>' in response.text


def test_templates_do_not_hot_reload_in_long_running_web_process():
    assert app.version == "0.7.0"
    from olivaw.web import templates

    assert templates.env.auto_reload is False


def test_home_route_renders_from_non_repo_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert "Operations Center" in response.text
    assert "Source Freshness" in response.text


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
    assert "Weather" in response.text
    assert "manual" in response.text
    assert "files" in response.text
    assert "prime_observer" in response.text
    assert "core_signal" in response.text
    assert "weather" in response.text
    assert "Raw available" in response.text
    assert "local_context" in response.text
    assert "Example item" in response.text
    assert "Demonstrates source plumbing." in response.text


def test_sources_route_renders_weather_diagnostics(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
        "OLIVAW_WEATHER_ENABLED",
        "OLIVAW_WEATHER_LATITUDE",
        "OLIVAW_WEATHER_LONGITUDE",
        "OLIVAW_WEATHER_LOCATION_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    monkeypatch.setenv("OLIVAW_WEATHER_ENABLED", "true")
    monkeypatch.setenv("OLIVAW_WEATHER_LATITUDE", "33.4484")
    monkeypatch.setenv("OLIVAW_WEATHER_LONGITUDE", "-112.074")
    monkeypatch.setenv("OLIVAW_WEATHER_LOCATION_NAME", "Phoenix")
    monkeypatch.setattr(
        "olivaw.sources.weather.OpenMeteoProvider.fetch_forecast",
        lambda self, *, latitude, longitude, units: {
            "current": {
                "time": "2026-06-17T08:00",
                "temperature_2m": 72,
                "weather_code": 0,
                "wind_speed_10m": 6,
            },
            "current_units": {
                "temperature_2m": "°F",
                "wind_speed_10m": "mph",
            },
            "daily": {
                "time": ["2026-06-17"],
                "temperature_2m_max": [86],
                "temperature_2m_min": [68],
                "precipitation_probability_max": [10],
                "weather_code": [0],
            },
            "daily_units": {
                "temperature_2m_max": "°F",
                "temperature_2m_min": "°F",
            },
        },
    )

    response = client.get("/sources")

    assert response.status_code == 200
    assert "Weather provider" in response.text
    assert "Open-Meteo" in response.text
    assert "Weather location" in response.text
    assert "Phoenix" in response.text
    assert "Weather last fetch" in response.text
    assert "Weather forecast date" in response.text


def test_sources_route_renders_investigations_events_and_metadata(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (prime_dir / "investigation_index.json").write_text(
        """
[
  {
    "id": "inv-20260608",
    "title": "June 8 WAN samples",
    "status": "available",
    "path": "viz/investigation.json"
  }
]
""",
        encoding="utf-8",
    )
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "status": "Attention",
  "summary": "The network had 1 sustained slowdown period.",
  "events": [
    {
      "summary": "1 sustained slowdown period was found.",
      "confidence": "0.82",
      "confidence_reason": "Matched sustained slowdown threshold.",
      "supporting_facts": [
        {"summary": "WAN p95 exceeded threshold.", "source": "prime_observer"}
      ],
      "recommendation_trace": {
        "recommendation": "Check provider status if symptoms matched.",
        "interpretation": "Core Signal classified the event."
      },
      "interpretation_source": "core_signal",
      "attribution_source": "prime_observer_incident"
    }
  ]
}
""",
        encoding="utf-8",
    )

    response = client.get("/sources")

    assert response.status_code == 200
    assert "Source path:" in response.text
    assert "Investigation index" in response.text
    assert "Investigation index loaded: 1 investigations." in response.text
    assert "Investigation index path" in response.text
    assert "Catalog entries" in response.text
    assert "Investigation entries: 1" in response.text
    assert "June 8 WAN samples" in response.text
    assert "Interpreted events" in response.text
    assert "Core Signal events loaded: 1." in response.text
    assert "Event objects found" in response.text
    assert "Interpreted events rendered" in response.text
    assert "1 sustained slowdown period was found." in response.text
    assert "Confidence" in response.text
    assert "0.82" in response.text
    assert "Why" in response.text
    assert "Matched sustained slowdown threshold." in response.text
    assert "Supporting facts" in response.text


def test_sources_route_renders_precise_empty_diagnostics(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    (prime_dir / "investigation_index.json").write_text("[]\n", encoding="utf-8")
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "status": "Healthy",
  "summary": "Stable."
}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))

    response = client.get("/sources")

    assert response.status_code == 200
    assert "Investigation index loaded but contains no catalog entries." in response.text
    assert "Core Signal reports loaded, but no event objects were emitted." in (
        response.text
    )
    assert "Event objects found" in response.text


def test_briefing_route_replaces_ambiguous_empty_states(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "No Core Signal report file found at configured path." in response.text
    assert "Investigation index file was not found at configured path." in response.text
    assert "No interpreted Core Signal events are available." not in response.text
    assert "No Prime Observer investigation index is available." not in response.text


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
    assert "Today&apos;s Assessment" in response.text
    assert "Intelligence briefing" in response.text
    assert "Generated" in response.text
    assert re.search(r"Generated (just now|\d+ minutes? ago|today at)", response.text)
    assert "Refresh briefing" in response.text
    assert "Current Status" in response.text
    assert "Healthy" in response.text
    assert "Sources do not report a condition needing attention." in response.text
    assert response.text.count('class="disclosure-card"') == 1
    assert '<details class="disclosure-card" id="evidence-package">' in response.text
    assert 'href="#evidence-package"' in response.text
    assert "Open Evidence Package" in response.text
    assert "<summary>What Matters" not in response.text
    assert "<summary>What We Know" not in response.text
    assert "<summary>What We Think" not in response.text
    assert "<summary>Why We Believe This" not in response.text
    assert "<summary>What Remains Uncertain" not in response.text
    assert "Recommended Action" in response.text
    assert "Health Review" in response.text
    assert "Weather:" not in response.text
    assert "Health review not generated yet." in response.text
    assert "Status: not_generated" in response.text
    assert "Refresh Health Review" in response.text
    assert "Evidence Package" in response.text
    assert "Facts" in response.text
    assert "Interpretation" in response.text
    assert "Uncertainty" in response.text
    assert "Sources" in response.text
    assert "Technical Details" in response.text
    assert "<summary>Source attribution and current facts" not in response.text
    assert "<summary>Core Signal events" not in response.text
    assert "Category:" not in response.text
    assert "Raw briefing" in response.text
    assert "manual, files" in response.text
    assert "Example item from manual source" in response.text
    assert "File found: status/system.txt" in response.text


def test_briefing_route_renders_compact_weather_context(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
        "OLIVAW_WEATHER_ENABLED",
        "OLIVAW_WEATHER_LATITUDE",
        "OLIVAW_WEATHER_LONGITUDE",
        "OLIVAW_WEATHER_LOCATION_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    monkeypatch.setenv("OLIVAW_WEATHER_ENABLED", "true")
    monkeypatch.setenv("OLIVAW_WEATHER_LATITUDE", "33.4484")
    monkeypatch.setenv("OLIVAW_WEATHER_LONGITUDE", "-112.074")
    monkeypatch.setenv("OLIVAW_WEATHER_LOCATION_NAME", "Phoenix")
    monkeypatch.setattr(
        "olivaw.sources.weather.OpenMeteoProvider.fetch_forecast",
        lambda self, *, latitude, longitude, units: {
            "current": {
                "time": "2026-06-17T08:00",
                "temperature_2m": 72,
                "weather_code": 0,
                "wind_speed_10m": 6,
            },
            "current_units": {
                "temperature_2m": "°F",
                "wind_speed_10m": "mph",
            },
            "daily": {
                "time": ["2026-06-17"],
                "temperature_2m_max": [86],
                "temperature_2m_min": [68],
                "precipitation_probability_max": [10],
                "weather_code": [0],
            },
            "daily_units": {
                "temperature_2m_max": "°F",
                "temperature_2m_min": "°F",
            },
        },
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert (
        "Weather: Currently 72°F and clear. High 86°F, low 68°F. "
        "Rain chance 10%."
    ) in response.text
    assert "This briefing is source-backed using: manual, weather." in response.text
    assert "Weather does not provide recommendations." in response.text
    assert response.text.count('class="disclosure-card"') == 1


def test_home_route_renders_weather_card_when_available(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
        "OLIVAW_WEATHER_ENABLED",
        "OLIVAW_WEATHER_LATITUDE",
        "OLIVAW_WEATHER_LONGITUDE",
        "OLIVAW_WEATHER_LOCATION_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    monkeypatch.setenv("OLIVAW_WEATHER_ENABLED", "true")
    monkeypatch.setenv("OLIVAW_WEATHER_LATITUDE", "33.4484")
    monkeypatch.setenv("OLIVAW_WEATHER_LONGITUDE", "-112.074")
    monkeypatch.setenv("OLIVAW_WEATHER_LOCATION_NAME", "Phoenix")
    monkeypatch.setattr(
        "olivaw.sources.weather.OpenMeteoProvider.fetch_forecast",
        lambda self, *, latitude, longitude, units: {
            "current": {
                "time": "2026-06-17T08:00",
                "temperature_2m": 72,
                "weather_code": 0,
                "wind_speed_10m": 6,
            },
            "current_units": {
                "temperature_2m": "°F",
                "wind_speed_10m": "mph",
            },
            "daily": {
                "time": ["2026-06-17"],
                "temperature_2m_max": [86],
                "temperature_2m_min": [68],
                "precipitation_probability_max": [10],
                "weather_code": [0],
            },
            "daily_units": {
                "temperature_2m_max": "°F",
                "temperature_2m_min": "°F",
            },
        },
    )

    response = client.get("/")

    assert response.status_code == 200
    assert "<h3>Weather</h3>" in response.text
    assert "Currently 72°F and clear. High 86°F, low 68°F. Rain chance 10%." in (
        response.text
    )
    assert "Weather updated" in response.text
    assert "Weather</dt>" in response.text


def test_briefing_route_omits_weather_when_unavailable(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
        "OLIVAW_WEATHER_ENABLED",
        "OLIVAW_WEATHER_LATITUDE",
        "OLIVAW_WEATHER_LONGITUDE",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    monkeypatch.setenv("OLIVAW_WEATHER_ENABLED", "true")

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Weather:" not in response.text
    assert "Weather source requires latitude and longitude." in response.text


def test_home_route_omits_weather_card_when_unavailable(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
        "OLIVAW_WEATHER_ENABLED",
        "OLIVAW_WEATHER_LATITUDE",
        "OLIVAW_WEATHER_LONGITUDE",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    monkeypatch.setenv("OLIVAW_WEATHER_ENABLED", "true")

    response = client.get("/")

    assert response.status_code == 200
    assert "<h3>Weather</h3>" not in response.text
    assert "Weather source requires latitude and longitude." in response.text


def test_briefing_route_renders_generated_health_review(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "date": "2026-06-12",
  "status": "Watch",
  "summary": "Sustained slowdown was detected.",
  "confidence": "medium",
  "confidence_reason": "WAN degraded while LAN remained healthy.",
  "uncertainties": [
    "Unable to distinguish ISP congestion from transient routing issues."
  ],
  "attribution_assessment": {
    "candidate": "upstream",
    "confidence": "medium",
    "reason": "WAN degraded while LAN remained healthy."
  },
  "evidence_strength": {
    "rating": "moderate",
    "reason": "Multiple sustained WAN periods were observed."
  }
}
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "olivaw.web.generate_health_review",
        lambda dashboard, *, config: HealthReviewResult(
            text=(
                "Prime Observer evidence and Core Signal interpretation describe "
                "a sustained slowdown. Core Signal reported medium confidence "
                "and noted routing uncertainty. The review is explanatory only."
            ),
            status="available",
            provider="fake-local",
            model="fake-model",
            latency_ms=123,
        ),
    )

    refresh = client.post(
        "/health-review/refresh",
        headers={"referer": "http://testserver/briefing"},
    )
    response = client.get("/briefing")

    assert refresh.status_code == 200
    assert response.status_code == 200
    executive_section = response.text.split('<section class="details-stack">', 1)[0]
    assert "Health Review" in executive_section
    assert "Prime Observer evidence and Core Signal interpretation" in executive_section
    assert "fake-local / fake-model" in executive_section
    assert "123 ms" in executive_section
    assert "Evidence Package" in executive_section
    assert response.text.count('class="disclosure-card"') == 1


def test_health_review_refresh_caches_rejected_result(monkeypatch):
    monkeypatch.setattr(
        "olivaw.web.generate_health_review",
        lambda dashboard, *, config: HealthReviewResult(
            text="Health review unavailable: guardrail rejected.",
            status="guardrail_rejected",
            reason="unsupported recommendation",
            provider="fake-local",
            model="fake-model",
            guardrail_rejected=True,
        ),
    )

    refresh = client.post(
        "/health-review/refresh",
        headers={"referer": "http://testserver/"},
    )
    response = client.get("/")

    assert refresh.status_code == 200
    assert response.status_code == 200
    assert "Health review unavailable: guardrail rejected." in response.text
    assert "unsupported recommendation" in response.text


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


def test_briefing_dashboard_extracts_core_signal_events_and_safe_references():
    dashboard = _briefing_dashboard(
        """# Source Briefing

## Core Signal
- Core Signal Morning Brief - 2026-06-08 (2026-06-08) [Attention]: Slowdown.
  - Event: 1 sustained slowdown period was found.
    - Event ID: core-signal-sustained_slowdown-abc123
    - Event kind: sustained_slowdown
    - Severity/status: Attention / attention
    - Affected window: 2026-06-08T11:11:30+00:00 to 2026-06-08T11:12:09+00:00
    - Confidence: High
    - Issue location: Likely upstream/ISP issue
    - Recommended action: Check provider status if symptoms matched.
    - Attribution source: Prime Observer incident attribution
    - View investigation: viz/investigate.html?start=1&end=2

## Attribution
This briefing is source-backed using: core_signal.
""",
        "2026-06-08T18:00:00+00:00",
        ("core_signal",),
    )

    event = dashboard["core_signal_events"][0]
    assert event["summary"] == "1 sustained slowdown period was found."
    assert event["severity_status"] == "Attention / attention"
    assert event["affected_window"] == (
        "2026-06-08T11:11:30+00:00 to 2026-06-08T11:12:09+00:00"
    )
    assert event["confidence"] == "High"
    assert event["recommended_action"] == "Check provider status if symptoms matched."
    assert event["investigation_reference"] == "viz/investigate.html?start=1&end=2"
    assert "investigation_href" not in event


def test_briefing_dashboard_extracts_core_signal_explanation_metadata():
    dashboard = _briefing_dashboard(
        """# Source Briefing

## Core Signal
- Core Signal Morning Brief - 2026-06-08 (2026-06-08) [Attention]: Slowdown.
  - Event: 1 sustained slowdown period was found.
    - Interpretation: Core Signal
    - Presentation: Olivaw
    - Confidence: High
    - Why it matters: Sustained slowdown was detected.
    - Confidence rationale: Matched a sustained slowdown policy threshold.
    - Supporting facts: 1
      - Fact: WAN p95 exceeded the sustained threshold.
        - Source: prime_observer
        - Reference: http://127.0.0.1:8000/investigate.html?start=1&end=2
    - Recommended action: Check provider status if symptoms matched.
    - Recommendation trace:
      - Recommendation: Check provider status if symptoms matched.
      - Supporting facts: WAN p95 exceeded the sustained threshold.
      - Interpretation: Core Signal classified this as sustained slowdown.
    - Evidence: Prime Observer incident attribution
    - Related events:
      - Related event: core-signal-dns-watch-def456 - same_window

## Attribution
This briefing is source-backed using: core_signal.
""",
        "2026-06-08T18:00:00+00:00",
        ("core_signal",),
    )

    event = dashboard["core_signal_events"][0]
    assert event["confidence"] == "High"
    assert event["confidence_reason"] == "Matched a sustained slowdown policy threshold."
    assert event["supporting_fact_count"] == "1"
    assert event["supporting_facts"] == [
        {
            "summary": "WAN p95 exceeded the sustained threshold.",
            "source": "prime_observer",
            "reference": "http://127.0.0.1:8000/investigate.html?start=1&end=2",
        }
    ]
    assert event["recommendation_trace"] == [
        {
            "stage": "Recommendation",
            "detail": "Check provider status if symptoms matched.",
        },
        {
            "stage": "Supporting facts",
            "detail": "WAN p95 exceeded the sustained threshold.",
        },
        {
            "stage": "Interpretation",
            "detail": "Core Signal classified this as sustained slowdown.",
        },
    ]
    assert event["evidence"] == "Prime Observer incident attribution"
    assert event["interpretation"] == "Core Signal"
    assert event["presentation"] == "Olivaw"
    assert event["related_events"] == [
        "core-signal-dns-watch-def456 - same_window"
    ]


def test_briefing_route_renders_compact_wave_metadata(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (prime_dir / "investigation_index.json").write_text(
        """
[
  {
    "title": "June 8 WAN samples",
    "created_at": "2026-06-08T18:00:00+00:00",
    "event_count": 3,
    "status": "available",
    "path": "viz/investigation.json"
  }
]
""",
        encoding="utf-8",
    )
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "date": "2026-06-08",
  "status": "Attention",
  "summary": "The network had 1 sustained slowdown period.",
  "events": [
    {
      "summary": "1 sustained slowdown period was found.",
      "confidence": "0.82",
      "confidence_reason": "Matched sustained slowdown threshold.",
      "supporting_facts": [
        {"summary": "WAN p95 exceeded threshold.", "source": "prime_observer"}
      ],
      "recommendation_trace": {
        "recommendation": "Check provider status if symptoms matched.",
        "interpretation": "Core Signal classified the event."
      },
      "related_events": [
        {"id": "core-signal-dns-watch-def456", "relationship": "same_window"}
      ],
      "interpretation_source": "core_signal",
      "attribution_source": "prime_observer_incident"
    }
  ]
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Evidence" in response.text
    assert "Prime Observer" in response.text
    assert "Interpretation" in response.text
    assert "Core Signal" in response.text
    assert "Presentation" in response.text
    assert "Olivaw" in response.text
    assert "Confidence" in response.text
    assert "0.82" in response.text
    assert "Matched sustained slowdown threshold." in response.text
    assert "Supporting facts" in response.text
    assert "WAN p95 exceeded threshold." in response.text
    assert "Recommendation trace" in response.text
    assert "Related events" in response.text
    assert "Core Signal finding" in response.text
    assert "<summary>Why We Believe This" not in response.text
    assert "June 8 WAN samples" in response.text


def test_briefing_route_renders_wave_3b_uncertainty_attribution_and_strength(
    monkeypatch,
    tmp_path,
):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (prime_dir / "network_attribution.json").write_text(
        """
{
  "generated_at": "2026-06-12T06:05:00+00:00",
  "current_attribution": {
    "label": "No active network issue detected",
    "status": "no_issue_detected",
    "confidence": "high"
  }
}
""",
        encoding="utf-8",
    )
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "date": "2026-06-12",
  "status": "Attention",
  "summary": "Sustained slowdown was detected.",
  "confidence": "medium",
  "confidence_reason": "WAN degraded while LAN remained healthy.",
  "supporting_facts": [
    {"summary": "WAN degradation exceeded the sustained threshold.", "source": "prime_observer"},
    {"summary": "LAN remained below local degradation thresholds.", "source": "prime_observer"}
  ],
  "uncertainties": [
    "Unable to distinguish ISP congestion from transient routing issues."
  ],
  "attribution_assessment": {
    "candidate": "upstream",
    "confidence": "medium",
    "reason": "WAN degraded while LAN remained healthy."
  },
  "evidence_strength": {
    "rating": "moderate",
    "reason": "Multiple sustained WAN periods were observed."
  },
  "recommended_action": "Check provider status if symptoms matched."
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Facts" in response.text
    assert "WAN degradation exceeded the sustained threshold." in response.text
    assert "LAN remained below local degradation thresholds." in response.text
    assert "Interpretation" in response.text
    assert "Attribution assessment" in response.text
    assert ">upstream<" in response.text
    assert "Evidence strength" in response.text
    assert ">moderate<" in response.text
    assert "Multiple sustained WAN periods were observed." in response.text
    assert "Uncertainty" in response.text
    assert "<summary>What We Know" not in response.text
    assert "<summary>What We Think" not in response.text
    assert "<summary>What Remains Uncertain" not in response.text
    assert (
        "Unable to distinguish ISP congestion from transient routing issues."
        in response.text
    )
    assert "Confidence" in response.text
    assert ">medium<" in response.text


def test_briefing_route_handles_mixed_old_and_new_core_signal_events(
    monkeypatch,
    tmp_path,
):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "status": "Watch",
  "summary": "Mixed event formats.",
  "events": [
    {
      "summary": "Older v0.6.0 event.",
      "confidence": "low",
      "confidence_reason": "Older confidence reason."
    },
    {
      "summary": "New v0.7.0 event.",
      "confidence": "medium",
      "uncertainties": ["Routing cause remains uncertain."],
      "attribution_assessment": {
        "candidate": "upstream",
        "confidence": "medium",
        "reason": "WAN degraded while LAN remained healthy."
      },
      "evidence_strength": {
        "rating": "moderate",
        "reason": "Multiple sustained WAN periods were observed."
      }
    }
  ]
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Older v0.6.0 event." in response.text
    assert "New v0.7.0 event." in response.text
    assert "Routing cause remains uncertain." in response.text
    assert "Attribution assessment" in response.text
    assert "Evidence strength" in response.text


def test_briefing_route_v06_events_do_not_infer_wave_3b_fields(
    monkeypatch,
    tmp_path,
):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "status": "Healthy",
  "summary": "Stable.",
  "events": [
    {
      "summary": "Older event shape.",
      "confidence": "low",
      "confidence_reason": "Available evidence does not clearly distinguish local Wi-Fi/router from upstream ISP/path."
    }
  ]
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Older event shape." in response.text
    assert "No explicit uncertainty information was provided." in response.text
    executive_section = response.text.split('<section class="details-stack">', 1)[0]
    assert "Available evidence does not clearly distinguish" in response.text
    assert "What Remains Uncertain" not in executive_section
    assert "Attribution assessment" not in executive_section


def test_briefing_route_renders_executive_summary_from_source_data(
    monkeypatch,
    tmp_path,
):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (prime_dir / "network_attribution.json").write_text(
        """
{
  "generated_at": "2026-06-08T18:00:00+00:00",
  "current_attribution": {
    "label": "No active network issue detected",
    "status": "no_issue_detected",
    "confidence": "high"
  }
}
""",
        encoding="utf-8",
    )
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "date": "2026-06-08",
  "status": "Healthy",
  "summary": "Current LAN and WAN state appears stable.",
  "recommended_action": "No action."
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Today&apos;s Assessment" in response.text
    assert "Current LAN and WAN state appears stable." in response.text
    assert "Priority signals" in response.text
    assert "<summary>What Matters" not in response.text
    assert "Recommended Action" in response.text
    assert "No action." in response.text


def test_briefing_separates_current_health_from_historical_slowdown(
    monkeypatch,
    tmp_path,
):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (prime_dir / "network_attribution.json").write_text(
        """
{
  "generated_at": "2026-06-12T06:05:00+00:00",
  "current_attribution": {
    "label": "No active network issue detected",
    "status": "no_issue_detected",
    "confidence": "high"
  }
}
""",
        encoding="utf-8",
    )
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Morning Brief",
  "date": "2026-06-12",
  "status": "Healthy",
  "summary": "No active network issue is currently detected.",
  "recommended_action": "Check whether symptoms matched the affected time.",
  "events": [
    {
      "summary": "A sustained slowdown was observed earlier. User impact was possible.",
      "status": "attention",
      "severity": "attention",
      "window_start": "2026-06-11 05:58",
      "window_end": "2026-06-12 05:58",
      "confidence": "low",
      "confidence_reason": "Available evidence does not clearly distinguish local Wi-Fi/router from upstream ISP/path.",
      "supporting_facts": [
        {
          "summary": "The affected window overlapped the Prime Observer investigation.",
          "source": "prime_observer",
          "reference": "viz/investigate.html?start=1&end=2"
        }
      ],
      "recommendation_trace": {
        "recommendation": "Check whether symptoms matched the affected time.",
        "interpretation": "Core Signal supplied the historical finding."
      },
      "prime_observer_investigation": "viz/investigate.html?start=1&end=2"
    }
  ]
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Current Status" in response.text
    assert "Healthy now" in response.text
    assert "No active network issue is currently detected." in response.text
    assert "Historical Finding" in response.text
    assert (
        "A sustained slowdown was observed earlier. User impact was possible."
        in response.text
    )
    assert "Affected window: 2026-06-11 05:58 to 2026-06-12 05:58." in response.text
    assert "Uncertainty" in response.text
    assert "<summary>What Remains Uncertain" not in response.text
    assert "No explicit uncertainty information was provided." in response.text
    assert (
        "No immediate network change is recommended. If people noticed symptoms "
        "during the affected window, compare reports with the evidence package."
    ) in response.text
    executive_section = response.text.split('<section class="details-stack">', 1)[0]
    assert "no_issue_detected" not in executive_section
    assert "Action Needed" not in executive_section
    assert "Prime Observer investigation" not in executive_section
    assert ">viz/investigate.html" not in executive_section


def test_briefing_investigate_further_renders_navigation_actions(
    monkeypatch,
    tmp_path,
):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_PRIME_OBSERVER_BASE_URL",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_root = tmp_path / "prime"
    prime_dir = prime_root / "viz"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir(parents=True)
    core_dir.mkdir()
    files_dir.mkdir()
    (prime_dir / "investigate.html").write_text("investigation", encoding="utf-8")
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_BASE_URL", "http://127.0.0.1:8766")
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (prime_dir / "investigation_index.json").write_text(
        """
[
  {
    "title": "June 8 WAN samples",
    "status": "available",
    "path": "viz/investigation.json"
  }
]
""",
        encoding="utf-8",
    )
    (prime_dir / "investigation.json").write_text(
        """
{
  "navigation": {
    "first_event": {"id": "event-1", "label": "First sample"},
    "last_event": {"id": "event-2", "label": "Last sample"}
  }
}
""",
        encoding="utf-8",
    )
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "status": "Watch",
  "summary": "Affected telemetry exists.",
  "events": [
    {
      "summary": "Sustained slowdown.",
      "prime_observer_investigation": "viz/investigate.html?start=1&end=2",
      "evidence_window": {"label": "2026-06-08T11:00:00Z to 2026-06-08T11:15:00Z"},
      "attribution_source": "prime_observer_incident"
    }
  ]
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Open Evidence Package" in response.text
    assert "Review affected telemetry window" in response.text
    assert "Inspect first event" in response.text
    assert "Inspect last event" in response.text
    assert "Technical references" in response.text
    main_section = response.text.split("Technical references", 1)[0]
    assert "viz/investigation.json" not in main_section
    assert ">viz/investigate.html?start=1&amp;end=2<" not in main_section
    assert (
        'href="http://127.0.0.1:8766/investigate.html?start=1&amp;end=2"'
        in response.text
    )
    assert "viz/investigation.json" in response.text
    assert 'href="file://' not in response.text


def test_briefing_resolves_relative_viz_reference_against_base_url(
    monkeypatch,
    tmp_path,
):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_PRIME_OBSERVER_BASE_URL",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime" / "viz"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir(parents=True)
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_BASE_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "status": "Watch",
  "summary": "Affected telemetry exists.",
  "events": [
    {
      "summary": "Sustained slowdown.",
      "prime_observer_investigation": "viz/investigate.html?start=2026-06-08T11%3A00%3A00Z&end=2026-06-08T11%3A15%3A00Z",
      "attribution_source": "prime_observer_incident"
    }
  ]
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert (
        'href="http://127.0.0.1:8000/investigate.html?'
        "start=2026-06-08T11%3A00%3A00Z&amp;"
        'end=2026-06-08T11%3A15%3A00Z"'
    ) in response.text


def test_briefing_investigate_further_disables_local_investigation_without_base_url(
    monkeypatch,
    tmp_path,
):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_PRIME_OBSERVER_BASE_URL",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime" / "viz"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir(parents=True)
    core_dir.mkdir()
    files_dir.mkdir()
    (prime_dir / "investigate.html").write_text("investigation", encoding="utf-8")
    config_path = tmp_path / "olivaw.toml"
    config_path.write_text(
        """
[sources.prime_observer]
enabled = true
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OLIVAW_CONFIG", str(config_path))
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "status": "Watch",
  "summary": "Affected telemetry exists.",
  "events": [
    {
      "summary": "Sustained slowdown.",
      "prime_observer_investigation": "viz/investigate.html?start=1&end=2",
      "attribution_source": "prime_observer_incident"
    }
  ]
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Open Evidence Package" in response.text
    assert (
        "Start the Prime Observer local server to open telemetry evidence."
        in response.text
    )
    assert '<a href="file://' not in response.text
    assert (
        '<a href="viz/investigate.html?start=1&amp;end=2">'
        not in response.text
    )
    main_section = response.text.split("Technical references", 1)[0]
    assert "viz/investigate.html?start=1&amp;end=2" not in main_section
    assert "viz/investigate.html?start=1&amp;end=2" in response.text


def test_sources_route_renders_prime_observer_base_url_diagnostics(
    monkeypatch,
    tmp_path,
):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_PRIME_OBSERVER_BASE_URL",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime" / "viz"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir(parents=True)
    core_dir.mkdir()
    files_dir.mkdir()
    (prime_dir / "investigation_index.json").write_text("[]", encoding="utf-8")
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))

    response = client.get("/sources")

    assert response.status_code == 200
    assert "Prime Observer base URL" in response.text
    assert "http://127.0.0.1:1" in response.text
    assert "Prime Observer investigate URL" in response.text
    assert "http://127.0.0.1:1/investigate.html" in response.text
    assert "Prime Observer investigate HTTP" in response.text
    assert "Investigation links enabled" in response.text
    assert ">yes<" in response.text


def test_briefing_investigate_further_empty_state(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "status": "Healthy",
  "summary": "No investigation metadata."
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert 'href="#evidence-package"' in response.text
    assert "Open Evidence Package" in response.text
    assert "No supporting evidence links are available for this briefing." in response.text


def test_briefing_route_does_not_invent_recommendation_or_confidence(
    monkeypatch,
    tmp_path,
):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "date": "2026-06-08",
  "status": "Healthy",
  "summary": "Stable."
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "No specific recommendation is available from Core Signal." in response.text
    assert "Confidence" not in response.text
    assert "Restart" not in response.text
    assert "Call provider" not in response.text


def test_briefing_route_handles_sparse_live_source_shapes(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (prime_dir / "investigation_index.json").write_text(
        """
[
  {
    "title": "Sparse investigation",
    "status": "",
    "path": ""
  }
]
""",
        encoding="utf-8",
    )
    (prime_dir / "investigation.json").write_text(
        """
{
  "generated_at": "2026-06-08T18:00:00+00:00",
  "navigation": {
    "first_event": {},
    "previous_event": null,
    "next_event": {},
    "last_event": null
  },
  "event_neighborhoods": [
    {
      "event": {},
      "nearby_events": null
    }
  ]
}
""",
        encoding="utf-8",
    )
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "date": "2026-06-08",
  "status": "Watch",
  "summary": "Sparse report loaded.",
  "confidence": null,
  "confidence_reason": null,
  "supporting_facts": null,
  "recommendation_trace": null,
  "related_events": null,
  "events": [
    {
      "summary": "Sparse event loaded.",
      "confidence": null,
      "confidence_reason": null,
      "supporting_facts": null,
      "recommendation_trace": null,
      "related_events": null,
      "prime_observer_reference": {"path": ""}
    }
  ]
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Today&apos;s Assessment" in response.text
    assert "Sparse report loaded." in response.text
    assert "Sparse event loaded." in response.text
    assert "Internal Server Error" not in response.text


def test_briefing_dashboard_normalizes_optional_collections():
    dashboard = _normalize_briefing_dashboard(
        {
            "core_signal_explanation": {
                "supporting_facts": None,
                "recommendation_trace": None,
            },
            "core_signal_events": [
                {
                    "summary": "Older event shape.",
                    "supporting_facts": None,
                    "recommendation_trace": None,
                    "related_events": None,
                }
            ],
            "prime_nearby_events": [{"anchor": "event-1", "events": None}],
            "investigation_references": [
                {"label": "Empty target", "target": "", "href": ""},
                {"label": "Investigation", "target": "viz/investigation.json"},
            ],
            "what_matters": None,
            "worth_knowing": None,
            "network_status": None,
            "dns_activity": None,
            "dns_details": None,
            "prime_investigations": None,
            "prime_investigation_navigation": None,
            "core_signal_findings": None,
            "source_details": None,
        }
    )

    assert dashboard["core_signal_explanation"]["supporting_facts"] == []
    assert dashboard["core_signal_explanation"]["recommendation_trace"] == []
    assert dashboard["core_signal_events"][0]["supporting_facts"] == []
    assert dashboard["core_signal_events"][0]["recommendation_trace"] == []
    assert dashboard["core_signal_events"][0]["related_events"] == []
    assert dashboard["prime_nearby_events"][0]["events"] == []
    assert dashboard["investigation_references"] == [
        {
            "label": "Investigation",
            "target": "viz/investigation.json",
            "href": "",
            "kind": "",
        }
    ]
    assert dashboard["what_matters"] == []


def test_briefing_route_preserves_attribution_boundaries(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (prime_dir / "investigation_index.json").write_text(
        """
[
  {
    "title": "WAN fact packet",
    "status": "available",
    "path": "viz/investigation.json"
  }
]
""",
        encoding="utf-8",
    )
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "status": "Attention",
  "summary": "Sustained slowdown was interpreted.",
  "events": [
    {
      "summary": "Slowdown event.",
      "confidence": "medium",
      "confidence_reason": "Core Signal matched the policy.",
      "recommended_action": "Monitor the next report.",
      "interpretation_source": "core_signal",
      "attribution_source": "prime_observer_incident"
    }
  ]
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Evidence" in response.text
    assert "Prime Observer incident attribution" in response.text
    assert "Interpretation" in response.text
    assert "Core Signal" in response.text
    assert "Presentation" in response.text
    assert "Olivaw" in response.text


def test_briefing_dashboard_links_absolute_prime_observer_investigation_url():
    dashboard = _briefing_dashboard(
        """# Source Briefing

## Core Signal
- Core Signal Morning Brief - 2026-06-08 (2026-06-08) [Attention]: Slowdown.
  - Event: 1 sustained slowdown period was found.
    - View investigation: http://127.0.0.1:8000/investigate.html?start=1&end=2
""",
        "2026-06-08T18:00:00+00:00",
        ("core_signal",),
    )

    event = dashboard["core_signal_events"][0]
    assert event["investigation_href"] == (
        "http://127.0.0.1:8000/investigate.html?start=1&end=2"
    )


def test_briefing_dashboard_extracts_prime_observer_investigations():
    dashboard = _briefing_dashboard(
        """# Source Briefing

## Prime Observer
- Current-state observations only; interpretation belongs to Core Signal.
- Investigation index data: from Prime Observer.
  - Investigation: June 8 WAN samples
    - Created at: 2026-06-08T18:00:00+00:00
    - Event count: 2
    - Status: available
    - Path: http://127.0.0.1:8000/investigation.json
- Investigation metadata: from Prime Observer.
  - Navigation metadata: from Prime Observer.
    - First event: First sample (id event-1, target #event-1)
    - Previous event: Previous sample (id event-0)
    - Next event: Next sample (id event-2, target #event-2)
    - Last event: Last sample (id event-3)
  - Nearby-event facts: from Prime Observer.
    - Events in the same investigation window for Next sample:
      - First sample (id event-1)
      - Later sample (id event-3)
- Latest sample timestamp: 2026-06-08T19:00:00+00:00

## Attribution
This briefing is source-backed using: prime_observer.
""",
        "2026-06-08T18:00:00+00:00",
        ("prime_observer",),
    )

    assert dashboard["prime_investigations"] == [
        {
            "title": "June 8 WAN samples",
            "created_at": "2026-06-08T18:00:00+00:00",
            "event_count": "2",
            "status": "available",
            "path": "http://127.0.0.1:8000/investigation.json",
            "href": "http://127.0.0.1:8000/investigation.json",
        }
    ]
    assert dashboard["prime_investigation_navigation"] == [
        {"label": "First event", "target": "First sample (id event-1, target #event-1)"},
        {"label": "Previous event", "target": "Previous sample (id event-0)"},
        {"label": "Next event", "target": "Next sample (id event-2, target #event-2)"},
        {"label": "Last event", "target": "Last sample (id event-3)"},
    ]
    assert dashboard["prime_nearby_events"] == [
        {
            "anchor": "Next sample",
            "events": ["First sample (id event-1)", "Later sample (id event-3)"],
        }
    ]
    assert dashboard["investigation_actions"]["event_navigation"][:2] == [
        {
            "label": "Inspect first event",
            "detail": "First sample (id event-1, target #event-1)",
            "href": "",
            "target": "First sample (id event-1, target #event-1)",
            "attribution": "Prime Observer",
        },
        {
            "label": "Inspect last event",
            "detail": "Last sample (id event-3)",
            "href": "",
            "target": "Last sample (id event-3)",
            "attribution": "Prime Observer",
        },
    ]
    assert dashboard["investigation_actions"]["event_navigation"][2]["label"] == (
        "View nearby events"
    )


def test_briefing_route_renders_core_signal_event_without_broken_local_link(
    monkeypatch,
    tmp_path,
):
    for name in ("OLIVAW_CONFIG", "OLIVAW_FILES_DIR"):
        monkeypatch.delenv(name, raising=False)
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

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Interpreted events" in response.text
    assert "The network had 1 sustained slowdown period(s)." in response.text
    assert "Severity/status" in response.text
    assert "Attention / attention" in response.text
    assert "Affected window" in response.text
    assert "Confidence" not in response.text
    assert "View investigation: viz/investigate.html?start=1&amp;end=2" in response.text
    assert 'href="viz/investigate.html?start=1&amp;end=2"' not in response.text


def test_briefing_route_renders_prime_observer_evidence_metadata(
    monkeypatch,
    tmp_path,
):
    for name in ("OLIVAW_CONFIG", "OLIVAW_FILES_DIR", "OLIVAW_CORE_SIGNAL_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_ENABLED", "false")
    prime_dir = tmp_path / "prime"
    prime_dir.mkdir()
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    (prime_dir / "investigation_index.json").write_text(
        """
[
  {
    "id": "inv-20260608",
    "title": "June 8 WAN samples",
    "created_at": "2026-06-08T18:00:00+00:00",
    "event_count": 2,
    "status": "available",
    "path": "viz/investigation.json"
  }
]
""",
        encoding="utf-8",
    )
    (prime_dir / "investigation.json").write_text(
        """
{
  "generated_at": "2026-06-08T18:00:00+00:00",
  "navigation": {
    "first_event": {"id": "event-1", "label": "First sample", "anchor": "#event-1"},
    "last_event": {"id": "event-2", "label": "Last sample", "anchor": "#event-2"}
  },
  "event_neighborhoods": [
    {
      "event": {"id": "event-2", "label": "Last sample"},
      "nearby_events": [
        {"id": "event-1", "label": "First sample"}
      ]
    }
  ]
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Artifact references" in response.text
    assert "<summary>Prime Observer evidence artifacts" not in response.text
    assert "June 8 WAN samples" in response.text
    assert "Prime Observer evidence index" in response.text
    assert "Evidence navigation" in response.text
    assert "Prime Observer navigation metadata" in response.text
    assert "Nearby events" in response.text
    assert "Events in the same evidence window for Last sample" in response.text
    assert "Prime Observer factual discovery" in response.text
    assert "Core Signal" in response.text
    forbidden = ("correlated", "caused by", "likely related", "root cause")
    assert not any(term in response.text.lower() for term in forbidden)


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


def test_dashboard_status_does_not_recalculate_from_confidence_metadata():
    core_lines = [
        "- Core Signal Morning Brief - 2026-06-05 (2026-06-05) [Healthy]: Stable.",
        "- Confidence: Low",
        "- Confidence rationale: Sparse data.",
        "- Recommended action: No action.",
    ]

    status = _dashboard_status(core_lines, [])

    assert status["label"] == "Healthy"
    assert status["tone"] == "healthy"


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
