from __future__ import annotations

import pytest

from olivaw.bootstrap import init_data
from olivaw.capabilities.sources import (
    SourceInspectionCapability,
    format_sources_report,
)
from olivaw.config import (
    CoreSignalSourceConfig,
    FileSourceConfig,
    OlivawConfig,
    PrimeObserverSourceConfig,
)
from olivaw.sources import (
    FileSource,
    ManualSource,
    SourceRegistry,
    WeatherSource,
    create_default_registry,
)
from olivaw.sources.base import SourceHealth
from olivaw.sources.registry import inspect_sources


class BrokenSource:
    source_id = "broken"
    display_name = "Broken source"

    def health(self):
        return SourceHealth(
            source_id=self.source_id,
            display_name=self.display_name,
            status="ok",
            message="Broken source appears available.",
        )

    def fetch(self):
        raise RuntimeError("boom")


class FakeWeatherProvider:
    def __init__(self, payload: dict[str, object] | None = None, exc: Exception | None = None):
        self.payload = payload or _weather_payload()
        self.exc = exc

    def fetch_forecast(self, *, latitude: float, longitude: float, units: str):
        if self.exc:
            raise self.exc
        return self.payload


def _weather_payload() -> dict[str, object]:
    return {
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
    }


def test_manual_source_health_is_ok():
    source = ManualSource()

    status = source.health()

    assert status.source_id == "manual"
    assert status.display_name == "Manual example source"
    assert status.status == "ok"
    assert "available" in status.message


def test_manual_source_fetch_returns_deterministic_data():
    payload = ManualSource().fetch()

    assert payload == {
        "source": "manual",
        "status": "ok",
        "items": [
            {
                "title": "Example item",
                "summary": "Demonstrates source plumbing.",
            }
        ],
    }


def test_source_registry_registers_lists_gets_and_fetches_sources():
    registry = SourceRegistry()
    source = ManualSource()

    registry.register(source)

    assert registry.list_sources() == (source,)
    assert registry.get_source("manual") is source
    assert registry.get_source("missing") is None
    assert registry.health_all()[0].status == "ok"
    assert registry.fetch_all()[0]["source"] == "manual"


def test_source_registry_rejects_duplicate_source_ids():
    registry = SourceRegistry()
    registry.register(ManualSource())

    with pytest.raises(ValueError, match="manual"):
        registry.register(ManualSource())


def test_default_registry_contains_manual_and_file_sources(tmp_path):
    registry = create_default_registry(
        OlivawConfig(files=FileSourceConfig(directory=tmp_path))
    )

    assert registry.get_source("manual") is not None
    assert registry.get_source("files") is not None
    assert registry.get_source("prime_observer") is not None
    assert registry.get_source("core_signal") is not None
    assert registry.get_source("weather") is not None


def test_inspect_sources_returns_status_and_sample_data(tmp_path):
    init_data(tmp_path)
    report = inspect_sources(
        config=OlivawConfig(files=FileSourceConfig(directory=tmp_path))
    )

    assert report["sources"][0]["source_id"] == "manual"
    assert report["sources"][0]["status"] == "ok"
    assert report["sources"][1]["source_id"] == "files"
    assert report["sources"][1]["status"] == "ok"
    assert report["sources"][3]["source_id"] == "core_signal"
    assert report["sources"][4]["source_id"] == "weather"
    assert report["data"][0]["items"][0]["title"] == "Example item"
    assert report["data"][1]["count"] == 3
    assert "aggregate" in report
    assert report["aggregate"]["sources"][0]["source_id"] == "manual"


def test_weather_source_disabled_returns_clear_diagnostics():
    source = WeatherSource(enabled=False)

    health = source.health()
    payload = source.fetch()

    assert health.status == "unavailable"
    assert "disabled" in health.message
    assert payload["status"] == "unavailable"
    assert payload["diagnostics"]["enabled"] == "no"
    assert payload["diagnostics"]["provider_status"] == "not called"


def test_weather_source_missing_location_returns_clear_diagnostics():
    source = WeatherSource(enabled=True)

    health = source.health()
    payload = source.fetch()

    assert health.status == "unavailable"
    assert "latitude and longitude" in health.message
    assert payload["diagnostics"]["configured"] == "no"


def test_weather_source_successful_fetch_normalizes_daily_facts():
    source = WeatherSource(
        enabled=True,
        latitude=33.4484,
        longitude=-112.074,
        location_name="Phoenix",
        provider=FakeWeatherProvider(),
    )

    health = source.health()
    payload = source.fetch()

    assert health.status == "ok"
    assert payload["status"] == "ok"
    assert payload["count"] == 1
    item = payload["items"][0]
    assert item["title"] == "Phoenix"
    assert item["summary"] == "Currently 72°F and clear. High 86°F, low 68°F. Rain chance 10%."
    assert item["forecast_date"] == "2026-06-17"
    assert item["facts"] == [
        {"kind": "current_temperature", "summary": "Current temperature: 72°F"},
        {"kind": "condition", "summary": "Condition: clear"},
        {"kind": "high_temperature", "summary": "High temperature: 86°F"},
        {"kind": "low_temperature", "summary": "Low temperature: 68°F"},
        {"kind": "precipitation_chance", "summary": "Rain chance: 10%"},
        {"kind": "wind", "summary": "Wind: 6 mph"},
    ]
    assert payload["diagnostics"]["provider"] == "Open-Meteo"
    assert payload["diagnostics"]["provider_status"] == "ok"


def test_weather_source_api_failure_returns_degraded_payload():
    source = WeatherSource(
        enabled=True,
        latitude=33.4484,
        longitude=-112.074,
        provider=FakeWeatherProvider(exc=RuntimeError("network down")),
    )

    payload = source.fetch()

    assert payload["status"] == "error"
    assert payload["count"] == 0
    assert "Weather fetch failed: RuntimeError: network down" in payload["errors"]
    assert payload["diagnostics"]["provider_status"] == "error"


def test_sources_report_distinguishes_investigation_index_load_status(tmp_path):
    prime_dir = tmp_path / "prime"
    prime_dir.mkdir()
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
    config = OlivawConfig(
        files=FileSourceConfig(directory=tmp_path / "files"),
        prime_observer=PrimeObserverSourceConfig(directory=prime_dir),
    )

    report = SourceInspectionCapability().run(config=config)
    text = format_sources_report(report)

    assert "Root: " in text
    assert "Investigation index: Investigation index loaded: 1 investigations." in text
    assert "Investigation index status: loaded-with-N" in text
    assert "Investigation entries: 1" in text
    assert "Investigation: June 8 WAN samples (viz/investigation.json)" in text


def test_sources_report_shows_prime_observer_link_diagnostics(tmp_path):
    prime_dir = tmp_path / "prime" / "viz"
    prime_dir.mkdir(parents=True)
    (prime_dir / "investigate.html").write_text("investigation", encoding="utf-8")
    (prime_dir / "investigation_index.json").write_text("[]", encoding="utf-8")
    config = OlivawConfig(
        files=FileSourceConfig(directory=tmp_path / "files"),
        prime_observer=PrimeObserverSourceConfig(
            directory=prime_dir,
            base_url="http://127.0.0.1:1",
        ),
    )

    report = SourceInspectionCapability().run(config=config)
    text = format_sources_report(report)

    assert "Prime Observer base URL: http://127.0.0.1:1" in text
    assert (
        "Prime Observer investigate URL: http://127.0.0.1:1/investigate.html"
        in text
    )
    assert "Investigation links enabled: yes" in text
    assert "Prime Observer investigate HTTP: not reachable" in text


def test_sources_report_gives_guidance_when_prime_observer_base_url_missing(tmp_path):
    prime_dir = tmp_path / "prime" / "viz"
    prime_dir.mkdir(parents=True)
    (prime_dir / "investigation_index.json").write_text("[]", encoding="utf-8")
    config = OlivawConfig(
        files=FileSourceConfig(directory=tmp_path / "files"),
        prime_observer=PrimeObserverSourceConfig(directory=prime_dir),
    )

    report = SourceInspectionCapability().run(config=config)
    text = format_sources_report(report)

    assert "Prime Observer base URL:" not in text
    assert (
        "Prime Observer investigate HTTP: not checked; Prime Observer base URL "
        "is not configured."
    ) in text
    assert "Investigation links enabled: no" in text
    assert "Configure sources.prime_observer.base_url" in text
    assert "OLIVAW_PRIME_OBSERVER_BASE_URL" in text


def test_sources_report_surfaces_core_signal_event_metadata(tmp_path):
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "status": "Attention",
  "summary": "The network had 1 sustained slowdown period.",
  "events": [
    {
      "id": "core-signal-sustained_slowdown-abc123",
      "summary": "1 sustained slowdown period was found.",
      "confidence": "0.82",
      "confidence_reason": "Matched sustained slowdown threshold.",
      "supporting_facts": [
        {"summary": "WAN p95 exceeded threshold.", "source": "prime_observer"}
      ],
      "interpretation_source": "core_signal",
      "attribution_source": "prime_observer_incident"
    }
  ]
}
""",
        encoding="utf-8",
    )
    config = OlivawConfig(
        files=FileSourceConfig(directory=tmp_path / "files"),
        core_signal=CoreSignalSourceConfig(directory=core_dir),
    )

    report = SourceInspectionCapability().run(config=config)
    text = format_sources_report(report)

    assert "Interpreted events: Core Signal events loaded: 1." in text
    assert "Event objects found: 1" in text
    assert "Interpreted events: 1" in text
    assert "Event: 1 sustained slowdown period was found." in text
    assert "Confidence: 0.82" in text
    assert "Supporting facts: 1" in text
    assert "Why: Matched sustained slowdown threshold." in text


def test_source_aggregation_separates_prime_observer_and_core_signal(tmp_path):
    prime_dir = tmp_path / "prime"
    prime_dir.mkdir()
    (prime_dir / "network_attribution.json").write_text(
        """
{
  "generated_at": "2026-06-17T12:00:00Z",
  "current_status": "mixed",
  "current_label": "Mixed evidence",
  "current_confidence": "medium"
}
""",
        encoding="utf-8",
    )
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "status": "Attention",
  "summary": "A sustained slowdown was found.",
  "recommended_action": "Check provider status if symptoms matched.",
  "events": [
    {
      "id": "event-1",
      "summary": "WAN degradation persisted.",
      "confidence_reason": "WAN degraded while LAN stayed healthy.",
      "supporting_facts": [
        {"summary": "WAN p95 exceeded threshold.", "source": "prime_observer"}
      ],
      "recommended_action": "Monitor matching symptom reports.",
      "prime_observer_reference": {"path": "viz/investigate.html?event=1"}
    }
  ]
}
""",
        encoding="utf-8",
    )
    registry = create_default_registry(
        OlivawConfig(
            files=FileSourceConfig(directory=tmp_path / "files"),
            prime_observer=PrimeObserverSourceConfig(directory=prime_dir),
            core_signal=CoreSignalSourceConfig(directory=core_dir),
        )
    )

    aggregate = registry.aggregate()

    source_ids = {source.source_id for source in aggregate.sources}
    assert {"prime_observer", "core_signal"}.issubset(source_ids)
    assert any(fact["source_id"] == "prime_observer" for fact in aggregate.facts)
    assert any(
        item["source_id"] == "core_signal"
        for item in aggregate.interpretation_items
    )
    assert any(action["source_id"] == "core_signal" for action in aggregate.actions)
    assert any(
        reference["target"] == "viz/investigate.html?event=1"
        for reference in aggregate.references
    )
    assert not any(
        item["source_id"] == "prime_observer"
        for item in aggregate.interpretation_items
    )


def test_source_aggregation_tolerates_missing_prime_observer(tmp_path):
    registry = create_default_registry(
        OlivawConfig(
            files=FileSourceConfig(directory=tmp_path / "files"),
            prime_observer=PrimeObserverSourceConfig(directory=tmp_path / "missing"),
            core_signal=CoreSignalSourceConfig(directory=tmp_path / "missing-core"),
        )
    )

    aggregate = registry.aggregate()
    prime = next(source for source in aggregate.sources if source.source_id == "prime_observer")
    core = next(source for source in aggregate.sources if source.source_id == "core_signal")

    assert prime.status == "unavailable"
    assert core.status == "unavailable"
    assert aggregate.facts


def test_source_aggregation_tolerates_failed_source():
    registry = SourceRegistry()
    registry.register(ManualSource())
    registry.register(BrokenSource())

    aggregate = registry.aggregate()

    assert any(source.source_id == "manual" for source in aggregate.sources)
    broken = next(source for source in aggregate.sources if source.source_id == "broken")
    assert broken.status == "error"
    assert "Fetch failed" in broken.message
    assert any(fact["source_id"] == "manual" for fact in aggregate.facts)


def test_source_aggregation_includes_weather_facts():
    registry = SourceRegistry()
    registry.register(
        WeatherSource(
            enabled=True,
            latitude=33.4484,
            longitude=-112.074,
            location_name="Phoenix",
            provider=FakeWeatherProvider(),
        )
    )

    aggregate = registry.aggregate()
    weather = aggregate.sources[0]

    assert weather.source_id == "weather"
    assert weather.source_type == "external_context"
    assert weather.raw_available is True
    assert weather.summary_items[0]["summary"].startswith("Currently 72°F")
    assert any(fact["summary"] == "Rain chance: 10%" for fact in aggregate.facts)
    assert any(
        fact["summary"] == "Rain chance: 10%"
        for fact in aggregate.health_review_context["facts"]
    )


def test_source_aggregation_marks_weather_fetch_failure_as_error():
    registry = SourceRegistry()
    registry.register(
        WeatherSource(
            enabled=True,
            latitude=33.4484,
            longitude=-112.074,
            provider=FakeWeatherProvider(exc=RuntimeError("network down")),
        )
    )

    aggregate = registry.aggregate()

    assert aggregate.sources[0].status == "error"
    assert "Weather fetch failed" in aggregate.sources[0].message
    assert aggregate.sources[0].diagnostics["provider_status"] == "error"


def test_inspect_sources_tolerates_failed_source():
    registry = SourceRegistry()
    registry.register(ManualSource())
    registry.register(BrokenSource())

    report = inspect_sources(registry=registry)

    assert report["sources"][1]["status"] == "error"
    assert report["data"][1]["status"] == "error"
    assert "Fetch failed" in report["data"][1]["errors"][0]
    assert report["aggregate"]["sources"][1]["status"] == "error"


def test_sources_report_includes_normalized_diagnostics(tmp_path):
    init_data(tmp_path)
    report = inspect_sources(
        config=OlivawConfig(files=FileSourceConfig(directory=tmp_path))
    )
    text = format_sources_report(report)

    assert "Normalized Sources:" in text
    assert "Manual example source (manual, manual): ok" in text
    assert "Raw available: yes" in text


def test_sources_report_shows_weather_diagnostics():
    registry = SourceRegistry()
    registry.register(
        WeatherSource(
            enabled=True,
            latitude=33.4484,
            longitude=-112.074,
            location_name="Phoenix",
            provider=FakeWeatherProvider(),
        )
    )

    report = inspect_sources(registry=registry)
    text = format_sources_report(report)

    assert "Weather (weather, external_context): ok" in text
    assert "Weather configured: yes" in text
    assert "Weather provider: Open-Meteo" in text
    assert "Weather location: Phoenix" in text
    assert "Weather last fetch: ok" in text


def test_file_source_scans_supported_files(tmp_path):
    (tmp_path / "status").mkdir()
    (tmp_path / "status" / "system.txt").write_text(
        "first line\nsecond line\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.md").write_text("# Notes\nDetails\n", encoding="utf-8")
    (tmp_path / "data.json").write_text('{"ok": true}\n', encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"not inspected")

    payload = FileSource(root=tmp_path).fetch()

    assert payload["source"] == "files"
    assert payload["status"] == "ok"
    assert payload["count"] == 3
    paths = [item["path"] for item in payload["items"]]
    assert paths == ["data.json", "notes.md", "status/system.txt"]
    assert payload["items"][2]["title"] == "system.txt"
    assert payload["items"][2]["preview"] == "first line\nsecond line"


def test_file_source_ignores_hidden_files(tmp_path):
    (tmp_path / ".secret.txt").write_text("hidden", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "note.md").write_text("hidden", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("visible", encoding="utf-8")

    payload = FileSource(root=tmp_path).fetch()

    assert payload["count"] == 1
    assert payload["items"][0]["path"] == "visible.txt"


def test_file_source_ignores_large_files(tmp_path):
    (tmp_path / "small.txt").write_text("small", encoding="utf-8")
    (tmp_path / "large.txt").write_text("too large", encoding="utf-8")

    payload = FileSource(root=tmp_path, max_bytes=5).fetch()

    assert payload["count"] == 1
    assert payload["items"][0]["path"] == "small.txt"


def test_file_source_reports_missing_directory(tmp_path):
    source = FileSource(root=tmp_path / "missing")

    health = source.health()
    payload = source.fetch()

    assert health.status == "unavailable"
    assert payload["status"] == "unavailable"
    assert payload["count"] == 0
