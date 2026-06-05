from __future__ import annotations

from olivaw.briefing import compose_source_briefing
from olivaw.config import OlivawConfig, PrimeObserverSourceConfig
from olivaw.sources import PrimeObserverSource
from olivaw.sources.registry import SourceRegistry, create_default_registry


def test_prime_observer_source_reports_missing_directory(tmp_path):
    source = PrimeObserverSource(directory=tmp_path / "missing")

    health = source.health()
    payload = source.fetch()

    assert health.status == "unavailable"
    assert "Directory does not exist" in health.message
    assert payload["status"] == "unavailable"
    assert payload["items"] == []


def test_prime_observer_source_reports_empty_directory(tmp_path):
    source = PrimeObserverSource(directory=tmp_path)

    health = source.health()
    payload = source.fetch()

    assert health.status == "unavailable"
    assert "No Prime Observer report files found" in health.message
    assert payload["status"] == "unavailable"
    assert payload["count"] == 0


def test_prime_observer_source_loads_valid_network_report(tmp_path):
    _write_network_attribution(tmp_path / "network_attribution.json")

    source = PrimeObserverSource(directory=tmp_path)
    health = source.health()
    payload = source.fetch()

    assert health.status == "ok"
    assert payload["source"] == "prime_observer"
    assert payload["status"] == "ok"
    assert payload["count"] == 1
    item = payload["items"][0]
    assert item["title"] == "Prime Observer network attribution"
    assert item["status"] == "no_issue_detected"
    assert item["summary"] == "LAN and WAN both look stable."
    assert item["report_type"] == "network_attribution"
    assert "Confidence: high" in item["findings"]


def test_prime_observer_source_loads_latest_csv_summary(tmp_path):
    (tmp_path / "latest.csv").write_text(
        "ts,phase_label,host,p95_ms,loss_pct\n"
        "2026-06-04T00:00:00-07:00,FIBER,1.1.1.1,25.0,0.0\n",
        encoding="utf-8",
    )

    payload = PrimeObserverSource(directory=tmp_path).fetch()

    assert payload["status"] == "ok"
    assert payload["items"][0]["title"] == "Prime Observer latest samples"
    assert "p95 25.0 ms" in payload["items"][0]["summary"]


def test_prime_observer_source_handles_malformed_report(tmp_path):
    (tmp_path / "network_attribution.json").write_text("{bad json", encoding="utf-8")

    source = PrimeObserverSource(directory=tmp_path)
    health = source.health()
    payload = source.fetch()

    assert health.status == "ok"
    assert payload["status"] == "error"
    assert payload["count"] == 0
    assert "JSONDecodeError" in payload["errors"][0]


def test_default_registry_registers_prime_observer_source(tmp_path):
    config = OlivawConfig(
        prime_observer=PrimeObserverSourceConfig(directory=tmp_path)
    )

    registry = create_default_registry(config)

    assert registry.get_source("prime_observer") is not None


def test_source_briefing_includes_prime_observer_section(tmp_path):
    _write_network_attribution(tmp_path / "network_attribution.json")
    registry = SourceRegistry()
    registry.register(PrimeObserverSource(directory=tmp_path))

    briefing = compose_source_briefing(registry=registry)

    assert "## Prime Observer" in briefing.text
    assert "Prime Observer network attribution" in briefing.text
    assert "LAN and WAN both look stable." in briefing.text
    assert "This briefing is source-backed using: prime_observer." in briefing.text


def _write_network_attribution(path):
    path.write_text(
        """
{
  "attribution_status": "no_network_issue_detected",
  "attribution_label": "No network issue detected",
  "attribution_confidence": "High",
  "generated_at": "2026-06-04T21:00:00Z",
  "attribution_evidence": {
    "summary": "LAN and WAN both look stable."
  },
  "current_attribution": {
    "status": "no_issue_detected",
    "label": "No network issue detected",
    "confidence": "high",
    "evidence": ["LAN and WAN both look stable."]
  }
}
""",
        encoding="utf-8",
    )

