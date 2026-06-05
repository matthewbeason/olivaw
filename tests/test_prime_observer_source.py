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
    assert payload["items"][0]["latest_sample_timestamp"] == "2026-06-04T00:00:00-07:00"


def test_prime_observer_source_surfaces_top_dns_domains(tmp_path):
    _write_nextdns_summary(
        tmp_path / "nextdns_summary.json",
        top_blocked_domain="ads.example.test",
        top_resolved_domain="api.example.test",
    )

    payload = PrimeObserverSource(directory=tmp_path).fetch()

    item = payload["items"][0]
    assert item["report_type"] == "nextdns_summary"
    assert item["top_blocked_domain"] == "ads.example.test"
    assert item["top_resolved_domain"] == "api.example.test"
    assert "Top blocked domain: ads.example.test" in item["findings"]
    assert "Top resolved domain: api.example.test" in item["findings"]


def test_prime_observer_source_marks_redacted_dns_entities(tmp_path):
    _write_nextdns_summary(tmp_path / "nextdns_summary.json")

    payload = PrimeObserverSource(directory=tmp_path).fetch()

    item = payload["items"][0]
    assert item["top_blocked_domain"] == "unavailable"
    assert item["top_resolved_domain"] == "unavailable"
    assert item["top_domain_entity"] == (
        "entity_1 (redacted by Prime Observer privacy settings)"
    )
    assert "Top redacted entity" not in str(item)


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
    assert "Current-state observations only" in briefing.text
    assert "Current LAN/WAN state: No network issue detected" in briefing.text
    assert "This briefing is source-backed using: prime_observer." in briefing.text


def test_source_briefing_prime_observer_is_current_state_focused(tmp_path):
    _write_network_attribution(tmp_path / "network_attribution.json")
    _write_nextdns_summary(
        tmp_path / "nextdns_summary.json",
        top_blocked_domain="ads.example.test",
        top_resolved_domain="api.example.test",
    )
    (tmp_path / "latest.csv").write_text(
        "ts,phase_label,host,p95_ms,loss_pct\n"
        "2026-06-04T00:00:00-07:00,FIBER,1.1.1.1,25.0,0.0\n",
        encoding="utf-8",
    )
    registry = SourceRegistry()
    registry.register(PrimeObserverSource(directory=tmp_path))

    briefing = compose_source_briefing(registry=registry)
    prime_section = _section(briefing.text, "## Prime Observer")

    assert "Current-state observations only" in prime_section
    assert "Latest sample timestamp: 2026-06-04T00:00:00-07:00" in prime_section
    assert "Current LAN/WAN state: No network issue detected" in prime_section
    assert "DNS summary: available from Prime Observer." in prime_section
    assert "Top blocked domain: ads.example.test" in prime_section
    assert "Top resolved domain: api.example.test" in prime_section
    assert "Top redacted entity" not in prime_section


def test_source_briefing_redacted_dns_values_are_clear(tmp_path):
    _write_nextdns_summary(tmp_path / "nextdns_summary.json")
    registry = SourceRegistry()
    registry.register(PrimeObserverSource(directory=tmp_path))

    briefing = compose_source_briefing(registry=registry)
    prime_section = _section(briefing.text, "## Prime Observer")

    assert "Top blocked domain: unavailable" in prime_section
    assert "Top resolved domain: unavailable" in prime_section
    assert "entity_1 (redacted by Prime Observer privacy settings)" in prime_section
    assert "Top redacted entity" not in prime_section


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


def _write_nextdns_summary(
    path,
    *,
    top_blocked_domain: str | None = None,
    top_resolved_domain: str | None = None,
):
    blocked_field = (
        f'"top_blocked_domain": "{top_blocked_domain}",'
        if top_blocked_domain
        else ""
    )
    resolved_field = (
        f'"top_resolved_domain": "{top_resolved_domain}",'
        if top_resolved_domain
        else ""
    )
    path.write_text(
        f"""
{{
  "generated_at": "2026-06-04T21:05:00Z",
  "status": "ok",
  "summary": {{
    "total_queries": 1000,
    "blocked_queries": 20,
    "allowed_queries": 900,
    "block_rate_pct": 2.0,
    "encrypted_rate_pct": 80.0,
    {blocked_field}
    {resolved_field}
    "top_entities": [
      {{
        "label": "entity_1",
        "count": 250,
        "share_of_total": 0.25,
        "name_redacted": true
      }}
    ]
  }}
}}
""",
        encoding="utf-8",
    )


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    rest = text[start + len(heading):]
    next_heading = rest.find("\n## ")
    if next_heading == -1:
        return rest
    return rest[:next_heading]
