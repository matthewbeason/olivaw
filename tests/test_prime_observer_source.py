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


def test_prime_observer_source_preserves_investigation_window_without_evidence(
    tmp_path,
):
    (tmp_path / "investigation.json").write_text(
        """
{
  "schema_version": 1,
  "generated_at": "2026-06-08T18:00:00+00:00",
  "event_window": {
    "start": "2026-06-08T11:11:30+00:00",
    "end": "2026-06-08T11:12:09+00:00",
    "context_start": "2026-06-08T10:41:30+00:00",
    "context_end": "2026-06-08T11:42:09+00:00"
  },
  "periods": {
    "during": {"wan": {"sample_count": 20}}
  },
  "timeline_samples": [
    {"ts": "2026-06-08T11:11:30+00:00", "p95_ms": 200}
  ]
}
""",
        encoding="utf-8",
    )

    payload = PrimeObserverSource(directory=tmp_path).fetch()

    item = payload["items"][0]
    assert item["report_type"] == "investigation"
    assert item["title"] == "Prime Observer investigation export"
    assert item["investigation_start"] == "2026-06-08T11:11:30+00:00"
    assert item["investigation_end"] == "2026-06-08T11:12:09+00:00"
    assert "timeline_samples" not in item
    assert "periods" not in item
    assert item["investigation_navigation"] == {}
    assert item["event_neighborhoods"] == []


def test_prime_observer_source_loads_optional_investigation_index(tmp_path):
    (tmp_path / "investigation_index.json").write_text(
        """
[
  {
    "id": "inv-20260608",
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

    payload = PrimeObserverSource(directory=tmp_path).fetch()

    item = payload["items"][0]
    assert item["report_type"] == "investigation_index"
    assert item["title"] == "Prime Observer investigation index"
    assert item["investigation_catalog"] == [
        {
            "id": "inv-20260608",
            "title": "June 8 WAN samples",
            "created_at": "2026-06-08T18:00:00+00:00",
            "event_count": "3",
            "status": "available",
            "path": "viz/investigation.json",
        }
    ]
    assert payload["diagnostics"]["investigation_index"] == (
        "Investigation index loaded: 1 investigations."
    )
    assert payload["diagnostics"]["investigation_index_status"] == "loaded-with-N"
    assert payload["diagnostics"]["catalog_entry_count"] == 1


def test_prime_observer_source_reports_missing_investigation_index(tmp_path):
    source = PrimeObserverSource(directory=tmp_path)

    payload = source.fetch()

    assert payload["status"] == "unavailable"
    assert payload["diagnostics"]["investigation_index"] == (
        "Investigation index file was not found at configured path."
    )
    assert payload["diagnostics"]["investigation_index_status"] == "missing"
    assert payload["diagnostics"]["catalog_entry_count"] == 0


def test_prime_observer_source_reports_empty_investigation_index(tmp_path):
    (tmp_path / "investigation_index.json").write_text("[]\n", encoding="utf-8")

    payload = PrimeObserverSource(directory=tmp_path).fetch()

    assert payload["status"] == "ok"
    assert payload["diagnostics"]["investigation_index"] == (
        "Investigation index loaded but contains no catalog entries."
    )
    assert payload["diagnostics"]["investigation_index_status"] == "loaded-empty"
    assert payload["diagnostics"]["catalog_entry_count"] == 0


def test_prime_observer_source_preserves_investigation_navigation_and_neighbors(
    tmp_path,
):
    (tmp_path / "investigation.json").write_text(
        """
{
  "generated_at": "2026-06-08T18:00:00+00:00",
  "event_window": {
    "start": "2026-06-08T11:11:30+00:00",
    "end": "2026-06-08T11:12:09+00:00"
  },
  "events": [
    {"id": "event-1", "kind": "sample", "ts": "2026-06-08T11:11:30+00:00"}
  ],
  "navigation": {
    "first_event": {"id": "event-1", "label": "First sample", "anchor": "#event-1"},
    "previous_event": {"id": "event-0", "label": "Previous sample"},
    "next_event": {"id": "event-2", "label": "Next sample"},
    "last_event": {"id": "event-3", "label": "Last sample"}
  },
  "event_neighborhoods": {
    "event-2": {
      "nearby_events": [
        {"id": "event-1", "label": "Earlier sample"},
        {"id": "event-3", "label": "Later sample"}
      ]
    }
  }
}
""",
        encoding="utf-8",
    )

    payload = PrimeObserverSource(directory=tmp_path).fetch()

    item = payload["items"][0]
    assert item["investigation_navigation"]["first_event"]["id"] == "event-1"
    assert item["investigation_navigation"]["first_event"]["anchor"] == "#event-1"
    assert item["investigation_navigation"]["next_event"]["label"] == "Next sample"
    assert item["investigation_events"][0]["id"] == "event-1"
    neighborhoods = item["event_neighborhoods"]
    assert neighborhoods[0]["event"]["id"] == "event-2"
    assert neighborhoods[0]["nearby_events"][0]["label"] == "Earlier sample"


def test_prime_observer_source_surfaces_top_dns_domains(tmp_path):
    _write_nextdns_summary(
        tmp_path / "nextdns_summary.json",
        top_queried_domain="www.example.test",
        top_blocked_domain="ads.example.test",
        top_resolved_domain="api.example.test",
    )

    payload = PrimeObserverSource(directory=tmp_path).fetch()

    item = payload["items"][0]
    assert item["report_type"] == "nextdns_summary"
    assert item["top_queried_domain"] == "www.example.test"
    assert item["top_queried_domain_count"] == 300
    assert item["top_queried_domain_share"] == 0.3
    assert item["top_blocked_domain"] == "ads.example.test"
    assert item["top_blocked_domain_count"] == 10
    assert item["top_blocked_domain_share"] == 0.5
    assert item["top_resolved_domain"] == "api.example.test"
    assert item["top_resolved_domain_count"] == 250
    assert item["top_resolved_domain_share"] == 0.25
    assert item["top_blocked_category"] == "OISD"
    assert item["blocked_query_count"] == 20
    assert item["dns_block_rate"] == 0.02
    assert item["dns_encrypted_queries"] == 800
    assert item["dns_encrypted_rate"] == 0.8
    assert "Top queried domain: www.example.test" in item["findings"]
    assert "Top blocked domain: ads.example.test" in item["findings"]
    assert "Top resolved domain: api.example.test" in item["findings"]


def test_prime_observer_source_marks_redacted_dns_entities(tmp_path):
    _write_nextdns_summary(tmp_path / "nextdns_summary.json")

    payload = PrimeObserverSource(directory=tmp_path).fetch()

    item = payload["items"][0]
    assert item["top_blocked_domain"] == "unavailable"
    assert item["top_resolved_domain"] == "unavailable"
    assert item["top_queried_domain"] == (
        "entity_1 (redacted by Prime Observer privacy settings)"
    )
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
    assert "Current network state: no active issue detected" in briefing.text
    assert "This briefing is source-backed using: prime_observer." in briefing.text


def test_source_briefing_prime_observer_is_current_state_focused(tmp_path):
    _write_network_attribution(tmp_path / "network_attribution.json")
    _write_nextdns_summary(
        tmp_path / "nextdns_summary.json",
        top_queried_domain="www.example.test",
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
    assert "Current network state: no active issue detected" in prime_section
    assert "DNS summary: available from Prime Observer." in prime_section
    assert (
        "Top queried domain: www.example.test (count 300, share 0.3)"
        in prime_section
    )
    assert "Top blocked domain: ads.example.test (count 10, share 0.5)" in prime_section
    assert (
        "Top resolved domain: api.example.test (count 250, share 0.25)"
        in prime_section
    )
    assert "Blocked queries: 20" in prime_section
    assert "Encrypted queries: 800" in prime_section
    assert "Block rate: 2.0%" in prime_section
    assert "Raw block rate: 0.02" in prime_section
    assert "Encrypted query rate: 80.0%" in prime_section
    assert "Raw encrypted query rate: 0.8" in prime_section
    assert "Top blocked category/reason: OISD" in prime_section
    assert "Top redacted entity" not in prime_section


def test_source_briefing_renders_prime_observer_investigation_metadata(tmp_path):
    (tmp_path / "investigation_index.json").write_text(
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
    (tmp_path / "investigation.json").write_text(
        """
{
  "generated_at": "2026-06-08T18:00:00+00:00",
  "event_window": {
    "start": "2026-06-08T11:11:30+00:00",
    "end": "2026-06-08T11:12:09+00:00"
  },
  "navigation": {
    "first_event": {"id": "event-1", "label": "First sample", "anchor": "#event-1"},
    "next_event": {"id": "event-2", "label": "Next sample", "anchor": "#event-2"}
  },
  "event_neighborhoods": [
    {
      "event": {"id": "event-2", "label": "Next sample"},
      "nearby_events": [
        {"id": "event-1", "label": "First sample"},
        {"id": "event-3", "label": "Later sample"}
      ]
    }
  ]
}
""",
        encoding="utf-8",
    )
    registry = SourceRegistry()
    registry.register(PrimeObserverSource(directory=tmp_path))

    briefing = compose_source_briefing(registry=registry)
    prime_section = _section(briefing.text, "## Prime Observer")

    assert "Investigation index data: from Prime Observer." in prime_section
    assert "Investigation: June 8 WAN samples" in prime_section
    assert "Path: viz/investigation.json" in prime_section
    assert "Navigation metadata: from Prime Observer." in prime_section
    assert "First event: First sample" in prime_section
    assert "Next event: Next sample" in prime_section
    assert "Nearby-event facts: from Prime Observer." in prime_section
    assert "Events in the same investigation window for Next sample" in prime_section
    assert "First sample" in prime_section
    forbidden = ("correlated", "caused by", "likely related", "root cause")
    assert not any(term in prime_section.lower() for term in forbidden)


def test_source_briefing_redacted_dns_values_are_clear(tmp_path):
    _write_nextdns_summary(tmp_path / "nextdns_summary.json")
    registry = SourceRegistry()
    registry.register(PrimeObserverSource(directory=tmp_path))

    briefing = compose_source_briefing(registry=registry)
    prime_section = _section(briefing.text, "## Prime Observer")

    assert "Top blocked domain: unavailable" in prime_section
    assert "Top resolved domain: unavailable" in prime_section
    assert (
        "Top queried domain: entity_1 (redacted by Prime Observer privacy settings)"
        in prime_section
    )
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
    top_queried_domain: str | None = None,
    top_blocked_domain: str | None = None,
    top_resolved_domain: str | None = None,
):
    queried_field = (
        f'"top_queried_domain": "{top_queried_domain}",'
        if top_queried_domain
        else ""
    )
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
    "encrypted_queries": 800,
    "block_rate_pct": 2.0,
    "dns_block_rate": 0.02,
    "encrypted_rate_pct": 80.0,
    "dns_encrypted_rate": 0.8,
    {queried_field}
    "top_queried_domain_count": 300,
    "top_queried_domain_share": 0.3,
    {blocked_field}
    "top_blocked_domain_count": 10,
    "top_blocked_domain_share": 0.5,
    "top_blocked_reason": "OISD",
    {resolved_field}
    "top_resolved_domain_count": 250,
    "top_resolved_domain_share": 0.25,
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
