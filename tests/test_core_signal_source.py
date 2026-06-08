from __future__ import annotations

from olivaw.briefing import compose_source_briefing
from olivaw.config import CoreSignalSourceConfig, OlivawConfig
from olivaw.sources import CoreSignalSource, PrimeObserverSource
from olivaw.sources.registry import SourceRegistry, create_default_registry


def test_core_signal_source_reports_missing_directory(tmp_path):
    source = CoreSignalSource(directory=tmp_path / "missing")

    health = source.health()
    payload = source.fetch()

    assert health.status == "unavailable"
    assert "Directory does not exist" in health.message
    assert payload["status"] == "unavailable"
    assert payload["items"] == []


def test_core_signal_source_reports_empty_directory(tmp_path):
    source = CoreSignalSource(directory=tmp_path)

    health = source.health()
    payload = source.fetch()

    assert health.status == "unavailable"
    assert "No Core Signal report files found" in health.message
    assert payload["status"] == "unavailable"
    assert payload["count"] == 0


def test_core_signal_source_handles_malformed_report(tmp_path):
    (tmp_path / "latest.json").write_text("{bad json", encoding="utf-8")

    source = CoreSignalSource(directory=tmp_path)
    health = source.health()
    payload = source.fetch()

    assert health.status == "ok"
    assert payload["status"] == "error"
    assert payload["count"] == 0
    assert "JSONDecodeError" in payload["errors"][0]


def test_core_signal_source_loads_valid_json_report(tmp_path):
    (tmp_path / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "date": "2026-06-05",
  "status": "Watch",
  "summary": "Performance was slower than usual.",
  "status_reason": "The network differed from historical norms.",
  "recommended_action": "No action unless people noticed issues.",
  "dns_status": "normal",
  "dns_interpretation": "DNS filtering looked normal.",
  "dns_recommended_action": "No DNS action needed.",
  "dns_findings": [
    "Block rate remained within the expected range.",
    "Encrypted DNS usage was steady."
  ],
  "noteworthy_findings": [
    "Performance was slower than usual.",
    "DNS filtering looked normal."
  ]
}
""",
        encoding="utf-8",
    )

    payload = CoreSignalSource(directory=tmp_path).fetch()

    assert payload["source"] == "core_signal"
    assert payload["status"] == "ok"
    assert payload["count"] == 1
    item = payload["items"][0]
    assert item["title"] == "Core Signal Summary"
    assert item["status"] == "Watch"
    assert item["summary"] == "Performance was slower than usual."
    assert item["status_reason"] == "The network differed from historical norms."
    assert item["recommended_action"] == "No action unless people noticed issues."
    assert item["dns_status"] == "normal"
    assert item["dns_meaning"] == "DNS filtering looked normal."
    assert item["dns_recommended_action"] == "No DNS action needed."
    assert "Block rate remained within the expected range." in item["dns_findings"]
    assert "Encrypted DNS usage was steady." in item["dns_findings"]
    assert "DNS filtering looked normal." in item["findings"]


def test_core_signal_source_preserves_json_event_metadata(tmp_path):
    (tmp_path / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "date": "2026-06-08",
  "status": "Attention",
  "summary": "The network had 1 sustained slowdown period.",
  "events": [
    {
      "id": "core-signal-sustained_slowdown-abc123",
      "kind": "sustained_slowdown",
      "status": "Attention",
      "severity": "attention",
      "confidence": "High",
      "window_start": "2026-06-08T11:11:30+00:00",
      "window_end": "2026-06-08T11:12:09+00:00",
      "summary": "1 sustained slowdown period was found.",
      "why": "Sustained slowdown was detected.",
      "recommended_action": "Check provider status if symptoms matched.",
      "confidence_reason": "Matched a sustained slowdown policy threshold.",
      "supporting_facts": [
        {
          "summary": "WAN p95 exceeded the sustained threshold.",
          "source": "prime_observer",
          "reference": {
            "url": "http://127.0.0.1:8000/investigate.html?start=1&end=2"
          },
          "raw_evidence": {"p95_ms": 221}
        }
      ],
      "recommendation_trace": {
        "recommendation": "Check provider status if symptoms matched.",
        "supporting_facts": ["WAN p95 exceeded the sustained threshold."],
        "interpretation": "Core Signal classified this as sustained slowdown."
      },
      "interpretation_source": "core_signal",
      "related_events": [
        {
          "event_id": "core-signal-dns-watch-def456",
          "relationship": "same_window",
          "summary": "DNS watch event in the same Core Signal output.",
          "reference": "events[1]"
        }
      ],
      "issue_location": "Likely upstream/ISP issue",
      "attribution_source": "prime_observer_incident",
      "prime_observer_reference": {
        "type": "event",
        "url": "http://127.0.0.1:8000/investigate.html?start=1&end=2",
        "window_start": "2026-06-08T11:11:30+00:00",
        "window_end": "2026-06-08T11:12:09+00:00"
      },
      "evidence_window": {
        "source": "prime_observer",
        "window_start": "2026-06-08T11:11:30+00:00",
        "window_end": "2026-06-08T11:12:09+00:00",
        "granularity": "15-minute bucket"
      }
    }
  ]
}
""",
        encoding="utf-8",
    )

    payload = CoreSignalSource(directory=tmp_path).fetch()

    event = payload["items"][0]["events"][0]
    assert event["id"] == "core-signal-sustained_slowdown-abc123"
    assert event["kind"] == "sustained_slowdown"
    assert event["severity"] == "attention"
    assert event["confidence"] == "High"
    assert event["confidence_reason"] == "Matched a sustained slowdown policy threshold."
    assert event["window_start"] == "2026-06-08T11:11:30+00:00"
    assert event["window_end"] == "2026-06-08T11:12:09+00:00"
    assert event["recommended_action"] == "Check provider status if symptoms matched."
    assert event["supporting_facts"] == [
        {
            "summary": "WAN p95 exceeded the sustained threshold.",
            "source": "prime_observer",
            "reference": "http://127.0.0.1:8000/investigate.html?start=1&end=2",
        }
    ]
    assert "raw_evidence" not in event["supporting_facts"][0]
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
    assert event["interpretation_source"] == "core_signal"
    assert event["related_events"] == [
        {
            "id": "core-signal-dns-watch-def456",
            "relationship": "same_window",
            "summary": "DNS watch event in the same Core Signal output.",
            "reference": "events[1]",
        }
    ]
    assert event["issue_location"] == "Likely upstream/ISP issue"
    assert event["attribution_source"] == "prime_observer_incident"
    assert event["prime_observer_investigation"].startswith("http://127.0.0.1:8000/")
    assert event["evidence_window"]["granularity"] == "15-minute bucket"


def test_core_signal_source_loads_markdown_morning_brief(tmp_path):
    (tmp_path / "latest.md").write_text(_morning_brief(), encoding="utf-8")

    payload = CoreSignalSource(directory=tmp_path).fetch()

    assert payload["status"] == "ok"
    item = payload["items"][0]
    assert item["title"] == "Core Signal Morning Brief - 2026-06-05"
    assert item["report_date"] == "2026-06-05"
    assert item["status"] == "Watch"
    assert item["summary"] == "Performance was unusually slow compared with normal."
    assert item["status_reason"] == (
        "Performance differed from historical norms but was not actionable."
    )
    assert item["recommended_action"] == "No action unless people noticed issues."
    assert "Performance was slower than usual." in item["findings"]
    assert item["dns_findings"] == ["DNS filtering looked normal."]
    assert item["report_type"] == "morning_brief"


def test_core_signal_source_parses_markdown_event_metadata(tmp_path):
    (tmp_path / "latest.md").write_text(_event_morning_brief(), encoding="utf-8")

    payload = CoreSignalSource(directory=tmp_path).fetch()

    event = payload["items"][0]["events"][0]
    assert event["id"] == "latest"
    assert event["status"] == "Attention"
    assert event["severity"] == "attention"
    assert event["summary"] == "The network had 1 sustained slowdown period(s)."
    assert event["window_start"] == "2026-06-07 11:00 UTC-07:00"
    assert event["window_end"] == "2026-06-08 11:00 UTC-07:00"
    assert event["recommended_action"] == "Check provider status if symptoms matched."
    assert event["issue_location"] == "Likely upstream/ISP issue"
    assert event["attribution_source"] == "Prime Observer incident attribution"
    assert event["prime_observer_investigation"].startswith("viz/investigate.html?")
    assert event["evidence_window"]["label"].startswith("2026-06-07 11:00")


def test_core_signal_source_loads_markdown_pattern_report(tmp_path):
    patterns = tmp_path / "patterns"
    patterns.mkdir()
    (patterns / "latest.md").write_text(_pattern_report(), encoding="utf-8")

    payload = CoreSignalSource(directory=tmp_path).fetch()

    assert payload["status"] == "ok"
    item = payload["items"][0]
    assert item["title"] == "Core Signal Pattern Report - 2026-06-02"
    assert item["report_date"] == "2026-06-02"
    assert item["report_type"] == "pattern_report"
    assert "7 observed pattern candidates" in item["summary"]
    assert "Business-hour WAN elevation" in item["findings"]
    assert "Concentration: OISD block reason" in item["dns_findings"]


def test_core_signal_source_prefers_latest_markdown_per_category(tmp_path):
    (tmp_path / "latest.md").write_text(_morning_brief(), encoding="utf-8")
    (tmp_path / "morning-brief-2026-06-04.md").write_text(
        _morning_brief().replace("2026-06-05", "2026-06-04"),
        encoding="utf-8",
    )
    patterns = tmp_path / "patterns"
    patterns.mkdir()
    (patterns / "latest.md").write_text(_pattern_report(), encoding="utf-8")
    (patterns / "pattern-report-2026-06-01.md").write_text(
        _pattern_report().replace("2026-06-02", "2026-06-01"),
        encoding="utf-8",
    )

    payload = CoreSignalSource(directory=tmp_path).fetch()

    assert payload["count"] == 2
    titles = [item["title"] for item in payload["items"]]
    assert titles == [
        "Core Signal Morning Brief - 2026-06-05",
        "Core Signal Pattern Report - 2026-06-02",
    ]


def test_core_signal_source_registration(tmp_path):
    config = OlivawConfig(core_signal=CoreSignalSourceConfig(directory=tmp_path))

    registry = create_default_registry(config)

    assert registry.get_source("core_signal") is not None


def test_core_signal_source_health_reporting(tmp_path):
    (tmp_path / "latest.md").write_text(_morning_brief(), encoding="utf-8")

    health = CoreSignalSource(directory=tmp_path).health()

    assert health.source_id == "core_signal"
    assert health.display_name == "Core Signal"
    assert health.status == "ok"
    assert "Core Signal report file" in health.message


def test_source_briefing_includes_core_signal_section(tmp_path):
    (tmp_path / "latest.md").write_text(_morning_brief(), encoding="utf-8")
    registry = SourceRegistry()
    registry.register(CoreSignalSource(directory=tmp_path))

    briefing = compose_source_briefing(registry=registry)

    assert "## Core Signal" in briefing.text
    assert "Core Signal Morning Brief" in briefing.text
    assert (
        "Why/status reasoning: Performance differed from historical norms but was not actionable."
        in briefing.text
    )
    assert "Recommended action: No action unless people noticed issues." in briefing.text
    assert "This briefing is source-backed using: core_signal." in briefing.text


def test_source_briefing_renders_core_signal_event_metadata(tmp_path):
    (tmp_path / "latest.md").write_text(_event_morning_brief(), encoding="utf-8")
    registry = SourceRegistry()
    registry.register(CoreSignalSource(directory=tmp_path))

    briefing = compose_source_briefing(registry=registry)

    assert "Event: The network had 1 sustained slowdown period(s)." in briefing.text
    assert "Severity/status: Attention / attention" in briefing.text
    assert (
        "Affected window: 2026-06-07 11:00 UTC-07:00 to 2026-06-08 11:00 UTC-07:00"
        in briefing.text
    )
    assert (
        "Recommended action: Check provider status if symptoms matched."
        in briefing.text
    )
    assert "Issue location: Likely upstream/ISP issue" in briefing.text
    assert "Evidence: Prime Observer incident attribution" in briefing.text
    assert "View investigation: viz/investigate.html?start=1&end=2" in briefing.text


def test_source_briefing_keeps_prime_observer_and_core_signal_semantics_separate(
    tmp_path,
):
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    prime_dir.mkdir()
    core_dir.mkdir()
    _write_prime_nextdns_summary(prime_dir / "nextdns_summary.json")
    (core_dir / "latest.md").write_text(_morning_brief(), encoding="utf-8")
    registry = SourceRegistry()
    registry.register(PrimeObserverSource(directory=prime_dir))
    registry.register(CoreSignalSource(directory=core_dir))

    briefing = compose_source_briefing(registry=registry)
    prime_section = _section(briefing.text, "## Prime Observer")
    core_section = _section(briefing.text, "## Core Signal")

    assert "DNS summary: available from Prime Observer." in prime_section
    assert "DNS filtering looked normal" not in prime_section
    assert "DNS filtering looked normal" in core_section
    assert "DNS interpretation: DNS filtering looked normal." in core_section
    assert "Why/status reasoning:" in core_section
    assert "Recommended action:" in core_section


def _morning_brief() -> str:
    return """# Core Signal Morning Brief - 2026-06-05

Status: Watch

Performance was unusually slow compared with normal.

Why This Status:
Performance differed from historical norms but was not actionable.

Recommended Action: No action unless people noticed issues.

Worth knowing:
- Performance was slower than usual.
- DNS filtering looked normal.
"""


def _event_morning_brief() -> str:
    return """# Core Signal Morning Brief - 2026-06-08

Status: Attention

The network had 1 sustained slowdown period(s).

Why This Status:
Sustained slowdown was detected, which means user impact was possible.

Issue Location: Likely upstream/ISP issue

Recommended Action: Check provider status if symptoms matched.

Worth knowing:
- 1 sustained slowdown period(s) were found.
- Evidence points to an upstream/ISP issue.

Technical Evidence:
- Window: 2026-06-07 11:00 UTC-07:00 to 2026-06-08 11:00 UTC-07:00
- Prime Observer investigation: viz/investigate.html?start=1&end=2
- Attribution source: Prime Observer incident attribution
- Prime Observer policy: v0.5.0-aligned
"""


def _pattern_report() -> str:
    return """# Core Signal Pattern Report - 2026-06-02

## Executive Summary
Core Signal found 7 observed pattern candidates in 59,310 WAN points across 31 days.

### Afternoon ramp

The afternoon ramp showed elevated latency.

### Business-hour WAN elevation

Weekday business hours showed higher WAN latency than other windows.

## Concentration Signals

### Concentration: OISD block reason

- Entity type: Blocked DNS reason
- OISD represented 95.7% of available blocked-reason activity.
"""


def _write_prime_nextdns_summary(path):
    path.write_text(
        """
{
  "generated_at": "2026-06-04T21:05:00Z",
  "status": "ok",
  "summary": {
    "total_queries": 1000,
    "blocked_queries": 20,
    "allowed_queries": 900,
    "block_rate_pct": 2.0,
    "encrypted_rate_pct": 80.0,
    "top_entities": [
      {
        "label": "entity_1",
        "count": 250,
        "share_of_total": 0.25,
        "name_redacted": true
      }
    ]
  }
}
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
