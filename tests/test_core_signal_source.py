from __future__ import annotations

from olivaw.briefing import compose_source_briefing
from olivaw.config import CoreSignalSourceConfig, OlivawConfig
from olivaw.sources import CoreSignalSource
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
  "recommended_action": "No action unless people noticed issues.",
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
    assert item["recommended_action"] == "No action unless people noticed issues."
    assert "DNS filtering looked normal." in item["findings"]


def test_core_signal_source_loads_markdown_morning_brief(tmp_path):
    (tmp_path / "latest.md").write_text(_morning_brief(), encoding="utf-8")

    payload = CoreSignalSource(directory=tmp_path).fetch()

    assert payload["status"] == "ok"
    item = payload["items"][0]
    assert item["title"] == "Core Signal Morning Brief - 2026-06-05"
    assert item["report_date"] == "2026-06-05"
    assert item["status"] == "Watch"
    assert item["summary"] == "Performance was unusually slow compared with normal."
    assert item["recommended_action"] == "No action unless people noticed issues."
    assert "Performance was slower than usual." in item["findings"]
    assert item["report_type"] == "morning_brief"


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
    assert "Recommended action: No action unless people noticed issues." in briefing.text
    assert "This briefing is source-backed using: core_signal." in briefing.text


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


def _pattern_report() -> str:
    return """# Core Signal Pattern Report - 2026-06-02

## Executive Summary
Core Signal found 7 observed pattern candidates in 59,310 WAN points across 31 days.

### Afternoon ramp

The afternoon ramp showed elevated latency.

### Business-hour WAN elevation

Weekday business hours showed higher WAN latency than other windows.
"""
