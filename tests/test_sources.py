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
from olivaw.sources import FileSource, ManualSource, SourceRegistry, create_default_registry
from olivaw.sources.registry import inspect_sources


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
    assert report["data"][0]["items"][0]["title"] == "Example item"
    assert report["data"][1]["count"] == 3


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
