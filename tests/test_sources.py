from __future__ import annotations

import pytest

from olivaw.bootstrap import init_data
from olivaw.config import OlivawConfig
from olivaw.config import FileSourceConfig
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


def test_inspect_sources_returns_status_and_sample_data(tmp_path):
    init_data(tmp_path)
    report = inspect_sources(
        config=OlivawConfig(files=FileSourceConfig(directory=tmp_path))
    )

    assert report["sources"][0]["source_id"] == "manual"
    assert report["sources"][0]["status"] == "ok"
    assert report["sources"][1]["source_id"] == "files"
    assert report["sources"][1]["status"] == "ok"
    assert report["data"][0]["items"][0]["title"] == "Example item"
    assert report["data"][1]["count"] == 3


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
