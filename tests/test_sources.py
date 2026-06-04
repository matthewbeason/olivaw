from __future__ import annotations

import pytest

from olivaw.sources import ManualSource, SourceRegistry, create_default_registry
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


def test_default_registry_contains_manual_source():
    registry = create_default_registry()

    assert registry.get_source("manual") is not None


def test_inspect_sources_returns_status_and_sample_data():
    report = inspect_sources()

    assert report["sources"][0]["source_id"] == "manual"
    assert report["sources"][0]["status"] == "ok"
    assert report["data"][0]["items"][0]["title"] == "Example item"
