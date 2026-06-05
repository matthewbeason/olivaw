from __future__ import annotations

from pathlib import Path

from olivaw.assistant.attribution import SOURCE_BACKED
from olivaw.briefing import compose_briefing_from_file, compose_source_briefing
from olivaw.sources.base import SourceHealth
from olivaw.sources import FileSource, ManualSource
from olivaw.sources.registry import SourceRegistry


def test_briefing_golden_output():
    output = compose_briefing_from_file(Path("examples/daily_context.json"))

    assert output == """# Daily Briefing

Date: 2026-06-04

## Focus
Stabilize Olivaw v0 as a local-first assistant foundation.

## Summary
Keep the first version small: health checks, deterministic briefing, provider routing, CLI, and web UI.

## Priorities
1. Ship the initial framework [in_progress] - The repository needs a clean base before additional assistant behavior is added.
2. Keep cloud fallback explicit [planned] - Local-first behavior should remain predictable and privacy-preserving.
3. Make health checks actionable [planned] - Missing local services should guide setup instead of causing crashes.

## Signals
- Briefing works without a model (local fixtures): The v0 briefing renderer is deterministic and testable offline.
- Local provider is preferred (provider router): Cloud models are ignored unless explicitly enabled.

## Projects
- Olivaw: foundation. Next: Run tests, verify CLI commands, and start the web app.

## Reminders
- Do not integrate Prime Observer or Core Signal in v0.
- Do not commit secrets or local configuration.
"""


class FakeSource:
    source_id = "fake"
    display_name = "Fake source"

    def health(self):
        return SourceHealth(
            source_id=self.source_id,
            display_name=self.display_name,
            status="ok",
            message="Fake source is available.",
        )

    def fetch(self):
        return {
            "source": self.source_id,
            "status": "ok",
            "items": [
                {"title": "Fake item", "summary": "Useful source-backed fact."}
            ],
        }


class EmptySource(FakeSource):
    source_id = "empty"
    display_name = "Empty source"

    def fetch(self):
        return {"source": self.source_id, "status": "ok", "items": []}


class FailedSource(FakeSource):
    source_id = "failed"
    display_name = "Failed source"

    def fetch(self):
        raise RuntimeError("source unavailable")


def test_source_backed_briefing_generation_includes_attribution():
    registry = SourceRegistry()
    registry.register(FakeSource())

    result = compose_source_briefing(registry=registry)

    assert result.attribution == SOURCE_BACKED
    assert result.sources == ("fake",)
    assert result.capability == "source-backed briefing"
    assert "# Source Briefing" in result.text
    assert "## Sources" in result.text
    assert "- fake: ok (Fake source) - Fake source is available." in result.text
    assert "Fake item from fake source: Useful source-backed fact." in result.text
    assert "This briefing is source-backed using: fake." in result.text


def test_source_backed_briefing_includes_manual_and_file_items(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "welcome.md").write_text(
        "# Welcome\nA source-backed note.\n",
        encoding="utf-8",
    )
    registry = SourceRegistry()
    registry.register(ManualSource())
    registry.register(FileSource(root=tmp_path))

    result = compose_source_briefing(registry=registry)

    assert result.sources == ("manual", "files")
    assert "Example item from manual source" in result.text
    assert "File found: notes/welcome.md" in result.text
    assert "- notes/welcome.md - # Welcome A source-backed note." in result.text
    assert "This briefing is source-backed using: manual, files." in result.text


def test_source_backed_briefing_handles_empty_sources():
    registry = SourceRegistry()
    registry.register(EmptySource())

    result = compose_source_briefing(registry=registry)

    assert result.attribution == SOURCE_BACKED
    assert "empty has no items." in result.text
    assert "empty: ok, no items returned." in result.text


def test_source_backed_briefing_handles_failed_sources():
    registry = SourceRegistry()
    registry.register(FailedSource())

    result = compose_source_briefing(registry=registry)

    assert result.attribution == SOURCE_BACKED
    assert "- failed: error (Failed source) - Fetch failed: RuntimeError: source unavailable" in result.text
    assert "failed unavailable: Fetch failed: RuntimeError: source unavailable" in result.text
