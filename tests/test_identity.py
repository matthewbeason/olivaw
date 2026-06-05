from __future__ import annotations

from olivaw.assistant.identity import capabilities_summary, get_identity


def test_identity_contains_current_implemented_capabilities():
    identity = get_identity()

    assert identity.name == "Olivaw"
    assert "R. Daneel Olivaw" in identity.origin_note
    assert "local-first personal assistant framework" in identity.purpose
    assert (
        "deterministic briefing generation from structured input"
        in identity.implemented_capabilities
    )
    assert "provider health reporting" in identity.implemented_capabilities
    assert "local Ollama provider access" in identity.implemented_capabilities
    assert "read-only configuration display" in identity.implemented_capabilities
    assert "source inspection" in identity.implemented_capabilities
    assert "file inspection" in identity.implemented_capabilities
    assert "source-backed briefing generation" in identity.implemented_capabilities
    assert "PrimeObserverSource" in identity.implemented_capabilities
    assert "CoreSignalSource" in identity.implemented_capabilities


def test_identity_contains_not_yet_implemented_capabilities():
    identity = get_identity()

    assert "persistent memory" in identity.not_yet_implemented_capabilities
    assert "calendar integration" in identity.not_yet_implemented_capabilities
    assert "email integration" in identity.not_yet_implemented_capabilities
    assert "weather lookup" in identity.not_yet_implemented_capabilities
    assert "Source aggregation" in identity.not_yet_implemented_capabilities
    assert "desktop automation" in identity.not_yet_implemented_capabilities


def test_capabilities_summary_separates_current_from_roadmap():
    summary = capabilities_summary()
    implemented, not_implemented = summary.split("Not implemented yet:")

    assert "deterministic briefing generation" in implemented
    assert "source inspection" in implemented
    assert "file inspection" in implemented
    assert "source-backed briefing generation" in implemented
    assert "calendar integration" not in implemented
    assert "weather lookup" not in implemented
    assert "calendar integration" in not_implemented
    assert "weather lookup" in not_implemented
