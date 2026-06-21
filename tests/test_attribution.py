from __future__ import annotations

from olivaw.assistant.attribution import (
    DERIVED,
    MODEL_KNOWLEDGE,
    MODEL_UNAVAILABLE,
    SOURCE_BACKED,
    AttributedResponse,
    UNKNOWN_OPERATIONAL_STATE,
    UNAVAILABLE_SOURCE_BACKED,
)
from olivaw.assistant.capability_registry import create_capability_registry


def test_attributed_response_carries_internal_metadata():
    response = AttributedResponse(
        text="Based on ManualSource...",
        attribution=SOURCE_BACKED,
        sources=("manual",),
        capability="source inspection",
    )

    assert response.text == "Based on ManualSource..."
    assert response.attribution == SOURCE_BACKED
    assert response.sources == ("manual",)
    assert response.capability == "source inspection"


def test_attribution_states_are_stable_strings():
    assert SOURCE_BACKED.value == "source_backed"
    assert DERIVED.value == "derived"
    assert MODEL_KNOWLEDGE.value == "model_knowledge"
    assert UNKNOWN_OPERATIONAL_STATE.value == "unknown_operational_state"
    assert UNAVAILABLE_SOURCE_BACKED.value == "unavailable_source_backed"
    assert MODEL_UNAVAILABLE.value == "model_unavailable"


def test_capability_registry_tracks_current_and_planned_sources():
    registry = create_capability_registry()

    assert "source inspection" in registry.implemented_capabilities
    assert "file inspection" in registry.implemented_capabilities
    assert registry.implemented_sources == (
        "manual",
        "files",
        "prime_observer",
        "core_signal",
        "weather",
    )
    assert registry.has_source("files") is True
    assert registry.has_source("prime_observer") is True
    assert registry.has_source("core_signal") is True
    assert registry.has_source("weather") is True
    assert "WeatherSource" not in registry.planned_sources
    assert "PrimeObserverSource" not in registry.planned_sources
    assert "CoreSignalSource" not in registry.planned_sources
    assert "persistent memory" in registry.planned_capabilities
