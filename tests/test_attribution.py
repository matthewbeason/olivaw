from __future__ import annotations

from olivaw.assistant.attribution import (
    CAPABILITY_UNAVAILABLE,
    MODEL_REASONED,
    SOURCE_BACKED,
    AttributedResponse,
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
    assert SOURCE_BACKED.value == "source-backed"
    assert MODEL_REASONED.value == "model-reasoned"
    assert CAPABILITY_UNAVAILABLE.value == "capability-unavailable"


def test_capability_registry_tracks_current_and_planned_sources():
    registry = create_capability_registry()

    assert "source inspection" in registry.implemented_capabilities
    assert "file inspection" in registry.implemented_capabilities
    assert registry.implemented_sources == (
        "manual",
        "files",
        "prime_observer",
        "core_signal",
    )
    assert registry.has_source("files") is True
    assert registry.has_source("prime_observer") is True
    assert registry.has_source("core_signal") is True
    assert registry.has_source("weather") is False
    assert "WeatherSource" in registry.planned_sources
    assert "PrimeObserverSource" not in registry.planned_sources
    assert "CoreSignalSource" not in registry.planned_sources
    assert "persistent memory" in registry.planned_capabilities
