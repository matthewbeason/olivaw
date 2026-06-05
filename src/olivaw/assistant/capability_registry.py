from __future__ import annotations

from dataclasses import dataclass, field

from olivaw.assistant.identity import get_identity


@dataclass(frozen=True)
class CapabilityRegistry:
    implemented_capabilities: tuple[str, ...] = field(default_factory=tuple)
    implemented_sources: tuple[str, ...] = field(default_factory=tuple)
    planned_capabilities: tuple[str, ...] = field(default_factory=tuple)
    planned_sources: tuple[str, ...] = field(default_factory=tuple)

    def has_source(self, source_id: str) -> bool:
        return source_id in self.implemented_sources


PLANNED_SOURCES = (
    "PrimeObserverSource",
    "CoreSignalSource",
    "WeatherSource",
    "CalendarSource",
    "EmailSource",
)


def create_capability_registry(
    implemented_sources: tuple[str, ...] = ("manual", "files"),
) -> CapabilityRegistry:
    identity = get_identity()
    planned_capabilities = tuple(
        item
        for item in identity.not_yet_implemented_capabilities
        if item not in PLANNED_SOURCES
    )
    return CapabilityRegistry(
        implemented_capabilities=identity.implemented_capabilities,
        implemented_sources=implemented_sources,
        planned_capabilities=planned_capabilities,
        planned_sources=PLANNED_SOURCES,
    )

