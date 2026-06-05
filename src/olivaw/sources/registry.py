from __future__ import annotations

from dataclasses import asdict

from olivaw.config import OlivawConfig, load_config
from olivaw.sources.base import Source, SourceHealth, SourcePayload
from olivaw.sources.file_source import FileSource
from olivaw.sources.manual import ManualSource
from olivaw.sources.prime_observer import PrimeObserverSource


class SourceRegistry:
    def __init__(self) -> None:
        self._sources: dict[str, Source] = {}

    def register(self, source: Source) -> None:
        if source.source_id in self._sources:
            raise ValueError(f"Source already registered: {source.source_id}")
        self._sources[source.source_id] = source

    def list_sources(self) -> tuple[Source, ...]:
        return tuple(self._sources.values())

    def get_source(self, source_id: str) -> Source | None:
        return self._sources.get(source_id)

    def health_all(self) -> tuple[SourceHealth, ...]:
        return tuple(source.health() for source in self.list_sources())

    def fetch_all(self) -> tuple[SourcePayload, ...]:
        return tuple(source.fetch() for source in self.list_sources())


def create_default_registry(config: OlivawConfig | None = None) -> SourceRegistry:
    resolved_config = config or load_config()
    registry = SourceRegistry()
    registry.register(ManualSource())
    registry.register(
        FileSource(
            root=resolved_config.files.directory,
            max_bytes=resolved_config.files.max_bytes,
        )
    )
    registry.register(
        PrimeObserverSource(
            directory=resolved_config.prime_observer.directory,
            enabled=resolved_config.prime_observer.enabled,
        )
    )
    return registry


def inspect_sources(
    registry: SourceRegistry | None = None,
    config: OlivawConfig | None = None,
) -> dict[str, object]:
    resolved = registry or create_default_registry(config)
    return {
        "sources": [asdict(status) for status in resolved.health_all()],
        "data": list(resolved.fetch_all()),
    }
