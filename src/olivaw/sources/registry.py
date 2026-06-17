from __future__ import annotations

from olivaw.config import OlivawConfig, load_config
from olivaw.sources.aggregation import AggregatedSources, aggregate_sources
from olivaw.sources.base import Source, SourceHealth, SourcePayload
from olivaw.sources.core_signal import CoreSignalSource
from olivaw.sources.file_source import FileSource
from olivaw.sources.manual import ManualSource
from olivaw.sources.prime_observer import PrimeObserverSource
from olivaw.sources.weather import WeatherSource


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

    def aggregate(self) -> AggregatedSources:
        return aggregate_sources(self.list_sources())


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
            base_url=resolved_config.prime_observer.base_url,
        )
    )
    registry.register(
        CoreSignalSource(
            directory=resolved_config.core_signal.directory,
            enabled=resolved_config.core_signal.enabled,
        )
    )
    registry.register(
        WeatherSource(
            enabled=resolved_config.weather.enabled,
            latitude=resolved_config.weather.latitude,
            longitude=resolved_config.weather.longitude,
            location_name=resolved_config.weather.location_name,
            units=resolved_config.weather.units,
        )
    )
    return registry


def inspect_sources(
    registry: SourceRegistry | None = None,
    config: OlivawConfig | None = None,
) -> dict[str, object]:
    resolved = registry or create_default_registry(config)
    aggregate = resolved.aggregate()
    return {
        "sources": [
            {
                "source_id": source.source_id,
                "display_name": source.source_name,
                "status": source.status,
                "message": source.message,
            }
            for source in aggregate.sources
        ],
        "data": [_safe_payload(source) for source in resolved.list_sources()],
        "aggregate": aggregate.as_dict(),
    }


def _safe_payload(source: Source) -> SourcePayload:
    try:
        health = source.health()
    except Exception as exc:
        return {
            "source": getattr(source, "source_id", "unknown"),
            "status": "error",
            "items": [],
            "count": 0,
            "errors": [f"Health check failed: {type(exc).__name__}: {exc}"],
        }
    if health.status != "ok":
        return {
            "source": health.source_id,
            "status": health.status,
            "items": [],
            "count": 0,
        }
    try:
        return source.fetch()
    except Exception as exc:
        return {
            "source": health.source_id,
            "status": "error",
            "items": [],
            "count": 0,
            "errors": [f"Fetch failed: {type(exc).__name__}: {exc}"],
        }
