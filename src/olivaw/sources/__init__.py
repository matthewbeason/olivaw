from olivaw.sources.aggregation import AggregatedSources, SourceResult, aggregate_sources
from olivaw.sources.base import Source, SourceHealth, SourcePayload, SourceStatus
from olivaw.sources.core_signal import CoreSignalSource
from olivaw.sources.file_source import FileSource
from olivaw.sources.manual import ManualSource
from olivaw.sources.prime_observer import PrimeObserverSource
from olivaw.sources.weather import OpenMeteoProvider, WeatherSource
from olivaw.sources.registry import (
    SourceRegistry,
    create_default_registry,
    inspect_sources,
)

__all__ = [
    "ManualSource",
    "FileSource",
    "PrimeObserverSource",
    "CoreSignalSource",
    "WeatherSource",
    "OpenMeteoProvider",
    "Source",
    "SourceResult",
    "SourceHealth",
    "SourcePayload",
    "SourceRegistry",
    "SourceStatus",
    "AggregatedSources",
    "aggregate_sources",
    "create_default_registry",
    "inspect_sources",
]
