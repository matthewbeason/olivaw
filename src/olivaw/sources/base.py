from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

SourceStatus = Literal["ok", "unavailable", "error"]
SourcePayload = dict[str, Any]


@dataclass(frozen=True)
class SourceHealth:
    source_id: str
    display_name: str
    status: SourceStatus
    message: str


class Source(Protocol):
    source_id: str
    display_name: str

    def health(self) -> SourceHealth:
        """Return source availability without raising for expected failures."""

    def fetch(self) -> SourcePayload:
        """Return structured source data."""

