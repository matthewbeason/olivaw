from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class AttributionState(StrEnum):
    SOURCE_BACKED = "source-backed"
    MODEL_REASONED = "model-reasoned"
    CAPABILITY_UNAVAILABLE = "capability-unavailable"


SOURCE_BACKED = AttributionState.SOURCE_BACKED
MODEL_REASONED = AttributionState.MODEL_REASONED
CAPABILITY_UNAVAILABLE = AttributionState.CAPABILITY_UNAVAILABLE


@dataclass(frozen=True)
class AttributedResponse:
    text: str
    attribution: AttributionState
    sources: tuple[str, ...] = field(default_factory=tuple)
    capability: str | None = None

