from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class AttributionState(StrEnum):
    SOURCE_BACKED = "source-backed"
    DERIVED = "derived"
    MODEL_REASONED = "model-reasoned"
    CAPABILITY_UNAVAILABLE = "capability-unavailable"


SOURCE_BACKED = AttributionState.SOURCE_BACKED
DERIVED = AttributionState.DERIVED
MODEL_REASONED = AttributionState.MODEL_REASONED
CAPABILITY_UNAVAILABLE = AttributionState.CAPABILITY_UNAVAILABLE


@dataclass(frozen=True)
class AttributedResponse:
    text: str
    attribution: AttributionState
    sources: tuple[str, ...] = field(default_factory=tuple)
    capability: str | None = None
    provenance_label: str = ""
    provenance_detail: str = ""
