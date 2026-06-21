from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class AttributionState(StrEnum):
    SOURCE_BACKED = "source_backed"
    DERIVED = "derived"
    MODEL_KNOWLEDGE = "model_knowledge"
    UNKNOWN_OPERATIONAL_STATE = "unknown_operational_state"
    UNAVAILABLE_SOURCE_BACKED = "unavailable_source_backed"
    MODEL_UNAVAILABLE = "model_unavailable"


SOURCE_BACKED = AttributionState.SOURCE_BACKED
DERIVED = AttributionState.DERIVED
MODEL_KNOWLEDGE = AttributionState.MODEL_KNOWLEDGE
UNKNOWN_OPERATIONAL_STATE = AttributionState.UNKNOWN_OPERATIONAL_STATE
UNAVAILABLE_SOURCE_BACKED = AttributionState.UNAVAILABLE_SOURCE_BACKED
MODEL_UNAVAILABLE = AttributionState.MODEL_UNAVAILABLE


@dataclass(frozen=True)
class AttributedResponse:
    text: str
    attribution: AttributionState
    sources: tuple[str, ...] = field(default_factory=tuple)
    capability: str | None = None
    provenance_label: str = ""
    provenance_detail: str = ""
