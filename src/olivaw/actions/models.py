from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from olivaw.config import OlivawConfig

RiskLevel = Literal[
    "safe_read",
    "local_state_change",
    "external_write",
    "system_control",
]


@dataclass(frozen=True)
class ActionRequest:
    action_id: str
    parameters: dict[str, object] = field(default_factory=dict)
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    executed_at: datetime | None = None


@dataclass(frozen=True)
class ActionResult:
    success: bool
    message: str
    started_at: datetime
    completed_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionDefinition:
    action_id: str
    label: str
    description: str
    category: str
    risk_level: RiskLevel
    requires_confirmation: bool
    handler: Callable[[ActionRequest, "ActionExecutionContext"], ActionResult]

    def as_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "label": self.label,
            "description": self.description,
            "category": self.category,
            "risk_level": self.risk_level,
            "requires_confirmation": self.requires_confirmation,
        }


@dataclass(frozen=True)
class ActionExecutionContext:
    config: OlivawConfig
    refresh_health_review: Callable[[], dict[str, object]]
    refresh_sources: Callable[[], dict[str, object]]
    source_diagnostics: Callable[[], dict[str, object]]
    evidence_package: Callable[[], dict[str, object]]
