from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from olivaw.actions.models import ActionExecutionContext, ActionRequest, ActionResult
from olivaw.actions.registry import ActionRegistry

SUPPORTED_RISK_LEVELS = {"safe_read", "local_state_change"}


@dataclass
class ActionHistory:
    last_action: ActionRequest | None = None
    last_result: ActionResult | None = None
    last_run: datetime | None = None

    def record(self, request: ActionRequest, result: ActionResult) -> None:
        self.last_action = request
        self.last_result = result
        self.last_run = result.completed_at

    def as_dict(self) -> dict[str, object]:
        return {
            "last_action": self.last_action,
            "last_result": self.last_result,
            "last_run": self.last_run,
        }


def execute_action(
    registry: ActionRegistry,
    request: ActionRequest,
    context: ActionExecutionContext,
    *,
    history: ActionHistory | None = None,
) -> ActionResult:
    started_at = datetime.now(timezone.utc)
    definition = registry.get(request.action_id)
    if definition is None:
        result = ActionResult(
            success=False,
            message=f"Unknown action: {request.action_id}",
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            metadata={"action_id": request.action_id},
        )
        if history is not None:
            history.record(request, result)
        return result

    if definition.risk_level not in SUPPORTED_RISK_LEVELS:
        result = ActionResult(
            success=False,
            message=f"Unsupported action risk level: {definition.risk_level}",
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            metadata={
                "action_id": definition.action_id,
                "risk_level": definition.risk_level,
            },
        )
        if history is not None:
            history.record(request, result)
        return result

    request = ActionRequest(
        action_id=request.action_id,
        parameters=request.parameters,
        requested_at=request.requested_at,
        executed_at=started_at,
    )
    result = definition.handler(request, context)
    if history is not None:
        history.record(request, result)
    return result
