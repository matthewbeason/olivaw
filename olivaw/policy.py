from __future__ import annotations

from olivaw.actions import get_action_spec, validate_action_params
from olivaw.models import Action, ApprovalRequest, PolicyDecision

FORBIDDEN_UNKNOWN_ACTION_MESSAGE = (
    "Unknown actions are forbidden in v1. "
    "Only registered action types may be proposed or executed."
)


def evaluate_action(action: Action) -> tuple[PolicyDecision, ApprovalRequest | None]:
    spec = get_action_spec(action.action_type)
    if spec is None:
        decision = PolicyDecision(
            status="denied",
            action_id=action.action_id,
            action_type=action.action_type,
            risk_class=None,
            requires_approval=False,
            reasons=[FORBIDDEN_UNKNOWN_ACTION_MESSAGE],
        )
        return decision, None

    is_valid, errors = validate_action_params(action.action_type, action.params)
    if not is_valid:
        decision = PolicyDecision(
            status="denied",
            action_id=action.action_id,
            action_type=action.action_type,
            risk_class=spec["risk_class"],
            requires_approval=False,
            reasons=["Parameter validation failed", *errors],
        )
        return decision, None

    if spec["requires_approval"]:
        approval = ApprovalRequest(
            action_id=action.action_id,
            action_type=action.action_type,
            title="Approval required",
            message=action.description,
            risk_class=spec["risk_class"],
            reasons=[
                f"Risk class is {spec['risk_class']}",
                "v1 policy requires explicit approval for this action type",
            ],
        )
        decision = PolicyDecision(
            status="pending_approval",
            action_id=action.action_id,
            action_type=action.action_type,
            risk_class=spec["risk_class"],
            requires_approval=True,
            reasons=[
                "Action is registered",
                "Parameters validated",
                "Approval is required before execution",
            ],
        )
        return decision, approval

    decision = PolicyDecision(
        status="approved",
        action_id=action.action_id,
        action_type=action.action_type,
        risk_class=spec["risk_class"],
        requires_approval=False,
        reasons=[
            "Action is registered",
            "Parameters validated",
            "Action is read-only or analysis-only and allowed in v1",
        ],
    )
    return decision, None
