from __future__ import annotations

from typing import Any

from olivaw.executor import ExecutorError, execute_action
from olivaw.models import Action, ApprovalRequest
from olivaw.policy import evaluate_action


class InMemoryApprovalStore:
    def __init__(self) -> None:
        self._items: dict[str, ApprovalRequest] = {}

    def add(self, approval: ApprovalRequest) -> None:
        self._items[approval.approval_id] = approval

    def get(self, approval_id: str) -> ApprovalRequest | None:
        return self._items.get(approval_id)

    def approve(self, approval_id: str) -> ApprovalRequest | None:
        approval = self.get(approval_id)
        if approval is not None:
            approval.status = "approved"
        return approval

    def reject(self, approval_id: str) -> ApprovalRequest | None:
        approval = self.get(approval_id)
        if approval is not None:
            approval.status = "rejected"
        return approval

    def all_pending(self) -> list[ApprovalRequest]:
        return [item for item in self._items.values() if item.status == "pending"]


class OlivawApp:
    def __init__(self) -> None:
        self.approvals = InMemoryApprovalStore()
        self.actions: dict[str, Action] = {}

    def submit_action(self, action_data: dict[str, Any]) -> dict[str, Any]:
        action = Action.model_validate(action_data)
        self.actions[action.action_id] = action

        decision, approval = evaluate_action(action)

        response: dict[str, Any] = {
            "action": action.model_dump(mode="json"),
            "policy_decision": decision.model_dump(mode="json"),
        }

        if decision.status == "approved":
            try:
                result = execute_action(action)
                response["execution_result"] = result
            except ExecutorError as exc:
                response["execution_result"] = {
                    "status": "failed",
                    "error": str(exc),
                }

        elif decision.status == "pending_approval" and approval is not None:
            self.approvals.add(approval)
            response["approval_request"] = approval.model_dump(mode="json")

        return response

    def approve_action(self, approval_id: str) -> dict[str, Any]:
        approval = self.approvals.approve(approval_id)
        if approval is None:
            return {
                "status": "error",
                "error": f"Unknown approval id: {approval_id}",
            }

        action = self.actions.get(approval.action_id)
        if action is None:
            return {
                "status": "error",
                "error": f"No action found for approval id: {approval_id}",
            }

        try:
            result = execute_action(action)
            return {
                "status": "approved_and_executed",
                "approval_request": approval.model_dump(mode="json"),
                "execution_result": result,
            }
        except ExecutorError as exc:
            return {
                "status": "approved_but_not_executed",
                "approval_request": approval.model_dump(mode="json"),
                "execution_result": {
                    "status": "failed",
                    "error": str(exc),
                },
            }


if __name__ == "__main__":
    app = OlivawApp()

    example_action = {
        "action_type": "state.read.snapshot",
        "description": "Read current state for key home entities",
        "params": {
            "entities": [
                "front_door.lock",
                "living_room.motion",
                "thermostat.mode",
            ]
        },
        "context": {
            "session_id": "demo-session",
            "goal": "inspect_home_state",
        },
    }

    result = app.submit_action(example_action)
    print(result)
