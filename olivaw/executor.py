from __future__ import annotations

from typing import Any

from olivaw.actions import get_action_spec
from olivaw.models import Action


class ExecutorError(Exception):
    pass


READ_ONLY_ACTIONS = {
    "state.read.snapshot",
    "observer.fetch_events",
    "analysis.summarize_events",
}


def execute_action(action: Action) -> dict[str, Any]:
    spec = get_action_spec(action.action_type)
    if spec is None:
        raise ExecutorError(
            "Execution forbidden: unknown action type. v1 only executes registered actions."
        )

    if action.action_type not in READ_ONLY_ACTIONS:
        raise ExecutorError(
            "Execution forbidden: this executor only supports approved read-only actions in v1."
        )

    handler = HANDLERS.get(action.action_type)
    if handler is None:
        raise ExecutorError(f"No handler registered for action type: {action.action_type}")

    return handler(action)


def handle_state_read_snapshot(action: Action) -> dict[str, Any]:
    entities = action.params.get("entities", [])
    return {
        "action_id": action.action_id,
        "action_type": action.action_type,
        "status": "completed",
        "result": {
            "entities": [
                {"entity_id": entity, "value": None, "timestamp": None}
                for entity in entities
            ],
            "note": "Stub result: integration lookup not implemented yet.",
        },
    }


def handle_observer_fetch_events(action: Action) -> dict[str, Any]:
    return {
        "action_id": action.action_id,
        "action_type": action.action_type,
        "status": "completed",
        "result": {
            "events": [],
            "note": "Stub result: Prime Observer integration not implemented yet.",
        },
    }


def handle_analysis_summarize_events(action: Action) -> dict[str, Any]:
    event_ids = action.params.get("event_ids", [])
    return {
        "action_id": action.action_id,
        "action_type": action.action_type,
        "status": "completed",
        "result": {
            "summary": f"Stub summary for {len(event_ids)} event(s).",
            "risk_flags": [],
            "note": "Stub result: analysis pipeline not implemented yet.",
        },
    }


HANDLERS = {
    "state.read.snapshot": handle_state_read_snapshot,
    "observer.fetch_events": handle_observer_fetch_events,
    "analysis.summarize_events": handle_analysis_summarize_events,
}
