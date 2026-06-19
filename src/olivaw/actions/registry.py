from __future__ import annotations

from olivaw.actions.models import ActionDefinition


class ActionRegistry:
    def __init__(self) -> None:
        self._actions: dict[str, ActionDefinition] = {}

    def register(self, definition: ActionDefinition) -> None:
        if definition.action_id in self._actions:
            raise ValueError(f"Action already registered: {definition.action_id}")
        self._actions[definition.action_id] = definition

    def get(self, action_id: str) -> ActionDefinition | None:
        return self._actions.get(action_id)

    def list_actions(self) -> tuple[ActionDefinition, ...]:
        return tuple(self._actions.values())
