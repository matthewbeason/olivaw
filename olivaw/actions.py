from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from olivaw.models import RiskClass


class StateReadSnapshotParams(BaseModel):
    entities: list[str]


class ObserverFetchEventsTimeRange(BaseModel):
    start: str
    end: str


class ObserverFetchEventsParams(BaseModel):
    source: str
    location: str | None = None
    time_range: ObserverFetchEventsTimeRange
    max_events: int = 100


class AnalysisSummarizeEventsParams(BaseModel):
    event_ids: list[str]
    include_risk_flags: bool = True
    output_style: str = "brief"


class NotifySendSummaryParams(BaseModel):
    channel: str
    title: str
    message: str
    priority: str = "normal"


class HomeActSetDeviceStateParams(BaseModel):
    device_id: str
    command: str
    arguments: dict[str, Any] = Field(default_factory=dict)


ACTION_REGISTRY: dict[str, dict[str, Any]] = {
    "state.read.snapshot": {
        "risk_class": "observe",
        "requires_approval": False,
        "params_model": StateReadSnapshotParams,
        "read_only": True,
        "description": "Read current state for key home entities.",
    },
    "observer.fetch_events": {
        "risk_class": "observe",
        "requires_approval": False,
        "params_model": ObserverFetchEventsParams,
        "read_only": True,
        "description": "Fetch Prime Observer events from an approved source.",
    },
    "analysis.summarize_events": {
        "risk_class": "analyze",
        "requires_approval": False,
        "params_model": AnalysisSummarizeEventsParams,
        "read_only": True,
        "description": "Summarize previously fetched events.",
    },
    "notify.send_summary": {
        "risk_class": "notify",
        "requires_approval": True,
        "params_model": NotifySendSummaryParams,
        "read_only": False,
        "description": "Send a summary through an approved notification channel.",
    },
    "home.act.set_device_state": {
        "risk_class": "physical_act",
        "requires_approval": True,
        "params_model": HomeActSetDeviceStateParams,
        "read_only": False,
        "description": "Change a device state in the home environment.",
    },
}


def get_action_spec(action_type: str) -> dict[str, Any] | None:
    return ACTION_REGISTRY.get(action_type)


def validate_action_params(action_type: str, params: dict[str, Any]) -> tuple[bool, list[str]]:
    spec = get_action_spec(action_type)
    if spec is None:
        return False, [f"Unknown action type: {action_type}"]

    params_model = spec["params_model"]
    try:
        params_model.model_validate(params)
    except ValidationError as exc:
        return False, [err["msg"] for err in exc.errors()]

    return True, []


def get_risk_class(action_type: str) -> RiskClass | None:
    spec = get_action_spec(action_type)
    if spec is None:
        return None
    return spec["risk_class"]
