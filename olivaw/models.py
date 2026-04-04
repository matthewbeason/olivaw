from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

RiskClass = Literal["observe", "analyze", "notify", "digital_act", "physical_act"]
PolicyStatus = Literal["approved", "pending_approval", "denied"]
RequestedBy = Literal["user", "system"]
ActionSource = Literal["chat", "ui", "scheduled", "integration"]
ApprovalStatus = Literal["pending", "approved", "rejected"]


class ActionContext(BaseModel):
    session_id: str
    goal: str | None = None
    correlation_id: str | None = None


class Action(BaseModel):
    action_id: str = Field(default_factory=lambda: str(uuid4()))
    action_type: str
    description: str
    params: dict[str, Any] = Field(default_factory=dict)
    requested_by: RequestedBy = "user"
    source: ActionSource = "chat"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    context: ActionContext


class PolicyDecision(BaseModel):
    status: PolicyStatus
    action_id: str
    action_type: str
    risk_class: RiskClass | None = None
    requires_approval: bool = False
    reasons: list[str] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    approval_id: str = Field(default_factory=lambda: str(uuid4()))
    action_id: str
    action_type: str
    status: ApprovalStatus = "pending"
    title: str
    message: str
    risk_class: RiskClass
    reasons: list[str] = Field(default_factory=list)
