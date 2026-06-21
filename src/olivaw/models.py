from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ProviderKind = Literal["local", "cloud", "router"]
ProviderState = Literal["available", "unavailable", "disabled", "unknown"]


@dataclass(frozen=True)
class ProviderStatus:
    name: str
    kind: ProviderKind
    state: ProviderState
    message: str
    detail: str | None = None
    model: str | None = None

    @property
    def available(self) -> bool:
        return self.state == "available"


@dataclass(frozen=True)
class CompletionRequest:
    prompt: str
    system_prompt: str | None = None


@dataclass(frozen=True)
class CompletionResponse:
    text: str
    provider: str
    model: str | None = None
    request_duration_ms: int | None = None
    ollama_total_duration_ms: int | None = None
    ollama_load_duration_ms: int | None = None
    ollama_prompt_eval_duration_ms: int | None = None
    ollama_eval_duration_ms: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None


@dataclass(frozen=True)
class HealthReport:
    local: ProviderStatus
    cloud: ProviderStatus
    selected_provider: str | None
    cloud_fallback: str
    notes: list[str] = field(default_factory=list)
