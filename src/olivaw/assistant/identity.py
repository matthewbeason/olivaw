from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AssistantIdentity:
    name: str
    origin_note: str
    purpose: str
    implemented_capabilities: tuple[str, ...] = field(default_factory=tuple)
    not_yet_implemented_capabilities: tuple[str, ...] = field(default_factory=tuple)
    operating_principles: tuple[str, ...] = field(default_factory=tuple)


OLIVAW_IDENTITY = AssistantIdentity(
    name="Olivaw",
    origin_note="Named after R. Daneel Olivaw from Isaac Asimov's fiction.",
    purpose="A local-first personal assistant framework.",
    implemented_capabilities=(
        "deterministic briefing generation from structured input",
        "provider health reporting",
        "local Ollama provider access",
        "cloud OpenAI provider support when explicitly enabled",
        "provider routing",
        "CLI interface",
        "lightweight web interface",
        "read-only configuration display",
        "source inspection",
        "file inspection",
        "source-aware response attribution",
        "source-backed briefing generation",
    ),
    not_yet_implemented_capabilities=(
        "persistent memory",
        "calendar integration",
        "email integration",
        "notifications/reminders",
        "weather lookup",
        "local business lookup",
        "Prime Observer integration",
        "Core Signal integration",
        "PrimeObserverSource",
        "CoreSignalSource",
        "WeatherSource",
        "CalendarSource",
        "EmailSource",
        "Source aggregation",
        "autonomous background tasks",
        "tool execution",
        "desktop automation",
    ),
    operating_principles=(
        "Prefer local execution and local providers by default.",
        "Use cloud providers only when explicitly enabled.",
        "Degrade gracefully when services or credentials are missing.",
        "Describe only implemented capabilities as current functionality.",
        "Clearly label roadmap items as not implemented yet.",
        "Avoid exposing secrets in user-facing output.",
    ),
)


def get_identity() -> AssistantIdentity:
    return OLIVAW_IDENTITY


def capabilities_summary(identity: AssistantIdentity | None = None) -> str:
    resolved = identity or get_identity()
    lines = [
        f"{resolved.name} can currently:",
        *[f"- {item}" for item in resolved.implemented_capabilities],
        "",
        "Not implemented yet:",
        *[f"- {item}" for item in resolved.not_yet_implemented_capabilities],
    ]
    return "\n".join(lines)
