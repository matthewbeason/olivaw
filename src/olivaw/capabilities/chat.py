from __future__ import annotations

from dataclasses import dataclass

from olivaw.assistant.attribution import (
    CAPABILITY_UNAVAILABLE,
    MODEL_REASONED,
    SOURCE_BACKED,
    AttributedResponse,
)
from olivaw.assistant.capability_registry import create_capability_registry
from olivaw.assistant.identity import capabilities_summary
from olivaw.assistant.prompts import build_chat_system_prompt
from olivaw.config import OlivawConfig, load_config
from olivaw.models import CompletionRequest
from olivaw.providers.router import RouterProvider
from olivaw.sources.registry import create_default_registry

HEALTH_HINT = (
    "Run `olivaw health` to inspect provider status. "
    "If using Ollama, verify it is installed and running at the configured endpoint."
)


@dataclass
class ChatCapability:
    name: str = "chat"
    description: str = "Minimal provider-routed chat placeholder."

    def run(self, prompt: str, config: OlivawConfig | None = None) -> str:
        return self.run_with_attribution(prompt, config=config).text

    def run_with_attribution(
        self, prompt: str, config: OlivawConfig | None = None
    ) -> AttributedResponse:
        if _is_capability_question(prompt):
            return AttributedResponse(
                text=capabilities_summary(),
                attribution=SOURCE_BACKED,
                sources=("capability-registry",),
                capability="capability inspection",
            )

        missing = _missing_capability_for_prompt(prompt)
        if missing:
            return _capability_unavailable_response(missing)

        resolved_config = config or load_config()
        if _is_source_question(prompt):
            return _source_status_response(resolved_config)

        router = RouterProvider(resolved_config)
        try:
            response = router.complete(
                CompletionRequest(
                    prompt=prompt,
                    system_prompt=build_chat_system_prompt(),
                )
            )
        except Exception as exc:
            return AttributedResponse(
                text=_format_chat_failure(exc),
                attribution=CAPABILITY_UNAVAILABLE,
                capability="model provider",
            )
        return AttributedResponse(
            text=response.text,
            attribution=MODEL_REASONED,
            capability="chat",
        )


def _format_chat_failure(exc: Exception) -> str:
    detail = str(exc).strip() or "provider failed without additional detail"
    return (
        "Chat provider unavailable: "
        f"{type(exc).__name__}: {detail}. {HEALTH_HINT}"
    )


def _is_capability_question(prompt: str) -> bool:
    normalized = " ".join(prompt.lower().split())
    capability_phrases = (
        "what can you do",
        "what can you currently do",
        "what are your capabilities",
        "what are you able to do",
    )
    return any(phrase in normalized for phrase in capability_phrases)


def _is_source_question(prompt: str) -> bool:
    normalized = " ".join(prompt.lower().split())
    source_phrases = (
        "what sources",
        "which sources",
        "registered sources",
        "available sources",
        "configured sources",
    )
    return any(phrase in normalized for phrase in source_phrases)


def _missing_capability_for_prompt(prompt: str) -> str | None:
    normalized = " ".join(prompt.lower().split())
    unavailable_checks = (
        ("weather source", ("weather", "forecast", "temperature")),
        ("calendar source", ("calendar", "schedule", "appointment", "meeting")),
        ("email source", ("email", "inbox", "mail")),
        ("reminder source", ("reminder", "remind me", "notification")),
        ("web search source", ("web search", "search the web", "look up online")),
        ("news source", ("news", "headlines")),
        ("sports scores source", ("sports score", "sports scores", "score of")),
        ("stock prices source", ("stock price", "stock prices", "share price")),
    )
    for capability, phrases in unavailable_checks:
        if any(phrase in normalized for phrase in phrases):
            return capability
    return None


def _capability_unavailable_response(capability: str) -> AttributedResponse:
    registry = create_capability_registry()
    planned = _planned_source_name(capability, registry.planned_sources)
    suffix = (
        f" {planned} is a planned source, but it is not implemented yet."
        if planned
        else ""
    )
    return AttributedResponse(
        text=(
            f"I do not currently have a {capability} configured, so I cannot "
            f"answer that from Olivaw sources yet.{suffix}"
        ),
        attribution=CAPABILITY_UNAVAILABLE,
        capability=capability,
    )


def _planned_source_name(capability: str, planned_sources: tuple[str, ...]) -> str | None:
    normalized = capability.lower()
    for source in planned_sources:
        stem = source.removesuffix("Source").lower()
        if stem in normalized:
            return source
    return None


def _source_status_response(config: OlivawConfig) -> AttributedResponse:
    registry = create_default_registry(config)
    health = registry.health_all()
    lines = ["Registered Olivaw sources:"]
    for source in health:
        lines.append(
            f"- {source.display_name} ({source.source_id}): {source.status} - {source.message}"
        )
    lines.extend(
        [
            "",
            "Planned sources are not implemented yet: PrimeObserverSource, "
            "CoreSignalSource, WeatherSource, CalendarSource, EmailSource.",
        ]
    )
    return AttributedResponse(
        text="\n".join(lines),
        attribution=SOURCE_BACKED,
        sources=tuple(source.source_id for source in health),
        capability="source inspection",
    )
