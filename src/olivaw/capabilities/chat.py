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
        resolved_config = config or load_config()
        if _is_capability_question(prompt):
            return AttributedResponse(
                text=capabilities_summary(),
                attribution=SOURCE_BACKED,
                sources=("capability-registry",),
                capability="capability inspection",
            )

        weather = _weather_response(prompt, resolved_config)
        if weather is not None:
            return weather

        missing = _missing_capability_for_prompt(prompt)
        if missing:
            return _capability_unavailable_response(missing)

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


def _weather_response(
    prompt: str,
    config: OlivawConfig,
) -> AttributedResponse | None:
    if not _is_weather_question(prompt):
        return None
    registry = create_default_registry(config)
    aggregate = registry.aggregate().as_dict()
    for source in _source_dicts(aggregate.get("sources")):
        if source.get("source_id") != "weather":
            continue
        status = str(source.get("status") or "unknown")
        if status == "ok":
            summary = _weather_summary_text(source)
            if summary:
                location = str(source.get("source_name") or "Weather")
                return AttributedResponse(
                    text=f"{location}: {summary}",
                    attribution=SOURCE_BACKED,
                    sources=("weather",),
                    capability="weather lookup",
                )
        message = str(source.get("message") or "").strip()
        details = (
            f" Weather source status: {message}."
            if message
            else " Weather source data is not available right now."
        )
        return AttributedResponse(
            text=(
                "I do not currently have weather context available from Olivaw sources yet."
                f"{details}"
            ),
            attribution=CAPABILITY_UNAVAILABLE,
            sources=("weather",),
            capability="weather source",
        )
    return _capability_unavailable_response("weather source")


def _is_weather_question(prompt: str) -> bool:
    normalized = " ".join(prompt.lower().split())
    return any(
        phrase in normalized
        for phrase in ("weather", "forecast", "temperature", "temp", "rain")
    )


def _source_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _weather_summary_text(source: dict[str, object]) -> str:
    for item in _source_dicts(source.get("summary_items")):
        summary = str(item.get("summary") or "").strip()
        if summary:
            return summary
    return ""


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
            "Planned sources are not implemented yet: CalendarSource, EmailSource.",
        ]
    )
    return AttributedResponse(
        text="\n".join(lines),
        attribution=SOURCE_BACKED,
        sources=tuple(source.source_id for source in health),
        capability="source inspection",
    )
