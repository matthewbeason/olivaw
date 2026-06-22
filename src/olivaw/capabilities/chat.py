from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace

from olivaw.actions import IntentResolver, create_builtin_action_registry
from olivaw.assistant.attribution import (
    DERIVED,
    MODEL_KNOWLEDGE,
    MODEL_UNAVAILABLE,
    SOURCE_BACKED,
    AttributedResponse,
    UNKNOWN_OPERATIONAL_STATE,
    UNAVAILABLE_SOURCE_BACKED,
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
UNKNOWN_SOURCE_ANSWER = "I don't currently have a source that can answer that."
_ACTION_INTENT_RESOLVER = IntentResolver(create_builtin_action_registry())


@dataclass
class ChatCapability:
    name: str = "chat"
    description: str = "Minimal provider-routed chat placeholder."

    def run(self, prompt: str, config: OlivawConfig | None = None) -> str:
        return self.run_with_attribution(prompt, config=config).text

    def run_with_attribution(
        self, prompt: str, config: OlivawConfig | None = None
    ) -> AttributedResponse:
        started = time.perf_counter()
        resolved_config = config or load_config()
        if _is_capability_question(prompt):
            return _with_chat_metrics(
                AttributedResponse(
                text=capabilities_summary(),
                attribution=SOURCE_BACKED,
                sources=("capability-registry",),
                capability="capability inspection",
                provenance_label="Source",
                provenance_detail="Capability Registry",
                ),
                started,
                model_invoked=False,
            )

        if _is_source_question(prompt):
            return _with_chat_metrics(
                _source_status_response(resolved_config),
                started,
                model_invoked=False,
            )

        action = _action_suggestion_response(prompt)
        if action is not None:
            return _with_chat_metrics(action, started, model_invoked=False)

        grounded = _grounded_operational_response(prompt, resolved_config)
        if grounded is not None:
            return _with_chat_metrics(grounded, started, model_invoked=False)

        missing = _missing_capability_for_prompt(prompt)
        if missing:
            return _with_chat_metrics(
                _capability_unavailable_response(missing),
                started,
                model_invoked=False,
            )

        router = RouterProvider(resolved_config)
        prompt_started = time.perf_counter()
        system_prompt = build_chat_system_prompt()
        prompt_construction_duration_ms = _elapsed_ms(prompt_started)
        model_started = time.perf_counter()
        try:
            response = router.complete(
                CompletionRequest(
                    prompt=prompt,
                    system_prompt=system_prompt,
                )
            )
        except Exception as exc:
            return _with_chat_metrics(
                AttributedResponse(
                    text=_format_chat_failure(exc),
                    attribution=MODEL_UNAVAILABLE,
                    capability="model provider",
                    provenance_label="Unavailable",
                    provenance_detail="Model provider",
                ),
                started,
                model_invoked=True,
                prompt_construction_duration_ms=prompt_construction_duration_ms,
                model_request_duration_ms=_elapsed_ms(model_started),
            )
        return _with_chat_metrics(
            AttributedResponse(
                text=_sanitize_model_knowledge_response(
                    response.text,
                    protected_terms=_registered_provenance_terms(resolved_config),
                ),
                attribution=MODEL_KNOWLEDGE,
                capability="chat",
                provenance_label="Knowledge mode",
                provenance_detail="Model knowledge",
            ),
            started,
            model_invoked=True,
            prompt_construction_duration_ms=prompt_construction_duration_ms,
            model_request_duration_ms=response.request_duration_ms
            if response.request_duration_ms is not None
            else _elapsed_ms(model_started),
            ollama_total_duration_ms=response.ollama_total_duration_ms,
            ollama_load_duration_ms=response.ollama_load_duration_ms,
            ollama_prompt_eval_duration_ms=response.ollama_prompt_eval_duration_ms,
            ollama_eval_duration_ms=response.ollama_eval_duration_ms,
            prompt_eval_count=response.prompt_eval_count,
            eval_count=response.eval_count,
        )


def _format_chat_failure(exc: Exception) -> str:
    detail = str(exc).strip() or "provider failed without additional detail"
    return (
        "Chat provider unavailable: "
        f"{type(exc).__name__}: {detail}. {HEALTH_HINT}"
    )


def _action_suggestion_response(prompt: str) -> AttributedResponse | None:
    match = _ACTION_INTENT_RESOLVER.resolve(prompt)
    if match is None:
        return None
    return AttributedResponse(
        text="I can do that.",
        attribution=SOURCE_BACKED,
        sources=("action-registry",),
        capability="action suggestion",
        provenance_label="Source",
        provenance_detail="Action Registry",
        metrics={"matched_action_id": match.action_id},
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
    started = time.perf_counter()
    registry = create_default_registry(config)
    aggregate = registry.aggregate().as_dict()
    source_retrieval_duration_ms = _elapsed_ms(started)
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
                    provenance_label="Source",
                    provenance_detail="Weather",
                    metrics={
                        "source_retrieval_duration_ms": source_retrieval_duration_ms
                    },
                )
        message = str(source.get("message") or "").strip()
        return _source_unavailable_response(
            subject="weather conditions",
            required_source="weather data",
            sources=("weather",),
            detail=message or "Weather source data is not available right now.",
            metrics={"source_retrieval_duration_ms": source_retrieval_duration_ms},
        )
    return _missing_source_response(
        subject="weather conditions",
        required_source="weather data",
        metrics={"source_retrieval_duration_ms": source_retrieval_duration_ms},
    )


def _is_weather_question(prompt: str) -> bool:
    normalized = " ".join(prompt.lower().split())
    return any(
        phrase in normalized
        for phrase in ("weather", "forecast", "temperature", "temp", "rain")
    )


def _grounded_operational_response(
    prompt: str,
    config: OlivawConfig,
) -> AttributedResponse | None:
    if _is_weather_question(prompt):
        return _weather_response(prompt, config)
    if _is_network_question(prompt):
        return _network_response(config)
    limitation = _operational_limitation(prompt)
    if limitation is not None:
        subject, required_source = limitation
        return _missing_source_response(
            subject=subject,
            required_source=required_source,
        )
    return None


def _is_network_question(prompt: str) -> bool:
    normalized = " ".join(prompt.lower().split())
    network_phrases = (
        "network",
        "slowdown",
        "latency",
        "packet loss",
        "connectivity",
        "internet",
        "wan",
        "lan",
        "overnight",
        "outage",
    )
    return any(phrase in normalized for phrase in network_phrases)


def _network_response(config: OlivawConfig) -> AttributedResponse:
    started = time.perf_counter()
    aggregate = create_default_registry(config).aggregate().as_dict()
    source_retrieval_duration_ms = _elapsed_ms(started)
    prime = _source_by_id(aggregate, "prime_observer")
    core = _source_by_id(aggregate, "core_signal")
    prime_summary = _source_summary(prime)
    core_summary = _source_interpretation_summary(aggregate, source_id="core_signal")

    if core_summary and prime_summary:
        return AttributedResponse(
            text=f"{core_summary} Prime Observer reports: {prime_summary}",
            attribution=DERIVED,
            sources=("prime_observer", "core_signal"),
            capability="network status",
            provenance_label="Derived from",
            provenance_detail="Prime Observer + Core Signal",
            metrics={"source_retrieval_duration_ms": source_retrieval_duration_ms},
        )
    if prime_summary:
        return AttributedResponse(
            text=f"Prime Observer: {prime_summary}",
            attribution=SOURCE_BACKED,
            sources=("prime_observer",),
            capability="network status",
            provenance_label="Source",
            provenance_detail="Prime Observer",
            metrics={"source_retrieval_duration_ms": source_retrieval_duration_ms},
        )
    if core_summary:
        return AttributedResponse(
            text=f"Core Signal: {core_summary}",
            attribution=SOURCE_BACKED,
            sources=("core_signal",),
            capability="network status",
            provenance_label="Source",
            provenance_detail="Core Signal",
            metrics={"source_retrieval_duration_ms": source_retrieval_duration_ms},
        )

    unavailable_sources = tuple(
        source_id
        for source_id, source in (
            ("prime_observer", prime),
            ("core_signal", core),
        )
        if str(source.get("status") or "").strip() in {"unavailable", "error"}
    )
    if unavailable_sources:
        detail = _first_non_empty(
            str(prime.get("message") or "").strip(),
            str(core.get("message") or "").strip(),
        )
        return _source_unavailable_response(
            subject="network conditions",
            required_source="Prime Observer evidence or Core Signal interpretation",
            sources=unavailable_sources,
            detail=detail,
            metrics={"source_retrieval_duration_ms": source_retrieval_duration_ms},
        )
    return _missing_source_response(
        subject="network conditions",
        required_source="Prime Observer evidence or Core Signal interpretation",
        metrics={"source_retrieval_duration_ms": source_retrieval_duration_ms},
    )


def _operational_limitation(prompt: str) -> tuple[str, str] | None:
    normalized = " ".join(prompt.lower().split())
    patterns = (
        (
            ("disk", "storage"),
            ("usage", "utilization", "space", "free", "used", "capacity"),
            ("disk usage", "disk utilization"),
        ),
        (
            ("memory", "ram"),
            ("usage", "utilization", "using", "available", "free"),
            ("memory usage", "memory utilization"),
        ),
        (
            ("cpu",),
            ("usage", "utilization", "load"),
            ("cpu usage", "cpu utilization"),
        ),
        (
            ("uptime", "downtime", "outage", "outages"),
            (),
            ("uptime or outage status", "uptime or outage telemetry"),
        ),
        (
            ("monitoring", "monitored", "watching"),
            (),
            ("ongoing monitoring status", "monitoring telemetry"),
        ),
        (
            ("provider", "infrastructure"),
            ("status", "health", "monitoring"),
            ("provider status", "provider telemetry"),
        ),
    )
    for subject_tokens, qualifier_tokens, response in patterns:
        if not any(token in normalized for token in subject_tokens):
            continue
        if qualifier_tokens and not any(token in normalized for token in qualifier_tokens):
            continue
        return response
    return None


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
        attribution=UNKNOWN_OPERATIONAL_STATE,
        capability=capability,
        provenance_label="Knowledge mode",
        provenance_detail="Unknown operational state",
    )


def _planned_source_name(capability: str, planned_sources: tuple[str, ...]) -> str | None:
    normalized = capability.lower()
    for source in planned_sources:
        stem = source.removesuffix("Source").lower()
        if stem in normalized:
            return source
    return None


def _source_status_response(config: OlivawConfig) -> AttributedResponse:
    started = time.perf_counter()
    registry = create_default_registry(config)
    health = registry.health_all()
    source_retrieval_duration_ms = _elapsed_ms(started)
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
        provenance_label="Source",
        provenance_detail="Registered sources",
        metrics={"source_retrieval_duration_ms": source_retrieval_duration_ms},
    )


def _missing_source_response(
    *,
    subject: str,
    required_source: str,
    metrics: dict[str, object] | None = None,
) -> AttributedResponse:
    return AttributedResponse(
        text=(
            f"{UNKNOWN_SOURCE_ANSWER} "
            f"I would need a source that provides {required_source}."
        ),
        attribution=UNKNOWN_OPERATIONAL_STATE,
        capability=subject,
        provenance_label="Knowledge mode",
        provenance_detail="Unknown operational state",
        metrics=metrics or {},
    )


def _source_unavailable_response(
    *,
    subject: str,
    required_source: str,
    sources: tuple[str, ...],
    detail: str,
    metrics: dict[str, object] | None = None,
) -> AttributedResponse:
    detail_text = f" {detail}" if detail else ""
    return AttributedResponse(
        text=(
            f"The source exists but is currently unavailable. "
            f"I would need a source that provides {required_source}.{detail_text}"
        ),
        attribution=UNAVAILABLE_SOURCE_BACKED,
        sources=sources,
        capability=subject,
        provenance_label="Knowledge mode",
        provenance_detail="Unavailable source-backed state",
        metrics=metrics or {},
    )


def _source_by_id(aggregate: dict[str, object], source_id: str) -> dict[str, object]:
    for source in _source_dicts(aggregate.get("sources")):
        if source.get("source_id") == source_id:
            return source
    return {}


def _source_summary(source: dict[str, object]) -> str:
    if str(source.get("status") or "").strip() != "ok":
        return ""
    return _weather_summary_text(source)


def _source_interpretation_summary(
    aggregate: dict[str, object],
    *,
    source_id: str,
) -> str:
    for item in _source_dicts(aggregate.get("interpretation_items")):
        if item.get("source_id") != source_id:
            continue
        summary = str(item.get("summary") or "").strip()
        if summary:
            return summary
    return ""


def _first_non_empty(*values: str) -> str:
    for value in values:
        if value:
            return value
    return ""


def _join_source_labels(sources: tuple[str, ...]) -> str:
    labels = []
    for source_id in sources:
        labels.append(
            {
                "prime_observer": "Prime Observer",
                "core_signal": "Core Signal",
                "weather": "Weather",
            }.get(source_id, source_id.replace("_", " ").title())
        )
    return " + ".join(label for label in labels if label)


def _registered_provenance_terms(config: OlivawConfig) -> tuple[str, ...]:
    registry = create_default_registry(config)
    terms: list[str] = []
    for source in registry.list_sources():
        terms.extend(
            (
                source.source_id,
                source.source_id.replace("_", " "),
                source.source_id.replace("_", " ").title(),
                source.display_name,
                type(source).__name__,
            )
        )
    capability_registry = create_capability_registry(
        implemented_sources=tuple(source.source_id for source in registry.list_sources())
    )
    terms.extend(capability_registry.planned_sources)
    terms.extend(("Action Registry", "action-registry"))
    return tuple(dict.fromkeys(term for term in terms if term.strip()))


def _sanitize_model_knowledge_response(
    text: str,
    *,
    protected_terms: tuple[str, ...] = (),
) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped

    sanitized = stripped
    sanitized = re.sub(
        r"\s*\((?:via|from|source:|according to|derived from)\s+[^)]*\)",
        "",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(
        r"\s*\([^)]*\b(?:sourced|source|provider|model knowledge|"
        r"knowledge[- ]base|access to)[^)]*\)",
        "",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(
        r"(?im)^[ \t]*(?:source-backed fact|source backed fact|source fact|"
        r"source-backed|source backed|sourced-backed|sourced backed|"
        r"from registered sources|"
        r"according to registered sources)[ \t:]+",
        "",
        sanitized,
    )
    sanitized = _remove_generic_provenance_claims(sanitized)
    sanitized = re.sub(
        r"\b(?:source-backed|source backed|sourced-backed|sourced backed)\b\s*",
        "",
        sanitized,
        flags=re.IGNORECASE,
    )
    if protected_terms:
        term_pattern = "|".join(
            re.escape(term)
            for term in sorted(protected_terms, key=len, reverse=True)
            if term.strip()
        )
        sanitized = _remove_provenance_claims(sanitized, term_pattern)
        sanitized = _remove_residual_internal_source_names(sanitized, protected_terms)
    sanitized = re.sub(r"[ \t]+\n", "\n", sanitized)
    sanitized = re.sub(r"(?<=[.!?])(?=[A-Z])", " ", sanitized)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    sanitized = re.sub(r" {2,}", " ", sanitized)
    return sanitized.strip()


def _remove_generic_provenance_claims(text: str) -> str:
    sanitized = re.sub(
        r"(?ims)(?:^|(?<=[.!?])\s*)"
        r"[^.!?\n]*\b(?:no access to|do not have access to|don't have access to|"
        r"do not have direct access to|don't have direct access to|"
        r"does not have access to|has access to|have access to|"
        r"accessed this knowledge)\b"
        r"[^.!?\n]*\b(?:sources?|databases?|records?|knowledge[- ]bases?|"
        r"tools?|texts?|providers?|model information|registered sources?)\b"
        r"[^.!?\n]*[.!?]?\s*",
        "",
        text,
    )
    sanitized = re.sub(
        r"(?ims)(?:^|(?<=[.!?])\s*)"
        r"[^.!?\n]*\bno specific (?:sources?|tools?|providers?|databases?)\b"
        r"[^.!?\n]*[.!?]?\s*",
        "",
        sanitized,
    )
    sanitized = re.sub(
        r"(?ims)(?:^|(?<=[.!?])\s*)"
        r"[^.!?\n]*\b(?:generated|generate|referencing|reference|citing|cite|"
        r"from my knowledge[- ]base|from a knowledge[- ]base)\b"
        r"[^.!?\n]*\b(?:sources?|knowledge[- ]bases?|model knowledge)\b"
        r"[^.!?\n]*[.!?]?\s*",
        "",
        sanitized,
    )
    sanitized = re.sub(
        r"(?ims)(?:^|(?<=[.!?])\s*)"
        r"[^.!?\n]*\b(?:based on my model knowledge|from my model knowledge|"
        r"uses model knowledge|drawn from model knowledge|"
        r"based on model knowledge)\b[^.!?\n]*[.!?]?\s*",
        "",
        sanitized,
    )
    sanitized = re.sub(
        r"(?ims)(?:^|(?<=[.!?])\s*)"
        r"[^.!?\n]*\bgeneral information based on model knowledge\b"
        r"[^.!?\n]*[.!?]?\s*",
        "",
        sanitized,
    )
    sanitized = re.sub(
        r"(?ims)(?:^|(?<=[.!?])\s*)"
        r"[^.!?\n]*\bmy knowledge\b[^.!?\n]*\b"
        r"(?:based on|sourced|sources?|providers?|databases?)\b"
        r"[^.!?\n]*[.!?]?\s*",
        "",
        sanitized,
    )
    sanitized = re.sub(
        r"(?ims)(?:^|(?<=[.!?])\s*)"
        r"(?:this is|this answer is|this overview is)[^.!?\n]*\b"
        r"(?:source-backed|source backed|sourced-backed|sourced backed)\b"
        r"[^.!?\n]*[.!?]?\s*",
        "",
        sanitized,
    )
    return re.sub(
        r"(?ims)(?:^|(?<=[.!?])\s*)"
        r"[^.!?\n]*\b(?:rely on my knowledge|from my knowledge)\b"
        r"[^.!?\n]*[.!?]?\s*",
        "",
        sanitized,
    )


def _remove_provenance_claims(text: str, term_pattern: str) -> str:
    line_provenance = re.compile(
        rf"(?im)^[ \t]*(?:source|sources|via|derived from|attribution)"
        rf"[ \t:]+(?:[^\n]*\b(?:{term_pattern})\b[^\n]*)(?:\n|$)"
    )
    sanitized = line_provenance.sub("", text)
    leading_claims = (
        rf"\bAccording to\s+(?:the\s+)?(?:{term_pattern})[,:]?\s*",
        rf"\bBased on\s+(?:the\s+)?(?:{term_pattern})[,:]?\s*",
        rf"\bFrom\s+(?:the\s+)?(?:{term_pattern})[,:]?\s*",
        rf"\bUsing\s+(?:the\s+)?(?:{term_pattern})[,:]?\s*",
        rf"\b(?:{term_pattern})\s+(?:reports|report|says|states|indicates|shows|"
        rf"provided|provides|supplied|supplies|confirms|notes)(?:\s+that)?[,:]?\s*",
    )
    for pattern in leading_claims:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(
        rf"\s+\bvia\s+(?:the\s+)?(?:{term_pattern})\b",
        "",
        sanitized,
        flags=re.IGNORECASE,
    )
    return sanitized


def _remove_residual_internal_source_names(
    text: str,
    protected_terms: tuple[str, ...],
) -> str:
    sanitized = text
    residual_terms = [
        term
        for term in protected_terms
        if (
            term.endswith("Source")
            or "_" in term
            or "-" in term
            or term in {"Prime Observer", "Core Signal", "Action Registry"}
        )
    ]
    for term in sorted(residual_terms, key=len, reverse=True):
        sanitized = re.sub(
            rf"\b{re.escape(term)}\b",
            "",
            sanitized,
            flags=re.IGNORECASE,
        )
    return sanitized


def _with_chat_metrics(
    response: AttributedResponse,
    started: float,
    *,
    model_invoked: bool,
    **metrics: object,
) -> AttributedResponse:
    merged = {
        "total_request_duration_ms": _elapsed_ms(started),
        "time_to_first_token_ms": None,
        "model_invoked": model_invoked,
        "prompt_construction_duration_ms": metrics.pop(
            "prompt_construction_duration_ms", 0
        ),
        "source_retrieval_duration_ms": response.metrics.get(
            "source_retrieval_duration_ms", 0
        ),
    }
    merged.update(response.metrics)
    merged.update({key: value for key, value in metrics.items() if value is not None})
    return replace(response, metrics=merged)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
