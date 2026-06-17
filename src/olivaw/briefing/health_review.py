from __future__ import annotations

import os
import re
import socket
import time
import urllib.error
from dataclasses import dataclass
from typing import Literal

from olivaw.config import OlivawConfig
from olivaw.models import CompletionRequest
from olivaw.providers.ollama import OllamaProvider

UNAVAILABLE_TEXT = "Health review unavailable"

HealthReviewStatus = Literal[
    "available",
    "disabled",
    "no_source_data",
    "provider_unavailable",
    "model_unavailable",
    "timeout",
    "generation_failed",
    "guardrail_rejected",
]


@dataclass(frozen=True)
class HealthReviewResult:
    text: str
    status: HealthReviewStatus
    reason: str = ""
    model: str = ""
    provider: str = ""
    latency_ms: int | None = None
    guardrail_rejected: bool = False

    @property
    def available(self) -> bool:
        return self.status == "available"


def generate_health_review(
    dashboard: dict[str, object],
    *,
    config: OlivawConfig,
    provider: object | None = None,
) -> HealthReviewResult:
    started = time.monotonic()
    model = config.local.model
    if not _health_review_enabled():
        return _unavailable(
            "disabled",
            "Health Review is disabled by OLIVAW_HEALTH_REVIEW_ENABLED.",
            model=model,
            latency_ms=_elapsed_ms(started),
        )

    digest = build_health_review_digest(dashboard)
    if not _digest_has_source_data(digest):
        return _unavailable(
            "no_source_data",
            "no source data available.",
            model=model,
            latency_ms=_elapsed_ms(started),
        )

    model_provider = provider or OllamaProvider(
        config.local,
        timeout=2.0,
        complete_timeout=45.0,
    )
    provider_name = str(getattr(model_provider, "name", "local model"))
    try:
        health = model_provider.health()
    except Exception as exc:
        return _unavailable(
            "provider_unavailable",
            _safe_failure_reason("Ollama health check failed", exc),
            provider=provider_name,
            model=model,
            latency_ms=_elapsed_ms(started),
        )

    provider_name = health.name
    model = health.model or model
    if not health.available:
        return _unavailable(
            "provider_unavailable",
            health.message,
            provider=provider_name,
            model=model,
            latency_ms=_elapsed_ms(started),
        )

    missing_model = _missing_model_reason(model_provider, model)
    if missing_model:
        return _unavailable(
            "model_unavailable",
            missing_model,
            provider=provider_name,
            model=model,
            latency_ms=_elapsed_ms(started),
        )

    request = CompletionRequest(
        prompt=build_health_review_prompt(digest),
        system_prompt=HEALTH_REVIEW_SYSTEM_PROMPT,
    )
    try:
        response = model_provider.complete(request)
    except TimeoutError as exc:
        return _unavailable(
            "timeout",
            _safe_failure_reason("local model timed out", exc),
            provider=provider_name,
            model=model,
            latency_ms=_elapsed_ms(started),
        )
    except (socket.timeout, TimeoutError) as exc:
        return _unavailable(
            "timeout",
            _safe_failure_reason("local model timed out", exc),
            provider=provider_name,
            model=model,
            latency_ms=_elapsed_ms(started),
        )
    except urllib.error.HTTPError as exc:
        status, reason = _http_generation_status(exc, model)
        return _unavailable(
            status,
            reason,
            provider=provider_name,
            model=model,
            latency_ms=_elapsed_ms(started),
        )
    except Exception as exc:
        return _unavailable(
            "generation_failed",
            _safe_failure_reason("local model generation failed", exc),
            provider=provider_name,
            model=model,
            latency_ms=_elapsed_ms(started),
        )

    provider_name = response.provider or provider_name
    model = response.model or model
    cleaned = _clean_review_text(response.text)
    if not cleaned:
        return _unavailable(
            "generation_failed",
            f"Ollama is reachable, but model {model} returned an empty response.",
            provider=provider_name,
            model=model,
            latency_ms=_elapsed_ms(started),
        )

    if not _review_is_safe(cleaned, digest):
        retry_text = _retry_guardrail_rewrite(model_provider, digest)
        if retry_text:
            return HealthReviewResult(
                text=retry_text,
                status="available",
                provider=provider_name,
                model=model,
                latency_ms=_elapsed_ms(started),
            )
        return _unavailable(
            "guardrail_rejected",
            "generated response was rejected by guardrails.",
            provider=provider_name,
            model=model,
            latency_ms=_elapsed_ms(started),
            guardrail_rejected=True,
        )

    return HealthReviewResult(
        text=cleaned,
        status="available",
        provider=provider_name,
        model=model,
        latency_ms=_elapsed_ms(started),
    )


HEALTH_REVIEW_SYSTEM_PROMPT = "\n".join(
    [
        "You are speaking directly to the operator as an experienced assistant.",
        "Lead with the answer: current state first, then recent history, uncertainty, and why the operator should care.",
        "Use conversational, voice-friendly language that can be spoken aloud naturally.",
        "Start with the current state; do not use a preamble.",
        "Say this matters because rather than you should care because.",
        "Do not say the operator needs to care; explain why it matters.",
        "Do not use we, our, or my; the assistant explains findings but does not own them.",
        "If there is no current condition needing attention, say that plainly instead of saying sources report it.",
        "Avoid source bookkeeping, internal architecture, raw metadata narration, counts, bullets, headings, and markdown.",
        "Do not mention source system names unless attribution is required for clarity.",
        "Weather is optional external context, not a recommendation or alert source.",
        "Do not invent weather alerts or safety recommendations.",
        "Do not discuss confidence or evidence strength directly; use those fields only to choose careful wording.",
        "You must not generate new findings, events, recommendations, confidence values, attribution, or causes.",
        "Only restate recommendations that are explicitly supplied.",
        "Return 2 to 4 concise sentences.",
    ]
)


def build_health_review_digest(dashboard: dict[str, object]) -> dict[str, object]:
    aggregate_context = _aggregate_health_review_context(dashboard)
    explanation = _dict_value(dashboard.get("core_signal_explanation"))
    events = _dict_list(dashboard.get("core_signal_events"))
    first_event = events[0] if events else {}
    attribution_assessment = _first_mapping(
        dashboard.get("attribution_assessment"),
        explanation.get("attribution_assessment"),
        first_event.get("attribution_assessment"),
    )
    evidence_strength = _first_mapping(
        dashboard.get("evidence_strength"),
        explanation.get("evidence_strength"),
        first_event.get("evidence_strength"),
    )

    digest = {
        "current_system_status": {
            "label": str(dashboard.get("current_status_label") or "").strip(),
            "explanation": str(dashboard.get("status_explanation") or "").strip(),
        },
        "prime_observer": {
            "current_attribution": _strings(dashboard.get("network_status"), limit=3),
            "target_group_summaries": _strings(dashboard.get("what_we_know"), limit=4),
            "noticeability": _strings(dashboard.get("worth_knowing"), limit=3),
            "investigation_counts": _investigation_count_summary(dashboard),
            "dns_summary": _strings(dashboard.get("dns_activity"), limit=3),
        },
        "core_signal": {
            "summary": _first_text(
                explanation.get("summary"),
                first_event.get("summary"),
                dashboard.get("executive_summary"),
            ),
            "why": _first_text(explanation.get("why"), first_event.get("why")),
            "confidence": _first_text(
                explanation.get("confidence"),
                first_event.get("confidence"),
            ),
            "confidence_reason": _first_text(
                explanation.get("confidence_reason"),
                first_event.get("confidence_reason"),
            ),
            "uncertainties": _strings(dashboard.get("uncertainty_items"), limit=5),
            "attribution_assessment": attribution_assessment,
            "evidence_strength": evidence_strength,
            "recommendation_trace": _strings_from_trace(
                explanation.get("recommendation_trace")
                or first_event.get("recommendation_trace"),
                limit=5,
            ),
        },
    }
    if aggregate_context:
        digest["aggregated_sources"] = aggregate_context
    return digest


def build_health_review_prompt(digest: dict[str, object]) -> str:
    return "\n".join(
        [
            "Create a concise Health Review from only the structured fields below.",
            "",
            "Rules:",
            "- Speak to the operator directly.",
            "- Lead with the current state.",
            "- Do not start with a preamble such as here is a review or I want to bring you up to speed.",
            "- Use 2 to 4 sentences, with a hard maximum of 4 sentences.",
            "- Prefer natural wording that can be spoken aloud.",
            "- Answer what is happening now, what happened recently, what remains uncertain, and whether the operator needs to care.",
            "- Prefer this matters because over you should care because.",
            "- Do not say the operator needs to care; explain why it matters.",
            "- Do not use we, our, or my.",
            "- Do not narrate field labels; say healthy now instead of labeled as Healthy now.",
            "- Summarize, explain, restate, compare, and provide context only from these fields.",
            "- Do not create new events, findings, confidence values, recommendations, attribution, or causes.",
            "- Only restate recommendation language that is explicitly present in the fields.",
            "- Weather facts are external context only; do not invent weather alerts or safety recommendations.",
            "- Avoid source bookkeeping language such as Prime Observer reports, Core Signal reports, or investigation counts show.",
            "- Do not describe internal architecture or repeat raw metadata labels.",
            "- Do not discuss confidence or evidence strength directly; use them only to tune wording.",
            "- If a value is absent, do not fill it in.",
            "",
            "Good example:",
            "Things look healthy right now. There were two slowdown periods earlier, but they appear to have cleared. It is not yet clear whether this was provider congestion or transient routing instability. You probably do not need to care unless users noticed symptoms during that window.",
            "",
            "Bad example:",
            "Prime Observer reports mixed evidence with moderate evidence strength. Investigation counts show sustained degradation, so the operator should investigate the root cause.",
            "",
            "Structured fields:",
            _format_operator_digest(digest),
        ]
    )


def _format_operator_digest(digest: dict[str, object]) -> str:
    aggregate = _dict_value(digest.get("aggregated_sources"))
    neutral_digest = {
        "current_state": digest.get("current_system_status"),
        "observed_facts": {
            key: value
            for key, value in _dict_value(digest.get("prime_observer")).items()
            if key not in {"investigation_counts", "dns_summary"}
        },
        "interpreted_findings": digest.get("core_signal"),
    }
    if aggregate:
        neutral_digest["aggregated_context"] = aggregate
    return _format_digest(neutral_digest)


def format_health_review_diagnostic(result: HealthReviewResult) -> str:
    accepted = "yes" if result.available else "no"
    rejected = "yes" if result.guardrail_rejected else "no"
    lines = [
        "Olivaw Health Review",
        "",
        f"Status: {result.status}",
        f"Provider: {result.provider or 'not selected'}",
        f"Model: {result.model or 'not configured'}",
        f"Accepted: {accepted}",
        f"Rejected: {rejected}",
        f"Reason: {result.reason or 'ok'}",
    ]
    if result.latency_ms is not None:
        lines.append(f"Latency: {result.latency_ms} ms")
    lines.extend(["", "Review:", result.text])
    return "\n".join(lines)


def _format_digest(value: object, *, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if item in ("", [], {}, None):
                continue
            if isinstance(item, dict):
                nested = _format_digest(item, indent=indent + 2)
                if nested:
                    lines.append(f"{prefix}{key}:")
                    lines.append(nested)
            elif isinstance(item, list):
                lines.append(f"{prefix}{key}:")
                for entry in item:
                    lines.append(f"{prefix}  - {entry}")
            else:
                lines.append(f"{prefix}{key}: {item}")
        return "\n".join(lines)
    return str(value)


def _health_review_enabled() -> bool:
    value = os.environ.get("OLIVAW_HEALTH_REVIEW_ENABLED")
    if value is None:
        return True
    return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _digest_has_source_data(digest: dict[str, object]) -> bool:
    prime = _dict_value(digest.get("prime_observer"))
    core = _dict_value(digest.get("core_signal"))
    aggregate = _dict_value(digest.get("aggregated_sources"))
    source_values: list[object] = [
        prime.get("current_attribution"),
        prime.get("target_group_summaries"),
        prime.get("noticeability"),
        prime.get("investigation_counts"),
        prime.get("dns_summary"),
        core.get("summary"),
        core.get("why"),
        core.get("confidence"),
        core.get("confidence_reason"),
        core.get("uncertainties"),
        core.get("attribution_assessment"),
        core.get("evidence_strength"),
        core.get("recommendation_trace"),
        aggregate.get("facts"),
        aggregate.get("interpretation_items"),
        aggregate.get("actions"),
    ]
    return any(_meaningful_source_value(value) for value in source_values)


def _meaningful_source_value(value: object) -> bool:
    if isinstance(value, dict):
        return any(_meaningful_source_value(item) for item in value.values())
    if isinstance(value, list):
        return any(_meaningful_source_value(item) for item in value)
    text = str(value or "").strip().lower()
    if not text:
        return False
    unavailable_markers = (
        "no source-backed facts or observations are available",
        "no explicit uncertainty information was provided",
        "no specific recommendation is available",
        "sources do not report a condition needing attention",
    )
    return text not in unavailable_markers


def _missing_model_reason(provider: object, model: str) -> str:
    models_method = getattr(provider, "models", None)
    if not callable(models_method):
        return ""
    try:
        models = tuple(str(item) for item in models_method())
    except Exception:
        return ""
    if not model or model in models:
        return ""
    available = ", ".join(models) or "none"
    return f"Ollama is reachable, but model {model} is not installed. Installed models: {available}."


def _review_is_safe(text: str, digest: dict[str, object]) -> bool:
    if not text:
        return False
    sentence_count = len(_sentences(text))
    if sentence_count < 1 or sentence_count > 4:
        return False

    lowered = text.lower()
    forbidden_phrases = (
        "i believe",
        "i'd like",
        "i would like",
        "bring you up to speed",
        "current health review",
        "we observed",
        "we detected",
        "our recommendation",
        "our current",
        "you should care",
        "operator should care",
        "should care because",
        "operator needs to care",
        "needs to care because",
        "matter matters",
        "this matter matters",
        "could have resulted",
        "could lead to",
        "may have caused",
        "causing possible",
        "be prepared to take action",
        "likely due to",
        "likely related",
        "significantly impact",
        "significant impact on user experience",
        "can have a significant impact",
        "require intervention",
        "requires intervention",
        "underlying network problem",
        "underlying network problems",
        "if not addressed promptly",
        "decreased productivity",
        "will be repeated",
        "root cause",
        "further investigation",
        "investigate further",
        "should investigate",
        "probably should investigate",
        "recommended that you investigate",
        "investigation should",
        "should be performed",
        "need for investigation",
        "can lead to user impact",
        "user impact is unlikely",
        "in order",
        "definitely",
        "was responsible",
        "is responsible",
        "should reboot",
        "reboot the router",
        "restart the router",
        "confidence should",
        "new recommendation",
        "new finding",
    )
    if any(phrase in lowered for phrase in forbidden_phrases):
        return False
    if re.search(r"\bhere(?:\s+is|'s)\b", lowered):
        return False
    if _uses_source_bookkeeping(lowered):
        return False
    if _narrates_raw_metadata(lowered):
        return False

    digest_text = _format_digest(digest).lower()
    if _mentions_unsupplied_confidence(lowered, digest_text):
        return False
    if _mentions_unsupplied_recommendation(lowered, digest_text):
        return False
    if _mentions_unsupplied_attribution(lowered, digest_text):
        return False
    if _mentions_unattributed_advice(text, digest_text):
        return False
    if _contradicts_supplied_slowdown(lowered, digest_text):
        return False
    return True


def _mentions_unsupplied_confidence(text: str, digest_text: str) -> bool:
    if "confidence" not in text:
        return False
    for value in ("high", "medium", "low", "moderate"):
        if value in text and value not in digest_text:
            return True
    return False


def _mentions_unsupplied_recommendation(text: str, digest_text: str) -> bool:
    recommendation_text = (
        text.replace("do not need to care", "")
        .replace("probably do not need to care", "")
        .replace("does not need to care", "")
        .replace("doesn't need to care", "")
    )
    recommendation_terms = (
        "should",
        "must",
        "need to",
        "recommended action",
        "warrant",
        "warrants",
        "warranting",
    )
    if not any(term in recommendation_text for term in recommendation_terms):
        return False
    return not _digest_has_supported_recommendation(digest_text)


def _uses_source_bookkeeping(text: str) -> bool:
    source_bookkeeping_terms = (
        "prime observer",
        "core signal",
        "investigation count",
        "investigation counts",
        "reports that",
        "reported that",
        "reports no",
        "reported no",
        "reported by sources",
        "sources not reporting",
        "sources do not report",
        "sources reported",
        "source not reporting",
        "source does not report",
        "detected by",
    )
    return any(term in text for term in source_bookkeeping_terms)


def _narrates_raw_metadata(text: str) -> bool:
    raw_metadata_terms = (
        "confidence is",
        "confidence level",
        "medium confidence",
        "low confidence",
        "high confidence",
        "evidence strength",
        "attribution assessment",
        "recommendation trace",
        "current_state",
        "observed_facts",
        "interpreted_findings",
        "observed facts",
        "interpreted findings",
        "noticeability mentions",
        "mentions that",
        "labeled as",
        "prevent user impact",
        "prevent potential user impact",
        "metadata",
        "essential",
        "warrant investigation",
        "warrants investigation",
        "warranting investigation",
        "current lan/wan state",
        "lan/wan state:",
        "current state:",
    )
    return any(term in text for term in raw_metadata_terms)


def _mentions_unsupplied_attribution(text: str, digest_text: str) -> bool:
    attribution_terms = ("cox", "comcast", "isp", "dns outage", "router", "wi-fi")
    for term in attribution_terms:
        if term in text and term not in digest_text:
            return True
    return False


def _mentions_unattributed_advice(text: str, digest_text: str) -> bool:
    advice_terms = (
        "should",
        "must",
        "need to",
        "essential",
        "warrant",
        "warrants",
        "warranting",
    )
    for sentence in _sentences(text):
        lowered = sentence.lower()
        if (
            "do not need to care" in lowered
            or "probably do not need to care" in lowered
            or "does not need to care" in lowered
            or "doesn't need to care" in lowered
        ):
            continue
        if not any(term in lowered for term in advice_terms):
            continue
        if not _digest_has_supported_recommendation(digest_text):
            return True
    return False


def _digest_has_supported_recommendation(digest_text: str) -> bool:
    if (
        "recommendation" not in digest_text
        and "recommended" not in digest_text
        and "recommend" not in digest_text
    ):
        return False
    unsupported_markers = (
        "no specific recommendation is available",
        "no recommendation",
        "not recommended",
    )
    return not any(marker in digest_text for marker in unsupported_markers)


def _contradicts_supplied_slowdown(text: str, digest_text: str) -> bool:
    if "sustained slowdown" not in digest_text:
        return False
    contradiction_terms = (
        "no sustained slowdown has been reported",
        "no sustained slowdowns have been reported",
        "no sustained slowdown was reported",
        "no sustained slowdowns were reported",
    )
    return any(term in text for term in contradiction_terms)


def _retry_guardrail_rewrite(
    model_provider: object,
    digest: dict[str, object],
) -> str:
    request = CompletionRequest(
        prompt="\n".join(
            [
                "Create a Health Review again with stricter boundaries.",
                "Use only the structured fields below.",
                "Return 2 to 4 concise sentences.",
                "Speak directly to the operator and lead with the current state.",
                "Do not mention source system names, internal architecture, confidence, evidence strength, or investigation counts.",
                "Only restate recommendations that are explicitly supplied.",
                "",
                "Structured fields:",
                _format_operator_digest(digest),
            ]
        ),
        system_prompt=HEALTH_REVIEW_SYSTEM_PROMPT,
    )
    complete = getattr(model_provider, "complete", None)
    if not callable(complete):
        return ""
    try:
        response = complete(request)
    except Exception:
        return ""
    cleaned = _clean_review_text(response.text)
    if cleaned and _review_is_safe(cleaned, digest):
        return cleaned
    return ""


def _http_generation_status(
    exc: urllib.error.HTTPError,
    model: str,
) -> tuple[HealthReviewStatus, str]:
    message = _http_error_message(exc)
    if exc.code == 404 or "model" in message.lower() and "not found" in message.lower():
        return (
            "model_unavailable",
            f"Ollama is reachable, but model {model} is unavailable. {message}".strip(),
        )
    return (
        "generation_failed",
        f"Ollama generation failed with HTTP {exc.code}. {message}".strip(),
    )


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8")
    except Exception:
        return str(exc)
    return _clean_review_text(raw) or str(exc)


def _safe_failure_reason(prefix: str, exc: Exception) -> str:
    detail = str(exc).strip()
    if not detail:
        return f"{prefix}."
    return f"{prefix}: {detail}."


def _unavailable(
    status: HealthReviewStatus,
    reason: str,
    *,
    provider: str = "",
    model: str = "",
    latency_ms: int | None = None,
    guardrail_rejected: bool = False,
) -> HealthReviewResult:
    return HealthReviewResult(
        text=f"{UNAVAILABLE_TEXT}: {reason}",
        status=status,
        reason=reason,
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        guardrail_rejected=guardrail_rejected,
    )


def _clean_review_text(text: str) -> str:
    lines = [line.strip().removeprefix("- ").strip() for line in text.splitlines()]
    cleaned = " ".join(line for line in lines if line)
    return re.sub(r"\s+", " ", cleaned).strip()


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))


def _sentences(text: str) -> list[str]:
    return [
        sentence for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence
    ]


def _strings(value: object, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result[:limit]


def _aggregate_health_review_context(
    dashboard: dict[str, object],
) -> dict[str, object]:
    aggregate = _dict_value(dashboard.get("source_aggregate"))
    context = _dict_value(aggregate.get("health_review_context"))
    if not context:
        return {}

    facts = _aggregate_summaries(context.get("facts"), limit=6)
    interpretations = _aggregate_summaries(
        context.get("interpretation_items"),
        limit=5,
    )
    actions = _aggregate_summaries(context.get("actions"), limit=4)
    references = _aggregate_reference_summaries(context.get("references"), limit=4)
    result = {
        "facts": facts,
        "interpretation_items": interpretations,
        "actions": actions,
        "references": references,
    }
    return {
        key: value
        for key, value in result.items()
        if _meaningful_source_value(value)
    }


def _aggregate_summaries(value: object, *, limit: int) -> list[str]:
    result: list[str] = []
    for item in _dict_list(value):
        summary = _first_text(item.get("summary"), item.get("title"))
        if summary and summary not in result:
            result.append(summary)
    return result[:limit]


def _aggregate_reference_summaries(value: object, *, limit: int) -> list[str]:
    result: list[str] = []
    for item in _dict_list(value):
        label = _first_text(item.get("label"), "Reference")
        target = _first_text(item.get("target"))
        if not target:
            continue
        summary = f"{label}: {target}"
        if summary not in result:
            result.append(summary)
    return result[:limit]


def _dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict_value(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _first_mapping(*values: object) -> dict[str, str]:
    for value in values:
        if not isinstance(value, dict):
            continue
        result = {
            key: str(value.get(key) or "").strip()
            for key in ("value", "confidence", "reason")
            if str(value.get(key) or "").strip()
        }
        if result:
            return result
    return {}


def _first_text(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _strings_from_trace(value: object, *, limit: int) -> list[str]:
    trace = _dict_list(value)
    result: list[str] = []
    for step in trace:
        stage = str(step.get("stage") or "").strip()
        detail = str(step.get("detail") or "").strip()
        if stage and detail:
            result.append(f"{stage}: {detail}")
        elif detail:
            result.append(detail)
    return result[:limit]


def _investigation_count_summary(dashboard: dict[str, object]) -> str:
    investigations = _dict_list(dashboard.get("prime_investigations"))
    events = _dict_list(dashboard.get("core_signal_events"))
    pieces = []
    if investigations:
        pieces.append(f"{len(investigations)} Prime Observer investigation artifact(s)")
    if events:
        pieces.append(f"{len(events)} interpreted Core Signal event(s)")
    return "; ".join(pieces)
