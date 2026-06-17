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
        "You are summarizing source-provided evidence and interpretation for Olivaw.",
        "Olivaw is a presentation layer downstream of Prime Observer and Core Signal.",
        "Prime Observer owns factual evidence, telemetry, investigations, DNS analytics, and target-group observations.",
        "Core Signal owns interpretation, attribution assessment, confidence, uncertainty, evidence strength, and recommendations.",
        "You must not generate new findings, events, recommendations, confidence values, attribution, or causes.",
        "Only explain the supplied findings using concise operator-facing prose.",
        "Do not tell the operator what to do.",
        "If you mention a recommendation, attribute it to Core Signal.",
        "Return 3 to 6 sentences. Do not use bullets, headings, or markdown.",
    ]
)


def build_health_review_digest(dashboard: dict[str, object]) -> dict[str, object]:
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

    return {
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


def build_health_review_prompt(digest: dict[str, object]) -> str:
    return "\n".join(
        [
            "Create a concise Health Review from only the structured fields below.",
            "",
            "Rules:",
            "- Summarize, explain, restate, compare, and provide context only from these fields.",
            "- Do not create new events, findings, confidence values, recommendations, attribution, or causes.",
            "- Do not tell the operator what to do.",
            "- Do not use advisory wording such as should, must, need to, essential, warrants, or warranting.",
            "- Attribute any recommendation language to Core Signal.",
            "- If a value is absent, do not fill it in.",
            "- Return 3 to 6 sentences of operator-facing prose.",
            "",
            "Structured fields:",
            _format_digest(digest),
        ]
    )


def format_health_review_diagnostic(result: HealthReviewResult) -> str:
    lines = [
        "Olivaw Health Review",
        "",
        f"Status: {result.status}",
        f"Provider: {result.provider or 'not selected'}",
        f"Model: {result.model or 'not configured'}",
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
    if sentence_count < 1 or sentence_count > 8:
        return False

    lowered = text.lower()
    forbidden_phrases = (
        "i believe",
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

    digest_text = _format_digest(digest).lower()
    if _mentions_unsupplied_confidence(lowered, digest_text):
        return False
    if _mentions_unsupplied_recommendation(lowered, digest_text):
        return False
    if _mentions_unsupplied_attribution(lowered, digest_text):
        return False
    if _mentions_unattributed_advice(text):
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
    recommendation_terms = (
        "should",
        "must",
        "need to",
        "recommended action",
        "warrant",
        "warrants",
        "warranting",
        "essential",
    )
    if not any(term in text for term in recommendation_terms):
        return False
    return "recommendation" not in digest_text and "recommended" not in digest_text


def _mentions_unsupplied_attribution(text: str, digest_text: str) -> bool:
    attribution_terms = ("cox", "comcast", "isp", "dns outage", "router", "wi-fi")
    for term in attribution_terms:
        if term in text and term not in digest_text:
            return True
    return False


def _mentions_unattributed_advice(text: str) -> bool:
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
        if not any(term in lowered for term in advice_terms):
            continue
        if "core signal" not in lowered:
            return True
    return False


def _retry_guardrail_rewrite(
    model_provider: object,
    digest: dict[str, object],
) -> str:
    request = CompletionRequest(
        prompt="\n".join(
            [
                "Create a Health Review again with stricter boundaries.",
                "Use only the structured fields below.",
                "Return 3 to 5 sentences.",
                "Do not tell the operator what to do.",
                "Do not use advisory wording such as should, must, need to, essential, warrants, or warranting.",
                "If source data includes a recommendation, phrase it only as Core Signal metadata.",
                "",
                "Structured fields:",
                _format_digest(digest),
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
