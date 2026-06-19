from __future__ import annotations

from dataclasses import dataclass

from olivaw.actions.registry import ActionRegistry


@dataclass(frozen=True)
class IntentMatch:
    action_id: str


class IntentResolver:
    def __init__(self, registry: ActionRegistry) -> None:
        self._registry = registry

    def resolve(self, prompt: str) -> IntentMatch | None:
        normalized = _normalize(prompt)
        if not normalized:
            return None

        matched_action_id = (
            _match_refresh_health_review(normalized)
            or _match_refresh_sources(normalized)
            or _match_source_diagnostics(normalized)
            or _match_open_evidence_package(normalized)
            or _match_open_prime_observer(normalized)
        )
        if matched_action_id is None:
            return None
        if self._registry.get(matched_action_id) is None:
            return None
        return IntentMatch(action_id=matched_action_id)


def _match_refresh_health_review(normalized: str) -> str | None:
    if (
        _contains_any(normalized, "refresh", "update", "rerun", "regenerate")
        and "health review" in normalized
    ):
        return "refresh_health_review"
    return None


def _match_refresh_sources(normalized: str) -> str | None:
    if _contains_any(normalized, "refresh", "update", "reload", "rebuild") and (
        "source" in normalized or "sources" in normalized
    ):
        return "refresh_sources"
    return None


def _match_source_diagnostics(normalized: str) -> str | None:
    if (
        "diagnostic" in normalized
        or "diagnostics" in normalized
        or _contains_all(normalized, "source", "status")
    ):
        return "source_diagnostics"
    return None


def _match_open_evidence_package(normalized: str) -> str | None:
    if ("evidence" in normalized or "evidence package" in normalized) and _contains_any(
        normalized,
        "open",
        "show",
    ):
        return "open_evidence_package"
    return None


def _match_open_prime_observer(normalized: str) -> str | None:
    if "prime observer" in normalized and _contains_any(normalized, "open", "show"):
        return "open_prime_observer"
    return None


def _normalize(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def _contains_all(normalized: str, *phrases: str) -> bool:
    return all(phrase in normalized for phrase in phrases)


def _contains_any(normalized: str, *phrases: str) -> bool:
    return any(phrase in normalized for phrase in phrases)
