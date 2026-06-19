from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlparse

from olivaw.actions.models import (
    ActionDefinition,
    ActionExecutionContext,
    ActionRequest,
    ActionResult,
)
from olivaw.actions.registry import ActionRegistry


def create_builtin_action_registry() -> ActionRegistry:
    registry = ActionRegistry()
    for definition in (
        ActionDefinition(
            action_id="refresh_health_review",
            label="Refresh Health Review",
            description="Generate and cache a fresh Health Review from current sources.",
            category="review",
            risk_level="local_state_change",
            requires_confirmation=True,
            handler=_refresh_health_review,
        ),
        ActionDefinition(
            action_id="refresh_sources",
            label="Refresh Sources",
            description="Rebuild the normalized source aggregate from configured sources.",
            category="sources",
            risk_level="safe_read",
            requires_confirmation=True,
            handler=_refresh_sources,
        ),
        ActionDefinition(
            action_id="open_evidence_package",
            label="Open Evidence Package",
            description="Return the existing Prime Observer evidence package link.",
            category="evidence",
            risk_level="safe_read",
            requires_confirmation=True,
            handler=_open_evidence_package,
        ),
        ActionDefinition(
            action_id="open_prime_observer",
            label="Open Prime Observer",
            description="Return the configured Prime Observer URL.",
            category="evidence",
            risk_level="safe_read",
            requires_confirmation=True,
            handler=_open_prime_observer,
        ),
        ActionDefinition(
            action_id="source_diagnostics",
            label="Source Diagnostics",
            description="Summarize normalized source status.",
            category="sources",
            risk_level="safe_read",
            requires_confirmation=True,
            handler=_source_diagnostics,
        ),
    ):
        registry.register(definition)
    return registry


def _refresh_health_review(
    request: ActionRequest,
    context: ActionExecutionContext,
) -> ActionResult:
    started_at = _started_at(request)
    summary = context.refresh_health_review()
    status = str(summary.get("status") or "unknown")
    message = "Health Review refreshed."
    if status and status != "available":
        message = f"Health Review refresh completed with status: {status}."
    return _result(True, message, started_at, summary)


def _refresh_sources(
    request: ActionRequest,
    context: ActionExecutionContext,
) -> ActionResult:
    started_at = _started_at(request)
    summary = context.refresh_sources()
    source_count = int(summary.get("source_count") or 0)
    ok_count = int(summary.get("ok_count") or 0)
    return _result(
        True,
        f"Sources refreshed: {ok_count}/{source_count} sources are ok.",
        started_at,
        summary,
    )


def _open_evidence_package(
    request: ActionRequest,
    context: ActionExecutionContext,
) -> ActionResult:
    started_at = _started_at(request)
    package = context.evidence_package()
    href = str(package.get("href") or "").strip()
    if href:
        return _result(
            True,
            "Evidence package is available.",
            started_at,
            package,
        )
    return _result(
        False,
        str(
            package.get("unavailable_reason")
            or "No HTTP-backed evidence package is available."
        ),
        started_at,
        package,
    )


def _open_prime_observer(
    request: ActionRequest,
    context: ActionExecutionContext,
) -> ActionResult:
    started_at = _started_at(request)
    base_url = str(context.config.prime_observer.base_url or "").strip()
    if _is_http_url(base_url):
        return _result(
            True,
            "Prime Observer URL is available.",
            started_at,
            {"href": base_url, "label": "Open Prime Observer"},
        )
    return _result(
        False,
        "Prime Observer URL is not configured with an HTTP target.",
        started_at,
        {"href": "", "label": "Open Prime Observer"},
    )


def _source_diagnostics(
    request: ActionRequest,
    context: ActionExecutionContext,
) -> ActionResult:
    started_at = _started_at(request)
    diagnostics = context.source_diagnostics()
    source_count = int(diagnostics.get("source_count") or 0)
    return _result(
        True,
        f"Source diagnostics ready for {source_count} sources.",
        started_at,
        diagnostics,
    )


def _started_at(request: ActionRequest) -> datetime:
    return request.executed_at or datetime.now(timezone.utc)


def _result(
    success: bool,
    message: str,
    started_at: datetime,
    metadata: dict[str, object],
) -> ActionResult:
    return ActionResult(
        success=success,
        message=message,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
        metadata=metadata,
    )


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
