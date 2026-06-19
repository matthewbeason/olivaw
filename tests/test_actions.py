from __future__ import annotations

from datetime import datetime, timezone

import pytest

from olivaw.actions import (
    ActionDefinition,
    ActionExecutionContext,
    ActionHistory,
    ActionRegistry,
    ActionRequest,
    ActionResult,
    create_builtin_action_registry,
    execute_action,
)
from olivaw.config import OlivawConfig, PrimeObserverSourceConfig


def test_action_registry_registers_lists_and_gets_actions():
    registry = ActionRegistry()
    definition = _definition("demo")

    registry.register(definition)

    assert registry.get("demo") is definition
    assert registry.get("missing") is None
    assert registry.list_actions() == (definition,)


def test_action_registry_rejects_duplicate_action_ids():
    registry = ActionRegistry()
    registry.register(_definition("demo"))

    with pytest.raises(ValueError, match="demo"):
        registry.register(_definition("demo"))


def test_builtin_registry_contains_initial_safe_operator_actions():
    registry = create_builtin_action_registry()
    action_ids = {action.action_id for action in registry.list_actions()}

    assert action_ids == {
        "refresh_health_review",
        "refresh_sources",
        "open_evidence_package",
        "open_prime_observer",
        "source_diagnostics",
    }
    assert all(
        action.risk_level in {"safe_read", "local_state_change"}
        for action in registry.list_actions()
    )
    assert all(action.requires_confirmation for action in registry.list_actions())


def test_refresh_health_review_action_returns_result_shape():
    result = execute_action(
        create_builtin_action_registry(),
        ActionRequest(action_id="refresh_health_review"),
        _context(
            refresh_health_review=lambda: {
                "status": "available",
                "provider": "fake-local",
                "model": "fake-model",
            }
        ),
    )

    assert result.success is True
    assert result.message == "Health Review refreshed."
    assert result.metadata["status"] == "available"
    assert result.started_at <= result.completed_at


def test_refresh_sources_action_summarizes_source_status():
    result = execute_action(
        create_builtin_action_registry(),
        ActionRequest(action_id="refresh_sources"),
        _context(
            refresh_sources=lambda: {
                "source_count": 2,
                "ok_count": 1,
                "statuses": {"ok": 1, "unavailable": 1},
            }
        ),
    )

    assert result.success is True
    assert result.message == "Sources refreshed: 1/2 sources are ok."
    assert result.metadata["statuses"] == {"ok": 1, "unavailable": 1}


def test_source_diagnostics_action_returns_normalized_summary():
    result = execute_action(
        create_builtin_action_registry(),
        ActionRequest(action_id="source_diagnostics"),
        _context(
            source_diagnostics=lambda: {
                "source_count": 1,
                "ok_count": 1,
                "sources": [{"source_id": "manual", "status": "ok"}],
            }
        ),
    )

    assert result.success is True
    assert result.message == "Source diagnostics ready for 1 sources."
    assert result.metadata["sources"] == [{"source_id": "manual", "status": "ok"}]


def test_open_prime_observer_requires_configured_http_url():
    registry = create_builtin_action_registry()

    missing = execute_action(
        registry,
        ActionRequest(action_id="open_prime_observer"),
        _context(config=OlivawConfig()),
    )
    available = execute_action(
        registry,
        ActionRequest(action_id="open_prime_observer"),
        _context(
            config=OlivawConfig(
                prime_observer=PrimeObserverSourceConfig(
                    base_url="http://127.0.0.1:8000"
                )
            )
        ),
    )

    assert missing.success is False
    assert "not configured" in missing.message
    assert available.success is True
    assert available.metadata["href"] == "http://127.0.0.1:8000"


def test_invalid_action_handling_records_bounded_failure():
    history = ActionHistory()

    result = execute_action(
        create_builtin_action_registry(),
        ActionRequest(action_id="missing"),
        _context(),
        history=history,
    )

    assert result.success is False
    assert result.message == "Unknown action: missing"
    assert history.last_action is not None
    assert history.last_action.action_id == "missing"
    assert history.last_result is result
    assert history.last_run == result.completed_at


def test_action_history_tracks_last_action_and_result():
    history = ActionHistory()

    result = execute_action(
        create_builtin_action_registry(),
        ActionRequest(action_id="refresh_sources"),
        _context(refresh_sources=lambda: {"source_count": 0, "ok_count": 0}),
        history=history,
    )

    assert history.as_dict()["last_action"] is not None
    assert history.as_dict()["last_result"] is result
    assert history.as_dict()["last_run"] == result.completed_at


def _definition(action_id: str) -> ActionDefinition:
    def handler(
        request: ActionRequest,
        context: ActionExecutionContext,
    ) -> ActionResult:
        now = datetime.now(timezone.utc)
        return ActionResult(
            success=True,
            message=request.action_id,
            started_at=now,
            completed_at=now,
        )

    return ActionDefinition(
        action_id=action_id,
        label="Demo",
        description="Demo action",
        category="test",
        risk_level="safe_read",
        requires_confirmation=True,
        handler=handler,
    )


def _context(
    *,
    config: OlivawConfig | None = None,
    refresh_health_review=None,
    refresh_sources=None,
    source_diagnostics=None,
    evidence_package=None,
) -> ActionExecutionContext:
    return ActionExecutionContext(
        config=config or OlivawConfig(),
        refresh_health_review=refresh_health_review or (lambda: {}),
        refresh_sources=refresh_sources or (lambda: {}),
        source_diagnostics=source_diagnostics or (lambda: {"source_count": 0}),
        evidence_package=evidence_package or (lambda: {}),
    )
