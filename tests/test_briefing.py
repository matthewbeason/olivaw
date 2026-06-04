from __future__ import annotations

from pathlib import Path

from olivaw.briefing import compose_briefing_from_file


def test_briefing_golden_output():
    output = compose_briefing_from_file(Path("examples/daily_context.json"))

    assert output == """# Daily Briefing

Date: 2026-06-04

## Focus
Stabilize Olivaw v0 as a local-first assistant foundation.

## Summary
Keep the first version small: health checks, deterministic briefing, provider routing, CLI, and web UI.

## Priorities
1. Ship the initial framework [in_progress] - The repository needs a clean base before additional assistant behavior is added.
2. Keep cloud fallback explicit [planned] - Local-first behavior should remain predictable and privacy-preserving.
3. Make health checks actionable [planned] - Missing local services should guide setup instead of causing crashes.

## Signals
- Briefing works without a model (local fixtures): The v0 briefing renderer is deterministic and testable offline.
- Local provider is preferred (provider router): Cloud models are ignored unless explicitly enabled.

## Projects
- Olivaw: foundation. Next: Run tests, verify CLI commands, and start the web app.

## Reminders
- Do not integrate Prime Observer or Core Signal in v0.
- Do not commit secrets or local configuration.
"""

