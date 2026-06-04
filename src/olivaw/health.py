from __future__ import annotations

from olivaw.config import OlivawConfig, load_config
from olivaw.models import HealthReport
from olivaw.providers.router import RouterProvider


def run_health_checks(config: OlivawConfig | None = None) -> HealthReport:
    resolved_config = config or load_config()
    return RouterProvider(resolved_config).health()


def format_health_report(report: HealthReport) -> str:
    lines = [
        "Olivaw Health",
        "",
        _format_provider("Local Provider", report.local),
        "",
        _format_provider("Cloud Provider", report.cloud),
        "",
        f"Selected Provider: {report.selected_provider or 'None'}",
        f"Cloud Fallback: {report.cloud_fallback}",
    ]
    if report.notes:
        lines.extend(["", "Notes:"])
        lines.extend(f"- {note}" for note in report.notes)
    return "\n".join(lines)


def _format_provider(title: str, status) -> str:
    lines = [
        f"{title}: {status.state.title()}",
        f"Name: {status.name}",
        f"Model: {status.model or 'not configured'}",
        f"Reason: {status.message}",
    ]
    if status.detail:
        lines.append(f"Suggested Action: {status.detail}")
    return "\n".join(lines)

