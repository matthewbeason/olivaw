from __future__ import annotations

from olivaw.assistant.attribution import SOURCE_BACKED, AttributedResponse
from olivaw.config import OlivawConfig
from olivaw.sources.base import Source, SourceHealth, SourcePayload
from olivaw.sources.registry import SourceRegistry, create_default_registry


def compose_source_briefing(
    registry: SourceRegistry | None = None,
    config: OlivawConfig | None = None,
) -> AttributedResponse:
    resolved = registry or create_default_registry(config)
    snapshots = [_source_snapshot(source) for source in resolved.list_sources()]
    used_sources = tuple(
        _health(snapshot).source_id
        for snapshot in snapshots
        if _health(snapshot).status == "ok" and _items(_payload(snapshot))
    )
    return AttributedResponse(
        text=render_source_briefing(snapshots),
        attribution=SOURCE_BACKED,
        sources=used_sources,
        capability="source-backed briefing",
    )


def render_source_briefing(snapshots: list[dict[str, object]]) -> str:
    lines = ["# Source Briefing", "", "## Sources"]
    if not snapshots:
        lines.append("- No sources registered.")
    for snapshot in snapshots:
        health = _health(snapshot)
        lines.append(
            f"- {health.source_id}: {health.status} ({health.display_name}) - {health.message}"
        )

    lines.extend(["", "## Highlights"])
    highlights = _highlight_lines(snapshots)
    if highlights:
        lines.extend(highlights)
    else:
        lines.append("- No source-backed highlights available.")

    prime_observer_lines = _prime_observer_lines(snapshots)
    if prime_observer_lines:
        lines.extend(["", "## Prime Observer"])
        lines.extend(prime_observer_lines)

    core_signal_lines = _core_signal_lines(snapshots)
    if core_signal_lines:
        lines.extend(["", "## Core Signal"])
        lines.extend(core_signal_lines)

    lines.extend(["", "## Files"])
    file_lines = _file_lines(snapshots)
    if file_lines:
        lines.extend(file_lines)
    else:
        lines.append("- No file source items available.")

    lines.extend(["", "## Source Notes"])
    notes = _source_note_lines(snapshots)
    if notes:
        lines.extend(notes)
    else:
        lines.append("- No source notes.")

    used = ", ".join(
        _health(snapshot).source_id
        for snapshot in snapshots
        if _health(snapshot).status == "ok" and _items(_payload(snapshot))
    )
    if not used:
        used = "none"
    lines.extend(
        [
            "",
            "## Attribution",
            f"This briefing is source-backed using: {used}.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _source_snapshot(source: Source) -> dict[str, object]:
    try:
        health = source.health()
    except Exception as exc:
        health = SourceHealth(
            source_id=getattr(source, "source_id", "unknown"),
            display_name=getattr(source, "display_name", "Unknown source"),
            status="error",
            message=f"Health check failed: {type(exc).__name__}: {exc}",
        )
        return {"health": health, "payload": None}

    payload: SourcePayload | None = None
    if health.status == "ok":
        try:
            payload = source.fetch()
        except Exception as exc:
            health = SourceHealth(
                source_id=health.source_id,
                display_name=health.display_name,
                status="error",
                message=f"Fetch failed: {type(exc).__name__}: {exc}",
            )
    return {"health": health, "payload": payload}


def _highlight_lines(snapshots: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for snapshot in snapshots:
        health = _health(snapshot)
        payload = _payload(snapshot)
        if health.status != "ok":
            lines.append(f"- {health.source_id} unavailable: {health.message}")
            continue
        items = _items(payload)
        if not items:
            lines.append(f"- {health.source_id} has no items.")
            continue
        for item in items:
            if health.source_id == "files":
                path = str(item.get("path") or item.get("title") or "unknown file")
                lines.append(f"- File found: {path}")
            else:
                title = str(item.get("title") or "Untitled item")
                summary = str(item.get("summary") or item.get("preview") or "").strip()
                if summary:
                    lines.append(f"- {title} from {health.source_id} source: {summary}")
                else:
                    lines.append(f"- {title} from {health.source_id} source")
    return lines


def _file_lines(snapshots: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for snapshot in snapshots:
        health = _health(snapshot)
        if health.source_id != "files" or health.status != "ok":
            continue
        for item in _items(_payload(snapshot)):
            path = str(item.get("path") or item.get("title") or "unknown file")
            preview = _one_line_preview(str(item.get("preview") or "No preview."))
            lines.append(f"- {path} - {preview}")
    return lines


def _prime_observer_lines(snapshots: list[dict[str, object]]) -> list[str]:
    for snapshot in snapshots:
        health = _health(snapshot)
        if health.source_id != "prime_observer":
            continue
        if health.status != "ok":
            return [f"- Status: {health.status} - {health.message}"]

        items = _items(_payload(snapshot))
        if not items:
            return ["- Status: ok, no Prime Observer items returned."]

        lines: list[str] = []
        for item in items[:3]:
            title = str(item.get("title") or "Prime Observer report")
            report_date = str(item.get("report_date") or "unknown date")
            status = str(item.get("status") or "unknown")
            summary = _one_line_preview(str(item.get("summary") or "No summary."))
            lines.append(f"- {title} ({report_date}) [{status}]: {summary}")
            findings = item.get("findings", [])
            if isinstance(findings, list):
                for finding in findings[:3]:
                    lines.append(f"  - {finding}")
        return lines
    return []


def _core_signal_lines(snapshots: list[dict[str, object]]) -> list[str]:
    for snapshot in snapshots:
        health = _health(snapshot)
        if health.source_id != "core_signal":
            continue
        if health.status != "ok":
            return [f"- Status: {health.status} - {health.message}"]

        items = _items(_payload(snapshot))
        if not items:
            return ["- Status: ok, no Core Signal items returned."]

        lines: list[str] = []
        for item in items[:3]:
            title = str(item.get("title") or "Core Signal report")
            report_date = str(item.get("report_date") or "unknown date")
            status = str(item.get("status") or "unknown")
            summary = _one_line_preview(str(item.get("summary") or "No summary."))
            lines.append(f"- {title} ({report_date}) [{status}]: {summary}")
            recommended_action = str(item.get("recommended_action") or "").strip()
            if recommended_action:
                lines.append(f"  - Recommended action: {recommended_action}")
            findings = item.get("findings", [])
            if isinstance(findings, list):
                for finding in findings[:3]:
                    lines.append(f"  - {finding}")
        return lines
    return []


def _source_note_lines(snapshots: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for snapshot in snapshots:
        health = _health(snapshot)
        payload = _payload(snapshot)
        if health.status != "ok":
            lines.append(f"- {health.source_id}: unavailable ({health.message})")
            continue
        items = _items(payload)
        if not items:
            lines.append(f"- {health.source_id}: ok, no items returned.")
            continue
        lines.append(f"- {health.source_id}: ok, {len(items)} item(s) returned.")
    return lines


def _health(snapshot: dict[str, object]) -> SourceHealth:
    return snapshot["health"]  # type: ignore[return-value]


def _payload(snapshot: dict[str, object]) -> SourcePayload | None:
    return snapshot.get("payload")  # type: ignore[return-value]


def _items(payload: SourcePayload | None) -> list[dict[str, object]]:
    if not payload:
        return []
    items = payload.get("items", [])
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _one_line_preview(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) > 160:
        return normalized[:157].rstrip() + "..."
    return normalized or "No preview."
