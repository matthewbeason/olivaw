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
            elif health.source_id == "prime_observer":
                report_type = str(item.get("report_type") or "")
                if report_type == "network_attribution":
                    current = str(item.get("current_label") or item.get("summary") or "").strip()
                    lines.append(f"- Prime Observer current network state: {current}")
                elif report_type == "nextdns_summary":
                    lines.append("- Prime Observer DNS summary is available.")
                elif report_type == "csv":
                    timestamp = str(
                        item.get("latest_sample_timestamp")
                        or item.get("report_date")
                        or "unknown time"
                    )
                    lines.append(f"- Prime Observer latest sample: {timestamp}")
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

        lines = ["- Current-state observations only; interpretation belongs to Core Signal."]
        for item in items:
            report_type = str(item.get("report_type") or "")
            if report_type == "network_attribution":
                lines.extend(_prime_network_lines(item))
            elif report_type == "csv":
                lines.extend(_prime_latest_sample_lines(item))
            elif report_type == "nextdns_summary":
                lines.extend(_prime_dns_lines(item))
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
            status_reason = str(item.get("status_reason") or "").strip()
            if status_reason:
                lines.append(f"  - Why/status reasoning: {status_reason}")
            recommended_action = str(item.get("recommended_action") or "").strip()
            if recommended_action:
                lines.append(f"  - Recommended action: {recommended_action}")
            findings = item.get("findings", [])
            if isinstance(findings, list):
                for finding in findings[:3]:
                    lines.append(f"  - {finding}")
        return lines
    return []


def _prime_network_lines(item: dict[str, object]) -> list[str]:
    report_date = str(item.get("report_date") or "unknown time")
    current = str(item.get("current_label") or item.get("summary") or "unknown").strip()
    status = str(item.get("current_status") or item.get("status") or "unknown")
    confidence = str(item.get("current_confidence") or "unknown")
    lines = [
        f"- Network attribution generated at {report_date}.",
        f"  - Current LAN/WAN state: {current}",
        f"  - Current status: {status}",
        f"  - Confidence: {confidence}",
    ]
    window = str(item.get("window_label") or "").strip()
    if window:
        lines.append(f"  - Latest window state: {window}")
    return lines


def _prime_latest_sample_lines(item: dict[str, object]) -> list[str]:
    timestamp = str(
        item.get("latest_sample_timestamp") or item.get("report_date") or "unknown time"
    )
    phase = str(item.get("latest_sample_phase") or "unknown phase")
    host = str(item.get("latest_sample_host") or "unknown host")
    p95 = str(item.get("latest_sample_p95_ms") or "unknown")
    loss = str(item.get("latest_sample_loss_pct") or "unknown")
    return [
        f"- Latest sample timestamp: {timestamp}",
        f"  - Phase/target: {phase} to {host}",
        f"  - Raw p95/loss: {p95} ms, {loss}%",
    ]


def _prime_dns_lines(item: dict[str, object]) -> list[str]:
    lines = ["- DNS summary: available from Prime Observer."]
    total = item.get("dns_total_queries")
    blocked = item.get("dns_blocked_queries")
    allowed = item.get("dns_allowed_queries")
    block_rate = item.get("dns_block_rate_pct")
    encrypted_rate = item.get("dns_encrypted_rate_pct")
    if total is not None:
        lines.append(f"  - Total queries: {total}")
    if blocked is not None:
        lines.append(f"  - Blocked queries: {blocked}")
    if allowed is not None:
        lines.append(f"  - Allowed queries: {allowed}")
    if block_rate is not None:
        lines.append(f"  - Block rate: {block_rate}%")
    if encrypted_rate is not None:
        lines.append(f"  - Encrypted query rate: {encrypted_rate}%")
    lines.append(
        f"  - Top blocked domain: {_domain_display(item.get('top_blocked_domain'))}"
    )
    lines.append(
        f"  - Top resolved domain: {_domain_display(item.get('top_resolved_domain'))}"
    )
    top_entity = _domain_display(item.get("top_domain_entity"))
    if top_entity != "unavailable":
        lines.append(f"  - Top domain/entity: {top_entity}")
    return lines


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


def _domain_display(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "null", "unavailable"}:
        return "unavailable"
    return text
