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
        lines.extend(_payload_diagnostic_lines(_payload(snapshot), source_id="prime_observer"))
        for item in items:
            report_type = str(item.get("report_type") or "")
            if report_type == "network_attribution":
                lines.extend(_prime_network_lines(item))
            elif report_type == "csv":
                lines.extend(_prime_latest_sample_lines(item))
            elif report_type == "nextdns_summary":
                lines.extend(_prime_dns_lines(item))
            elif report_type == "investigation_index":
                lines.extend(_prime_investigation_index_lines(item))
            elif report_type == "investigation":
                lines.extend(_prime_investigation_lines(item))
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
        lines.extend(_payload_diagnostic_lines(_payload(snapshot), source_id="core_signal"))
        for item in items[:3]:
            title = str(item.get("title") or "Core Signal report")
            report_date = str(item.get("report_date") or "unknown date")
            status = str(item.get("status") or "unknown")
            summary = _one_line_preview(str(item.get("summary") or "No summary."))
            lines.append(f"- {title} ({report_date}) [{status}]: {summary}")
            lines.append("  - Interpretation: Core Signal")
            lines.append("  - Presentation: Olivaw")
            report_confidence = str(item.get("confidence") or "").strip()
            if report_confidence:
                lines.append(f"  - Confidence: {report_confidence}")
            confidence_reason = str(item.get("confidence_reason") or "").strip()
            if confidence_reason:
                lines.append(f"  - Confidence rationale: {confidence_reason}")
            supporting_facts = _dicts(item.get("supporting_facts"))
            if supporting_facts:
                lines.append(f"  - Supporting facts: {len(supporting_facts)}")
                for fact in supporting_facts[:3]:
                    lines.extend(_supporting_fact_lines(fact, indent="    "))
            recommendation_trace = _dicts(item.get("recommendation_trace"))
            if recommendation_trace:
                lines.append("  - Recommendation trace:")
                for step in recommendation_trace[:6]:
                    stage = str(step.get("stage") or "Trace").strip()
                    detail = _one_line_preview(str(step.get("detail") or ""))
                    if detail:
                        lines.append(f"    - {stage}: {detail}")
            interpretation_source = str(item.get("interpretation_source") or "").strip()
            if interpretation_source:
                lines.append(f"  - Interpretation source: {interpretation_source}")
            related_events = _dicts(item.get("related_events"))
            if related_events:
                lines.append("  - Related events:")
                for related_event in related_events[:3]:
                    lines.append(f"    - Related event: {_related_event_label(related_event)}")
            events = item.get("events", [])
            if isinstance(events, list):
                for event in [event for event in events if isinstance(event, dict)][:3]:
                    lines.extend(_core_signal_event_lines(event))
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
            dns_findings = item.get("dns_findings", [])
            if isinstance(dns_findings, list):
                for finding in dns_findings[:3]:
                    lines.append(f"  - DNS interpretation: {finding}")
            dns_status = str(item.get("dns_status") or "").strip()
            if dns_status:
                lines.append(f"  - DNS status: {dns_status}")
            dns_meaning = str(item.get("dns_meaning") or "").strip()
            if dns_meaning:
                lines.append(f"  - DNS meaning: {dns_meaning}")
            dns_action = str(item.get("dns_recommended_action") or "").strip()
            if dns_action:
                lines.append(f"  - DNS recommended action: {dns_action}")
        return lines
    return []


def _core_signal_event_lines(event: dict[str, object]) -> list[str]:
    summary = _one_line_preview(str(event.get("summary") or "Core Signal event."))
    lines = [f"  - Event: {summary}"]
    lines.append("    - Interpretation: Core Signal")
    lines.append("    - Presentation: Olivaw")
    event_id = str(event.get("id") or "").strip()
    if event_id:
        lines.append(f"    - Event ID: {event_id}")
    kind = str(event.get("kind") or "").strip()
    if kind:
        lines.append(f"    - Event kind: {kind}")
    status = str(event.get("status") or "").strip()
    severity = str(event.get("severity") or "").strip()
    if status or severity:
        label = status or "unknown"
        if severity:
            label = f"{label} / {severity}"
        lines.append(f"    - Severity/status: {label}")
    affected = _event_window_label(event)
    if affected:
        lines.append(f"    - Affected window: {affected}")
    confidence = str(event.get("confidence") or "").strip()
    if confidence:
        lines.append(f"    - Confidence: {confidence}")
    why = str(event.get("why") or "").strip()
    if why:
        lines.append(f"    - Why it matters: {why}")
    confidence_reason = str(event.get("confidence_reason") or "").strip()
    if confidence_reason:
        lines.append(f"    - Confidence rationale: {confidence_reason}")
    supporting_facts = _dicts(event.get("supporting_facts"))
    if supporting_facts:
        lines.append(f"    - Supporting facts: {len(supporting_facts)}")
        for fact in supporting_facts[:3]:
            lines.extend(_supporting_fact_lines(fact, indent="      "))
    issue_location = str(event.get("issue_location") or "").strip()
    if issue_location:
        lines.append(f"    - Issue location: {issue_location}")
    recommended_action = str(event.get("recommended_action") or "").strip()
    if recommended_action:
        lines.append(f"    - Recommended action: {recommended_action}")
    recommendation_trace = _dicts(event.get("recommendation_trace"))
    if recommendation_trace:
        lines.append("    - Recommendation trace:")
        for step in recommendation_trace[:6]:
            stage = str(step.get("stage") or "Trace").strip()
            detail = _one_line_preview(str(step.get("detail") or ""))
            if detail:
                lines.append(f"      - {stage}: {detail}")
    attribution = str(event.get("attribution_source") or "").strip()
    if attribution:
        lines.append(f"    - Evidence: {_source_label(attribution)}")
    interpretation_source = str(event.get("interpretation_source") or "").strip()
    if interpretation_source:
        lines.append(f"    - Interpretation source: {interpretation_source}")
    investigation = _investigation_reference(event)
    if investigation:
        lines.append(f"    - View investigation: {investigation}")
    evidence = _evidence_window_label(event)
    if evidence:
        lines.append(f"    - Evidence window: {evidence}")
    related_events = _dicts(event.get("related_events"))
    if related_events:
        lines.append("    - Related events:")
        for related_event in related_events[:3]:
            lines.append(f"      - Related event: {_related_event_label(related_event)}")
    return lines


def _supporting_fact_lines(fact: dict[str, object], *, indent: str) -> list[str]:
    summary = _one_line_preview(str(fact.get("summary") or "Supporting fact."))
    lines = [f"{indent}- Fact: {summary}"]
    source = str(fact.get("source") or "").strip()
    reference = str(fact.get("reference") or "").strip()
    if source:
        lines.append(f"{indent}  - Source: {source}")
    if reference:
        lines.append(f"{indent}  - Reference: {reference}")
    return lines


def _related_event_label(event: dict[str, object]) -> str:
    pieces = []
    event_id = str(event.get("id") or "").strip()
    relationship = str(event.get("relationship") or "").strip()
    summary = str(event.get("summary") or "").strip()
    reference = str(event.get("reference") or "").strip()
    if event_id:
        pieces.append(event_id)
    if relationship:
        pieces.append(relationship)
    if summary:
        pieces.append(summary)
    if reference:
        pieces.append(reference)
    return " - ".join(pieces) or "Related event"


def _event_window_label(event: dict[str, object]) -> str:
    start = str(event.get("window_start") or "").strip()
    end = str(event.get("window_end") or "").strip()
    if start and end:
        return f"{start} to {end}"
    return start or end


def _evidence_window_label(event: dict[str, object]) -> str:
    evidence = event.get("evidence_window")
    if not isinstance(evidence, dict):
        return ""
    label = str(evidence.get("label") or "").strip()
    if label:
        return label
    start = str(evidence.get("window_start") or "").strip()
    end = str(evidence.get("window_end") or "").strip()
    granularity = str(evidence.get("granularity") or "").strip()
    pieces = []
    if start and end:
        pieces.append(f"{start} to {end}")
    elif start or end:
        pieces.append(start or end)
    if granularity:
        pieces.append(granularity)
    return "; ".join(pieces)


def _investigation_reference(event: dict[str, object]) -> str:
    reference = event.get("prime_observer_reference")
    if isinstance(reference, dict):
        for key in ("url", "path", "id"):
            value = str(reference.get(key) or "").strip()
            if value:
                return value
    return str(event.get("prime_observer_investigation") or "").strip()


def _source_label(value: str) -> str:
    labels = {
        "prime_observer_incident": "Prime Observer incident attribution",
        "prime_observer_window": "Prime Observer window attribution",
        "prime_observer_current": "Prime Observer current attribution",
        "core_signal_fallback": "Core Signal fallback",
    }
    return labels.get(value, value)


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
    encrypted = item.get("dns_encrypted_queries")
    if encrypted is not None:
        lines.append(f"  - Encrypted queries: {encrypted}")
    if block_rate is not None:
        lines.append(f"  - Block rate: {block_rate}%")
    raw_block_rate = item.get("dns_block_rate")
    if raw_block_rate is not None:
        lines.append(f"  - Raw block rate: {raw_block_rate}")
    if encrypted_rate is not None:
        lines.append(f"  - Encrypted query rate: {encrypted_rate}%")
    raw_encrypted_rate = item.get("dns_encrypted_rate")
    if raw_encrypted_rate is not None:
        lines.append(f"  - Raw encrypted query rate: {raw_encrypted_rate}")
    lines.append(
        "  - Top queried domain: "
        f"{_domain_display(item.get('top_queried_domain'))}"
        f"{_count_share(item, 'top_queried_domain')}"
    )
    lines.append(
        "  - Top blocked domain: "
        f"{_domain_display(item.get('top_blocked_domain'))}"
        f"{_count_share(item, 'top_blocked_domain')}"
    )
    lines.append(
        "  - Top resolved domain: "
        f"{_domain_display(item.get('top_resolved_domain'))}"
        f"{_count_share(item, 'top_resolved_domain')}"
    )
    top_blocked_category = _domain_display(item.get("top_blocked_category"))
    if top_blocked_category != "unavailable":
        lines.append(f"  - Top blocked category/reason: {top_blocked_category}")
    top_queried = _domain_display(item.get("top_queried_domain"))
    top_entity = _domain_display(item.get("top_domain_entity"))
    if top_entity != "unavailable" and top_queried == "unavailable":
        lines.append(f"  - Top domain/entity: {top_entity}")
    return lines


def _prime_investigation_index_lines(item: dict[str, object]) -> list[str]:
    catalog = _dicts(item.get("investigation_catalog"))
    if not catalog:
        return ["- Investigation index data: from Prime Observer, no entries listed."]
    lines = ["- Investigation index data: from Prime Observer."]
    for entry in catalog[:5]:
        title = str(entry.get("title") or entry.get("id") or "Investigation").strip()
        lines.append(f"  - Investigation: {title}")
        for key, label in (
            ("created_at", "Created at"),
            ("event_count", "Event count"),
            ("status", "Status"),
            ("path", "Path"),
        ):
            value = str(entry.get(key) or "").strip()
            if value:
                lines.append(f"    - {label}: {value}")
    return lines


def _prime_investigation_lines(item: dict[str, object]) -> list[str]:
    lines = ["- Investigation metadata: from Prime Observer."]
    start = str(item.get("investigation_start") or "").strip()
    end = str(item.get("investigation_end") or "").strip()
    if start or end:
        lines.append(f"  - Investigation window: {_window_label(start, end)}")

    navigation = item.get("investigation_navigation")
    if isinstance(navigation, dict) and navigation:
        lines.append("  - Navigation metadata: from Prime Observer.")
        for key, label in (
            ("first_event", "First event"),
            ("previous_event", "Previous event"),
            ("next_event", "Next event"),
            ("last_event", "Last event"),
        ):
            event = navigation.get(key)
            if isinstance(event, dict):
                lines.append(f"    - {label}: {_event_reference_label(event)}")

    neighborhoods = _dicts(item.get("event_neighborhoods"))
    if neighborhoods:
        lines.append("  - Nearby-event facts: from Prime Observer.")
        for neighborhood in neighborhoods[:3]:
            anchor = neighborhood.get("event")
            if isinstance(anchor, dict) and anchor:
                lines.append(
                    "    - Events in the same investigation window for "
                    f"{_event_reference_label(anchor)}:"
                )
            else:
                lines.append("    - Nearby events:")
            nearby_events = _dicts(neighborhood.get("nearby_events"))
            for event in nearby_events[:5]:
                lines.append(f"      - {_event_reference_label(event)}")
    return lines


def _window_label(start: str, end: str) -> str:
    if start and end:
        return f"{start} to {end}"
    return start or end or "unavailable"


def _event_reference_label(event: dict[str, object]) -> str:
    label = str(event.get("label") or event.get("id") or "Event").strip()
    event_id = str(event.get("id") or "").strip()
    timestamp = str(event.get("timestamp") or "").strip()
    target = str(event.get("path") or event.get("anchor") or "").strip()
    details = []
    if event_id and event_id != label:
        details.append(f"id {event_id}")
    if timestamp:
        details.append(timestamp)
    if target:
        details.append(f"target {target}")
    if details:
        return f"{label} ({', '.join(details)})"
    return label


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
        lines.extend(_payload_diagnostic_lines(payload, source_id=health.source_id))
    return lines


def _payload_diagnostic_lines(
    payload: SourcePayload | None,
    *,
    source_id: str,
) -> list[str]:
    if not payload:
        return []
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return []
    if source_id == "prime_observer":
        return _prime_observer_diagnostic_lines(diagnostics)
    if source_id == "core_signal":
        return _core_signal_diagnostic_lines(diagnostics)
    return []


def _prime_observer_diagnostic_lines(diagnostics: dict[str, object]) -> list[str]:
    lines: list[str] = []
    for key, label in (
        ("configured_path", "Configured path"),
        ("selection", "Selected Prime Observer files"),
        ("investigation_index_path", "Investigation index path"),
        ("investigation_index", "Investigation index"),
        ("investigation_path", "Investigation export path"),
        ("investigation", "Investigation export"),
        ("catalog_entry_count", "Catalog entry count"),
        ("investigation_event_count", "Investigation event count"),
        ("latest_investigation_timestamp", "Latest investigation timestamp"),
        ("investigation_index_modified", "Investigation index modified"),
        ("investigation_modified", "Investigation export modified"),
        ("investigation_index_generated_at", "Investigation index generated"),
        ("investigation_generated_at", "Investigation export generated"),
    ):
        value = diagnostics.get(key)
        if value not in (None, "", []):
            lines.append(f"- {label}: {value}")
    return lines


def _core_signal_diagnostic_lines(diagnostics: dict[str, object]) -> list[str]:
    lines: list[str] = []
    for key, label in (
        ("configured_path", "Configured reports path"),
        ("selection", "Selected Core Signal files"),
        ("interpreted_events", "Core Signal events"),
        ("event_objects_found", "Event objects found"),
        ("interpreted_events_rendered", "Interpreted events rendered"),
        ("latest_event_timestamp", "Latest event timestamp"),
    ):
        value = diagnostics.get(key)
        if value not in (None, "", []):
            lines.append(f"- {label}: {value}")
    generated = diagnostics.get("generated_timestamps")
    if isinstance(generated, list) and generated:
        lines.append(
            "- Generated timestamps: " + ", ".join(str(item) for item in generated)
        )
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


def _dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


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


def _count_share(item: dict[str, object], prefix: str) -> str:
    parts: list[str] = []
    count = item.get(f"{prefix}_count")
    if count is not None:
        parts.append(f"count {count}")
    share = item.get(f"{prefix}_share")
    if share is not None:
        parts.append(f"share {share}")
    if not parts:
        return ""
    return f" ({', '.join(parts)})"
