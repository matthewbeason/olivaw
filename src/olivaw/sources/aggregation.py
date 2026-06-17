from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from olivaw.sources.base import Source, SourceHealth, SourcePayload, SourceStatus


@dataclass(frozen=True)
class SourceResult:
    source_id: str
    source_name: str
    source_type: str
    status: SourceStatus
    message: str = ""
    generated_at: str = ""
    observed_at: str = ""
    freshness: str = ""
    summary_items: list[dict[str, object]] = field(default_factory=list)
    facts: list[dict[str, object]] = field(default_factory=list)
    interpretation_items: list[dict[str, object]] = field(default_factory=list)
    actions: list[dict[str, object]] = field(default_factory=list)
    references: list[dict[str, object]] = field(default_factory=list)
    diagnostics: dict[str, object] = field(default_factory=dict)
    raw_available: bool = False


@dataclass(frozen=True)
class AggregatedSources:
    sources: list[SourceResult]
    facts: list[dict[str, object]]
    interpretation_items: list[dict[str, object]]
    actions: list[dict[str, object]]
    references: list[dict[str, object]]
    diagnostics: list[dict[str, object]]
    health_review_context: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "sources": [asdict(source) for source in self.sources],
            "facts": self.facts,
            "interpretation_items": self.interpretation_items,
            "actions": self.actions,
            "references": self.references,
            "diagnostics": self.diagnostics,
            "health_review_context": self.health_review_context,
        }


def aggregate_sources(sources: tuple[Source, ...] | list[Source]) -> AggregatedSources:
    results = [_source_result(source) for source in sources]
    facts = _merge_owned_lists(results, "facts")
    interpretation_items = _merge_owned_lists(results, "interpretation_items")
    actions = _merge_owned_lists(results, "actions")
    references = _merge_owned_lists(results, "references")
    diagnostics = [
        {
            "source_id": result.source_id,
            "source_name": result.source_name,
            "status": result.status,
            "message": result.message,
            "diagnostics": result.diagnostics,
        }
        for result in results
    ]
    return AggregatedSources(
        sources=results,
        facts=facts,
        interpretation_items=interpretation_items,
        actions=actions,
        references=references,
        diagnostics=diagnostics,
        health_review_context={
            "facts": facts,
            "interpretation_items": interpretation_items,
            "actions": actions,
            "references": references,
        },
    )


def normalize_source_payload(
    health: SourceHealth,
    payload: SourcePayload | None,
) -> SourceResult:
    items = _items(payload)
    diagnostics = _dict(payload.get("diagnostics")) if payload else {}
    generated_at = _first_text(
        *(item.get("generated_at") for item in items),
        *_list_value(diagnostics.get("generated_timestamps")),
    )
    observed_at = _first_text(
        diagnostics.get("latest_event_timestamp"),
        diagnostics.get("latest_investigation_timestamp"),
        *(item.get("observed_at") for item in items),
        *(item.get("forecast_date") for item in items),
        *(item.get("report_date") for item in items),
        *(item.get("latest_sample_timestamp") for item in items),
    )

    facts: list[dict[str, object]] = []
    interpretation_items: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    references: list[dict[str, object]] = []
    summary_items: list[dict[str, object]] = []

    for item in items:
        summary = _summary_item(health, item)
        if summary:
            summary_items.append(summary)
        if health.source_id == "prime_observer":
            facts.extend(_prime_observer_facts(health, item))
            references.extend(_prime_observer_references(health, item))
        elif health.source_id == "core_signal":
            interpretation_items.extend(_core_signal_interpretations(health, item))
            actions.extend(_core_signal_actions(health, item))
            references.extend(_core_signal_references(health, item))
            facts.extend(_core_signal_supporting_facts(health, item))
        else:
            facts.extend(_generic_facts(health, item))

    return SourceResult(
        source_id=health.source_id,
        source_name=health.display_name,
        source_type=_source_type(health.source_id),
        status=health.status,
        message=health.message,
        generated_at=generated_at,
        observed_at=observed_at,
        freshness=_freshness(generated_at or observed_at),
        summary_items=summary_items,
        facts=facts,
        interpretation_items=interpretation_items,
        actions=actions,
        references=references,
        diagnostics=diagnostics,
        raw_available=bool(payload and items),
    )


def _source_result(source: Source) -> SourceResult:
    try:
        health = source.health()
    except Exception as exc:
        return SourceResult(
            source_id=getattr(source, "source_id", "unknown"),
            source_name=getattr(source, "display_name", "Unknown source"),
            source_type=_source_type(getattr(source, "source_id", "unknown")),
            status="error",
            message=f"Health check failed: {type(exc).__name__}: {exc}",
        )

    payload: SourcePayload | None = None
    if health.status == "ok":
        try:
            payload = source.fetch()
            payload_status = str(payload.get("status") or "").strip()
            if payload_status in {"unavailable", "error"}:
                errors = payload.get("errors")
                if isinstance(errors, list) and errors:
                    message = str(errors[0])
                else:
                    message = health.message
                health = SourceHealth(
                    source_id=health.source_id,
                    display_name=health.display_name,
                    status=payload_status,  # type: ignore[arg-type]
                    message=message,
                )
        except Exception as exc:
            health = SourceHealth(
                source_id=health.source_id,
                display_name=health.display_name,
                status="error",
                message=f"Fetch failed: {type(exc).__name__}: {exc}",
            )
    return normalize_source_payload(health, payload)


def _summary_item(
    health: SourceHealth,
    item: dict[str, object],
) -> dict[str, object]:
    title = _first_text(item.get("title"), item.get("report_type"), item.get("path"))
    summary = _first_text(item.get("summary"), item.get("preview"))
    if not title and not summary:
        return {}
    return _owned(
        health,
        {
            "title": title or "Source item",
            "summary": summary,
            "generated_at": _first_text(item.get("generated_at"), item.get("report_date")),
            "observed_at": _first_text(
                item.get("latest_sample_timestamp"),
                item.get("latest_investigation_timestamp"),
                item.get("report_date"),
            ),
            "raw_type": _first_text(item.get("report_type"), item.get("path")),
        },
    )


def _prime_observer_facts(
    health: SourceHealth,
    item: dict[str, object],
) -> list[dict[str, object]]:
    facts: list[dict[str, object]] = []
    report_type = str(item.get("report_type") or "")
    summary = _first_text(item.get("summary"))
    if summary:
        facts.append(_owned(health, {"summary": summary, "kind": report_type or "fact"}))
    if report_type == "network_attribution":
        for key, label in (
            ("current_label", "Current network attribution"),
            ("current_status", "Current network status"),
            ("current_confidence", "Current network confidence"),
            ("window_label", "Latest window state"),
        ):
            value = _first_text(item.get(key))
            if value:
                facts.append(_owned(health, {"summary": f"{label}: {value}", "kind": key}))
    elif report_type == "nextdns_summary":
        for key, label in (
            ("dns_total_queries", "DNS total queries"),
            ("dns_blocked_queries", "DNS blocked queries"),
            ("dns_allowed_queries", "DNS allowed queries"),
            ("dns_block_rate_pct", "DNS block rate"),
            ("top_queried_domain", "Top queried domain"),
            ("top_blocked_domain", "Top blocked domain"),
            ("top_resolved_domain", "Top resolved domain"),
        ):
            value = _first_text(item.get(key))
            if value:
                facts.append(_owned(health, {"summary": f"{label}: {value}", "kind": key}))
    elif report_type == "csv":
        timestamp = _first_text(item.get("latest_sample_timestamp"), item.get("report_date"))
        if timestamp:
            facts.append(
                _owned(
                    health,
                    {
                        "summary": f"Latest telemetry sample: {timestamp}",
                        "kind": "latest_sample",
                    },
                )
            )
    elif report_type == "investigation_index":
        catalog = _dicts(item.get("investigation_catalog"))
        facts.append(
            _owned(
                health,
                {
                    "summary": f"Investigation index entries: {len(catalog)}",
                    "kind": "investigation_index",
                },
            )
        )
    elif report_type == "investigation":
        count = len(_dicts(item.get("investigation_events")))
        facts.append(
            _owned(
                health,
                {
                    "summary": f"Investigation event references: {count}",
                    "kind": "investigation",
                },
            )
        )
    return facts


def _prime_observer_references(
    health: SourceHealth,
    item: dict[str, object],
) -> list[dict[str, object]]:
    references: list[dict[str, object]] = []
    source_path = _first_text(item.get("source_path"))
    if source_path:
        references.append(_owned(health, {"label": "Source file", "target": source_path}))
    for entry in _dicts(item.get("investigation_catalog")):
        target = _first_text(entry.get("path"), entry.get("id"))
        if target:
            references.append(
                _owned(
                    health,
                    {
                        "label": _first_text(entry.get("title"), entry.get("id"), "Investigation"),
                        "target": target,
                        "kind": "investigation",
                    },
                )
            )
    return references


def _core_signal_interpretations(
    health: SourceHealth,
    item: dict[str, object],
) -> list[dict[str, object]]:
    interpretations: list[dict[str, object]] = []
    summary = _first_text(item.get("summary"))
    if summary:
        interpretations.append(
            _owned(
                health,
                {
                    "summary": summary,
                    "status": _first_text(item.get("status")),
                    "confidence": _first_text(item.get("confidence")),
                    "confidence_reason": _first_text(item.get("confidence_reason")),
                    "attribution_assessment": _dict(item.get("attribution_assessment")),
                    "evidence_strength": _dict(item.get("evidence_strength")),
                    "uncertainties": _list_value(item.get("uncertainties")),
                    "kind": "report",
                },
            )
        )
    for event in _dicts(item.get("events")):
        interpretations.append(
            _owned(
                health,
                {
                    "summary": _first_text(event.get("summary"), event.get("id"), "Core Signal event"),
                    "id": _first_text(event.get("id")),
                    "status": _first_text(event.get("status")),
                    "severity": _first_text(event.get("severity")),
                    "confidence": _first_text(event.get("confidence")),
                    "confidence_reason": _first_text(event.get("confidence_reason")),
                    "why": _first_text(event.get("why")),
                    "attribution_assessment": _dict(event.get("attribution_assessment")),
                    "evidence_strength": _dict(event.get("evidence_strength")),
                    "uncertainties": _list_value(event.get("uncertainties")),
                    "kind": "event",
                },
            )
        )
    return interpretations


def _core_signal_actions(
    health: SourceHealth,
    item: dict[str, object],
) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    action = _first_text(item.get("recommended_action"))
    if action:
        actions.append(_owned(health, {"summary": action, "kind": "recommendation"}))
    for event in _dicts(item.get("events")):
        event_action = _first_text(event.get("recommended_action"))
        if event_action:
            actions.append(
                _owned(
                    health,
                    {
                        "summary": event_action,
                        "event_id": _first_text(event.get("id")),
                        "kind": "recommendation",
                    },
                )
            )
    return actions


def _core_signal_references(
    health: SourceHealth,
    item: dict[str, object],
) -> list[dict[str, object]]:
    references: list[dict[str, object]] = []
    source_path = _first_text(item.get("source_path"))
    if source_path:
        references.append(_owned(health, {"label": "Source file", "target": source_path}))
    for event in _dicts(item.get("events")):
        reference = event.get("prime_observer_reference")
        if isinstance(reference, dict):
            target = _first_text(reference.get("url"), reference.get("path"), reference.get("id"))
            if target:
                references.append(
                    _owned(
                        health,
                        {
                            "label": "Prime Observer evidence reference",
                            "target": target,
                            "event_id": _first_text(event.get("id")),
                            "kind": "evidence",
                        },
                    )
                )
    return references


def _core_signal_supporting_facts(
    health: SourceHealth,
    item: dict[str, object],
) -> list[dict[str, object]]:
    facts: list[dict[str, object]] = []
    for fact in _dicts(item.get("supporting_facts")):
        facts.append(_owned(health, {**fact, "reported_source": fact.get("source", "")}))
    for event in _dicts(item.get("events")):
        for fact in _dicts(event.get("supporting_facts")):
            facts.append(
                _owned(
                    health,
                    {
                        **fact,
                        "event_id": _first_text(event.get("id")),
                        "reported_source": fact.get("source", ""),
                    },
                )
            )
    return facts


def _generic_facts(
    health: SourceHealth,
    item: dict[str, object],
) -> list[dict[str, object]]:
    facts = [
        _owned(health, fact)
        for fact in _dicts(item.get("facts"))
        if _first_text(fact.get("summary"))
    ]
    summary = _first_text(item.get("summary"), item.get("preview"))
    if summary:
        facts.insert(0, _owned(health, {"summary": summary, "kind": "source_item"}))
    return facts


def _merge_owned_lists(
    results: list[SourceResult],
    attribute: str,
) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for result in results:
        merged.extend(getattr(result, attribute))
    return merged


def _owned(
    health: SourceHealth,
    values: dict[str, object],
) -> dict[str, object]:
    return {
        "source_id": health.source_id,
        "source_name": health.display_name,
        **{key: value for key, value in values.items() if value not in ("", [], {})},
    }


def _source_type(source_id: str) -> str:
    return {
        "prime_observer": "evidence",
        "core_signal": "interpretation",
        "files": "local_context",
        "manual": "manual",
    }.get(source_id, "external_context")


def _freshness(timestamp: str) -> str:
    if not timestamp:
        return ""
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return "available"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)
    if age.days < 1:
        return "fresh"
    if age.days <= 7:
        return "recent"
    return "stale"


def _items(payload: SourcePayload | None) -> list[dict[str, object]]:
    if not payload:
        return []
    items = payload.get("items", [])
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    return {}


def _dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _list_value(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    text = str(value or "").strip()
    return [text] if text else []


def _first_text(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""
