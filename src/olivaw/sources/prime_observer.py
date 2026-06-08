from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from olivaw.sources.base import SourceHealth, SourcePayload

JSON_PRIORITY = (
    "network_attribution.json",
    "nextdns_summary.json",
    "investigation_index.json",
    "investigation.json",
    "latest.json",
)
SUPPORTED_EXTENSIONS = {".json", ".md", ".txt", ".csv"}
PREVIEW_CHARS = 600


@dataclass(frozen=True)
class PrimeObserverSource:
    directory: Path
    enabled: bool = True
    source_id: str = "prime_observer"
    display_name: str = "Prime Observer"

    def health(self) -> SourceHealth:
        if not self.enabled:
            return SourceHealth(
                source_id=self.source_id,
                display_name=self.display_name,
                status="unavailable",
                message="PrimeObserverSource is disabled in configuration.",
            )
        if not self.directory.exists():
            return SourceHealth(
                source_id=self.source_id,
                display_name=self.display_name,
                status="unavailable",
                message=f"Directory does not exist: {self.directory}",
            )
        if not self.directory.is_dir():
            return SourceHealth(
                source_id=self.source_id,
                display_name=self.display_name,
                status="error",
                message=f"Configured path is not a directory: {self.directory}",
            )

        reports = self._discover_reports()
        if not reports:
            return SourceHealth(
                source_id=self.source_id,
                display_name=self.display_name,
                status="unavailable",
                message=f"No Prime Observer report files found in {self.directory}",
            )
        return SourceHealth(
            source_id=self.source_id,
            display_name=self.display_name,
            status="ok",
            message=f"Found {len(reports)} Prime Observer report file(s).",
        )

    def fetch(self) -> SourcePayload:
        health = self.health()
        if health.status != "ok":
            return self._payload(status=health.status, items=[])

        reports = self._discover_reports()
        items: list[dict[str, object]] = []
        errors: list[str] = []
        for path in reports:
            try:
                item = self._item_from_report(path)
            except Exception as exc:
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
                continue
            if item:
                items.append(item)

        status = "ok" if items else "error"
        payload = self._payload(status=status, items=items)
        if errors:
            payload["errors"] = errors
        return payload

    def _payload(self, status: str, items: list[dict[str, object]]) -> SourcePayload:
        return {
            "source": self.source_id,
            "status": status,
            "root": str(self.directory),
            "count": len(items),
            "items": items,
        }

    def _discover_reports(self) -> list[Path]:
        files = [
            path
            for path in self.directory.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        prioritized: list[Path] = []
        by_name = {path.name: path for path in files}
        for name in JSON_PRIORITY:
            if name in by_name:
                prioritized.append(by_name[name])

        remaining_json = sorted(
            (
                path
                for path in files
                if path.suffix.lower() == ".json" and path not in prioritized
            ),
            key=_mtime,
            reverse=True,
        )
        markdown = sorted(
            (path for path in files if path.suffix.lower() in {".md", ".txt"}),
            key=_mtime,
            reverse=True,
        )
        latest_csv = [path for path in files if path.name == "latest.csv"]
        other_csv = sorted(
            (
                path
                for path in files
                if path.suffix.lower() == ".csv" and path.name != "latest.csv"
            ),
            key=_mtime,
            reverse=True,
        )
        csv_files = [*latest_csv, *other_csv]
        return [*prioritized, *remaining_json, *markdown, *csv_files]

    def _item_from_report(self, path: Path) -> dict[str, object]:
        suffix = path.suffix.lower()
        if suffix == ".json":
            return self._json_item(path)
        if suffix == ".csv":
            return self._csv_item(path)
        return self._text_item(path)

    def _json_item(self, path: Path) -> dict[str, object]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if path.name == "investigation_index.json":
            return _investigation_index_item(path, data)
        if not isinstance(data, dict):
            raise ValueError("JSON report must be an object")

        if path.name == "network_attribution.json":
            return _network_attribution_item(path, data)
        if path.name == "nextdns_summary.json":
            return _nextdns_item(path, data)
        if path.name == "investigation.json":
            return _investigation_item(path, data)
        return _generic_json_item(path, data)

    def _csv_item(self, path: Path) -> dict[str, object]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return _base_item(
                path=path,
                title="Prime Observer CSV report",
                summary="CSV report is empty.",
                report_type="csv",
            )

        latest = rows[-1]
        ts = latest.get("ts") or _modified(path)
        phase = latest.get("phase_label") or "unknown phase"
        host = latest.get("host") or "unknown host"
        p95 = latest.get("p95_ms") or "unknown"
        loss = latest.get("loss_pct") or "unknown"
        return _base_item(
            path=path,
            title="Prime Observer latest samples",
            summary=(
                f"Latest sample {ts}: {phase} to {host}, "
                f"p95 {p95} ms, loss {loss}%."
            ),
            report_date=ts,
            status="ok",
            latest_sample_timestamp=ts,
            latest_sample_phase=phase,
            latest_sample_host=host,
            latest_sample_p95_ms=p95,
            latest_sample_loss_pct=loss,
            findings=[f"{len(rows)} sample row(s) in {path.name}."],
            report_type="csv",
        )

    def _text_item(self, path: Path) -> dict[str, object]:
        preview = _preview(path)
        return _base_item(
            path=path,
            title=path.stem.replace("_", " ").replace("-", " ").title(),
            summary=preview or "Text report is empty.",
            report_date=_modified(path),
            status="ok",
            preview=preview,
            report_type=path.suffix.lower().lstrip("."),
        )


def _network_attribution_item(path: Path, data: dict[str, Any]) -> dict[str, object]:
    current = _dict(data.get("current_attribution"))
    window = _dict(data.get("window_attribution"))
    evidence = _dict(data.get("attribution_evidence"))
    summary = str(
        evidence.get("summary")
        or _first_text(current.get("evidence"))
        or data.get("attribution_label")
        or "Prime Observer network attribution is available."
    )
    status = str(
        current.get("status")
        or data.get("attribution_status")
        or "unknown"
    )
    confidence = str(
        current.get("confidence")
        or data.get("attribution_confidence")
        or "unknown"
    )
    findings = [
        f"Current: {current.get('label') or data.get('attribution_label') or status}",
        f"Confidence: {confidence}",
    ]
    if window:
        findings.append(f"Window: {window.get('label') or window.get('status')}")
    return _base_item(
        path=path,
        title="Prime Observer network attribution",
        summary=summary,
        report_date=str(data.get("generated_at") or _modified(path)),
        status=status,
        current_status=status,
        current_label=str(
            current.get("label")
            or data.get("attribution_label")
            or status
        ),
        current_confidence=confidence,
        window_label=str(window.get("label") or window.get("status") or ""),
        findings=findings,
        report_type="network_attribution",
    )


def _nextdns_item(path: Path, data: dict[str, Any]) -> dict[str, object]:
    summary = _dict(data.get("summary"))
    status = str(data.get("status") or "unknown")
    block_rate = summary.get("block_rate_pct")
    block_rate_fraction = summary.get("dns_block_rate")
    encrypted_rate = summary.get("encrypted_rate_pct")
    encrypted_rate_fraction = summary.get("dns_encrypted_rate")
    top_entities = summary.get("top_entities")
    top_queried_domain = _domain_value(
        summary.get("top_queried_domain"),
        summary.get("top_queried_domains"),
    ) or _top_entity_value(top_entities)
    top_blocked_domain = _domain_value(
        summary.get("top_blocked_domain"),
        summary.get("top_blocked_domains"),
        summary.get("top_blocked"),
    )
    top_resolved_domain = _domain_value(
        summary.get("top_resolved_domain"),
        summary.get("top_resolved_domains"),
        summary.get("top_resolved"),
        summary.get("top_allowed_domain"),
        summary.get("top_allowed_domains"),
    )
    top_entity = _top_entity_value(top_entities)
    top_blocked_category = _first_present(
        summary.get("top_blocked_category"),
        summary.get("top_blocked_reason"),
    )
    findings = []
    if block_rate is not None:
        findings.append(f"Block rate: {block_rate}%")
    if encrypted_rate is not None:
        findings.append(f"Encrypted query rate: {encrypted_rate}%")
    if top_queried_domain:
        findings.append(f"Top queried domain: {top_queried_domain}")
    if top_blocked_domain:
        findings.append(f"Top blocked domain: {top_blocked_domain}")
    if top_resolved_domain:
        findings.append(f"Top resolved domain: {top_resolved_domain}")
    if top_entity and not top_blocked_domain and not top_resolved_domain:
        findings.append(f"Top domain/entity: {top_entity}")
    return _base_item(
        path=path,
        title="Prime Observer NextDNS summary",
        summary="NextDNS summary is available from Prime Observer.",
        report_date=str(data.get("generated_at") or _modified(path)),
        status=status,
        dns_total_queries=summary.get("total_queries"),
        dns_blocked_queries=summary.get("blocked_queries"),
        blocked_query_count=summary.get("blocked_query_count")
        or summary.get("blocked_queries"),
        dns_allowed_queries=summary.get("allowed_queries"),
        dns_encrypted_queries=summary.get("encrypted_query_count")
        or summary.get("encrypted_queries"),
        dns_block_rate_pct=block_rate,
        dns_block_rate=block_rate_fraction,
        dns_encrypted_rate_pct=encrypted_rate,
        dns_encrypted_rate=encrypted_rate_fraction,
        top_queried_domain=top_queried_domain or "unavailable",
        top_queried_domain_count=summary.get("top_queried_domain_count"),
        top_queried_domain_share=summary.get("top_queried_domain_share"),
        top_blocked_domain=top_blocked_domain or "unavailable",
        top_blocked_domain_count=summary.get("top_blocked_domain_count"),
        top_blocked_domain_share=summary.get("top_blocked_domain_share"),
        top_blocked_domain_share_of_blocked=summary.get(
            "top_blocked_domain_share_of_blocked"
        ),
        top_blocked_category=top_blocked_category or "unavailable",
        top_blocked_reason_queries=summary.get("top_blocked_reason_queries"),
        top_resolved_domain=top_resolved_domain or "unavailable",
        top_resolved_domain_count=summary.get("top_resolved_domain_count"),
        top_resolved_domain_share=summary.get("top_resolved_domain_share"),
        top_resolved_domain_share_of_resolved=summary.get(
            "top_resolved_domain_share_of_resolved"
        ),
        top_domain_entity=top_entity or "unavailable",
        findings=findings,
        report_type="nextdns_summary",
    )


def _investigation_item(path: Path, data: dict[str, Any]) -> dict[str, object]:
    event_window = _dict(data.get("event_window"))
    start = event_window.get("start") or _dict(data.get("input")).get("start")
    end = event_window.get("end") or _dict(data.get("input")).get("end")
    navigation = _investigation_navigation(data.get("navigation"))
    neighborhoods = _event_neighborhoods(data.get("event_neighborhoods"))
    events = _investigation_events(data.get("events"))
    summary = "Prime Observer investigation export is available."
    if start and end:
        summary = f"Prime Observer investigation export covers {start} to {end}."
    return _base_item(
        path=path,
        title="Prime Observer investigation export",
        summary=summary,
        report_date=str(data.get("generated_at") or _modified(path)),
        status="ok",
        investigation_start=str(start or ""),
        investigation_end=str(end or ""),
        investigation_context_start=str(event_window.get("context_start") or ""),
        investigation_context_end=str(event_window.get("context_end") or ""),
        investigation_navigation=navigation,
        event_neighborhoods=neighborhoods,
        investigation_events=events,
        report_type="investigation",
    )


def _investigation_index_item(path: Path, data: object) -> dict[str, object]:
    entries = _investigation_index_entries(data)
    return _base_item(
        path=path,
        title="Prime Observer investigation index",
        summary=(
            f"Prime Observer investigation index lists {len(entries)} investigation(s)."
        ),
        report_date=_modified(path),
        status="ok",
        investigation_catalog=entries,
        report_type="investigation_index",
    )


def _generic_json_item(path: Path, data: dict[str, Any]) -> dict[str, object]:
    title = str(data.get("title") or path.stem.replace("_", " ").title())
    summary = str(
        data.get("summary")
        or data.get("description")
        or data.get("status")
        or "Structured Prime Observer report is available."
    )
    return _base_item(
        path=path,
        title=title,
        summary=summary,
        report_date=str(data.get("date") or data.get("generated_at") or _modified(path)),
        status=str(data.get("status") or "ok"),
        report_type="json",
    )


def _base_item(
    *,
    path: Path,
    title: str,
    summary: str,
    report_date: str | None = None,
    status: str | None = None,
    findings: list[str] | None = None,
    preview: str | None = None,
    report_type: str,
    **extra: object,
) -> dict[str, object]:
    item = {
        "title": title,
        "summary": summary,
        "path": path.name,
        "report_date": report_date or _modified(path),
        "status": status or "unknown",
        "findings": findings or [],
        "preview": preview or summary,
        "report_type": report_type,
    }
    item.update(extra)
    return item


def _preview(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return " ".join(text.split())[:PREVIEW_CHARS].strip()


def _modified(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _mtime(path: Path) -> float:
    return path.stat().st_mtime


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _domain_value(*values: object) -> str | None:
    for value in values:
        candidate = _first_domain(value)
        if candidate:
            return candidate
    return None


def _first_domain(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, dict):
        if value.get("name_redacted") is True:
            label = str(value.get("label") or "domain").strip()
            return f"{label} (redacted)"
        for key in ("domain", "name", "label", "host"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
        return None
    if isinstance(value, list):
        for item in value:
            candidate = _first_domain(item)
            if candidate:
                return candidate
    return None


def _top_entity_value(value: object) -> str | None:
    if not isinstance(value, list) or not value:
        return None
    first = _dict(value[0])
    if not first:
        return None
    if first.get("name_redacted") is True:
        label = str(first.get("label") or "entity").strip()
        return f"{label} (redacted by Prime Observer privacy settings)"
    return _first_domain(first)


def _first_present(*values: object) -> object | None:
    for value in values:
        if value:
            return value
    return None


def _first_text(value: object) -> str | None:
    if isinstance(value, list):
        for item in value:
            if item:
                return str(item)
    if value:
        return str(value)
    return None


def _investigation_index_entries(data: object) -> list[dict[str, str]]:
    if isinstance(data, list):
        raw_entries = data
    elif isinstance(data, dict):
        raw_entries = _first_list(
            data.get("investigations"),
            data.get("items"),
            data.get("entries"),
        )
    else:
        raw_entries = []

    entries: list[dict[str, str]] = []
    for raw in raw_entries:
        entry = _dict(raw)
        if not entry:
            continue
        normalized = {
            "id": str(entry.get("id") or "").strip(),
            "title": str(entry.get("title") or entry.get("id") or "Investigation").strip(),
            "created_at": str(entry.get("created_at") or "").strip(),
            "event_count": str(entry.get("event_count") or "").strip(),
            "status": str(entry.get("status") or "").strip(),
            "path": str(entry.get("path") or "").strip(),
        }
        entries.append(normalized)
    return entries


def _investigation_events(value: object) -> list[dict[str, str]]:
    events = []
    for raw in _list(value):
        event = _event_reference(raw)
        if event:
            events.append(event)
    return events


def _investigation_navigation(value: object) -> dict[str, dict[str, str]]:
    navigation = _dict(value)
    if not navigation:
        return {}
    aliases = {
        "first_event": ("first_event", "first"),
        "previous_event": ("previous_event", "previous", "prev"),
        "next_event": ("next_event", "next"),
        "last_event": ("last_event", "last"),
    }
    normalized: dict[str, dict[str, str]] = {}
    for target, keys in aliases.items():
        for key in keys:
            event = _event_reference(navigation.get(key))
            if event:
                normalized[target] = event
                break
    return normalized


def _event_neighborhoods(value: object) -> list[dict[str, object]]:
    neighborhoods: list[dict[str, object]] = []
    if isinstance(value, dict):
        raw_neighborhoods: list[object] = []
        for key, raw in value.items():
            neighborhood = _dict(raw)
            if neighborhood:
                neighborhood.setdefault("event_id", key)
                raw_neighborhoods.append(neighborhood)
            else:
                raw_neighborhoods.append(
                    {"event_id": key, "nearby_events": _list(raw)}
                )
    else:
        raw_neighborhoods = _list(value)

    for raw in raw_neighborhoods:
        neighborhood = _dict(raw)
        if not neighborhood:
            continue
        anchor = _event_reference(
            neighborhood.get("event")
            or neighborhood.get("anchor_event")
            or neighborhood.get("center_event")
            or {
                "id": neighborhood.get("event_id")
                or neighborhood.get("id")
                or neighborhood.get("anchor_event_id")
            }
        )
        nearby_events = [
            event
            for event in (
                _event_reference(raw_event)
                for raw_event in _first_list(
                    neighborhood.get("nearby_events"),
                    neighborhood.get("events"),
                    neighborhood.get("neighbors"),
                )
            )
            if event
        ]
        if anchor or nearby_events:
            neighborhoods.append(
                {
                    "event": anchor,
                    "nearby_events": nearby_events,
                }
            )
    return neighborhoods


def _event_reference(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        return {"id": text, "label": text} if text else {}
    data = _dict(value)
    if not data:
        return {}
    event_id = str(
        data.get("id")
        or data.get("event_id")
        or data.get("anchor")
        or data.get("slug")
        or ""
    ).strip()
    timestamp = str(data.get("ts") or data.get("timestamp") or data.get("time") or "").strip()
    label = str(
        data.get("label")
        or data.get("title")
        or data.get("summary")
        or data.get("kind")
        or event_id
        or timestamp
        or "Event"
    ).strip()
    event = {
        "id": event_id,
        "label": label,
        "timestamp": timestamp,
        "kind": str(data.get("kind") or data.get("type") or "").strip(),
        "status": str(data.get("status") or "").strip(),
        "path": str(data.get("path") or data.get("url") or data.get("href") or "").strip(),
        "anchor": str(data.get("anchor") or data.get("fragment") or "").strip(),
    }
    return {key: value for key, value in event.items() if value}


def _first_list(*values: object) -> list[object]:
    for value in values:
        items = _list(value)
        if items:
            return items
    return []


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []
